from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

from .config import RunConfig
from .env_local import retrieve_local
from .logger import JsonlLogger, read_jsonl, summarize, write_summary
from .metrics import em_f1
from .parsing import parse_final_answer, parse_first_action
from .prompts import make_step0_query
from .services import (
    AdapterRuntime,
    build_generation_service,
    build_memory_service,
    finalize_dist,
    maybe_init_dist,
)

@dataclass
class WorkerContext:
    gpu_tag: str
    out_dir: str
    rank: int = 0
    world_size: int = 1
    dist_init_file: str = ""


@dataclass
class StepTraceRecord:
    episode_id: str
    step_id: int
    raw_model_output: str
    action_type: str
    search_query: Optional[str]
    information: Optional[str]
    snippet: Optional[str]
    mem_updated: bool
    mem_loss: Optional[float]
    time_gen: float
    time_update: float
    forced_terminate: bool
    global_round: int
    local_version_id: int
    adapter_path: Optional[str]
    inference_mode: str
    final_answer_raw_model_output: Optional[str] = None

    def to_dict(self) -> Dict:
        d = {
            "episode_id": self.episode_id,
            "step_id": self.step_id,
            "raw_model_output": self.raw_model_output,
            "action_type": self.action_type,
            "search_query": self.search_query,
            "information": self.information,
            "snippet": self.snippet,
            "mem_updated": self.mem_updated,
            "mem_loss": self.mem_loss,
            "time_gen": self.time_gen,
            "time_update": self.time_update,
            "forced_terminate": self.forced_terminate,
            "global_round": self.global_round,
            "local_version_id": self.local_version_id,
            "adapter_path": self.adapter_path,
            "inference_mode": self.inference_mode,
        }
        if self.final_answer_raw_model_output is not None:
            d["final_answer_raw_model_output"] = self.final_answer_raw_model_output
        return d


def _episode_key(ex: Dict) -> str:
    return str(ex.get("episode_id") or ex.get("_id") or ex.get("id") or "")


def sanitize_query(q: str, max_len: int = 96) -> str:
    q = (q or "").strip()
    q = q.splitlines()[0].strip()
    q = " ".join(q.split())
    if len(q) > max_len:
        q = q[:max_len].rstrip()
    return q


def _sum_all_ranks_int(value: int) -> int:
    if not (dist.is_available() and dist.is_initialized()):
        return int(value)

    if torch.cuda.is_available():
        rank = dist.get_rank()
        device = f"cuda:{rank % torch.cuda.device_count()}"
    else:
        device = "cpu"

    t = torch.tensor([value], dtype=torch.int64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return int(t.item())


def _write_trace(trace_logger: JsonlLogger, record: StepTraceRecord):
    trace_logger.write(record.to_dict())


def _finalize_episode(
    eval_logger: JsonlLogger,
    episode_id: str,
    pred: str,
    gold: str,
    step_count: int,
    search_count: int,
    updates: int,
    forced_terminate: bool,
    global_round: int,
    inference_mode: str,
):
    em, f1 = em_f1(pred, gold)
    eval_logger.write(
        {
            "episode_id": episode_id,
            "pred_answer": pred,
            "gold_answer": gold,
            "em": em,
            "f1": f1,
            "steps": step_count,
            "searches": search_count,
            "updates": updates,
            "forced_terminate": forced_terminate,
            "global_round": global_round,
            "inference_mode": inference_mode,
        }
    )
    eval_logger.flush()


def _write_periodic_summary(out_dir: Path, gpu_tag: str, config: RunConfig, global_round: int):
    eval_records = read_jsonl(out_dir / f"eval_results.{gpu_tag}.jsonl")
    summary = summarize(
        eval_records,
        {
            "gpu_tag": gpu_tag,
            "avg_steps": sum(r.get("steps", 0) for r in eval_records) / max(1, len(eval_records)),
            "avg_searches": sum(r.get("searches", 0) for r in eval_records) / max(1, len(eval_records)),
            "update_rate": sum(1 for r in eval_records if r.get("updates", 0) > 0) / max(1, len(eval_records)),
            "args": config.to_dict(),
            "global_round": global_round,
        },
    )
    write_summary(str(out_dir / f"summary.{gpu_tag}.json"), summary)


def run_worker(config: RunConfig, examples: List[Dict], ctx: WorkerContext):
    out_dir = Path(ctx.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    adapter_root = config.adapter_root_path
    staging_root = config.adapter_staging_path
    publish_root = config.adapter_publish_path
    adapter_root.mkdir(parents=True, exist_ok=True)
    staging_root.mkdir(parents=True, exist_ok=True)
    publish_root.mkdir(parents=True, exist_ok=True)

    trace_logger = JsonlLogger(str(out_dir / f"episode_trace.{ctx.gpu_tag}.jsonl"))
    eval_logger = JsonlLogger(str(out_dir / f"eval_results.{ctx.gpu_tag}.jsonl"))

    maybe_init_dist(ctx.world_size, ctx.rank, ctx.dist_init_file)

    generation_service = build_generation_service(config)
    memory_service = build_memory_service(config, out_dir, ctx.gpu_tag)
    runtime = AdapterRuntime(
        config=config,
        rank=ctx.rank,
        world_size=ctx.world_size,
        publish_root=publish_root,
        staging_root=staging_root,
        memory_service=memory_service,
    )

    if ctx.world_size > 1:
        device = f"cuda:{ctx.rank % torch.cuda.device_count()}" if torch.cuda.is_available() else "cpu"
        local_len = len(examples)
        local_len_t = torch.tensor([local_len], dtype=torch.int64, device=device)
        dist.all_reduce(local_len_t, op=dist.ReduceOp.MAX)
        max_len = int(local_len_t.item())
        if local_len < max_len:
            examples.extend([None] * (max_len - local_len))

    completed = 0

    try:
        for ex in examples:
            if ex is None:
                runtime.run_sync_if_needed(force=False)
                continue

            episode_id = _episode_key(ex)
            question = ex.get("question", "")
            gold = ex.get("answer", "")
            context = ex.get("context", [])

            history: List[Tuple[str, str]] = []
            used_queries = set()

            forced_terminate = False
            pred = "unknown"

            step_count = 0
            search_count = 0
            updates = 0
            use_local_inference = False

            for step_id in range(config.max_steps):
                step_count += 1
                t0 = time.time()

                adapter_ref = runtime.current_adapter_ref(use_local_inference, episode_id)

                if step_id == 0:
                    action_type = "search"
                    query = sanitize_query(make_step0_query(question))
                    raw = "[forced_step0_search]"
                    info_block = ""
                    paras = []
                else:
                    raw = generation_service.generate_action(question, history, adapter_ref)
                    parsed = parse_first_action(raw)
                    action_type = parsed.action_type
                    forced_terminate = forced_terminate or parsed.forced_terminate

                    if parsed.forced_terminate:
                        pred = "unknown"
                        _write_trace(
                            trace_logger,
                            StepTraceRecord(
                                episode_id=episode_id,
                                step_id=step_id,
                                raw_model_output=raw,
                                action_type="finish",
                                search_query=None,
                                information=None,
                                snippet=None,
                                mem_updated=False,
                                mem_loss=None,
                                time_gen=time.time() - t0,
                                time_update=0.0,
                                forced_terminate=True,
                                global_round=runtime.global_round,
                                local_version_id=runtime.local_version_id,
                                adapter_path=adapter_ref.path if adapter_ref else None,
                                inference_mode=generation_service.inference_mode,
                            ),
                        )
                        break

                    if action_type == "finish":
                        answer_raw = generation_service.generate_final_answer(question, history, adapter_ref)
                        answer_parsed = parse_final_answer(answer_raw)
                        pred = (answer_parsed.content or "").strip() or "unknown"
                        forced_terminate = forced_terminate or answer_parsed.forced_terminate

                        _write_trace(
                            trace_logger,
                            StepTraceRecord(
                                episode_id=episode_id,
                                step_id=step_id,
                                raw_model_output=raw,
                                final_answer_raw_model_output=answer_raw,
                                action_type="finish",
                                search_query=None,
                                information=None,
                                snippet=None,
                                mem_updated=False,
                                mem_loss=None,
                                time_gen=time.time() - t0,
                                time_update=0.0,
                                forced_terminate=forced_terminate,
                                global_round=runtime.global_round,
                                local_version_id=runtime.local_version_id,
                                adapter_path=adapter_ref.path if adapter_ref else None,
                                inference_mode=generation_service.inference_mode,
                            ),
                        )
                        break

                    if action_type != "search":
                        action_type = "search"
                        query = sanitize_query(make_step0_query(question))
                    else:
                        query = sanitize_query(parsed.content)

                    info_block = ""
                    paras = []

                if action_type == "search":
                    if len(query) < 3:
                        query = sanitize_query(make_step0_query(question))

                    qnorm = query.lower().strip()
                    search_count += 1

                    is_repeat = qnorm in used_queries
                    used_queries.add(qnorm)

                    if is_repeat:
                        paras = []
                        info_block = (
                            "[System Warning: You have already searched this exact query. "
                            "No new information found. You should finish soon based on existing evidence.]"
                        )
                    else:
                        paras, info_block = retrieve_local(
                            question,
                            query,
                            context,
                            topk=config.topk,
                            max_chars=config.max_chars,
                        )

                    history.append((query, info_block))

                snippet = None
                mem_updated = False
                mem_loss = None
                t_update = 0.0

                if memory_service.enabled and action_type == "search":
                    tu = time.time()
                    next_local_save_dir = runtime.local_stage_dir(episode_id, runtime.local_version_id + 1)
                    mem_updated, snippet, mem_loss = memory_service.maybe_update(
                        gen_backend=generation_service.backend,
                        question=question,
                        info_block=info_block,
                        top1_paragraph=paras[0] if paras else "",
                        adapter_ref=adapter_ref,
                        save_dir=next_local_save_dir,
                    )
                    if mem_updated:
                        runtime.mark_local_update()
                        updates += 1
                        use_local_inference = True
                    t_update = time.time() - tu

                _write_trace(
                    trace_logger,
                    StepTraceRecord(
                        episode_id=episode_id,
                        step_id=step_id,
                        raw_model_output=raw,
                        action_type="search",
                        search_query=query,
                        information=info_block,
                        snippet=snippet,
                        mem_updated=mem_updated,
                        mem_loss=mem_loss,
                        time_gen=time.time() - t0,
                        time_update=t_update,
                        forced_terminate=forced_terminate,
                        global_round=runtime.global_round,
                        local_version_id=runtime.local_version_id,
                        adapter_path=adapter_ref.path if adapter_ref else None,
                        inference_mode=generation_service.inference_mode,
                    ),
                )

            else:
                forced_terminate = True
                pred = "unknown"

            _finalize_episode(
                eval_logger=eval_logger,
                episode_id=episode_id,
                pred=pred,
                gold=gold,
                step_count=step_count,
                search_count=search_count,
                updates=updates,
                forced_terminate=forced_terminate,
                global_round=runtime.global_round,
                inference_mode=generation_service.inference_mode,
            )
            trace_logger.flush()

            runtime.increment_completed_episode()
            runtime.run_sync_if_needed(force=False)

            completed += 1
            if completed % 10 == 0:
                _write_periodic_summary(out_dir, ctx.gpu_tag, config, runtime.global_round)

        if memory_service.enabled and config.sync_every_episodes > 0:
            dirty = 1 if runtime.local_dirty_since_sync else 0
            global_dirty = _sum_all_ranks_int(dirty) if ctx.world_size > 1 else dirty
            if global_dirty > 0:
                runtime.run_sync_if_needed(force=True)

        if memory_service.enabled and config.sync_every_episodes > 0 and ctx.rank == 0:
            final_round_dir = runtime.published_global_round_dir(runtime.global_round)
            if final_round_dir.exists():
                memory_service.merge_and_save_final(str(final_round_dir), str(out_dir / "final_merged_model"))

        _write_periodic_summary(out_dir, ctx.gpu_tag, config, runtime.global_round)

    finally:
        trace_logger.close()
        eval_logger.close()
        generation_service.shutdown()
        finalize_dist(ctx.world_size)