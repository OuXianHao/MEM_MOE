from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

from .config import RunConfig
from .env_local import retrieve_local
from .llm_vllm import AdapterRef, VLLMConfig, VLLMEngine
from .logger import JsonlLogger, read_jsonl, summarize, write_summary
from .mem_injector_ntp import MemConfig, MemInjectorNTP
from .metrics import em_f1
from .parsing import parse_first_action
from .prompts import build_state_prompt, make_step0_query


@dataclass
class WorkerContext:
    gpu_tag: str
    out_dir: str
    rank: int = 0
    world_size: int = 1
    dist_init_file: str = ""


class InferenceClient:
    """
    Thin HTTP client for centralized inference.

    Expected server API:
      POST /generate
      request json:
        {
          "prompt": str,
          "max_tokens": int,
          "temperature": float,
          "stop": List[str] | null,
          "adapter": {
              "name": str,
              "int_id": int,
              "path": str
          } | null
        }

      response json:
        {
          "text": str
        }
    """

    def __init__(self, host: str, port: int, timeout: int = 600):
        self.host = host
        self.port = int(port)
        self.timeout = int(timeout)
        self.base_url = f"http://{host}:{port}"

    @staticmethod
    def _coerce_adapter(
        adapter: Optional[AdapterRef],
        lora_name: Optional[str],
        lora_int_id: Optional[int],
        lora_path: Optional[str],
    ) -> Optional[AdapterRef]:
        if adapter is not None:
            return adapter
        if lora_name is None or lora_int_id is None or lora_path is None:
            return None
        return AdapterRef(name=str(lora_name), int_id=int(lora_int_id), path=str(lora_path))

    def generate(
        self,
        prompt: str,
        max_tokens: int = 64,
        lora_name: Optional[str] = None,
        lora_int_id: Optional[int] = None,
        lora_path: Optional[str] = None,
        adapter: Optional[AdapterRef] = None,
        temperature: Optional[float] = None,
        stop: Optional[List[str]] = None,
    ) -> str:
        adapter = self._coerce_adapter(adapter, lora_name, lora_int_id, lora_path)

        payload = {
            "prompt": prompt,
            "max_tokens": int(max_tokens),
            "temperature": 0.0 if temperature is None else float(temperature),
            "stop": stop,
            "adapter": None
            if adapter is None
            else {
                "name": adapter.name,
                "int_id": adapter.int_id,
                "path": adapter.path,
            },
        }

        req = urllib.request.Request(
            url=f"{self.base_url}/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else str(e)
            raise RuntimeError(f"Inference server HTTPError: {e.code} {detail}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Cannot reach inference server at {self.base_url}. "
                f"Make sure the centralized inference service is running."
            ) from e

        try:
            obj = json.loads(body)
        except Exception as e:
            raise RuntimeError(f"Inference server returned non-JSON response: {body[:300]}") from e

        text = obj.get("text", "")
        return text if isinstance(text, str) else str(text)

    def shutdown(self):
        # no-op for client mode
        return


def _episode_key(ex: Dict) -> str:
    return str(ex.get("episode_id") or ex.get("_id") or ex.get("id") or "")


def _parse_round_num(path: Path, prefix: str) -> Optional[int]:
    if not path.name.startswith(prefix):
        return None
    tail = path.name[len(prefix):]
    return int(tail) if tail.isdigit() else None


def _find_latest_global_round(publish_root: Path) -> int:
    global_dir = publish_root / "global"
    if not global_dir.exists():
        return 0
    rounds: List[int] = []
    for p in global_dir.glob("round_*"):
        rn = _parse_round_num(p, "round_")
        if rn is not None:
            rounds.append(rn)
    return max(rounds) if rounds else 0


def _atomic_torch_save(obj: Dict[str, torch.Tensor], dest_path: Path):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_name = tempfile.mkstemp(prefix=f"{dest_path.name}.tmp.", dir=str(dest_path.parent))
    os.close(tmp_fd)
    Path(tmp_name).unlink(missing_ok=True)
    tmp_path = Path(tmp_name)
    try:
        torch.save(obj, str(tmp_path))
        tmp_path.replace(dest_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _atomic_write_text(text: str, dest_path: Path, encoding: str = "utf-8"):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_name = tempfile.mkstemp(prefix=f"{dest_path.name}.tmp.", dir=str(dest_path.parent))
    os.close(tmp_fd)
    tmp_path = Path(tmp_name)
    try:
        tmp_path.write_text(text, encoding=encoding)
        tmp_path.replace(dest_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _atomic_write_json(obj: Dict, dest_path: Path):
    text = json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    _atomic_write_text(text, dest_path)


def _read_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def sanitize_query(q: str, max_len: int = 96) -> str:
    q = (q or "").strip()
    q = q.splitlines()[0].strip()
    q = " ".join(q.split())
    if len(q) > max_len:
        q = q[:max_len].rstrip()
    return q


def _read_latest_global_meta(publish_root: Path, latest_round: int) -> Dict:
    if latest_round <= 0:
        return {}
    meta_path = publish_root / "global" / f"round_{latest_round}" / "META.json"
    meta = _read_json(meta_path)
    return meta or {}


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


def _average_adapter_states(paths: List[Path], decay_factor: float = 0.98) -> Dict[str, torch.Tensor]:
    summed: Dict[str, torch.Tensor] = {}
    count = 0
    for p in paths:
        state = torch.load(str(p), map_location="cpu")
        if not summed:
            summed = {k: v.float().clone() for k, v in state.items()}
        else:
            if set(state.keys()) != set(summed.keys()):
                raise RuntimeError("Adapter key mismatch during sync aggregation")
            for k, v in state.items():
                summed[k].add_(v.float())
        count += 1

    if count == 0:
        raise RuntimeError("No adapter states found to aggregate")

    for k in summed:
        summed[k].div_(float(count)).mul_(decay_factor)
    return summed


def _cleanup_after_sync(staging_root: Path, publish_root: Path, rank: int, keep_global: int = 2):
    worker_stage_dir = staging_root / f"worker_{rank}"
    if worker_stage_dir.exists():
        for child in worker_stage_dir.iterdir():
            if child.is_dir() and (child.name.startswith("ep") or child.name.startswith("sync_round_")):
                shutil.rmtree(child, ignore_errors=True)

    if rank == 0:
        global_dir = publish_root / "global"
        if global_dir.exists():
            rounds: List[Tuple[int, Path]] = []
            for p in global_dir.glob("round_*"):
                rn = _parse_round_num(p, "round_")
                if rn is not None:
                    rounds.append((rn, p))
            rounds.sort(key=lambda x: x[0])
            if len(rounds) > keep_global:
                for _, p in rounds[: len(rounds) - keep_global]:
                    shutil.rmtree(p, ignore_errors=True)


def _maybe_init_dist(ctx: WorkerContext):
    if ctx.world_size <= 1:
        return
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(
        backend=backend,
        init_method=f"file://{ctx.dist_init_file}",
        rank=ctx.rank,
        world_size=ctx.world_size,
    )


def _finalize_dist(ctx: WorkerContext):
    if ctx.world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()


def _maybe_write_global_pointer(config: RunConfig, adapter_dir: Path, round_id: int, source: str = "global"):
    payload = {
        "source": source,
        "round": int(round_id),
        "adapter_dir": str(adapter_dir),
        "updated_at_unix": int(time.time()),
    }
    _atomic_write_json(payload, config.adapter_pointer_path)


def _build_generation_backend(config: RunConfig):
    if config.uses_centralized_inference():
        return InferenceClient(
            host=config.inference_host,
            port=config.inference_port,
            timeout=600,
        )

    return VLLMEngine(
        VLLMConfig(
            model=config.model,
            temperature=config.temperature,
            gpu_memory_utilization=config.gpu_memory_utilization,
            tensor_parallel_size=config.tensor_parallel_size,
            max_model_len=config.max_model_len,
            dtype=config.dtype,
            max_loras=config.max_loras,
            max_lora_rank=config.max_lora_rank,
        )
    )


def _build_injector(config: RunConfig, out_dir: Path, ctx: WorkerContext) -> Optional[MemInjectorNTP]:
    if not config.train_mem:
        return None

    train_device_map = None
    if config.train_backend.lower() == "qlora":
        train_device_map = "auto"
    elif config.train_gpus:
        train_device_map = "cuda"
    else:
        train_device_map = "cuda" if torch.cuda.is_available() else "cpu"

    return MemInjectorNTP(
        MemConfig(
            base_model=config.model,
            cache_dir=str(out_dir / f"cache_{ctx.gpu_tag}"),
            mem_steps=config.mem_steps,
            mem_lr=config.mem_lr,
            mem_r=config.mem_r,
            mem_alpha=config.mem_alpha,
            mem_dropout=config.mem_dropout,
            mem_max_tokens=config.mem_max_tokens,
            train_backend=config.train_backend,
            quantize_4bit=config.quantize_4bit,
            mem_target_mode=config.mem_target_mode,
            mem_train_router=config.mem_train_router,
            device_map=train_device_map,
            torch_dtype=config.dtype,
        )
    )


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

    _maybe_init_dist(ctx)

    gen_backend = _build_generation_backend(config)
    injector = _build_injector(config, out_dir, ctx)

    global_round = _find_latest_global_round(publish_root)
    latest_meta = _read_latest_global_meta(publish_root, global_round)
    global_completed_offset = int(latest_meta.get("global_completed", 0)) if latest_meta else 0

    if injector is not None and global_round > 0:
        injector.load_adapter(str(publish_root / "global" / f"round_{global_round}"))

    local_completed_episodes = 0
    sync_round = global_round
    next_sync_at = config.sync_every_episodes * (sync_round + 1)

    local_version_id = 0
    local_dirty_since_sync = False

    def _local_stage_dir(episode_id: str, version_id: int) -> Path:
        return staging_root / f"worker_{ctx.rank}" / f"ep{episode_id}" / f"v{version_id}"

    def _sync_state_dir(target_round: int) -> Path:
        return staging_root / f"worker_{ctx.rank}" / f"sync_round_{target_round}"

    def _published_global_round_dir(round_id: int) -> Path:
        return publish_root / "global" / f"round_{round_id}"

    def current_adapter_ref(use_local: bool, episode_id: str) -> Optional[AdapterRef]:
        if injector is None:
            return None

        if use_local and local_version_id > 0:
            p = _local_stage_dir(episode_id, local_version_id)
            if (p / "adapter_config.json").exists():
                return AdapterRef(
                    name=f"local_rank{ctx.rank}_ep{episode_id}",
                    int_id=200000 + local_version_id,
                    path=str(p),
                )

        p = _published_global_round_dir(global_round)
        if global_round > 0 and (p / "adapter_config.json").exists():
            return AdapterRef(
                name="global",
                int_id=100000 + global_round,
                path=str(p),
            )

        return None

    def _compute_global_completed_total() -> int:
        if ctx.world_size > 1:
            since_start = _sum_all_ranks_int(local_completed_episodes)
        else:
            since_start = local_completed_episodes
        return global_completed_offset + since_start

    def run_sync_if_needed(force: bool = False):
        nonlocal sync_round, next_sync_at, global_round, local_dirty_since_sync

        if injector is None or config.sync_every_episodes <= 0:
            return

        global_completed_total = _compute_global_completed_total()
        if not force and global_completed_total < next_sync_at:
            return

        target_round = sync_round + 1

        local_sync_dir = _sync_state_dir(target_round)
        state_path = local_sync_dir / "adapter_state.pt"
        local_state = injector.get_adapter_state_dict()
        _atomic_torch_save(local_state, state_path)

        if ctx.world_size > 1:
            dist.barrier()

        global_round_dir = _published_global_round_dir(target_round)
        done_marker = global_round_dir / "DONE"
        meta_path = global_round_dir / "META.json"

        if ctx.rank == 0:
            paths = [
                staging_root / f"worker_{r}" / f"sync_round_{target_round}" / "adapter_state.pt"
                for r in range(ctx.world_size)
            ]
            avg_state = _average_adapter_states(paths)
            meta = {
                "round": target_round,
                "global_completed": global_completed_total,
                "sync_every_episodes": config.sync_every_episodes,
                "world_size": ctx.world_size,
                "created_at_unix": int(time.time()),
                "source_states": [str(p) for p in paths],
            }
            injector.save_avg_adapter_dir_atomic(avg_state, str(global_round_dir), meta)
            _maybe_write_global_pointer(config, global_round_dir, target_round, source="global")

        if ctx.world_size > 1:
            dist.barrier()

        if not done_marker.exists():
            raise RuntimeError(f"Global adapter round missing DONE marker: {done_marker}")
        if not meta_path.exists():
            raise RuntimeError(f"Global adapter round missing META.json: {meta_path}")

        injector.load_adapter(str(global_round_dir))
        global_round = target_round
        sync_round = target_round
        next_sync_at = config.sync_every_episodes * (sync_round + 1)
        local_dirty_since_sync = False

        _cleanup_after_sync(staging_root, publish_root, ctx.rank, keep_global=2)

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
                run_sync_if_needed(force=False)
                continue

            episode_id = _episode_key(ex)
            question = ex.get("question", "")
            gold = ex.get("answer", "")
            context = ex.get("context", [])

            history = []
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

                adapter_ref = current_adapter_ref(use_local_inference, episode_id)

                if step_id == 0:
                    action_type = "search"
                    query = sanitize_query(make_step0_query(question))
                    raw = "[forced_step0_search]"
                else:
                    prompt = build_state_prompt(question, history)
                    raw = gen_backend.generate(
                        prompt,
                        max_tokens=64,
                        temperature=0.0,
                        adapter=adapter_ref,
                        stop=["</search>", "</answer>"],
                    )

                    raw = (raw or "").strip()
                    if raw.startswith("<search>") and not raw.endswith("</search>"):
                        raw += "</search>"
                    elif raw.startswith("<answer>") and not raw.endswith("</answer>"):
                        raw += "</answer>"

                    parsed = parse_first_action(raw)
                    action_type = parsed.action_type
                    forced_terminate = forced_terminate or parsed.forced_terminate

                    if action_type not in ("search", "answer"):
                        action_type = "search"
                        query = sanitize_query(make_step0_query(question))
                    elif action_type == "answer":
                        pred = (parsed.content or "").strip() or "unknown"
                        trace_logger.write(
                            {
                                "episode_id": episode_id,
                                "step_id": step_id,
                                "raw_model_output": raw,
                                "action_type": "answer",
                                "search_query": None,
                                "information": None,
                                "snippet": None,
                                "mem_updated": False,
                                "mem_loss": None,
                                "time_gen": time.time() - t0,
                                "time_update": 0.0,
                                "forced_terminate": forced_terminate,
                                "global_round": global_round,
                                "local_version_id": local_version_id,
                                "adapter_path": adapter_ref.path if adapter_ref else None,
                                "inference_mode": "remote" if config.uses_centralized_inference() else "local",
                            }
                        )
                        break
                    else:
                        query = sanitize_query(parsed.content)

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
                            "No new information found. You MUST output <answer> in your next turn "
                            "based on what you already know.]"
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

                if injector is not None:
                    tu = time.time()
                    snippet = injector.compress_snippet(
                        gen_backend,
                        question,
                        info_block,
                        paras[0] if paras else "",
                        lora_name=adapter_ref.name if adapter_ref else None,
                        lora_int_id=adapter_ref.int_id if adapter_ref else None,
                        lora_path=adapter_ref.path if adapter_ref else None,
                    )
                    if injector.should_update(question, snippet):
                        ok, mem_loss = injector.train_adapter(snippet)
                        if ok:
                            local_version_id += 1
                            local_dir = _local_stage_dir(episode_id, local_version_id)
                            injector.save_adapter_atomic(str(local_dir))
                            mem_updated = True
                            updates += 1
                            use_local_inference = True
                            local_dirty_since_sync = True
                    t_update = time.time() - tu

                trace_logger.write(
                    {
                        "episode_id": episode_id,
                        "step_id": step_id,
                        "raw_model_output": raw,
                        "action_type": "search",
                        "search_query": query,
                        "information": info_block,
                        "snippet": snippet,
                        "mem_updated": mem_updated,
                        "mem_loss": mem_loss,
                        "time_gen": time.time() - t0,
                        "time_update": t_update,
                        "forced_terminate": forced_terminate,
                        "global_round": global_round,
                        "local_version_id": local_version_id,
                        "adapter_path": adapter_ref.path if adapter_ref else None,
                        "inference_mode": "remote" if config.uses_centralized_inference() else "local",
                    }
                )
            else:
                forced_terminate = True
                pred = "unknown"

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
                    "inference_mode": "remote" if config.uses_centralized_inference() else "local",
                }
            )
            eval_logger.flush()
            trace_logger.flush()

            local_completed_episodes += 1
            run_sync_if_needed(force=False)

            completed += 1
            if completed % 10 == 0:
                eval_records = read_jsonl(out_dir / f"eval_results.{ctx.gpu_tag}.jsonl")
                summary = summarize(
                    eval_records,
                    {
                        "gpu_tag": ctx.gpu_tag,
                        "avg_steps": sum(r.get("steps", 0) for r in eval_records) / max(1, len(eval_records)),
                        "avg_searches": sum(r.get("searches", 0) for r in eval_records) / max(1, len(eval_records)),
                        "update_rate": sum(1 for r in eval_records if r.get("updates", 0) > 0)
                        / max(1, len(eval_records)),
                        "args": config.to_dict(),
                        "global_round": global_round,
                    },
                )
                write_summary(str(out_dir / f"summary.{ctx.gpu_tag}.json"), summary)

        if injector is not None and config.sync_every_episodes > 0:
            dirty = 1 if local_dirty_since_sync else 0
            global_dirty = _sum_all_ranks_int(dirty) if ctx.world_size > 1 else dirty
            if global_dirty > 0:
                run_sync_if_needed(force=True)

        if injector is not None and config.sync_every_episodes > 0 and ctx.rank == 0:
            final_round_dir = _published_global_round_dir(global_round)
            if final_round_dir.exists():
                injector.merge_and_save_final(str(final_round_dir), str(out_dir / "final_merged_model"))

        eval_records = read_jsonl(out_dir / f"eval_results.{ctx.gpu_tag}.jsonl")
        summary = summarize(
            eval_records,
            {
                "gpu_tag": ctx.gpu_tag,
                "avg_steps": sum(r.get("steps", 0) for r in eval_records) / max(1, len(eval_records)),
                "avg_searches": sum(r.get("searches", 0) for r in eval_records) / max(1, len(eval_records)),
                "update_rate": sum(1 for r in eval_records if r.get("updates", 0) > 0) / max(1, len(eval_records)),
                "args": config.to_dict(),
                "global_round": global_round,
            },
        )
        write_summary(str(out_dir / f"summary.{ctx.gpu_tag}.json"), summary)

    finally:
        trace_logger.close()
        eval_logger.close()
        try:
            gen_backend.shutdown()
        except Exception:
            pass
        _finalize_dist(ctx)