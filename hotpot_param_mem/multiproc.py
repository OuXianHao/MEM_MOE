from __future__ import annotations

import glob
import json
import multiprocessing as mp

try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .config import RunConfig
from .data import dedupe_and_sort_by_episode, sort_trace_records
from .llm_vllm import VLLMConfig, start_inference_server_process, wait_for_server_ready
from .logger import read_jsonl, summarize, write_summary
from .runner import WorkerContext, run_worker


def _parse_gpu_list(spec: str) -> List[str]:
    spec = (spec or "").strip()
    if not spec:
        return []
    return [x.strip() for x in spec.split(",") if x.strip()]


def _visible_gpu_ids() -> List[str]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    gpu_ids = [g.strip() for g in visible.split(",") if g.strip()]
    return gpu_ids or ["0"]


def _resolve_infer_gpu_ids(cfg: RunConfig) -> List[str]:
    if cfg.infer_gpus:
        ids = _parse_gpu_list(cfg.infer_gpus)
        if ids:
            return ids
    return _visible_gpu_ids()


def _resolve_train_gpu_ids(cfg: RunConfig) -> List[str]:
    if cfg.train_gpus:
        ids = _parse_gpu_list(cfg.train_gpus)
        if ids:
            return ids
    return []


def _worker_entry(
    visible_devices: str,
    gpu_tag: str,
    cfg_dict: Dict,
    examples: List[Dict],
    rank: int,
    world_size: int,
    dist_init_file: str,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
    os.environ["VLLM_USE_RAY"] = "0"
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    import torch

    visible_list = [x.strip() for x in visible_devices.split(",") if x.strip()]
    if torch.cuda.is_available() and len(visible_list) == 1:
        torch.cuda.set_device(0)

    cfg = RunConfig(**cfg_dict)
    run_worker(
        cfg,
        examples,
        WorkerContext(
            gpu_tag=gpu_tag,
            out_dir=cfg.out,
            rank=rank,
            world_size=world_size,
            dist_init_file=dist_init_file,
        ),
    )


def _find_done_ids(out_dir: Path) -> Set[str]:
    done: Set[str] = set()
    for path in glob.glob(str(out_dir / "eval_results*.jsonl")):
        for rec in read_jsonl(Path(path)):
            if "episode_id" in rec:
                done.add(str(rec["episode_id"]))
    return done


def _merge_jsonl(out_dir: Path, pattern: str, output_name: str):
    records: List[Dict] = []
    for path in glob.glob(str(out_dir / pattern)):
        records.extend(read_jsonl(Path(path)))
    merged = dedupe_and_sort_by_episode(records)
    out_path = out_dir / output_name
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in merged:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return merged


def _merge_trace_jsonl(out_dir: Path, pattern: str, output_name: str):
    records: List[Dict] = []
    for path in glob.glob(str(out_dir / pattern)):
        records.extend(read_jsonl(Path(path)))
    merged = sort_trace_records(records)
    out_path = out_dir / output_name
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in merged:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return merged


def _build_worker_plan(cfg: RunConfig, pending: Sequence[Dict]) -> List[Tuple[str, str, List[Dict]]]:
    """
    Returns a list of worker launch tuples:
        (visible_devices, gpu_tag, examples_for_this_worker)

    Modes:
    1) legacy local mode:
       one worker per visible GPU, each worker sees exactly one GPU.
    2) centralized inference mode:
       workers should NOT hold the TP inference model. They only do control logic
       and optional local training, while generation is sent to the remote server.

       If train_gpus is explicitly set, shard workers across those GPUs.
       Otherwise, default to a single worker to avoid accidentally spawning many
       full training-model copies.
    """
    if cfg.uses_centralized_inference():
        train_gpu_ids = _resolve_train_gpu_ids(cfg)

        if not train_gpu_ids:
            return [("", "client0", list(pending))]

        chunks = [[] for _ in range(len(train_gpu_ids))]
        for i, ex in enumerate(pending):
            chunks[i % len(train_gpu_ids)].append(ex)

        active: List[Tuple[str, str, List[Dict]]] = []
        for wi, gpu_id in enumerate(train_gpu_ids):
            if chunks[wi]:
                active.append((gpu_id, f"train{gpu_id}", chunks[wi]))
        return active

    gpu_ids = _visible_gpu_ids()
    chunks = [[] for _ in range(len(gpu_ids))]
    for i, ex in enumerate(pending):
        chunks[i % len(gpu_ids)].append(ex)

    active: List[Tuple[str, str, List[Dict]]] = []
    for wi, gpu_id in enumerate(gpu_ids):
        if chunks[wi]:
            active.append((gpu_id, f"gpu{gpu_id}", chunks[wi]))
    return active


def _print_launch_plan(cfg: RunConfig, worker_plan: List[Tuple[str, str, List[Dict]]]):
    if cfg.uses_centralized_inference():
        infer_gpu_ids = _resolve_infer_gpu_ids(cfg)
        train_gpu_ids = _resolve_train_gpu_ids(cfg)
        print(
            "[multiproc] centralized_inference=True | "
            f"workers={len(worker_plan)} | "
            f"infer_gpus={infer_gpu_ids} | "
            f"train_gpus={train_gpu_ids or '(cpu / inherited env)'} | "
            f"tensor_parallel_size={cfg.tensor_parallel_size} | "
            f"launch_inference_server={cfg.launch_inference_server}"
        )
    else:
        gpu_ids = [vis for vis, _, _ in worker_plan]
        print(
            "[multiproc] centralized_inference=False | "
            f"workers={len(worker_plan)} | worker_gpus={gpu_ids}"
        )


def _build_server_vllm_config(cfg: RunConfig) -> VLLMConfig:
    return VLLMConfig(
        model=cfg.model,
        temperature=cfg.temperature,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        tensor_parallel_size=cfg.tensor_parallel_size,
        max_model_len=cfg.max_model_len,
        dtype=cfg.dtype,
        max_loras=cfg.max_loras,
        max_lora_rank=cfg.max_lora_rank,
    )


def _maybe_launch_inference_server(cfg: RunConfig) -> Optional[mp.Process]:
    if not cfg.uses_centralized_inference():
        return None
    if not cfg.launch_inference_server:
        return None

    infer_gpu_ids = _resolve_infer_gpu_ids(cfg)
    visible_devices = ",".join(infer_gpu_ids)

    if cfg.tensor_parallel_size != len(infer_gpu_ids):
        print(
            "[multiproc] warning: tensor_parallel_size does not match number of infer_gpus | "
            f"tp={cfg.tensor_parallel_size} infer_gpus={infer_gpu_ids}"
        )

    print(
        "[multiproc] launching centralized inference server | "
        f"host={cfg.inference_host} port={cfg.inference_port} visible_devices={visible_devices}"
    )

    server_proc = start_inference_server_process(
        host=cfg.inference_host,
        port=cfg.inference_port,
        cfg=_build_server_vllm_config(cfg),
        visible_devices=visible_devices,
    )

    wait_for_server_ready(
        host=cfg.inference_host,
        port=cfg.inference_port,
        timeout=600.0,
        poll_interval=1.0,
    )

    print("[multiproc] inference server is ready")
    return server_proc


def _shutdown_process(proc: Optional[mp.Process], name: str):
    if proc is None:
        return
    if not proc.is_alive():
        return

    print(f"[multiproc] shutting down {name}...")
    proc.terminate()
    proc.join(timeout=20)

    if proc.is_alive():
        print(f"[multiproc] force-killing {name}...")
        proc.kill()
        proc.join(timeout=10)


def run_multiproc(cfg: RunConfig, all_examples: Sequence[Dict]):
    out_dir = Path(cfg.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    done_ids = _find_done_ids(out_dir)
    pending = [ex for ex in all_examples if str(ex["episode_id"]) not in done_ids]

    if cfg.limit is not None:
        pending = pending[: cfg.limit]

    if not pending:
        print("No pending episodes. Merging existing outputs.")
        merged_eval = _merge_jsonl(out_dir, "eval_results*.jsonl", "eval_results.jsonl")
        _merge_trace_jsonl(out_dir, "episode_trace*.jsonl", "episode_trace.jsonl")
        write_summary(
            str(out_dir / "summary.json"),
            summarize(merged_eval, {"args": cfg.to_dict(), "newly_completed": 0}),
        )
        return

    worker_plan = _build_worker_plan(cfg, pending)
    if not worker_plan:
        raise RuntimeError("No active workers planned. Check GPU visibility / infer_gpus / train_gpus settings.")

    _print_launch_plan(cfg, worker_plan)

    world_size = len(worker_plan)
    dist_init_file = str((out_dir / "dist_init").resolve())
    Path(dist_init_file).unlink(missing_ok=True)

    ctx = mp.get_context("spawn")
    procs = []
    cfg_dict = cfg.to_dict()
    server_proc: Optional[mp.Process] = None

    try:
        server_proc = _maybe_launch_inference_server(cfg)

        for rank, (visible_devices, gpu_tag, chunk) in enumerate(worker_plan):
            p = ctx.Process(
                target=_worker_entry,
                args=(visible_devices, gpu_tag, cfg_dict, chunk, rank, world_size, dist_init_file),
            )
            p.start()
            procs.append(p)

        for p in procs:
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(f"Worker failed with code {p.exitcode}")

    finally:
        for p in procs:
            if p.is_alive():
                _shutdown_process(p, f"worker(pid={p.pid})")
        _shutdown_process(server_proc, "inference_server")

    merged_trace = _merge_trace_jsonl(out_dir, "episode_trace*.jsonl", "episode_trace.jsonl")
    merged_eval = _merge_jsonl(out_dir, "eval_results*.jsonl", "eval_results.jsonl")

    prev_completed = len(done_ids)
    total_completed = len(merged_eval)
    newly_completed = max(0, total_completed - prev_completed)

    checkpoints = max(1, newly_completed // 10)
    for i in range(checkpoints):
        upto = prev_completed + min(newly_completed, (i + 1) * 10)
        partial = merged_eval[:upto]
        write_summary(
            str(out_dir / "summary.json"),
            summarize(
                partial,
                {
                    "args": cfg.to_dict(),
                    "newly_completed": min(newly_completed, (i + 1) * 10),
                    "trace_count": len(merged_trace),
                },
            ),
        )

    write_summary(
        str(out_dir / "summary.json"),
        summarize(
            merged_eval,
            {"args": cfg.to_dict(), "newly_completed": newly_completed, "trace_count": len(merged_trace)},
        ),
    )