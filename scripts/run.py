#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]  # ~/MEM3
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hotpot_param_mem.config import RunConfig
from hotpot_param_mem.data import load_hotpot_examples
from hotpot_param_mem.multiproc import run_multiproc


def _str2bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def parse_args():
    p = argparse.ArgumentParser()

    # ---- core io ----
    p.add_argument("--data", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)

    # ---- task / decoding ----
    p.add_argument("--max_steps", type=int, default=7)
    p.add_argument("--topk", type=int, default=3)
    p.add_argument("--max_chars", type=int, default=1200)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=None)

    # ---- memory / online update ----
    p.add_argument("--train_mem", action="store_true")
    p.add_argument("--mem_steps", type=int, default=10)
    p.add_argument("--mem_lr", type=float, default=1e-4)
    p.add_argument("--mem_r", type=int, default=3)
    p.add_argument("--mem_alpha", type=int, default=16)
    p.add_argument("--mem_dropout", type=float, default=0.1)
    p.add_argument("--mem_max_tokens", type=int, default=200)
    p.add_argument("--sync_every_episodes", type=int, default=0)

    # ---- centralized inference / MoE serving ----
    p.add_argument("--centralized_inference", action="store_true")
    p.add_argument("--inference_host", type=str, default="127.0.0.1")
    p.add_argument("--inference_port", type=int, default=18000)
    p.add_argument("--launch_inference_server", action="store_true")
    p.add_argument("--inference_backend", type=str, default="vllm")
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.55)
    p.add_argument("--max_model_len", type=int, default=8192)
    p.add_argument("--dtype", type=str, default="bfloat16")
    p.add_argument("--max_loras", type=int, default=8)
    p.add_argument("--max_lora_rank", type=int, default=64)

    # ---- adapter publication / versioning ----
    p.add_argument("--adapter_root", type=str, default=None)
    p.add_argument("--adapter_publish_subdir", type=str, default="published")
    p.add_argument("--adapter_staging_subdir", type=str, default="staging")
    p.add_argument("--adapter_pointer_file", type=str, default="CURRENT_ADAPTER.json")

    # ---- memory training backend / MoE target control ----
    p.add_argument("--train_backend", type=str, default="lora", choices=["lora", "qlora"])
    p.add_argument("--quantize_4bit", action="store_true")
    p.add_argument(
        "--mem_target_mode",
        type=str,
        default="attn_mlp",
        choices=["attn", "attn_mlp", "attn_mlp_router"],
    )
    p.add_argument("--mem_train_router", action="store_true")

    # ---- resource split ----
    p.add_argument("--infer_gpus", type=str, default="")
    p.add_argument("--train_gpus", type=str, default="")

    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)

    cfg = RunConfig(
        # ---- core io ----
        data=args.data,
        model=args.model,
        out=args.out,

        # ---- task / decoding ----
        max_steps=args.max_steps,
        topk=args.topk,
        max_chars=args.max_chars,
        temperature=args.temperature,
        seed=args.seed,
        limit=args.limit,

        # ---- memory / online update ----
        train_mem=args.train_mem,
        mem_steps=args.mem_steps,
        mem_lr=args.mem_lr,
        mem_r=args.mem_r,
        mem_alpha=args.mem_alpha,
        mem_dropout=args.mem_dropout,
        mem_max_tokens=args.mem_max_tokens,
        sync_every_episodes=args.sync_every_episodes,

        # ---- centralized inference / MoE serving ----
        centralized_inference=args.centralized_inference,
        inference_host=args.inference_host,
        inference_port=args.inference_port,
        launch_inference_server=args.launch_inference_server,
        inference_backend=args.inference_backend,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
        max_loras=args.max_loras,
        max_lora_rank=args.max_lora_rank,

        # ---- adapter publication / versioning ----
        adapter_root=args.adapter_root,
        adapter_publish_subdir=args.adapter_publish_subdir,
        adapter_staging_subdir=args.adapter_staging_subdir,
        adapter_pointer_file=args.adapter_pointer_file,

        # ---- memory training backend / MoE target control ----
        train_backend=args.train_backend,
        quantize_4bit=args.quantize_4bit,
        mem_target_mode=args.mem_target_mode,
        mem_train_router=args.mem_train_router,

        # ---- resource split ----
        infer_gpus=args.infer_gpus,
        train_gpus=args.train_gpus,
    )

    examples = load_hotpot_examples(cfg.data)
    run_multiproc(cfg, examples)