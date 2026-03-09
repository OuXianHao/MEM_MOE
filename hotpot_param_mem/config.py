from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


@dataclass
class RunConfig:
    # ---- core io ----
    data: str
    model: str
    out: str

    # ---- task / decoding ----
    max_steps: int = 7
    topk: int = 3
    max_chars: int = 1200
    temperature: float = 0.0
    seed: int = 42
    limit: Optional[int] = None

    # ---- memory / online update ----
    train_mem: bool = False
    mem_steps: int = 10
    mem_lr: float = 1e-4
    mem_r: int = 3
    mem_alpha: int = 16
    mem_dropout: float = 0.1
    mem_max_tokens: int = 200
    sync_every_episodes: int = 0

    # ---- centralized inference / MoE serving ----
    centralized_inference: bool = False
    inference_host: str = "127.0.0.1"
    inference_port: int = 18000
    launch_inference_server: bool = False
    inference_backend: str = "vllm"   # reserved for future extension
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.55
    max_model_len: int = 8192
    dtype: str = "bfloat16"
    max_loras: int = 8
    max_lora_rank: int = 64

    # ---- adapter publication / versioning ----
    adapter_root: Optional[str] = None
    adapter_publish_subdir: str = "published"
    adapter_staging_subdir: str = "staging"
    adapter_pointer_file: str = "CURRENT_ADAPTER.json"

    # ---- memory training backend / MoE target control ----
    train_backend: str = "lora"       # lora | qlora
    quantize_4bit: bool = False
    mem_target_mode: str = "attn_mlp" # attn | attn_mlp | attn_mlp_router
    mem_train_router: bool = False

    # ---- resource split (reserved; used by later files) ----
    infer_gpus: str = ""
    train_gpus: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def out_path(self) -> Path:
        return Path(self.out)

    @property
    def adapter_root_path(self) -> Path:
        if self.adapter_root:
            return Path(self.adapter_root)
        return self.out_path / "adapters"

    @property
    def adapter_publish_path(self) -> Path:
        return self.adapter_root_path / self.adapter_publish_subdir

    @property
    def adapter_staging_path(self) -> Path:
        return self.adapter_root_path / self.adapter_staging_subdir

    @property
    def adapter_pointer_path(self) -> Path:
        return self.adapter_root_path / self.adapter_pointer_file

    def uses_centralized_inference(self) -> bool:
        return bool(self.centralized_inference)

    def uses_distributed_serving(self) -> bool:
        return int(self.tensor_parallel_size) > 1

    def uses_online_memory(self) -> bool:
        return bool(self.train_mem)

    def uses_quantized_training(self) -> bool:
        return self.train_backend.lower() == "qlora" or bool(self.quantize_4bit)