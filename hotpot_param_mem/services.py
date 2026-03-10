from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

from .config import RunConfig
from .llm_vllm import AdapterRef, VLLMConfig, VLLMEngine
from .mem_injector_ntp import MemConfig, MemInjectorNTP
from .prompts import build_final_answer_prompt, build_state_prompt


class InferenceClient:
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
        return


class GenerationService:
    def __init__(self, backend, centralized: bool):
        self.backend = backend
        self.centralized = centralized

    @property
    def inference_mode(self) -> str:
        return "remote" if self.centralized else "local"

    def generate_action(
        self,
        question: str,
        history: List[Tuple[str, str]],
        adapter_ref: Optional[AdapterRef],
    ) -> str:
        prompt = build_state_prompt(question, history)
        raw = self.backend.generate(
            prompt,
            max_tokens=64,
            temperature=0.0,
            adapter=adapter_ref,
            stop=["</search>"],  # do NOT stop on <finish/>
        )
        raw = (raw or "").strip()
        if raw.startswith("<search>") and not raw.endswith("</search>"):
            raw += "</search>"
        return raw

    def generate_final_answer(
        self,
        question: str,
        history: List[Tuple[str, str]],
        adapter_ref: Optional[AdapterRef],
    ) -> str:
        prompt = build_final_answer_prompt(question, history)
        raw = self.backend.generate(
            prompt,
            max_tokens=64,
            temperature=0.0,
            adapter=adapter_ref,
            stop=["</answer>"],
        )
        raw = (raw or "").strip()
        if raw.startswith("<answer>") and not raw.endswith("</answer>"):
            raw += "</answer>"
        return raw

    def shutdown(self):
        try:
            self.backend.shutdown()
        except Exception:
            pass


class MemoryUpdateService:
    def __init__(self, injector: Optional[MemInjectorNTP]):
        self.injector = injector

    @property
    def enabled(self) -> bool:
        return self.injector is not None

    def load_adapter(self, adapter_dir: str):
        if self.injector is not None:
            self.injector.load_adapter(adapter_dir)

    def get_adapter_state_dict(self) -> Dict[str, torch.Tensor]:
        if self.injector is None:
            return {}
        return self.injector.get_adapter_state_dict()

    def save_avg_adapter_dir_atomic(self, avg_state: Dict[str, torch.Tensor], adapter_dir: str, meta: Dict):
        if self.injector is None:
            raise RuntimeError("MemoryUpdateService is disabled")
        self.injector.save_avg_adapter_dir_atomic(avg_state, adapter_dir, meta)

    def merge_and_save_final(self, adapter_dir: str, output_dir: str):
        if self.injector is not None:
            self.injector.merge_and_save_final(adapter_dir, output_dir)

    def maybe_update(
        self,
        gen_backend,
        question: str,
        info_block: str,
        top1_paragraph: str,
        adapter_ref: Optional[AdapterRef],
        save_dir: Path,
    ) -> Tuple[bool, Optional[str], Optional[float]]:
        if self.injector is None:
            return False, None, None

        snippet = self.injector.compress_snippet(
            gen_backend,
            question,
            info_block,
            top1_paragraph,
            lora_name=adapter_ref.name if adapter_ref else None,
            lora_int_id=adapter_ref.int_id if adapter_ref else None,
            lora_path=adapter_ref.path if adapter_ref else None,
        )

        if not self.injector.should_update(question, snippet):
            return False, snippet, None

        ok, mem_loss = self.injector.train_adapter(snippet)
        if ok:
            self.injector.save_adapter_atomic(str(save_dir))
            return True, snippet, mem_loss
        return False, snippet, mem_loss


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


def maybe_init_dist(world_size: int, rank: int, dist_init_file: str):
    if world_size <= 1:
        return
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(
        backend=backend,
        init_method=f"file://{dist_init_file}",
        rank=rank,
        world_size=world_size,
    )


def finalize_dist(world_size: int):
    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()


def _maybe_write_global_pointer(config: RunConfig, adapter_dir: Path, round_id: int, source: str = "global"):
    payload = {
        "source": source,
        "round": int(round_id),
        "adapter_dir": str(adapter_dir),
        "updated_at_unix": int(time.time()),
    }
    _atomic_write_json(payload, config.adapter_pointer_path)


class AdapterRuntime:
    def __init__(
        self,
        config: RunConfig,
        rank: int,
        world_size: int,
        publish_root: Path,
        staging_root: Path,
        memory_service: MemoryUpdateService,
    ):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.publish_root = publish_root
        self.staging_root = staging_root
        self.memory_service = memory_service

        self.global_round = _find_latest_global_round(publish_root)
        latest_meta = _read_latest_global_meta(publish_root, self.global_round)
        self.global_completed_offset = int(latest_meta.get("global_completed", 0)) if latest_meta else 0

        self.sync_round = self.global_round
        self.next_sync_at = config.sync_every_episodes * (self.sync_round + 1)

        self.local_version_id = 0
        self.local_dirty_since_sync = False
        self.local_completed_episodes = 0

        if self.memory_service.enabled and self.global_round > 0:
            self.memory_service.load_adapter(str(self.published_global_round_dir(self.global_round)))

    def local_stage_dir(self, episode_id: str, version_id: int) -> Path:
        return self.staging_root / f"worker_{self.rank}" / f"ep{episode_id}" / f"v{version_id}"

    def sync_state_dir(self, target_round: int) -> Path:
        return self.staging_root / f"worker_{self.rank}" / f"sync_round_{target_round}"

    def published_global_round_dir(self, round_id: int) -> Path:
        return self.publish_root / "global" / f"round_{round_id}"

    def current_adapter_ref(self, use_local: bool, episode_id: str) -> Optional[AdapterRef]:
        if not self.memory_service.enabled:
            return None

        if use_local and self.local_version_id > 0:
            p = self.local_stage_dir(episode_id, self.local_version_id)
            if (p / "adapter_config.json").exists():
                return AdapterRef(
                    name=f"local_rank{self.rank}_ep{episode_id}",
                    int_id=200000 + self.local_version_id,
                    path=str(p),
                )

        p = self.published_global_round_dir(self.global_round)
        if self.global_round > 0 and (p / "adapter_config.json").exists():
            return AdapterRef(
                name="global",
                int_id=100000 + self.global_round,
                path=str(p),
            )

        return None

    def mark_local_update(self):
        self.local_version_id += 1
        self.local_dirty_since_sync = True

    def increment_completed_episode(self):
        self.local_completed_episodes += 1

    def compute_global_completed_total(self) -> int:
        if self.world_size > 1:
            since_start = _sum_all_ranks_int(self.local_completed_episodes)
        else:
            since_start = self.local_completed_episodes
        return self.global_completed_offset + since_start

    def run_sync_if_needed(self, force: bool = False):
        if not self.memory_service.enabled or self.config.sync_every_episodes <= 0:
            return

        global_completed_total = self.compute_global_completed_total()
        if not force and global_completed_total < self.next_sync_at:
            return

        target_round = self.sync_round + 1

        local_sync_dir = self.sync_state_dir(target_round)
        state_path = local_sync_dir / "adapter_state.pt"
        local_sync_dir.mkdir(parents=True, exist_ok=True)

        local_state = self.memory_service.get_adapter_state_dict()
        _atomic_torch_save(local_state, state_path)

        if self.world_size > 1:
            dist.barrier()

        global_round_dir = self.published_global_round_dir(target_round)
        done_marker = global_round_dir / "DONE"
        meta_path = global_round_dir / "META.json"

        if self.rank == 0:
            paths = [
                self.staging_root / f"worker_{r}" / f"sync_round_{target_round}" / "adapter_state.pt"
                for r in range(self.world_size)
            ]
            avg_state = _average_adapter_states(paths)
            meta = {
                "round": target_round,
                "global_completed": global_completed_total,
                "sync_every_episodes": self.config.sync_every_episodes,
                "world_size": self.world_size,
                "created_at_unix": int(time.time()),
                "source_states": [str(p) for p in paths],
            }
            self.memory_service.save_avg_adapter_dir_atomic(avg_state, str(global_round_dir), meta)
            _maybe_write_global_pointer(self.config, global_round_dir, target_round, source="global")

        if self.world_size > 1:
            dist.barrier()

        if not done_marker.exists():
            raise RuntimeError(f"Global adapter round missing DONE marker: {done_marker}")
        if not meta_path.exists():
            raise RuntimeError(f"Global adapter round missing META.json: {meta_path}")

        self.memory_service.load_adapter(str(global_round_dir))
        self.global_round = target_round
        self.sync_round = target_round
        self.next_sync_at = self.config.sync_every_episodes * (self.sync_round + 1)
        self.local_dirty_since_sync = False

        _cleanup_after_sync(self.staging_root, self.publish_root, self.rank, keep_global=2)


def build_generation_service(config: RunConfig) -> GenerationService:
    if config.uses_centralized_inference():
        backend = InferenceClient(
            host=config.inference_host,
            port=config.inference_port,
            timeout=600,
        )
    else:
        backend = VLLMEngine(
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
    return GenerationService(backend, config.uses_centralized_inference())


def build_memory_service(config: RunConfig, out_dir: Path, gpu_tag: str) -> MemoryUpdateService:
    if not config.train_mem:
        return MemoryUpdateService(None)

    train_device_map = None
    if config.train_backend.lower() == "qlora":
        train_device_map = "auto"
    elif config.train_gpus:
        train_device_map = "cuda"
    else:
        train_device_map = "cuda" if torch.cuda.is_available() else "cpu"

    injector = MemInjectorNTP(
        MemConfig(
            base_model=config.model,
            cache_dir=str(out_dir / f"cache_{gpu_tag}"),
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
    return MemoryUpdateService(injector)