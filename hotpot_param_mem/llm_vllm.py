from __future__ import annotations

import gc
import json
import multiprocessing as mp
import socket
import time
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional, Sequence

import torch
import vllm
from vllm import SamplingParams
from vllm.lora.request import LoRARequest

STOP_STRINGS = [
    "</search>",
    "</answer>",
    "\n</search>",
    "\n</answer>",
    " </search>",
    " </answer>",
]

DEFAULT_STOP_STRINGS = [
    "</search>",
    "</answer>",
    "\n</search>",
    "\n</answer>",
    " </search>",
    " </answer>",
]


@dataclass
class VLLMConfig:
    model: str
    temperature: float = 0.0
    gpu_memory_utilization: float = 0.55

    # ---- multi-gpu / long-context / lora serving ----
    tensor_parallel_size: int = 1
    max_model_len: int = 8192
    dtype: str = "bfloat16"
    max_loras: int = 8
    max_lora_rank: int = 64

    # ---- runtime behavior ----
    enforce_eager: bool = True
    distributed_executor_backend: str = "mp"


@dataclass
class AdapterRef:
    """
    A lightweight adapter reference shared across worker / client / server.
    """
    name: str
    int_id: int
    path: str


class VLLMEngine:
    """
    Local vLLM engine wrapper.

    Supports:
    - local direct generate(...)
    - multi-GPU tensor-parallel serving in one process
    - per-request LoRA adapter switching via LoRARequest
    """

    def __init__(self, cfg: VLLMConfig):
        self.cfg = cfg
        self.llm = self._build(cfg.model)

    def _build(self, model_path: str):
        kwargs = dict(
            model=model_path,
            enable_lora=True,
            tensor_parallel_size=max(1, int(self.cfg.tensor_parallel_size)),
            gpu_memory_utilization=float(self.cfg.gpu_memory_utilization),
            distributed_executor_backend=self.cfg.distributed_executor_backend,
            enforce_eager=bool(self.cfg.enforce_eager),
            max_model_len=int(self.cfg.max_model_len),
            max_loras=int(self.cfg.max_loras),
            max_lora_rank=int(self.cfg.max_lora_rank),
            trust_remote_code=True,
        )
        if self.cfg.dtype:
            kwargs["dtype"] = self.cfg.dtype
        return vllm.LLM(**kwargs)

    @staticmethod
    def build_adapter_ref(
        lora_name: Optional[str],
        lora_int_id: Optional[int],
        lora_path: Optional[str],
    ) -> Optional[AdapterRef]:
        if lora_name is None or lora_int_id is None or lora_path is None:
            return None
        return AdapterRef(name=str(lora_name), int_id=int(lora_int_id), path=str(lora_path))

    @staticmethod
    def _build_lora_request(adapter: Optional[AdapterRef]) -> Optional[LoRARequest]:
        if adapter is None:
            return None
        return LoRARequest(
            lora_name=adapter.name,
            lora_int_id=int(adapter.int_id),
            lora_path=adapter.path,
        )

    def generate(
        self,
        prompt: str,
        max_tokens: int = 64,
        lora_name: Optional[str] = None,
        lora_int_id: Optional[int] = None,
        lora_path: Optional[str] = None,
        adapter: Optional[AdapterRef] = None,
        temperature: Optional[float] = None,
        stop: Optional[Sequence[str]] = None,
    ) -> str:
        temp = self.cfg.temperature if temperature is None else float(temperature)

        params = SamplingParams(
            temperature=temp,
            max_tokens=int(max_tokens),
            stop=list(stop) if stop is not None else STOP_STRINGS,
            repetition_penalty=1.1,
            top_p=1.0 if temp == 0 else 0.95,
        )

        if adapter is None:
            adapter = self.build_adapter_ref(
                lora_name=lora_name,
                lora_int_id=lora_int_id,
                lora_path=lora_path,
            )

        lora_request = self._build_lora_request(adapter)
        out = self.llm.generate(
            [prompt],
            sampling_params=params,
            lora_request=lora_request,
        )
        return out[0].outputs[0].text if out and out[0].outputs else ""

    def reload(self, model_path: Optional[str] = None):
        model_path = self.cfg.model if model_path is None else str(model_path)

        del self.llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.llm = self._build(model_path)

    def shutdown(self):
        if hasattr(self, "llm") and self.llm is not None:
            del self.llm
            self.llm = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class InferenceHTTPServer(ThreadingHTTPServer):
    """
    HTTP server that owns exactly one VLLMEngine.
    """
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, server_address, engine: VLLMEngine):
        super().__init__(server_address, InferenceRequestHandler)
        self.engine = engine


class InferenceRequestHandler(BaseHTTPRequestHandler):
    server_version = "HotpotParamMemInference/0.1"

    def log_message(self, format: str, *args):
        # quiet default logging
        return

    def _send_json(self, status_code: int, payload: dict):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send_json(
                200,
                {
                    "ok": True,
                    "model": getattr(self.server.engine.cfg, "model", ""),
                    "tensor_parallel_size": int(getattr(self.server.engine.cfg, "tensor_parallel_size", 1)),
                },
            )
            return

        self._send_json(404, {"ok": False, "error": f"Unknown path: {self.path}"})

    def do_POST(self):
        if self.path != "/generate":
            self._send_json(404, {"ok": False, "error": f"Unknown path: {self.path}"})
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except Exception:
            content_length = 0

        if content_length <= 0:
            self._send_json(400, {"ok": False, "error": "Empty request body"})
            return

        try:
            raw = self.rfile.read(content_length).decode("utf-8")
            obj = json.loads(raw)
        except Exception as e:
            self._send_json(400, {"ok": False, "error": f"Invalid JSON body: {e}"})
            return

        try:
            prompt = str(obj.get("prompt", ""))
            max_tokens = int(obj.get("max_tokens", 64))
            temperature = obj.get("temperature", None)
            stop = obj.get("stop", None)

            adapter_obj = obj.get("adapter", None)
            adapter = None
            if adapter_obj is not None:
                adapter = AdapterRef(
                    name=str(adapter_obj["name"]),
                    int_id=int(adapter_obj["int_id"]),
                    path=str(adapter_obj["path"]),
                )

            text = self.server.engine.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                adapter=adapter,
                temperature=temperature,
                stop=stop,
            )
            self._send_json(200, {"ok": True, "text": text})
        except Exception as e:
            self._send_json(500, {"ok": False, "error": f"{type(e).__name__}: {e}"})


def serve_forever(host: str, port: int, cfg: VLLMConfig):
    """
    Blocking entrypoint for a centralized inference server.
    """
    engine = VLLMEngine(cfg)
    server = InferenceHTTPServer((host, int(port)), engine)

    try:
        print(
            "[llm_vllm] inference server ready | "
            f"host={host} port={port} model={cfg.model} tp={cfg.tensor_parallel_size}"
        )
        server.serve_forever(poll_interval=0.5)
    finally:
        try:
            server.server_close()
        except Exception:
            pass
        engine.shutdown()


def _server_entry(host: str, port: int, cfg_dict: dict):
    cfg = VLLMConfig(**cfg_dict)
    serve_forever(host=host, port=port, cfg=cfg)
    
def _wrapped_server_entry(host: str, port: int, cfg_dict: dict, visible_devices: Optional[str]):
    import os

    if visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
    os.environ["VLLM_USE_RAY"] = "0"
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    _server_entry(host, port, cfg_dict)

def start_inference_server_process(
    host: str,
    port: int,
    cfg: VLLMConfig,
    visible_devices: Optional[str] = None,
) -> mp.Process:
    """
    Launch a separate inference server process.

    visible_devices:
      - None   -> inherit current env
      - "0"    -> single GPU
      - "0,1,2,3,4,5,6,7" -> centralized TP serving across 8 GPUs
    """
    ctx = mp.get_context("spawn")

    p = ctx.Process(
        target=_wrapped_server_entry,
        args=(host, int(port), asdict(cfg), visible_devices),
    )
    p.start()
    return p


def wait_for_server_ready(host: str, port: int, timeout: float = 600.0, poll_interval: float = 1.0):
    """
    Wait until the HTTP inference server accepts /health requests.
    """
    import urllib.error
    import urllib.request

    deadline = time.time() + float(timeout)
    url = f"http://{host}:{int(port)}/health"

    last_err = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                body = resp.read().decode("utf-8", errors="replace")
                obj = json.loads(body)
                if obj.get("ok") is True:
                    return
        except (urllib.error.URLError, urllib.error.HTTPError, socket.timeout, json.JSONDecodeError) as e:
            last_err = e
        time.sleep(float(poll_interval))

    raise RuntimeError(
        f"Inference server did not become ready within {timeout:.1f}s at {url}. "
        f"Last error: {last_err}"
    )