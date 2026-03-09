from __future__ import annotations

import json
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, get_peft_model

try:
    from peft import prepare_model_for_kbit_training
except Exception:  # older peft
    prepare_model_for_kbit_training = None

from transformers import AutoModelForCausalLM, AutoTokenizer

from .env_local import keyword_overlap_ratio
from .prompts import build_compression_prompt

_SNIPPET_RE = re.compile(r"<snippet>\s*(.*?)\s*</snippet>", re.DOTALL | re.IGNORECASE)


def _extract_snippet(text: str) -> str:
    """
    Extract content inside <snippet>...</snippet>.
    Returns "" if not found or if snippet is NONE.
    Keeps at most 6 non-empty lines.
    """
    m = _SNIPPET_RE.search(text or "")
    if not m:
        return ""
    s = (m.group(1) or "").strip()
    if not s:
        return ""
    if s.upper() == "NONE":
        return ""
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    lines = lines[:6]
    return "\n".join(lines).strip()


@dataclass
class MemConfig:
    base_model: str
    cache_dir: str
    mem_steps: int = 20
    mem_lr: float = 3e-4
    mem_r: int = 8
    mem_alpha: int = 16
    mem_dropout: float = 0.05
    mem_max_tokens: int = 200

    # ---- new: training backend / MoE target control ----
    train_backend: str = "lora"          # lora | qlora
    quantize_4bit: bool = False
    mem_target_mode: str = "attn_mlp"    # attn | attn_mlp | attn_mlp_router
    mem_train_router: bool = False

    # ---- new: device / load control ----
    device_map: Optional[str] = None     # e.g. "cuda", "auto", "cpu"
    torch_dtype: Optional[str] = None    # "bfloat16" | "float16"
    gradient_checkpointing: bool = False


class MemInjectorNTP:
    def __init__(self, cfg: MemConfig):
        self.cfg = cfg

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.dtype = self._resolve_dtype(cfg.torch_dtype)
        self.device = self._resolve_device(cfg.device_map)

        self.base_model = self._load_base_model()
        self.base_model.eval()

        self.target_modules = self._detect_target_modules()
        self.adapter_cfg = LoraConfig(
            r=self.cfg.mem_r,
            lora_alpha=self.cfg.mem_alpha,
            lora_dropout=self.cfg.mem_dropout,
            target_modules=self.target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.peft_model = get_peft_model(self.base_model, self.adapter_cfg)

        if self.cfg.gradient_checkpointing and hasattr(self.peft_model, "gradient_checkpointing_enable"):
            self.peft_model.gradient_checkpointing_enable()

        self._print_target_summary()

    def _resolve_dtype(self, torch_dtype: Optional[str]):
        if torch_dtype:
            s = str(torch_dtype).strip().lower()
            if s in {"bf16", "bfloat16"}:
                return torch.bfloat16
            if s in {"fp16", "float16", "half"}:
                return torch.float16

        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16

    def _resolve_device(self, device_map: Optional[str]) -> str:
        if device_map:
            return str(device_map)
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _load_base_model(self):
        use_qlora = self.cfg.train_backend.lower() == "qlora" or bool(self.cfg.quantize_4bit)

        common_kwargs = dict(
            pretrained_model_name_or_path=self.cfg.base_model,
            trust_remote_code=True,
        )

        if use_qlora:
            try:
                import bitsandbytes  # noqa: F401
                from transformers import BitsAndBytesConfig
            except Exception as e:
                raise RuntimeError(
                    "QLoRA requested but bitsandbytes / BitsAndBytesConfig is unavailable. "
                    "Install bitsandbytes or switch --train_backend lora."
                ) from e

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self.dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

            model = AutoModelForCausalLM.from_pretrained(
                **common_kwargs,
                torch_dtype=self.dtype,
                device_map="auto" if self.device in {"cuda", "auto"} else self.device,
                quantization_config=bnb_config,
            )
            if prepare_model_for_kbit_training is None:
                raise RuntimeError(
                    "QLoRA requested but prepare_model_for_kbit_training is unavailable in current peft version."
                )
            model = prepare_model_for_kbit_training(model)
            return model

        return AutoModelForCausalLM.from_pretrained(
            **common_kwargs,
            torch_dtype=self.dtype,
            device_map=self.device,
        )

    def _detect_target_modules(self) -> List[str]:
        """
        MoE-aware target module selection.

        Modes:
        - attn: q/k/v/o only
        - attn_mlp: attention + ffn/expert mlp
        - attn_mlp_router: attention + ffn/expert mlp + router-ish modules

        Notes:
        - PEFT target_modules usually expects the leaf module names.
        - We intentionally avoid the old "all linear" behavior here because
          for MoE it becomes too implicit and hard to debug.
        """
        attn_names = {"q_proj", "k_proj", "v_proj", "o_proj"}
        mlp_names = {"gate_proj", "up_proj", "down_proj"}
        router_hint_names = {
            "router",
            "router_proj",
            "gate",
            "router_gate",
            "moe_gate",
            "expert_gate",
        }

        mode = str(self.cfg.mem_target_mode).strip().lower()
        include_router = bool(self.cfg.mem_train_router) or mode == "attn_mlp_router"

        wanted_leaf_names = set(attn_names)
        if mode in {"attn_mlp", "attn_mlp_router"}:
            wanted_leaf_names.update(mlp_names)

        exact_hits = set()
        router_hits = set()

        for name, module in self.base_model.named_modules():
            leaf = name.split(".")[-1]
            low_name = name.lower()
            low_leaf = leaf.lower()

            if leaf == "lm_head":
                continue
            if not isinstance(module, nn.Linear):
                continue

            if low_leaf in wanted_leaf_names:
                exact_hits.add(leaf)
                continue

            if include_router:
                if low_leaf in router_hint_names or any(tok in low_name for tok in router_hint_names):
                    router_hits.add(leaf)

        target_modules = set(exact_hits)
        if include_router:
            target_modules.update(router_hits)

        if not target_modules:
            raise RuntimeError(
                f"No LoRA target modules detected for mode={self.cfg.mem_target_mode}, "
                f"mem_train_router={self.cfg.mem_train_router}."
            )

        return sorted(target_modules)

    def _print_target_summary(self):
        mode = self.cfg.mem_target_mode
        router_flag = self.cfg.mem_train_router
        print(
            "[MemInjectorNTP] target_modules="
            f"{self.target_modules} | mode={mode} | mem_train_router={router_flag} "
            f"| train_backend={self.cfg.train_backend}"
        )

    def _fallback_snippet(self, top1_paragraph: str) -> str:
        pieces = re.split(r"(?<=[.!?])\s+", top1_paragraph)
        snippet = " ".join(pieces[:2]).strip()
        toks = self.tokenizer(snippet, add_special_tokens=False)["input_ids"]
        if len(toks) > self.cfg.mem_max_tokens:
            toks = toks[: self.cfg.mem_max_tokens]
            snippet = self.tokenizer.decode(toks, skip_special_tokens=True)
        return snippet

    def compress_snippet(
        self,
        llm_engine,
        question: str,
        info_block: str,
        top1_paragraph: str,
        lora_name: Optional[str],
        lora_int_id: Optional[int],
        lora_path: Optional[str],
    ) -> str:
        prompt = build_compression_prompt(question, info_block)

        output = llm_engine.generate(
            prompt,
            max_tokens=220,
            temperature=0.0,
            lora_name=lora_name,
            lora_int_id=lora_int_id,
            lora_path=lora_path,
            stop=["</snippet>"],
        )

        raw = (output or "").strip()
        if raw.startswith("<snippet>") and not raw.endswith("</snippet>"):
            raw += "</snippet>"
        snippet = _extract_snippet(raw)

        if not snippet:
            snippet = self._fallback_snippet(top1_paragraph)

        return snippet

    def should_update(self, question: str, snippet: str) -> bool:
        if not snippet.strip():
            return False
        token_count = len(self.tokenizer(snippet, add_special_tokens=False)["input_ids"])
        if token_count == 0 or token_count > self.cfg.mem_max_tokens:
            return False
        if keyword_overlap_ratio(question, snippet) < 0.05:
            return False
        return True

    def _atomic_save_dir(self, dest_dir: Path, save_fn):
        dest_dir.parent.mkdir(parents=True, exist_ok=True)
        tmp_dir = Path(tempfile.mkdtemp(prefix=f"{dest_dir.name}.tmp.", dir=str(dest_dir.parent)))
        try:
            save_fn(tmp_dir)
            if dest_dir.exists():
                shutil.rmtree(dest_dir)
            tmp_dir.rename(dest_dir)
        except Exception:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise

    def save_adapter_atomic(self, adapter_dir: str):
        dest_dir = Path(adapter_dir)
        self._atomic_save_dir(dest_dir, lambda p: self.peft_model.save_pretrained(str(p)))

    def save_avg_adapter_dir_atomic(self, avg_state: Dict[str, torch.Tensor], adapter_dir: str, meta: Dict):
        """
        Atomically publish a PEFT-loadable adapter directory for the averaged LoRA state,
        including META.json and DONE marker in the same atomic directory rename.
        """
        dest_dir = Path(adapter_dir)

        def _save(tmp_dir: Path):
            self.peft_model.save_pretrained(str(tmp_dir))

            with torch.no_grad():
                self.peft_model.load_state_dict(avg_state, strict=False)

            self.peft_model.save_pretrained(str(tmp_dir))

            (tmp_dir / "META.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            (tmp_dir / "DONE").write_text("ok\n", encoding="utf-8")

        self._atomic_save_dir(dest_dir, _save)

    def _training_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def train_adapter(self, snippet: str) -> Tuple[bool, Optional[float]]:
        self.peft_model.train()

        tokens = self.tokenizer(
            snippet,
            return_tensors="pt",
            truncation=True,
            max_length=self.cfg.mem_max_tokens,
        )

        train_device = self._training_device()
        input_ids = tokens["input_ids"].to(train_device)
        attn = tokens["attention_mask"].to(train_device)
        labels = input_ids.clone()

        optim = torch.optim.SGD(
            (p for p in self.peft_model.parameters() if p.requires_grad),
            lr=self.cfg.mem_lr,
            weight_decay=0.01,
        )

        loss_val = None
        for _ in range(self.cfg.mem_steps):
            out = self.peft_model(input_ids=input_ids, attention_mask=attn, labels=labels)
            loss = out.loss
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            loss_val = float(loss.detach().cpu().item())

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return True, loss_val

    def load_adapter(self, adapter_dir: str):
        self.peft_model = PeftModel.from_pretrained(self.base_model, adapter_dir, is_trainable=True)
        if self.cfg.gradient_checkpointing and hasattr(self.peft_model, "gradient_checkpointing_enable"):
            self.peft_model.gradient_checkpointing_enable()

    def get_adapter_state_dict(self) -> Dict[str, torch.Tensor]:
        state = self.peft_model.state_dict()
        out: Dict[str, torch.Tensor] = {}
        for k, v in state.items():
            if "lora_" in k:
                out[k] = v.detach().cpu().clone()
        return out

    def load_adapter_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        own_state = self.peft_model.state_dict()
        for k, v in state_dict.items():
            if k in own_state:
                own_state[k].copy_(v)

    def merge_and_save_final(self, adapter_dir: str, output_dir: str):
        model = AutoModelForCausalLM.from_pretrained(
            self.cfg.base_model,
            torch_dtype=self.dtype,
            device_map="cpu",
            trust_remote_code=True,
        )
        peft_model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=False)
        merged = peft_model.merge_and_unload()
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        merged.save_pretrained(str(out))
        self.tokenizer.save_pretrained(str(out))