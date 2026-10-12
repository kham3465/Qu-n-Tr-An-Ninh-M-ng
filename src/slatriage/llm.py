"""LLM backends: mock (CI/dev) and HuggingFace (GPU train/infer)."""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any


def extract_json_object(text: str) -> Any:
    """Parse first JSON object or array from model text."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"Cannot parse JSON from model output: {text[:200]!r}")


class LLMBackend(ABC):
    @abstractmethod
    def generate(self, system: str, user: str, *, temperature: float = 0.0, max_new_tokens: int = 512) -> str:
        raise NotImplementedError


class MockLLM(LLMBackend):
    """Deterministic stub: keep if detector name contains 'reentrancy' else drop."""

    def generate(self, system: str, user: str, *, temperature: float = 0.0, max_new_tokens: int = 512) -> str:
        # Expert single-alert path
        if '"alert_id"' in user and "Alerts:" not in user and "Expert decisions" not in user:
            try:
                # find alert json block
                data = extract_json_object(user[user.find("{") :])
                if isinstance(data, dict) and "alert_id" in data:
                    det = (data.get("detector") or "").lower()
                    decision = "keep" if "reentrancy" in det or "tx-origin" in det else "drop"
                    line = (data.get("lines") or [None])[0]
                    return json.dumps(
                        {
                            "alert_id": data["alert_id"],
                            "decision": decision,
                            "line": line,
                            "reason": "mock heuristic",
                        }
                    )
            except Exception:
                pass
        # Batch / judge
        if "Expert decisions" in user:
            try:
                chunk = user.split("Original alerts:", 1)[0]
                experts = extract_json_object(chunk[chunk.find("[") :])
                if not isinstance(experts, list):
                    experts = [experts] if isinstance(experts, dict) else []
                findings = [
                    {
                        "alert_id": e.get("alert_id"),
                        "detector": e.get("detector"),
                        "line": e.get("line"),
                        "decision": "keep",
                    }
                    for e in experts
                    if isinstance(e, dict) and e.get("decision") == "keep"
                ]
                return json.dumps({"findings": findings, "invented_count": 0})
            except Exception:
                # Non-JSON so JudgeAgent falls back to expert keeps (do not emit empty findings).
                return "PARSE_FAIL"
        if "Alerts:" in user:
            # crude: emit drop for all alert_ids found
            ids = re.findall(r'"alert_id"\s*:\s*"([^"]+)"', user)
            arr = [{"alert_id": i, "decision": "drop", "line": None, "reason": "mock"} for i in ids]
            return json.dumps(arr)
        return json.dumps({"alert_id": "unknown", "decision": "drop", "line": None, "reason": "mock fallback"})


_HF_POOLS: dict[str, "HFModelPool"] = {}


class HFModelPool:
    """One 4-bit base per model_id. Swap LoRA adapters instead of loading 7B four times."""

    def __init__(self, model_id: str, *, load_in_4bit: bool = True, device_map: Any = None):
        self.model_id = model_id
        self.load_in_4bit = load_in_4bit
        self.device_map = device_map if device_map is not None else {"": 0}
        self._model = None
        self._tokenizer = None
        self._peft = None
        self._loaded: set[str] = set()
        self._current: str | None = None

    def _load_base(self) -> None:
        if self._model is not None:
            return
        import os

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
        os.environ["ACCELERATE_MIXED_PRECISION"] = "no"
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        kwargs: dict[str, Any] = {"device_map": self.device_map, "trust_remote_code": True}
        if self.load_in_4bit:
            try:
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
            except TypeError:
                kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
            kwargs["dtype"] = torch.float16
        try:
            self._model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
        except TypeError:
            kwargs.pop("dtype", None)
            kwargs["torch_dtype"] = torch.float16
            try:
                self._model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
            except TypeError:
                kwargs.pop("torch_dtype", None)
                self._model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)

    def select(self, adapter_name: str | None, adapter_path: str | None) -> None:
        self._load_base()
        if not adapter_path:
            if self._peft is not None:
                try:
                    self._peft.disable_adapter()
                except Exception:
                    pass
            self._current = None
            return
        name = adapter_name or "default"
        from peft import PeftModel

        if self._peft is None:
            self._peft = PeftModel.from_pretrained(self._model, adapter_path, adapter_name=name)
            self._model = self._peft
            self._loaded.add(name)
        elif name not in self._loaded:
            self._peft.load_adapter(adapter_path, adapter_name=name)
            self._loaded.add(name)
        if hasattr(self._peft, "enable_adapter_layers"):
            try:
                self._peft.enable_adapter_layers()
            except Exception:
                pass
        self._peft.set_adapter(name)
        self._current = name

    def generate(self, system: str, user: str, *, temperature: float = 0.0, max_new_tokens: int = 512) -> str:
        assert self._tokenizer is not None and self._model is not None
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        prompt = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        try:
            device = next(self._model.parameters()).device
        except StopIteration:
            device = "cpu"
        inputs = self._tokenizer(prompt, return_tensors="pt").to(device)
        do_sample = temperature is not None and temperature > 0
        out = self._model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            pad_token_id=self._tokenizer.pad_token_id,
        )
        gen = out[0][inputs["input_ids"].shape[-1] :]
        return self._tokenizer.decode(gen, skip_special_tokens=True)


def get_hf_pool(model_id: str, **kwargs: Any) -> HFModelPool:
    key = str(model_id)
    pool = _HF_POOLS.get(key)
    if pool is None:
        pool = HFModelPool(model_id, **kwargs)
        _HF_POOLS[key] = pool
    return pool


def release_hf_pools() -> None:
    import gc

    _HF_POOLS.clear()
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


class HuggingFaceLLM(LLMBackend):
    def __init__(
        self,
        model_id: str,
        *,
        adapter_path: str | None = None,
        load_in_4bit: bool = True,
        device_map: Any = None,
        adapter_name: str | None = None,
        pool: HFModelPool | None = None,
    ):
        self.model_id = model_id
        self.adapter_path = adapter_path
        self.adapter_name = adapter_name or (str(adapter_path).replace("\\", "/").rstrip("/").split("/")[-1] if adapter_path else "base")
        self._pool = pool or get_hf_pool(model_id, load_in_4bit=load_in_4bit, device_map=device_map)

    def generate(self, system: str, user: str, *, temperature: float = 0.0, max_new_tokens: int = 512) -> str:
        self._pool.select(self.adapter_name if self.adapter_path else None, self.adapter_path)
        return self._pool.generate(system, user, temperature=temperature, max_new_tokens=max_new_tokens)


def get_backend(name: str = "mock", **kwargs: Any) -> LLMBackend:
    if name == "mock":
        return MockLLM()
    if name in ("hf", "huggingface"):
        return HuggingFaceLLM(**kwargs)
    raise ValueError(f"Unknown backend: {name}")
