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
                experts = extract_json_object(user[user.find("[") :])
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
                return json.dumps({"findings": [], "invented_count": 0})
        if "Alerts:" in user:
            # crude: emit drop for all alert_ids found
            ids = re.findall(r'"alert_id"\s*:\s*"([^"]+)"', user)
            arr = [{"alert_id": i, "decision": "drop", "line": None, "reason": "mock"} for i in ids]
            return json.dumps(arr)
        return json.dumps({"alert_id": "unknown", "decision": "drop", "line": None, "reason": "mock fallback"})


class HuggingFaceLLM(LLMBackend):
    def __init__(
        self,
        model_id: str,
        *,
        adapter_path: str | None = None,
        load_in_4bit: bool = True,
        device_map: str = "auto",
    ):
        self.model_id = model_id
        self.adapter_path = adapter_path
        self.load_in_4bit = load_in_4bit
        self.device_map = device_map
        self._model = None
        self._tokenizer = None

    def _ensure(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        kwargs: dict[str, Any] = {"device_map": self.device_map, "trust_remote_code": True}
        if self.load_in_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
            kwargs["torch_dtype"] = torch.float16
        self._model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
        if self.adapter_path:
            from peft import PeftModel

            self._model = PeftModel.from_pretrained(self._model, self.adapter_path)

    def generate(self, system: str, user: str, *, temperature: float = 0.0, max_new_tokens: int = 512) -> str:
        self._ensure()
        assert self._tokenizer is not None and self._model is not None
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        prompt = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
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


def get_backend(name: str = "mock", **kwargs: Any) -> LLMBackend:
    if name == "mock":
        return MockLLM()
    if name in ("hf", "huggingface"):
        return HuggingFaceLLM(**kwargs)
    raise ValueError(f"Unknown backend: {name}")
