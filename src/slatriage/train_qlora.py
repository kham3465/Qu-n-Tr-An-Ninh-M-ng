"""QLoRA trainer that survives TRL / Transformers API drift."""

from __future__ import annotations

import gc
import inspect
import json
from pathlib import Path
from typing import Any

from .dataset import load_sft_jsonl


def require_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. Enable a GPU, then retry.")
    print(
        f"cuda={torch.cuda.get_device_name(0)} "
        f"vram_gb={torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}"
    )


def _filter_kwargs(fn, kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return kwargs
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in params}


def _try_init(cls, attempts: list[dict[str, Any]]):
    last: Exception | None = None
    for kw in attempts:
        try:
            return cls(**_filter_kwargs(cls.__init__, kw))
        except (TypeError, ValueError) as e:
            last = e
            continue
    raise TypeError(f"{cls.__name__} API mismatch: {last}")


def _messages_to_text(tokenizer, messages: list[dict[str, str]]) -> str:
    if getattr(tokenizer, "apply_chat_template", None) and getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        except Exception:
            pass
    parts = []
    for m in messages:
        parts.append(f"<|im_start|>{m.get('role', 'user')}\n{m.get('content', '')}<|im_end|>")
    return "\n".join(parts)


def _build_bnb():
    import torch
    from transformers import BitsAndBytesConfig

    return _try_init(
        BitsAndBytesConfig,
        [
            dict(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_storage=torch.float16,
            ),
            dict(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            ),
            dict(load_in_4bit=True),
        ],
    )


def adapter_complete(out_dir: Path) -> bool:
    if not (out_dir / "train_meta.json").exists():
        return False
    return any(
        (out_dir / name).exists()
        for name in ("adapter_model.safetensors", "adapter_model.bin", "pytorch_model.bin")
    ) or (out_dir / "adapter_config.json").exists()


def latest_checkpoint(out_dir: Path) -> Path | None:
    d = out_dir / "checkpoints"
    if not d.is_dir():
        return None
    ckpts = []
    for p in d.glob("checkpoint-*"):
        tail = p.name.rsplit("-", 1)[-1]
        if tail.isdigit():
            ckpts.append((int(tail), p))
    if not ckpts:
        return None
    ckpts.sort()
    return ckpts[-1][1]


def _make_train_args(
    out_dir: Path,
    max_seq: int,
    epochs: float,
    lr: float,
    seed: int,
    optim: str,
    *,
    save_steps: int = 50,
    has_eval: bool = False,
    fp16: bool = False,
    bf16: bool = False,
):
    common = dict(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=lr,
        logging_steps=1,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=2,
        fp16=fp16,
        bf16=bf16,
        gradient_checkpointing=True,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        seed=seed,
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        optim=optim,
    )
    if has_eval:
        common["eval_steps"] = save_steps
        extras_eval = [
            dict(evaluation_strategy="steps", eval_strategy="steps"),
            dict(eval_strategy="steps"),
            dict(evaluation_strategy="steps"),
            {},
        ]
    else:
        extras_eval = [{}]
    extras = []
    for ev in extras_eval:
        extras.extend(
            [
                {**ev, "max_length": max_seq, "packing": False, "dataset_text_field": "text"},
                {**ev, "max_seq_length": max_seq, "packing": False, "dataset_text_field": "text"},
                {**ev, "max_length": max_seq, "packing": False},
                {**ev, "max_seq_length": max_seq, "packing": False},
                ev,
            ]
        )
    extras.append({})
    try:
        from trl import SFTConfig

        return _try_init(SFTConfig, [{**common, **e} for e in extras])
    except (ImportError, TypeError):
        from transformers import TrainingArguments

        return _try_init(TrainingArguments, [{**common, **e} for e in extras])


def _make_trainer(model, tokenizer, dataset, peft_config, args, eval_dataset=None):
    from trl import SFTTrainer

    bases = [
        dict(
            model=model,
            args=args,
            train_dataset=dataset,
            processing_class=tokenizer,
            dataset_text_field="text",
        ),
        dict(
            model=model,
            args=args,
            train_dataset=dataset,
            processing_class=tokenizer,
        ),
        dict(
            model=model,
            args=args,
            train_dataset=dataset,
            tokenizer=tokenizer,
            dataset_text_field="text",
        ),
        dict(
            model=model,
            args=args,
            train_dataset=dataset,
            tokenizer=tokenizer,
        ),
    ]
    if peft_config is not None:
        bases = [{**b, "peft_config": peft_config} for b in bases] + bases
    attempts = []
    if eval_dataset is not None:
        for b in bases:
            attempts.append({**b, "eval_dataset": eval_dataset})
    attempts.extend(bases)
    return _try_init(SFTTrainer, attempts)


def pin_single_gpu() -> None:
    """One visible GPU; never let Accelerate turn on fp16/bf16 AMP."""
    import os

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ["ACCELERATE_MIXED_PRECISION"] = "no"


def _train_device_map():
    return {"": 0}


def _load_causal(base_model: str, bnb: Any):
    import torch
    from transformers import AutoModelForCausalLM

    pin_single_gpu()
    device_map = _train_device_map()
    common = dict(quantization_config=bnb, device_map=device_map, trust_remote_code=True)
    last = None
    for extra in (dict(dtype=torch.float16), dict(torch_dtype=torch.float16), {}):
        try:
            return AutoModelForCausalLM.from_pretrained(base_model, **common, **extra)
        except TypeError as e:
            last = e
            continue
    raise TypeError(last)


def _is_quant_weight(weight) -> bool:
    return bool(getattr(weight, "is_quantized", False) or getattr(weight, "quant_state", None) is not None)


def _strip_bfloat16(model):
    """Qwen checkpoints keep lm_head/norms in bf16. T4 GradScaler cannot unscale that."""
    import torch

    for _name, param in model.named_parameters():
        if _is_quant_weight(param):
            continue
        if param.dtype == torch.bfloat16:
            param.data = param.data.to(dtype=torch.float32)
    for _name, buf in model.named_buffers():
        if getattr(buf, "dtype", None) == torch.bfloat16:
            buf.data = buf.data.to(dtype=torch.float32)
    if getattr(model, "config", None) is not None:
        model.config.torch_dtype = torch.float32
    return model


def _cast_non_quant_to_fp16(model):
    """Back-compat name: leftover bf16 goes to fp32 (AMP stays off)."""
    return _strip_bfloat16(model)


def _force_fp32_trainable(model) -> int:
    import torch

    n = 0
    for param in model.parameters():
        if param.requires_grad and param.dtype != torch.float32:
            param.data = param.data.to(dtype=torch.float32)
            n += 1
    return n


def _disable_amp(trainer) -> None:
    args = getattr(trainer, "args", None)
    if args is not None:
        args.fp16 = False
        args.bf16 = False
        for key in ("bf16_full_eval", "fp16_full_eval"):
            if hasattr(args, key):
                setattr(args, key, False)
    acc = getattr(trainer, "accelerator", None)
    if acc is not None:
        acc.native_amp = False
        if hasattr(acc, "amp"):
            acc.amp = False
        state = getattr(acc, "state", None)
        if state is not None and hasattr(state, "mixed_precision"):
            try:
                state.mixed_precision = "no"
            except Exception:
                pass
        if hasattr(acc, "scaler"):
            acc.scaler = None
    if hasattr(trainer, "scaler"):
        trainer.scaler = None


def train_one_role(
    *,
    sft_path: Path,
    out_dir: Path,
    base_model: str,
    lora_cfg: dict[str, Any],
    max_seq: int = 2048,
    epochs: float = 2.0,
    lr: float = 2e-4,
    seed: int = 0,
    resume: bool = True,
    force: bool = False,
) -> Path:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import AutoTokenizer

    out_dir.mkdir(parents=True, exist_ok=True)
    if resume and not force and adapter_complete(out_dir):
        print(f"RESUME skip {out_dir.name} (adapter already complete)")
        return out_dir

    rows = load_sft_jsonl(sft_path)
    print(f"role_sft={sft_path.name} n={len(rows)} out={out_dir.name}")

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    texts = [_messages_to_text(tokenizer, row["messages"]) for row in rows]
    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(texts), generator=rng).tolist()
    n_val = min(max(8, int(len(texts) * 0.1)), max(0, len(texts) - 8)) if len(texts) >= 16 else 0
    val_idx = set(perm[:n_val]) if n_val else set()
    train_texts = [t for i, t in enumerate(texts) if i not in val_idx]
    val_texts = [t for i, t in enumerate(texts) if i in val_idx]
    dataset = Dataset.from_dict({"text": train_texts})
    eval_dataset = Dataset.from_dict({"text": val_texts}) if val_texts else None
    steps_est = max(1, (len(train_texts) + 7) // 8 * max(1, int(epochs)))
    save_steps = max(5, steps_est // 6)
    ckpt = latest_checkpoint(out_dir) if resume and not force else None
    if ckpt:
        print(f"RESUME checkpoint {ckpt.name}")

    pin_single_gpu()
    print("QLORA_GUARD amp=off leftover=fp32 (T4 has no BF16 GradScaler)")
    model = _strip_bfloat16(_load_causal(base_model, _build_bnb()))
    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    try:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    except TypeError:
        model = prepare_model_for_kbit_training(model)
    model = _strip_bfloat16(model)

    peft_config = LoraConfig(
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("lora_alpha", 32)),
        lora_dropout=float(lora_cfg.get("lora_dropout", 0.05)),
        target_modules=list(lora_cfg.get("target_modules") or ["q_proj", "v_proj"]),
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_for_trainer: Any = peft_config
    try:
        from peft import PeftModel, get_peft_model

        if not isinstance(model, PeftModel):
            model = get_peft_model(model, peft_config)
        peft_for_trainer = None
    except Exception as e:
        print(f"WARN get_peft_model deferred to trainer: {type(e).__name__}: {e}")
    model = _strip_bfloat16(model)
    n_cast = _force_fp32_trainable(model)
    print(f"trainable_cast_fp32={n_cast}")

    trainer = None
    last_err: Exception | None = None
    for optim in ("paged_adamw_8bit", "adamw_torch"):
        try:
            args = _make_train_args(
                out_dir,
                max_seq,
                epochs,
                lr,
                seed,
                optim,
                save_steps=save_steps,
                has_eval=eval_dataset is not None,
                fp16=False,
                bf16=False,
            )
            trainer = _make_trainer(model, tokenizer, dataset, peft_for_trainer, args, eval_dataset=eval_dataset)
            _disable_amp(trainer)
            if getattr(trainer, "model", None) is not None:
                _strip_bfloat16(trainer.model)
                _force_fp32_trainable(trainer.model)
            if ckpt:
                trainer.train(resume_from_checkpoint=str(ckpt))
            else:
                trainer.train()
            last_err = None
            break
        except Exception as e:
            last_err = e
            print(f"WARN train retry optim={optim} amp=off: {type(e).__name__}: {e}")
            trainer = None
            continue
    if last_err is not None or trainer is None:
        raise RuntimeError(f"QLoRA failed after retries: {last_err}")

    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    logs = []
    try:
        logs = list(trainer.state.log_history or [])
    except Exception:
        logs = []
    loss = [x.get("loss") for x in logs if isinstance(x, dict) and "loss" in x]
    eval_loss = [x.get("eval_loss") for x in logs if isinstance(x, dict) and "eval_loss" in x]
    (out_dir / "train_meta.json").write_text(
        json.dumps(
            {
                "base": base_model,
                "sft": sft_path.name,
                "n": len(rows),
                "n_train": len(train_texts),
                "n_val": len(val_texts),
                "max_seq": max_seq,
                "epochs": epochs,
                "resumed_from": ckpt.name if ckpt else None,
                "loss": loss,
                "eval_loss": eval_loss,
                "logs": logs,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    del trainer, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out_dir


def load_lora_cfg(models_yaml: Path) -> dict[str, Any]:
    import yaml

    cfg = yaml.safe_load(models_yaml.read_text(encoding="utf-8")) or {}
    return cfg.get("lora") or {}


def default_sft_dir(root: Path) -> Path:
    """Prefer curated+SolidiFI combined set when all four roles exist; else Curated-only."""
    combined = root / "data" / "sft" / "combined"
    if all((combined / f"sft_{role}.jsonl").exists() for role in ("R", "A", "C", "J")):
        return combined
    return root / "data" / "sft"


def find_sft(sft_dir: Path, role: str) -> Path:
    candidates = [
        sft_dir / f"sft_{role}.jsonl",
        sft_dir / role / f"sft_{role}.jsonl",
        sft_dir / f"{role}.jsonl",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"missing SFT for {role}")
