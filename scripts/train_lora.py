#!/usr/bin/env python3
"""Export SFT JSONL then optionally train one LoRA with TRL+PEFT."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from slatriage.dataset import export_judge_sft, export_sft_jsonl


def train_with_trl(sft_path: Path, out_dir: Path, base_model: str, models_yaml: Path) -> None:
    import yaml
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer
    import torch

    cfg = yaml.safe_load(models_yaml.read_text(encoding="utf-8"))
    lora_cfg = cfg.get("lora", {})
    bnb = BitsAndBytesConfig(load_in_4bit=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    peft_config = LoraConfig(
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("lora_alpha", 32)),
        lora_dropout=float(lora_cfg.get("lora_dropout", 0.05)),
        target_modules=lora_cfg.get("target_modules", ["q_proj", "v_proj"]),
        bias="none",
        task_type="CAUSAL_LM",
    )
    ds = load_dataset("json", data_files=str(sft_path), split="train")

    def formatting(example):
        msgs = example["messages"]
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)

    args_sft = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        logging_steps=5,
        save_steps=50,
        fp16=True,
        max_seq_length=2048,
        packing=False,
    )
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        peft_config=peft_config,
        formatting_func=formatting,
        args=args_sft,
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    (out_dir / "train_meta.json").write_text(
        json.dumps({"base": base_model, "sft": str(sft_path)}, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", choices=["R", "A", "C", "J"], required=True)
    ap.add_argument("--data", type=Path, required=True, help="labels.jsonl")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--base", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    ap.add_argument("--models-yaml", type=Path, default=ROOT / "configs" / "models.yaml")
    ap.add_argument("--prepare-only", action="store_true", help="Only export SFT jsonl")
    ap.add_argument("--do-train", action="store_true", help="Run QLoRA (needs GPU)")
    args = ap.parse_args()

    out = args.out or (ROOT / "adapters" / args.role)
    out.mkdir(parents=True, exist_ok=True)
    sft_path = out / f"sft_{args.role}.jsonl"

    if args.role == "J":
        n = export_judge_sft(args.data, sft_path)
    else:
        n = export_sft_jsonl(args.data, sft_path, family=args.role)
    print(f"Prepared {n} SFT rows → {sft_path}")

    if args.prepare_only and not args.do_train:
        (out / "README.txt").write_text(
            f"SFT prepared for LoRA-{args.role}. Run with --do-train when labels frozen.\n",
            encoding="utf-8",
        )
        return

    if args.do_train:
        if n == 0:
            print("No rows to train. Abort.")
            sys.exit(1)
        print(f"Training LoRA-{args.role} on {args.base} …")
        train_with_trl(sft_path, out, args.base, args.models_yaml)
        print(f"Saved adapter → {out}")
    else:
        print("Tip: add --do-train to start QLoRA, or --prepare-only to stop after SFT export.")
        (out / "README.txt").write_text(
            f"Placeholder/SFT for LoRA-{args.role}. Use --do-train after LABELS_V1_FROZEN.\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
