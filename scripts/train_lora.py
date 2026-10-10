#!/usr/bin/env python3
"""Export SFT if needed, then QLoRA one role or all of R/A/C/J."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from slatriage.dataset import export_judge_sft, export_sft_jsonl
from slatriage.train_qlora import (
    adapter_complete,
    default_sft_dir,
    find_sft,
    load_lora_cfg,
    require_cuda,
    train_one_role,
)


def _export_if_needed(role: str, labels: Path | None, sft_path: Path) -> int:
    if sft_path.exists() and sft_path.stat().st_size > 2:
        n = sum(1 for line in sft_path.open(encoding="utf-8") if line.strip())
        print(f"use existing SFT {sft_path.name} ({n} rows)")
        return n
    if labels is None or not labels.exists():
        print(f"ERROR: no SFT at {sft_path} and no --data labels")
        sys.exit(1)
    if role == "J":
        n = export_judge_sft(labels, sft_path)
    else:
        n = export_sft_jsonl(labels, sft_path, family=role)
    print(f"exported {n} rows -> {sft_path}")
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", choices=["R", "A", "C", "J"], default=None)
    ap.add_argument("--all", action="store_true", help="Train R then A then C then J")
    ap.add_argument("--data", type=Path, default=None, help="labels_v1.jsonl (only if SFT missing)")
    ap.add_argument("--sft", type=Path, default=None, help="one SFT jsonl")
    ap.add_argument("--sft-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--base", default=os.environ.get("SLATRIAGE_BASE", "Qwen/Qwen2.5-Coder-7B-Instruct"))
    ap.add_argument("--models-yaml", type=Path, default=ROOT / "configs" / "models.yaml")
    ap.add_argument("--max-seq", type=int, default=2048)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--do-train", action="store_true")
    ap.add_argument("--resume", action="store_true", default=True, help="skip finished roles / continue checkpoint")
    ap.add_argument("--no-resume", action="store_false", dest="resume")
    ap.add_argument("--force", action="store_true", help="retrain even if adapter exists")
    args = ap.parse_args()

    if args.all:
        roles = ["R", "A", "C", "J"]
    elif args.role:
        roles = [args.role]
    else:
        print("ERROR: pass --role R|A|C|J or --all")
        sys.exit(1)

    if args.sft_dir is None:
        args.sft_dir = default_sft_dir(ROOT)
        print(f"sft_dir={args.sft_dir}")
    out_root = args.out_dir or args.out or (ROOT / "adapters")
    lora_cfg = load_lora_cfg(args.models_yaml) if args.models_yaml.exists() else {}

    prepared = []
    for role in roles:
        if args.sft and not args.all:
            sft_path = args.sft
        else:
            sft_path = args.sft_dir / f"sft_{role}.jsonl"
            try:
                sft_path = find_sft(args.sft_dir, role)
            except FileNotFoundError:
                pass
        n = _export_if_needed(role, args.data, sft_path)
        if n == 0:
            print(f"ERROR: zero rows for {role}")
            sys.exit(1)
        prepared.append((role, sft_path, n))

    if args.prepare_only and not args.do_train:
        print("prepare-only done")
        return
    if not args.do_train:
        print("SFT ready. Add --do-train to start QLoRA.")
        return

    jobs = []
    for role, sft_path, n in prepared:
        dest = out_root / role if args.all or args.out_dir or args.out is None else Path(args.out)
        if args.all:
            dest = out_root / role
        elif args.out and not args.all:
            dest = args.out
        else:
            dest = out_root / role
        if args.resume and not args.force and adapter_complete(dest):
            print(f"RESUME skip {role} (adapter already complete)")
            continue
        jobs.append((role, sft_path, n, dest))

    if not jobs:
        print("DONE all roles already complete")
        return

    require_cuda()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        os.environ.setdefault("HF_TOKEN", token)

    for role, sft_path, n, dest in jobs:
        print(f"TRAIN {role} n={n} base={args.base} resume={args.resume} force={args.force}")
        train_one_role(
            sft_path=sft_path,
            out_dir=dest,
            base_model=args.base,
            lora_cfg=lora_cfg,
            max_seq=args.max_seq,
            epochs=args.epochs,
            lr=args.lr,
            seed=args.seed,
            resume=args.resume,
            force=args.force,
        )
        print(f"SAVED {role} -> {dest.name}")
    print("DONE roles=" + ",".join(roles))


if __name__ == "__main__":
    main()
