#!/usr/bin/env python3
"""Colab / any T4: remaining HF eval only — B, C, D, E_noR, G. No train. No re-run A/E.

Full Curated (343 alerts / 112 files), same set as pred_E. Resume per source if
the tab disconnects. G may OOM on T4 — that is expected; B/C/D/E_noR are required.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REMAINING = "B,C,D,E_noR,G"


def main() -> None:
    ap = argparse.ArgumentParser(description="Run remaining council configs on all Curated sources")
    ap.add_argument("--adapters", type=Path, default=ROOT / "adapters")
    ap.add_argument("--pred-dir", type=Path, default=ROOT / "reports")
    ap.add_argument("--gold", type=Path, default=ROOT / "data" / "labels" / "labels_v1.jsonl")
    ap.add_argument("--configs", default=REMAINING)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ["ACCELERATE_MIXED_PRECISION"] = "no"
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if not args.gold.exists():
        raise SystemExit(f"missing gold {args.gold}")
    if not (args.adapters / "R" / "adapter_config.json").exists():
        raise SystemExit(f"missing adapters at {args.adapters} (need R/A/C/J)")

    args.pred_dir.mkdir(parents=True, exist_ok=True)
    print(f"REMAINING configs={args.configs} gold={args.gold} adapters={args.adapters} pred={args.pred_dir}", flush=True)
    print("no train / no A / no E — full 343/112 — resume incomplete pred_*.jsonl by source", flush=True)

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "after_train.py"),
        "--adapters",
        str(args.adapters),
        "--pred-dir",
        str(args.pred_dir),
        "--gold",
        str(args.gold),
        "--run-hf",
        "--resume",
        "--skip-baseline-a",
        "--configs",
        args.configs,
    ]
    if args.force:
        cmd.append("--force")
    rc = subprocess.call(cmd)
    if rc != 0:
        sys.exit(rc)
    print("REMAINING DONE — each of pred_B/C/D/E_noR.jsonl must be n=343 sources=112", flush=True)


if __name__ == "__main__":
    main()
