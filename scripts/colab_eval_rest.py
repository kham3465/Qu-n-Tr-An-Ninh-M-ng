#!/usr/bin/env python3
"""Colab/Kaggle: remaining HF configs on the same 112 files as pred_E.

No train. No rerun of complete pred_E. Resume per source if the tab dies.
Order: B, C, D, E_noR, G.

Env:
  SLATRIAGE_ADAPTERS   default /content/adapters then ./adapters
  SLATRIAGE_PRED_DIR   default /content/drive/MyDrive/slatriage_reports
  SLATRIAGE_CONFIGS    default remaining (B,C,D,E_noR,G)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _adapters() -> Path:
    env = os.environ.get("SLATRIAGE_ADAPTERS")
    candidates = []
    if env:
        candidates.append(Path(env))
    candidates.extend(
        [
            Path("/content/adapters"),
            Path("/kaggle/working/adapters"),
            ROOT / "adapters",
        ]
    )
    for d in candidates:
        if (d / "R" / "adapter_config.json").exists():
            return d
    raise SystemExit(
        "MISSING adapters (need R/A/C/J). Unzip slatriage_adapters.zip to /content/adapters"
    )


def _pred_dir() -> Path:
    env = os.environ.get("SLATRIAGE_PRED_DIR")
    if env:
        d = Path(env)
    elif Path("/content/drive/MyDrive").exists():
        d = Path("/content/drive/MyDrive/slatriage_reports")
    elif Path("/kaggle/working").exists():
        d = Path("/kaggle/working/reports")
    else:
        d = ROOT / "reports"
    d.mkdir(parents=True, exist_ok=True)
    return d


def main() -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ["ACCELERATE_MIXED_PRECISION"] = "no"
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    gold = ROOT / "data" / "labels" / "labels_v1.jsonl"
    if not gold.exists():
        raise SystemExit(f"MISSING gold {gold}")
    adapters = _adapters()
    pred_dir = _pred_dir()
    configs = os.environ.get("SLATRIAGE_CONFIGS", "remaining")
    print(f"gold={gold}")
    print(f"adapters={adapters}")
    print(f"pred_dir={pred_dir}")
    print(f"configs={configs}  (full labels_v1 = 343 alerts / 112 files, same as pred_E)")
    print("resume=on  force=off  holdout-only=off  train=off")
    for cfg in [c.strip() for c in configs.replace("remaining", "B,C,D,E_noR,G").split(",") if c.strip()]:
        p = pred_dir / f"pred_{cfg}.jsonl"
        if not p.exists():
            print(f"PREFLIGHT {p.name}: missing (will start 0/112)")
            continue
        n = 0
        srcs: set[str] = set()
        with p.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                n += 1
                try:
                    row = json.loads(line)
                    if row.get("source"):
                        srcs.add(str(row["source"]))
                except json.JSONDecodeError:
                    pass
        print(f"PREFLIGHT {p.name}: n={n} sources={len(srcs)} → resume SKIP {len(srcs)} files")
        if cfg == "G" and n < 50:
            raise SystemExit(
                f"pred_G.jsonl only {n} lines — sai folder, dừng để không ghi đè. "
                "Chọn đúng slatriage_reports trên Drive cũ."
            )

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "after_train.py"),
        "--adapters",
        str(adapters),
        "--pred-dir",
        str(pred_dir),
        "--gold",
        str(gold),
        "--run-hf",
        "--skip-baseline-a",
        "--resume",
        "--configs",
        configs,
    ]
    rc = subprocess.call(cmd)
    if rc != 0:
        sys.exit(rc)
    print("EVAL REST DONE — check pred_B/C/D/E_noR.jsonl each n=343 sources=112")


if __name__ == "__main__":
    main()
