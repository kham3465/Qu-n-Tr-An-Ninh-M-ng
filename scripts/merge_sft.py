#!/usr/bin/env python3
"""Merge Curated SFT + SolidiFI SFT → data/sft/combined (F1 stays on Curated)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from slatriage.dataset import merge_sft_jsonl
from slatriage.split import read_holdout


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft-dir", type=Path, default=ROOT / "data" / "sft")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--holdout", type=Path, default=ROOT / "data" / "labels" / "holdout_sources.json")
    args = ap.parse_args()
    held = read_holdout(args.holdout)
    print(f"holdout files excluded from train: {len(held)}")
    out = args.out_dir or (args.sft_dir / "combined")
    for role in ("R", "A", "C", "J"):
        n = merge_sft_jsonl(
            [args.sft_dir / f"sft_{role}.jsonl", args.sft_dir / f"sft_solidifi_{role}.jsonl"],
            out / f"sft_{role}.jsonl",
            exclude_sources=held,
        )
        print(f"combined {role}: {n}")


if __name__ == "__main__":
    main()
