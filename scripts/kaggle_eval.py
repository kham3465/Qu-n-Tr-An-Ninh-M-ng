#!/usr/bin/env python3
"""Kaggle eval after train: council A–E–G + figures. Needs labels_v1 + adapters."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _first_existing(paths: list[Path]) -> Path | None:
    for p in paths:
        if p and p.exists() and (p.is_file() or p.is_dir()):
            return p
    return None


def _find_gold(root: Path) -> Path:
    found = _first_existing(
        [
            root / "data" / "labels" / "labels_v1.jsonl",
            Path("/kaggle/working/labels_v1.jsonl"),
            Path("/kaggle/working/slatriage/data/labels/labels_v1.jsonl"),
        ]
    )
    if found:
        return found
    inp = Path("/kaggle/input")
    if inp.exists():
        hits = sorted(inp.glob("**/labels_v1.jsonl"))
        if hits:
            return hits[0]
    raise FileNotFoundError(
        "missing labels_v1.jsonl — upload it to /kaggle/working or as a dataset"
    )


def _find_adapters(root: Path) -> Path:
    candidates = [
        Path(os.environ.get("SLATRIAGE_ADAPTERS") or ""),
        Path("/kaggle/working/adapters"),
        root / "adapters",
    ]
    for c in candidates:
        if c and (c / "R" / "adapter_config.json").exists():
            return c
    inp = Path("/kaggle/input")
    if inp.exists():
        for p in sorted(inp.glob("**/R/adapter_config.json")):
            return p.parent.parent
    raise FileNotFoundError("missing adapters/R — upload slatriage_adapters.zip and unzip")


def main() -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ["ACCELERATE_MIXED_PRECISION"] = "no"
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    here = Path(__file__).resolve()
    root = here.parents[1]
    src = root / "src"
    if src.exists():
        sys.path.insert(0, str(src))

    gold = _find_gold(root)
    adapters = _find_adapters(root)
    work = Path(os.environ.get("SLATRIAGE_WORK", "/kaggle/working" if Path("/kaggle/working").exists() else str(root)))
    pred_dir = work / "reports"
    pred_dir.mkdir(parents=True, exist_ok=True)
    print(f"gold={gold}")
    print(f"adapters={adapters}")
    print(f"pred_dir={pred_dir}")

    cmd = [
        sys.executable,
        str(root / "scripts" / "after_train.py"),
        "--adapters",
        str(adapters),
        "--pred-dir",
        str(pred_dir),
        "--gold",
        str(gold),
        "--run-hf",
        "--resume",
    ]
    rc = subprocess.call(cmd)
    if rc != 0:
        sys.exit(rc)
    print("EVAL DONE — download reports/ from Output")


if __name__ == "__main__":
    main()
