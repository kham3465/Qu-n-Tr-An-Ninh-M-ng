#!/usr/bin/env python3
"""Kaggle eval after train: council A–E–G + figures. Needs labels_v1 + adapters."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import zipfile
from pathlib import Path


def _first_existing(paths: list[Path]) -> Path | None:
    for p in paths:
        if p and p.exists() and (p.is_file() or p.is_dir()):
            return p
    return None


def _is_adapters(d: Path) -> bool:
    return (d / "R" / "adapter_config.json").exists() or (d / "R" / "adapter_model.safetensors").exists()


def _dump_kaggle() -> None:
    for base in (Path("/kaggle/input"), Path("/kaggle/working")):
        if not base.exists():
            print(f"ls {base}: missing")
            continue
        print(f"ls {base}:")
        for p in sorted(base.iterdir())[:40]:
            print(f"  {p.name}{'/' if p.is_dir() else ''}")


def _unzip_adapters(zpath: Path, dest: Path) -> Path | None:
    print(f"unzip {zpath} -> {dest}")
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zpath) as zf:
        names = zf.namelist()
        if not any(n.replace("\\", "/").endswith("R/adapter_config.json") or n.replace("\\", "/").startswith("R/") for n in names):
            print(f"WARN zip has no adapters/R: {zpath.name}")
            return None
        zf.extractall(dest)
    if _is_adapters(dest):
        return dest
    nested = dest / "adapters"
    if _is_adapters(nested):
        return nested
    for p in dest.rglob("R/adapter_config.json"):
        return p.parent.parent
    return None


def _find_gold(root: Path) -> Path:
    searched = [
        root / "data" / "labels" / "labels_v1.jsonl",
        Path("/kaggle/working/labels_v1.jsonl"),
        Path("/kaggle/working/slatriage/data/labels/labels_v1.jsonl"),
        Path("/kaggle/working/results/labels_v1.jsonl"),
    ]
    found = _first_existing(searched)
    if found:
        return found
    inp = Path("/kaggle/input")
    if inp.exists():
        hits = sorted(inp.glob("**/labels_v1.jsonl"))
        if hits:
            return hits[0]
    raise FileNotFoundError(
        "missing labels_v1.jsonl — git pull mới hoặc upload file vào /kaggle/working"
    )


def _find_adapters(root: Path) -> Path:
    work = Path("/kaggle/working") if Path("/kaggle/working").exists() else root
    candidates = [
        Path(os.environ.get("SLATRIAGE_ADAPTERS") or ""),
        work / "adapters",
        work / "results" / "adapters",
        root / "adapters",
    ]
    for c in candidates:
        if c and _is_adapters(c):
            return c
    for base in (Path("/kaggle/input"), work):
        if not base.exists():
            continue
        for p in sorted(base.rglob("R/adapter_config.json")):
            return p.parent.parent
    zips: list[Path] = []
    for base in (Path("/kaggle/input"), work):
        if not base.exists():
            continue
        zips.extend(sorted(base.rglob("slatriage_adapters.zip")))
        zips.extend(sorted(base.glob("*.zip")))
    seen: set[str] = set()
    dest = work / "adapters"
    for z in zips:
        key = str(z.resolve())
        if key in seen:
            continue
        seen.add(key)
        found = _unzip_adapters(z, dest)
        if found:
            return found
    _dump_kaggle()
    raise FileNotFoundError(
        "missing adapters/R — Add Data: zip train cũ (Output) hoặc upload slatriage_adapters.zip"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--configs",
        default="E,B",
        help="HF configs to run. Default E,B fits a 3h Kaggle quota. All: B,C,D,E,E_noR,G",
    )
    ap.add_argument("--force", action="store_true", help="re-run even if pred_B.jsonl already exists")
    args = ap.parse_args()
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
        "--configs",
        args.configs,
    ]
    if args.force:
        cmd.append("--force")
    rc = subprocess.call(cmd)
    if rc != 0:
        sys.exit(rc)
    print("EVAL DONE — download reports/ from Output")


if __name__ == "__main__":
    main()
