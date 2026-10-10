#!/usr/bin/env python3
"""Kaggle one-shot: train LoRA R/A/C/J from git-cloned SFT (combined), then zip adapters."""

from __future__ import annotations

import os
import subprocess
import sys
import zipfile
from pathlib import Path


def _root() -> Path:
    here = Path(__file__).resolve()
    repo = here.parents[1]
    if (repo / "data" / "sft" / "sft_R.jsonl").exists() or (repo / "src" / "slatriage").exists():
        return repo
    inp = Path("/kaggle/input")
    if inp.exists():
        for p in sorted(inp.glob("*")):
            if (p / "data" / "sft" / "sft_R.jsonl").exists() or (p / "sft_R.jsonl").exists():
                return p
            if (p / "slatriage" / "data" / "sft" / "sft_R.jsonl").exists():
                return p / "slatriage"
    return repo


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    root = _root()
    src = root / "src"
    if src.exists():
        sys.path.insert(0, str(src))
    work = Path(os.environ.get("SLATRIAGE_WORK", "/kaggle/working" if Path("/kaggle/working").exists() else str(root)))
    try:
        from slatriage.train_qlora import default_sft_dir

        sft_dir = default_sft_dir(root)
    except Exception:
        sft_dir = root / "data" / "sft" / "combined"
        if not (sft_dir / "sft_R.jsonl").exists():
            sft_dir = root / "data" / "sft"
    if not (sft_dir / "sft_R.jsonl").exists():
        sft_dir = root
    out_dir = work / "adapters"
    models_yaml = root / "configs" / "models.yaml"
    train_py = root / "scripts" / "train_lora.py"
    if not train_py.exists():
        train_py = Path(__file__).resolve().parent / "train_lora.py"

    print(f"root={root.name}")
    print(f"sft_dir={sft_dir.name}")
    print(f"out_dir={out_dir.name}")
    cmd = [
        sys.executable,
        str(train_py),
        "--all",
        "--sft-dir",
        str(sft_dir),
        "--out-dir",
        str(out_dir),
        "--models-yaml",
        str(models_yaml if models_yaml.exists() else sft_dir / "models.yaml"),
        "--do-train",
        "--resume",
    ]
    pre = root / "scripts" / "preflight.py"
    if pre.exists():
        pre_cmd = [
            sys.executable,
            str(pre),
            "--sft-dir",
            str(sft_dir),
            "--require-cuda",
            "--out",
            str(work / "preflight.json"),
        ]
        print("preflight...", flush=True)
        if subprocess.call(pre_cmd) != 0:
            sys.exit(1)
    print("run train", flush=True)
    rc = subprocess.call(cmd)
    if rc != 0:
        sys.exit(rc)

    zpath = work / "slatriage_adapters.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
        for role in ("R", "A", "C", "J"):
            d = out_dir / role
            if not d.exists():
                print(f"WARN missing {d}")
                continue
            for f in d.rglob("*"):
                if f.is_file() and "checkpoints" not in f.parts:
                    zf.write(f, f.relative_to(out_dir).as_posix())
    print(f"zip={zpath.name} size={zpath.stat().st_size}")
    after = root / "scripts" / "after_train.py"
    if after.exists():
        print("build figures...", flush=True)
        subprocess.call(
            [
                sys.executable,
                str(after),
                "--adapters",
                str(out_dir),
                "--pred-dir",
                str(work / "reports") if (work / "reports").exists() else str(root / "reports"),
                "--skip-baseline-a",
            ]
        )
    print("Download slatriage_adapters.zip from Output.")


if __name__ == "__main__":
    main()
