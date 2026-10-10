#!/usr/bin/env python3
"""Run on the GPU machine BEFORE train. Exit 1 if anything required is missing."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from slatriage.dataset import load_sft_jsonl
from slatriage.train_qlora import default_sft_dir, find_sft, load_lora_cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft-dir", type=Path, default=None)
    ap.add_argument("--models-yaml", type=Path, default=ROOT / "configs" / "models.yaml")
    ap.add_argument("--require-cuda", action="store_true")
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "preflight.json")
    args = ap.parse_args()

    if args.sft_dir is None:
        args.sft_dir = default_sft_dir(ROOT)
    report: dict = {"ok": True, "errors": [], "warns": [], "sft": {}, "pkgs": {}, "sft_dir": str(args.sft_dir)}

    for role in ("R", "A", "C", "J"):
        try:
            path = find_sft(args.sft_dir, role)
            rows = load_sft_jsonl(path)
            splits = {}
            labels = {}
            bad_fam = 0
            for row in rows:
                splits[row.get("split") or "curated"] = splits.get(row.get("split") or "curated", 0) + 1
                lab = row.get("label")
                if lab:
                    labels[lab] = labels.get(lab, 0) + 1
                if role != "J" and row.get("family") != role:
                    bad_fam += 1
            report["sft"][role] = {"n": len(rows), "file": path.name, "splits": splits, "labels": labels}
            if len(rows) < 8:
                report["warns"].append(f"{role} only {len(rows)} rows")
            if bad_fam:
                report["ok"] = False
                report["errors"].append(f"{role}: {bad_fam} rows with wrong family")
            if role != "J" and labels.get("keep", 0) == 0:
                report["ok"] = False
                report["errors"].append(f"{role}: zero keep")
            if role != "J" and labels.get("drop", 0) < 8:
                report["warns"].append(f"{role}: only {labels.get('drop', 0)} drop")
        except Exception as e:
            report["ok"] = False
            report["errors"].append(f"SFT {role}: {e}")

    if args.models_yaml.exists():
        cfg = load_lora_cfg(args.models_yaml)
        report["lora"] = cfg
        if not cfg.get("target_modules"):
            report["warns"].append("lora target_modules empty, will use q_proj/v_proj")
    else:
        report["ok"] = False
        report["errors"].append("missing configs/models.yaml")

    labels = ROOT / "data" / "labels" / "labels_v1.jsonl"
    report["labels_v1"] = labels.exists()
    if not labels.exists():
        report["warns"].append("labels_v1.jsonl missing (train still OK; eval after train needs it)")

    for pkg in ("yaml", "torch", "transformers", "peft", "datasets", "trl", "bitsandbytes"):
        try:
            m = __import__(pkg)
            report["pkgs"][pkg] = getattr(m, "__version__", "ok")
        except ImportError:
            report["pkgs"][pkg] = None
            if pkg == "yaml":
                report["ok"] = False
                report["errors"].append("pyyaml missing")
            elif args.require_cuda:
                report["ok"] = False
                report["errors"].append(f"{pkg} not installed")
            else:
                report["warns"].append(f"{pkg} not installed (needed on the GPU machine)")

    try:
        import torch

        report["cuda"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            report["gpu"] = torch.cuda.get_device_name(0)
            report["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
            if report["vram_gb"] < 10:
                report["warns"].append("VRAM < 10GB: 7B QLoRA may OOM")
        elif args.require_cuda:
            report["ok"] = False
            report["errors"].append("cuda_available=False")
    except ImportError:
        report["cuda"] = False
        if args.require_cuda:
            report["ok"] = False
            report["errors"].append("torch not installed")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["ok"]:
        print("PREFLIGHT FAIL")
        sys.exit(1)
    print("PREFLIGHT OK")


if __name__ == "__main__":
    main()
