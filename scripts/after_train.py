#!/usr/bin/env python3
"""After adapters exist: baseline A + full report/figures. HF configs if pred already run."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from slatriage.report import NEED_FOR_FULL, write_report


def _run(cmd: list[str]) -> int:
    print(" ".join(cmd), flush=True)
    return subprocess.call(cmd)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", type=Path, default=ROOT / "adapters")
    ap.add_argument("--pred-dir", type=Path, default=ROOT / "reports")
    ap.add_argument("--gold", type=Path, default=ROOT / "data" / "labels" / "labels_v1.jsonl")
    ap.add_argument("--skip-baseline-a", action="store_true")
    ap.add_argument("--run-hf", action="store_true", help="run B/C/D/E/G with adapters (needs GPU)")
    ap.add_argument("--resume", action="store_true", default=True, help="skip pred/adapters already done")
    ap.add_argument("--force", action="store_true", help="re-run even if pred exists")
    ap.add_argument("--base", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    ap.add_argument("--g-model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")
    args = ap.parse_args()

    def _exists(p: Path) -> bool:
        return p.exists() and p.stat().st_size > 2

    council = ROOT / "scripts" / "run_council.py"
    pred_a = args.pred_dir / "pred_A.jsonl"
    if not args.skip_baseline_a and (args.force or not (args.resume and _exists(pred_a))):
        _run(
            [
                sys.executable,
                str(council),
                "--alerts",
                str(args.gold),
                "--config",
                "A",
                "--backend",
                "mock",
                "--out",
                str(pred_a),
            ]
        )
    elif _exists(pred_a):
        print(f"RESUME skip pred_A.jsonl")

    if args.run_hf:
        ad = args.adapters
        hf_jobs = [
            ("B", []),
            ("C", []),
            (
                "D",
                [
                    "--adapter-r",
                    str(ad / "R"),
                    "--adapter-a",
                    str(ad / "A"),
                    "--adapter-c",
                    str(ad / "C"),
                ],
            ),
            (
                "E",
                [
                    "--adapter-r",
                    str(ad / "R"),
                    "--adapter-a",
                    str(ad / "A"),
                    "--adapter-c",
                    str(ad / "C"),
                    "--adapter-j",
                    str(ad / "J"),
                ],
            ),
            (
                "E_noR",
                [
                    "--adapter-a",
                    str(ad / "A"),
                    "--adapter-c",
                    str(ad / "C"),
                    "--adapter-j",
                    str(ad / "J"),
                ],
            ),
            ("G", ["--g-model", args.g_model]),
        ]
        for cfg, extra in hf_jobs:
            out = args.pred_dir / f"pred_{cfg}.jsonl"
            if args.resume and not args.force and _exists(out):
                print(f"RESUME skip pred_{cfg}.jsonl")
                continue
            cfg_flag = "E" if cfg == "E_noR" else ("G" if cfg == "G" else cfg)
            cmd = [
                sys.executable,
                str(council),
                "--alerts",
                str(args.gold),
                "--config",
                cfg_flag,
                "--backend",
                "hf",
                "--model",
                args.base,
                "--out",
                str(out),
                *extra,
            ]
            rc = _run(cmd)
            if rc != 0:
                print(f"WARN council {cfg} rc={rc}")

    payload = write_report(root=ROOT, pred_dir=args.pred_dir, adapters=args.adapters)
    ready = [k for k, v in (payload.get("figures") or {}).items() if v]
    miss = payload.get("figures_missing") or []
    print(f"REPORT figures_ready={len(ready)} missing={len(miss)}")
    for k in miss:
        print(f"  wait {k}: {NEED_FOR_FULL.get(k)}")
    print("files: reports/BAO-CAO-CHI-SO.md reports/paper_stats.json reports/figures/")


if __name__ == "__main__":
    main()
