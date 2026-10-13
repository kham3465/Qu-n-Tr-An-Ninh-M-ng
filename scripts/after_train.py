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
    ap.add_argument(
        "--holdout",
        type=Path,
        default=ROOT / "data" / "labels" / "holdout_sources.json",
        help="optional file filter; ignored with --all-sources",
    )
    ap.add_argument(
        "--all-sources",
        action="store_true",
        help="run all Curated alerts (same 343/112 as pred_E). Use this for remaining B/C/D/E_noR/G.",
    )
    ap.add_argument("--skip-baseline-a", action="store_true")
    ap.add_argument("--run-hf", action="store_true", help="run B/C/D/E/G with adapters (needs GPU)")
    ap.add_argument("--resume", action="store_true", default=True, help="skip pred/adapters already done")
    ap.add_argument("--force", action="store_true", help="re-run even if pred exists")
    ap.add_argument("--base", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    ap.add_argument("--g-model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")
    ap.add_argument(
        "--configs",
        default="",
        help="comma list of HF jobs: B,C,D,E,E_noR,G. Empty = all. Remaining: B,C,D,E_noR,G",
    )
    args = ap.parse_args()
    want = {x.strip() for x in args.configs.split(",") if x.strip()}

    def _n_preds(p: Path) -> int:
        if not p.exists():
            return 0
        return sum(1 for line in p.open(encoding="utf-8") if line.strip())

    def _complete(p: Path, min_n: int = 300) -> bool:
        return _n_preds(p) >= min_n

    def _exists(p: Path) -> bool:
        return _complete(p)

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
                *([] if args.all_sources else ["--holdout", str(args.holdout)]),
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
            ("B", []),
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
            ("C", []),
            ("G", ["--g-model", args.g_model]),
        ]
        if want:
            by = {c: e for c, e in hf_jobs}
            hf_jobs = [(c, by[c]) for c in args.configs.split(",") if c.strip() in by]
        for cfg, extra in hf_jobs:
            out = args.pred_dir / f"pred_{cfg}.jsonl"
            n_have = _n_preds(out)
            if args.resume and not args.force and _complete(out):
                print(f"RESUME skip pred_{cfg}.jsonl n={n_have}")
                continue
            if n_have:
                print(f"RESUME continue pred_{cfg}.jsonl n={n_have} (need >=300)")
            cfg_flag = "E" if cfg == "E_noR" else ("G" if cfg == "G" else cfg)
            cmd = [
                sys.executable,
                str(council),
                "--alerts",
                str(args.gold),
                "--config",
                cfg_flag,
                *([] if args.all_sources else ["--holdout", str(args.holdout)]),
                "--backend",
                "hf",
                "--model",
                args.base,
                "--out",
                str(out),
                *extra,
            ]
            if args.force:
                cmd.append("--force")
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
