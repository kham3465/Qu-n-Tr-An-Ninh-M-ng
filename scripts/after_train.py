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

from slatriage.io_utils import read_jsonl
from slatriage.report import NEED_FOR_FULL, write_report

FULL_PRED_MIN = 300
HOLDOUT_PRED_MIN = 100
REMAINING = ("B", "C", "D", "E_noR", "G")


def _run(cmd: list[str]) -> int:
    print(" ".join(cmd), flush=True)
    return subprocess.call(cmd)


def pred_n(path: Path) -> int:
    if not path.exists() or path.stat().st_size < 2:
        return 0
    return sum(1 for line in path.open(encoding="utf-8") if line.strip())


def pred_complete(path: Path, *, holdout_only: bool) -> bool:
    return pred_n(path) >= (HOLDOUT_PRED_MIN if holdout_only else FULL_PRED_MIN)


def print_pred_status(pred_dir: Path) -> None:
    print("PRED STATUS", flush=True)
    for p in sorted(pred_dir.glob("pred_*.jsonl")):
        rows = read_jsonl(p)
        srcs = {str(r.get("source") or "") for r in rows if r.get("source")}
        pf = sum(1 for r in rows if r.get("reason") == "parse_fail")
        print(f"  {p.name}: n={len(rows)} sources={len(srcs)} parse_fail={pf}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", type=Path, default=ROOT / "adapters")
    ap.add_argument("--pred-dir", type=Path, default=ROOT / "reports")
    ap.add_argument("--gold", type=Path, default=ROOT / "data" / "labels" / "labels_v1.jsonl")
    ap.add_argument(
        "--holdout",
        type=Path,
        default=ROOT / "data" / "labels" / "holdout_sources.json",
        help="used only with --holdout-only; report F1 is always hold-out",
    )
    ap.add_argument(
        "--holdout-only",
        action="store_true",
        help="council only on 34 hold-out files (~120 alerts). Default = full labels_v1 (343/112 like pred_E)",
    )
    ap.add_argument(
        "--all-sources",
        action="store_true",
        help="compat no-op: full labels_v1 is already the default",
    )
    ap.add_argument("--skip-baseline-a", action="store_true")
    ap.add_argument("--run-hf", action="store_true", help="run B/C/D/E/G with adapters (needs GPU)")
    ap.add_argument("--resume", action="store_true", default=True, help="skip complete preds; continue partial files")
    ap.add_argument("--force", action="store_true", help="re-run from scratch even if pred exists")
    ap.add_argument("--base", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    ap.add_argument("--g-model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")
    ap.add_argument(
        "--configs",
        default="",
        help="comma list of HF jobs: B,C,D,E,E_noR,G. Empty = all. Remaining: B,C,D,E_noR,G",
    )
    args = ap.parse_args()
    raw = args.configs.strip()
    if raw.lower() in {"remaining", "rest", "con-lai"}:
        want = set(REMAINING)
    else:
        want = {x.strip() for x in raw.split(",") if x.strip()}

    council = ROOT / "scripts" / "run_council.py"
    pred_a = args.pred_dir / "pred_A.jsonl"
    if not args.skip_baseline_a and (args.force or not pred_complete(pred_a, holdout_only=args.holdout_only)):
        cmd_a = [
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
        if args.holdout_only:
            cmd_a.extend(["--holdout", str(args.holdout)])
        if args.resume and not args.force:
            cmd_a.append("--resume")
        _run(cmd_a)
    elif pred_complete(pred_a, holdout_only=args.holdout_only):
        print(f"RESUME skip pred_A.jsonl n={pred_n(pred_a)}")

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
            ("G", ["--g-model", args.g_model]),
        ]
        if want:
            hf_jobs = [(c, e) for c, e in hf_jobs if c in want]
        for cfg, extra in hf_jobs:
            out = args.pred_dir / f"pred_{cfg}.jsonl"
            n = pred_n(out)
            if args.resume and not args.force and pred_complete(out, holdout_only=args.holdout_only):
                print(f"RESUME skip pred_{cfg}.jsonl n={n} (already full)")
                continue
            if n and not args.force:
                print(f"CONTINUE pred_{cfg}.jsonl n={n} (need {FULL_PRED_MIN if not args.holdout_only else HOLDOUT_PRED_MIN})")
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
            if args.holdout_only:
                cmd.extend(["--holdout", str(args.holdout)])
            if args.resume and not args.force:
                cmd.append("--resume")
            rc = _run(cmd)
            if rc != 0:
                print(f"WARN council {cfg} rc={rc} — next config still runs")
            print_pred_status(args.pred_dir)

    print_pred_status(args.pred_dir)
    payload = write_report(root=ROOT, pred_dir=args.pred_dir, adapters=args.adapters)
    ready = [k for k, v in (payload.get("figures") or {}).items() if v]
    miss = payload.get("figures_missing") or []
    print(f"REPORT figures_ready={len(ready)} missing={len(miss)}")
    for k in miss:
        print(f"  wait {k}: {NEED_FOR_FULL.get(k)}")
    print("files: reports/BAO-CAO-CHI-SO.md reports/paper_stats.json reports/figures/")


if __name__ == "__main__":
    main()
