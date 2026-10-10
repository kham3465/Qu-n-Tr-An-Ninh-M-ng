#!/usr/bin/env python3
"""Build paper stats + figures from labels, SFT, and any pred_*.jsonl."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from slatriage.report import write_report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, default=ROOT / "data" / "labels" / "labels_v1.jsonl")
    ap.add_argument("--sft-dir", type=Path, default=None)
    ap.add_argument("--solidifi-labels", type=Path, default=ROOT / "data" / "labels" / "labels_solidifi.jsonl")
    ap.add_argument("--closed", type=Path, default=ROOT / "configs" / "closed_list.yaml")
    ap.add_argument("--curated-root", type=Path, default=ROOT / "data" / "raw" / "smartbugs-curated")
    ap.add_argument("--pred", type=Path, nargs="*", default=None)
    ap.add_argument("--pred-dir", type=Path, default=ROOT / "reports")
    ap.add_argument("--adapters", type=Path, default=ROOT / "adapters")
    ap.add_argument("--out-json", type=Path, default=ROOT / "reports" / "paper_stats.json")
    ap.add_argument("--out-md", type=Path, default=ROOT / "reports" / "BAO-CAO-CHI-SO.md")
    ap.add_argument("--fig-dir", type=Path, default=ROOT / "reports" / "figures")
    args = ap.parse_args()

    payload = write_report(
        root=ROOT,
        labels=args.labels,
        solidifi=args.solidifi_labels,
        sft_dir=args.sft_dir,
        closed=args.closed,
        curated_root=args.curated_root,
        pred_dir=args.pred_dir,
        pred_files=list(args.pred) if args.pred else None,
        adapters=args.adapters,
        out_json=args.out_json,
        out_md=args.out_md,
        fig_dir=args.fig_dir,
    )
    ready = [k for k, v in (payload.get("figures") or {}).items() if v]
    miss = payload.get("figures_missing") or []
    print(f"wrote {args.out_md.name} figures={len(ready)} missing={len(miss)}")
    if miss:
        print("missing: " + ", ".join(miss))


if __name__ == "__main__":
    main()
