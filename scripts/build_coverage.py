#!/usr/bin/env python3
"""P3: ma trận đóng (contract × hạng DASP) → KEEP|DROP|ABSENT|UNKNOWN."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from slatriage.coverage import build_matrix, load_closed, matrix_summary
from slatriage.io_utils import read_jsonl, write_json, write_jsonl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alerts", type=Path, required=True, help="labels hoặc pred jsonl (có family + label/decision)")
    ap.add_argument("--closed", type=Path, default=ROOT / "configs" / "closed_list.yaml")
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "coverage_matrix.jsonl")
    args = ap.parse_args()

    closed = load_closed(args.closed)
    alerts = read_jsonl(args.alerts)
    rows = build_matrix(alerts, closed)
    write_jsonl(args.out, rows)
    summary = matrix_summary(rows)
    write_json(args.out.with_suffix(".summary.json"), summary)
    print(json.dumps(summary, indent=2))
    print(f"cells={len(rows)} -> {args.out}")


if __name__ == "__main__":
    main()
