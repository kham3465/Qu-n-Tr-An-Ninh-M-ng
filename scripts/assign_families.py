#!/usr/bin/env python3
"""P1: gán họ R/A/C/O cho alerts.jsonl (theo detectors_map.yaml)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from slatriage.config_utils import load_yaml
from slatriage.families import assign_family
from slatriage.io_utils import read_jsonl, write_jsonl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alerts", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--map", type=Path, default=ROOT / "configs" / "detectors_map.yaml")
    args = ap.parse_args()

    mapping = load_yaml(args.map)
    rows = []
    counts = {"R": 0, "A": 0, "C": 0, "O": 0, "none": 0}
    for r in read_jsonl(args.alerts):
        fam = assign_family(r.get("detector") or "", mapping, allow_other=True)
        r = {**r, "family": fam}
        rows.append(r)
        counts[fam if fam else "none"] += 1
    write_jsonl(args.out, rows)
    print(f"wrote {len(rows)} -> {args.out} counts={counts}")


if __name__ == "__main__":
    main()
