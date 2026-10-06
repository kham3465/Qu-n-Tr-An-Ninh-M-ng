#!/usr/bin/env python3
"""P1: một file .sol → Slither JSON + alerts.jsonl đã gán họ."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from slatriage.config_utils import load_yaml
from slatriage.families import assign_family
from slatriage.io_utils import write_jsonl
from slatriage.slither_runner import normalize_detectors, run_slither_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sol", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--map", type=Path, default=ROOT / "configs" / "detectors_map.yaml")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    slither_json = args.out_dir / f"{args.sol.stem}.slither.json"
    try:
        result = run_slither_json(args.sol, slither_json)
    except FileNotFoundError:
        print("ERROR: slither not on PATH. pip install slither-analyzer")
        sys.exit(1)

    mapping = load_yaml(args.map)
    alerts = result.get("detectors") or []
    if not alerts and slither_json.exists():
        payload = json.loads(slither_json.read_text(encoding="utf-8"))
        alerts = normalize_detectors(payload)

    rows = []
    for a in alerts:
        det = a.get("detector") or ""
        fam = assign_family(det, mapping, allow_other=True)
        rows.append({**a, "family": fam, "in_scope": fam in ("R", "A", "C")})

    out_alerts = args.out_dir / f"{args.sol.stem}.alerts.jsonl"
    write_jsonl(out_alerts, rows)
    n_scope = sum(1 for r in rows if r["in_scope"])
    print(f"ok={result.get('ok')} alerts={len(rows)} in_scope_RAC={n_scope} -> {out_alerts}")


if __name__ == "__main__":
    main()
