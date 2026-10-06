#!/usr/bin/env python3
"""P1: batch-run Slither on a directory of .sol files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from slatriage.slither_runner import run_slither_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True, help="Folder containing .sol")
    ap.add_argument("--out", type=Path, required=True, help="Output folder for JSON + summary")
    ap.add_argument("--limit", type=int, default=20, help="Max files (default 20 for tuần 1)")
    args = ap.parse_args()

    sols = sorted(args.input.rglob("*.sol"))[: args.limit]
    args.out.mkdir(parents=True, exist_ok=True)
    summary = []
    for sol in sols:
        out_json = args.out / f"{sol.stem}.slither.json"
        print(f"[slither] {sol}")
        try:
            r = run_slither_json(sol, out_json)
        except FileNotFoundError:
            print("ERROR: `slither` not on PATH. pip install slither-analyzer")
            sys.exit(1)
        except Exception as e:
            r = {"sol_path": str(sol), "ok": False, "error": str(e), "detectors": []}
        summary.append(
            {
                "sol": str(sol),
                "ok": r.get("ok"),
                "n_alerts": len(r.get("detectors") or []),
                "returncode": r.get("returncode"),
            }
        )
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    ok = sum(1 for s in summary if s["ok"])
    print(f"Done: {ok}/{len(summary)} ok → {args.out / 'summary.json'}")


if __name__ == "__main__":
    main()
