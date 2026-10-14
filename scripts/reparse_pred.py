#!/usr/bin/env python3
"""CPU: re-parse pred_*.jsonl from stored raw (R1 <think> + JSON). No GPU."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from slatriage.agents import _parse_single_shot
from slatriage.io_utils import read_jsonl, write_jsonl
from slatriage.schema import Alert


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or args.pred
    rows = read_jsonl(args.pred)
    ok = fail = no_raw = 0
    new_rows = []
    for r in rows:
        if r.get("reason") != "parse_fail":
            new_rows.append(r)
            ok += 1
            continue
        raw = r.get("raw")
        if not raw:
            no_raw += 1
            new_rows.append(r)
            continue
        fake = [Alert(alert_id=str(r.get("alert_id") or ""), detector="", family=r.get("family"))]
        parsed = _parse_single_shot(str(raw), fake)
        hit = next((p for p in parsed if p.get("reason") != "parse_fail"), None)
        if hit:
            r = {**r, **hit}
            r.pop("raw", None)
            ok += 1
        else:
            fail += 1
        new_rows.append(r)
    write_jsonl(out, new_rows)
    print(f"n={len(new_rows)} recovered_or_ok={ok} still_fail={fail} no_raw={no_raw} out={out}")


if __name__ == "__main__":
    main()
