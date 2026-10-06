#!/usr/bin/env python3
"""P3: evaluate configs from prediction JSONL + gold labels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from slatriage.io_utils import read_jsonl, write_json
from slatriage.metrics import binary_prf, relative_gain


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", type=Path, required=True)
    ap.add_argument("--pred", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "summary.json")
    args = ap.parse_args()

    gold_rows = {r.get("alert_id"): r for r in read_jsonl(args.gold)}
    summary: dict = {}
    invent_notes: dict = {}

    for pred_path in args.pred:
        name = pred_path.stem.replace("pred_", "").replace(".jsonl", "")
        if name.endswith(".jsonl"):
            name = pred_path.stem
        # normalize names like pred_E → E
        if name.startswith("pred_"):
            name = name[5:]
        preds = read_jsonl(pred_path)
        y_true, y_pred = [], []
        for p in preds:
            g = gold_rows.get(p.get("alert_id"))
            if not g or g.get("label") == "unknown":
                continue
            y_true.append(g["label"])
            y_pred.append(p.get("decision") or p.get("label"))
        summary[name] = binary_prf(y_true, y_pred) if y_true else {"error": "no overlap", "n": 0}
        summary[name]["n_scored"] = len(y_true)

        report_side = pred_path.with_suffix(".report.json")
        if not str(report_side).endswith(".report.json"):
            report_side = Path(str(pred_path) + ".report.json")
            alt = pred_path.parent / (pred_path.stem + ".report.json")
            report_side = alt if alt.exists() else report_side
        if report_side.exists():
            rep = json.loads(report_side.read_text(encoding="utf-8"))
            j = rep.get("judge") or {}
            invent_notes[name] = j.get("invented_count")

    if "E" in summary and "B" in summary:
        if "precision" in summary["E"] and "precision" in summary["B"]:
            summary["relative_gain_E_vs_B"] = relative_gain(
                float(summary["E"]["precision"]), float(summary["B"]["precision"])
            )
    if invent_notes:
        summary["invented_count_by_config"] = invent_notes

    write_json(args.out, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
