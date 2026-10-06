#!/usr/bin/env python3
"""Evaluate configs from prediction JSONL + gold labels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from slatriage.io_utils import read_jsonl, write_json
from slatriage.metrics import binary_prf, relative_gain


def row_key(row: dict) -> tuple[str, str]:
    return (str(row.get("source") or ""), str(row.get("alert_id") or ""))


def config_name(pred_path: Path) -> str:
    name = pred_path.stem
    if name.startswith("pred_"):
        name = name[5:]
    return name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", type=Path, required=True)
    ap.add_argument("--pred", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "summary.json")
    args = ap.parse_args()

    gold_list = read_jsonl(args.gold)
    gold_by_pair = {row_key(r): r for r in gold_list}
    gold_by_id: dict[str, list] = {}
    for r in gold_list:
        gold_by_id.setdefault(str(r.get("alert_id") or ""), []).append(r)

    summary: dict = {}
    invent_notes: dict = {}

    for pred_path in args.pred:
        name = config_name(pred_path)
        preds = read_jsonl(pred_path)
        y_true, y_pred = [], []
        for p in preds:
            g = gold_by_pair.get(row_key(p))
            if g is None:
                hits = gold_by_id.get(str(p.get("alert_id") or ""), [])
                g = hits[0] if len(hits) == 1 else None
            if not g or g.get("label") == "unknown":
                continue
            y_true.append(g["label"])
            y_pred.append(p.get("decision") or p.get("label"))
        summary[name] = binary_prf(y_true, y_pred) if y_true else {"error": "no overlap", "n": 0}
        summary[name]["n_scored"] = len(y_true)

        alt = pred_path.parent / (pred_path.stem + ".report.json")
        if alt.exists():
            rep = json.loads(alt.read_text(encoding="utf-8"))
            j = rep.get("judge") or {}
            invent_notes[name] = j.get("invented_count", rep.get("invented_count"))

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
