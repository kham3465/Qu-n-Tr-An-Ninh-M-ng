#!/usr/bin/env python3
"""Evaluate configs from prediction JSONL + gold labels (overall + per family)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from slatriage.io_utils import read_jsonl, write_json
from slatriage.metrics import relative_gain, score_aligned


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
    summary: dict = {}
    invent_notes: dict = {}

    for pred_path in args.pred:
        name = config_name(pred_path)
        preds = read_jsonl(pred_path)
        summary[name] = score_aligned(gold_list, preds)
        alt = pred_path.parent / (pred_path.stem + ".report.json")
        if alt.exists():
            rep = json.loads(alt.read_text(encoding="utf-8"))
            j = rep.get("judge") or {}
            invent_notes[name] = j.get("invented_count", rep.get("invented_count"))

    if "E" in summary and "B" in summary:
        if "precision" in summary["E"] and "precision" in summary["B"]:
            summary["relative_gain_E_vs_B"] = {
                "precision": relative_gain(float(summary["E"]["precision"]), float(summary["B"]["precision"])),
                "f1": relative_gain(float(summary["E"].get("f1") or 0), float(summary["B"].get("f1") or 0)),
            }
    if invent_notes:
        summary["invented_count_by_config"] = invent_notes

    write_json(args.out, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
