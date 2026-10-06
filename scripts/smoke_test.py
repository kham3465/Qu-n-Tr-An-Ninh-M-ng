#!/usr/bin/env python3
"""Smoke test without GPU: mock council A/B/C/D/E/G + metrics."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from slatriage.council import run_config
from slatriage.io_utils import read_jsonl, write_jsonl
from slatriage.llm import MockLLM
from slatriage.metrics import binary_prf
from slatriage.schema import Alert


def main() -> None:
    labels_path = ROOT / "data" / "labels" / "demo_labels.jsonl"
    map_path = ROOT / "configs" / "detectors_map.yaml"
    out_dir = ROOT / "reports" / "smoke"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(labels_path)
    alerts = [Alert.from_dict(r) for r in rows]
    gold = {r["alert_id"]: r["label"] for r in rows}
    llm = MockLLM()

    summary = {}
    for cfg in ["A", "B", "C", "D", "E", "G"]:
        report = run_config(
            cfg,  # type: ignore[arg-type]
            alerts,
            map_path=map_path,
            base_llm=llm,
            g_llm=llm,
            sol_path="demo",
        )
        pred_path = out_dir / f"pred_{cfg}.jsonl"
        write_jsonl(pred_path, report.predictions)
        y_true, y_pred = [], []
        for p in report.predictions:
            aid = p["alert_id"]
            if aid not in gold or gold[aid] == "unknown":
                continue
            y_true.append(gold[aid])
            y_pred.append(p["decision"])
        summary[cfg] = binary_prf(y_true, y_pred)
        inv = report.judge.invented_count if report.judge else None
        print(f"[{cfg}] scored={len(y_true)} metrics={summary[cfg]} invent={inv}")

    assert "A" in summary and summary["A"]["precision"] >= 0
    print("SMOKE OK")
    print("out_dir=", str(out_dir))


if __name__ == "__main__":
    main()
