#!/usr/bin/env python3
"""Build labels.jsonl from existing Slither JSON + SmartBugs vulnerabilities.json."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from slatriage.config_utils import load_yaml
from slatriage.curated import (
    human_findings_for_entry,
    index_by_path,
    index_by_stem,
    load_curated_index,
    lookup_entry,
)
from slatriage.families import assign_family
from slatriage.io_utils import write_jsonl
from slatriage.label_match import match_alert
from slatriage.slither_runner import make_alert_id, normalize_detectors, snippet_around


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slither-dir", type=Path, required=True)
    ap.add_argument("--vuln-json", type=Path, required=True)
    ap.add_argument("--map", type=Path, default=ROOT / "configs" / "detectors_map.yaml")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    mapping = load_yaml(args.map)
    window = int(mapping.get("match", {}).get("line_window", 5))
    entries = load_curated_index(args.vuln_json)
    by_stem = index_by_stem(entries)
    by_path = index_by_path(entries)
    dataset = args.vuln_json.parent / "dataset"

    rows = []
    for path in sorted(args.slither_dir.glob("*.slither.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        alerts = payload.get("detectors")
        if alerts is None:
            alerts = normalize_detectors(payload)
        sol_hint = Path(payload.get("sol_path") or path.name)
        entry = lookup_entry(sol_hint, dataset, by_path, by_stem)
        humans = human_findings_for_entry(entry) if entry else []
        src = (entry or {}).get("path") or str(payload.get("sol_path") or path.name)
        sol_file = Path(payload["sol_path"]) if payload.get("sol_path") else None
        for i, alert in enumerate(alerts):
            det = alert.get("detector") or ""
            fam = assign_family(det, mapping, allow_other=False)
            if fam is None:
                continue
            label = match_alert(alert, humans, line_window=window, family=fam)
            snippet = ""
            if sol_file and sol_file.exists():
                snippet = snippet_around(sol_file, list(alert.get("lines") or []))
            rows.append(
                {
                    **alert,
                    "alert_id": make_alert_id(src, det, i),
                    "family": fam,
                    "label": label,
                    "source": src,
                    "source_slither": str(path),
                    "snippet": snippet,
                    "curated_categories": [h.get("category") for h in humans],
                }
            )
    n = write_jsonl(args.out, rows)
    print(f"Wrote {n} rows -> {args.out}")


if __name__ == "__main__":
    main()
