#!/usr/bin/env python3
"""P1: build labels.jsonl from Slither JSON + human vulnerability file (skeleton)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from slatriage.config_utils import load_yaml
from slatriage.families import assign_family
from slatriage.label_match import match_alert


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slither-dir", type=Path, required=True)
    ap.add_argument("--vuln-json", type=Path, help="SmartBugs vulnerabilities.json (optional tuần 1)")
    ap.add_argument("--map", type=Path, default=ROOT / "configs" / "detectors_map.yaml")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    mapping = load_yaml(args.map)
    window = int(mapping.get("match", {}).get("line_window", 5))
    human_by_file: dict = {}
    if args.vuln_json and args.vuln_json.exists():
        # Format varies; P1 adapts after inspecting Curated layout.
        human_by_file = json.loads(args.vuln_json.read_text(encoding="utf-8"))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with args.out.open("w", encoding="utf-8") as fout:
        for path in sorted(args.slither_dir.glob("*.slither.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            # Prefer normalized list if we stored full slither; else results.detectors
            from slatriage.slither_runner import normalize_detectors

            alerts = payload.get("detectors")
            if alerts is None:
                alerts = normalize_detectors(payload)
            # human findings lookup — placeholder: empty → all drop/unknown by rule
            humans = human_by_file.get(path.stem, []) if isinstance(human_by_file, dict) else []
            for alert in alerts:
                fam = assign_family(alert.get("detector") or "", mapping, allow_other=False)
                if fam is None:
                    continue
                label = match_alert(alert, humans, line_window=window, family=fam)
                row = {
                    **alert,
                    "family": fam,
                    "label": label,
                    "source_slither": str(path),
                }
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                n += 1
    print(f"Wrote {n} rows → {args.out}")


if __name__ == "__main__":
    main()
