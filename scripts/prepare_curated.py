#!/usr/bin/env python3
"""P1 end-to-end until train-ready: Slither Curated → labels_v1 → SFT jsonl (no GPU train)."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from slatriage.config_utils import load_yaml
from slatriage.curated import (
    human_findings_for_entry,
    index_by_path,
    index_by_stem,
    load_compiled_versions,
    load_curated_index,
    lookup_entry,
    pick_solc_version,
    snap_solc,
)
from slatriage.dataset import export_judge_sft, export_sft_jsonl
from slatriage.families import assign_family
from slatriage.io_utils import write_json, write_jsonl
from slatriage.label_match import match_alert
from slatriage.slither_runner import (
    make_alert_id,
    normalize_detectors,
    run_slither_json,
    snippet_around,
)


def out_name(sol: Path, dataset: Path) -> str:
    try:
        rel = sol.resolve().relative_to(dataset.resolve())
        return f"{rel.parent.name}_{sol.stem}.slither.json"
    except ValueError:
        return f"{sol.parent.name}_{sol.stem}.slither.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--curated-root",
        type=Path,
        default=ROOT / "data" / "raw" / "smartbugs-curated",
    )
    ap.add_argument("--limit", type=int, default=0, help="0 = all .sol")
    ap.add_argument("--skip-slither", action="store_true")
    args = ap.parse_args()

    vuln_json = args.curated_root / "vulnerabilities.json"
    dataset = args.curated_root / "dataset"
    slither_dir = ROOT / "data" / "raw" / "slither_json"
    labels_path = ROOT / "data" / "labels" / "labels_v1.jsonl"
    stats_path = ROOT / "data" / "labels" / "stats.md"
    map_path = ROOT / "configs" / "detectors_map.yaml"

    if not vuln_json.exists():
        print("ERROR: clone SmartBugs Curated into data/raw/smartbugs-curated")
        sys.exit(1)

    entries = load_curated_index(vuln_json)
    by_stem = index_by_stem(entries)
    by_path = index_by_path(entries)
    compiled_map = load_compiled_versions(args.curated_root / "versions.csv")
    mapping = load_yaml(map_path)
    window = int(mapping.get("match", {}).get("line_window", 5))

    sols = sorted(p for p in dataset.rglob("*.sol") if p.is_file())
    if args.limit:
        sols = sols[: args.limit]
    slither_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    if not args.skip_slither:
        for i, sol in enumerate(sols, 1):
            entry = lookup_entry(sol, dataset, by_path, by_stem)
            ver = pick_solc_version(entry, sol, compiled_map)
            fallbacks = [snap_solc(ver), "0.4.25"]
            dest = slither_dir / out_name(sol, dataset)
            print(f"[{i}/{len(sols)}] solc {ver} {sol.parent.name}/{sol.name}", flush=True)
            try:
                r = run_slither_json(sol, dest, solc_version=ver, solc_fallbacks=fallbacks)
            except FileNotFoundError:
                print("ERROR: slither not on PATH. pip install slither-analyzer solc-select")
                sys.exit(1)
            except Exception as e:
                r = {"ok": False, "detectors": [], "error": str(e), "returncode": -1, "solc": ver}
            summary.append(
                {
                    "sol": f"{sol.parent.name}/{sol.name}",
                    "ok": bool(r.get("ok")),
                    "n_alerts": len(r.get("detectors") or []),
                    "n_rac": sum(
                        1
                        for d in (r.get("detectors") or [])
                        if assign_family(d.get("detector") or "", mapping, allow_other=False)
                    ),
                    "solc": r.get("solc") or ver,
                    "returncode": r.get("returncode"),
                }
            )
        write_json(slither_dir / "summary.json", summary)
        ok = sum(1 for s in summary if s["ok"])
        rac = sum(s.get("n_rac", 0) for s in summary)
        print(f"Slither: {ok}/{len(summary)} ok; RAC alerts: {rac}")
    else:
        print("skip slither, use existing JSON")

    rows = []
    detector_counts: Counter = Counter()
    for path in sorted(slither_dir.glob("*.slither.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        alerts = payload.get("detectors")
        if alerts is None:
            alerts = normalize_detectors(payload)
        sol_hint = Path(payload.get("sol_path") or path.name)
        entry = lookup_entry(sol_hint, dataset, by_path, by_stem)
        if entry is None:
            # filename access_control_phishable.slither.json → try stem after first _
            name = path.name.replace(".slither.json", "")
            parts = name.split("_", 1)
            entry = by_stem.get(Path(parts[-1]).stem if parts else name)
        humans = human_findings_for_entry(entry) if entry else []
        src = (entry or {}).get("path") or str(payload.get("sol_path") or "")
        sol_file = Path(payload["sol_path"]) if payload.get("sol_path") else None
        for i, alert in enumerate(alerts):
            det = alert.get("detector") or ""
            detector_counts[det] += 1
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
    write_jsonl(labels_path, rows)
    write_json(slither_dir / "detector_counts.json", dict(detector_counts.most_common()))

    counts: Counter = Counter()
    for r in rows:
        counts[(r["family"], r["label"])] += 1
    if not summary:
        sum_path = slither_dir / "summary.json"
        if sum_path.exists():
            summary = json.loads(sum_path.read_text(encoding="utf-8"))
    n_ok = sum(1 for s in summary if s.get("ok")) if summary else "?"
    n_all = len(summary) if summary else (len(sols) if sols else "?")
    lines = [
        "# Curated label stats",
        "",
        f"- slither ok: {n_ok}/{n_all}",
        f"- in-scope RAC alerts: {len(rows)}",
        "",
        "| family | keep | drop | unknown |",
        "| --- | --- | --- | --- |",
    ]
    for fam in ("R", "A", "C"):
        lines.append(
            f"| {fam} | {counts[(fam, 'keep')]} | {counts[(fam, 'drop')]} | {counts[(fam, 'unknown')]} |"
        )
    stats_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    for role in ("R", "A", "C"):
        n = export_sft_jsonl(labels_path, ROOT / "adapters" / role / f"sft_{role}.jsonl", family=role)
        print(f"SFT {role}: {n} rows")
    n_j = export_judge_sft(labels_path, ROOT / "adapters" / "J" / "sft_J.jsonl")
    print(f"SFT J: {n_j} rows")
    print("READY_TO_TRAIN: run train_lora.py --do-train on GPU")


if __name__ == "__main__":
    main()
