#!/usr/bin/env python3
"""SolidiFI RAC folders → labels_solidifi.jsonl (+ optional SFT). Does not mix Curated."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from slatriage.config_utils import load_yaml
from slatriage.dataset import export_judge_sft, export_sft_jsonl, merge_sft_jsonl
from slatriage.families import assign_family
from slatriage.io_utils import write_json, write_jsonl
from slatriage.label_match import match_alert
from slatriage.slither_runner import make_alert_id, normalize_detectors, run_slither_json, snippet_around
from slatriage.solidifi import FOLDER_TO_FAMILY, injection_stats, iter_solidifi_contracts, load_buglog


def _slither_cache_ok(dest: Path) -> dict | None:
    if not dest.exists() or dest.stat().st_size < 2:
        return None
    try:
        payload = json.loads(dest.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if payload.get("detectors") is None and payload.get("success") is None:
        return None
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT / "data" / "raw" / "SolidiFI-benchmark" / "buggy_contracts")
    ap.add_argument("--limit-per-type", type=int, default=0, help="0 = all files in each RAC folder")
    ap.add_argument("--skip-slither", action="store_true")
    ap.add_argument("--force-slither", action="store_true", help="re-run Slither even if JSON exists")
    ap.add_argument("--export-sft", action="store_true")
    ap.add_argument("--no-merge", action="store_true", help="do not write data/sft/combined")
    args = ap.parse_args()

    if not args.root.exists():
        print("ERROR: clone SolidiFI-benchmark into data/raw/SolidiFI-benchmark")
        sys.exit(1)

    print("injection_sites", json.dumps(injection_stats(args.root)))
    mapping = load_yaml(ROOT / "configs" / "detectors_map.yaml")
    window = int(mapping.get("match", {}).get("line_window", 5))
    slither_dir = ROOT / "data" / "raw" / "solidifi_slither"
    slither_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    seen: Counter = Counter()
    for item in iter_solidifi_contracts(args.root):
        if args.limit_per_type and seen[item["folder"]] >= args.limit_per_type:
            continue
        seen[item["folder"]] += 1
        jobs.append(item)

    summary = []
    if not args.skip_slither:
        for i, job in enumerate(jobs, 1):
            dest = slither_dir / f"{job['folder'].replace(' ', '_')}_{job['sol_path'].stem}.slither.json"
            cached = None if args.force_slither else _slither_cache_ok(dest)
            if cached is not None:
                print(f"[{i}/{len(jobs)}] skip {job['source']}", flush=True)
                r = {
                    "ok": bool(cached.get("success", True)),
                    "detectors": cached.get("detectors") or [],
                    "solc": cached.get("solc"),
                }
            else:
                print(f"[{i}/{len(jobs)}] {job['source']}", flush=True)
                try:
                    r = run_slither_json(
                        job["sol_path"],
                        dest,
                        solc_version="0.4.25",
                        solc_fallbacks=["0.5.17"],
                    )
                except Exception as e:
                    r = {"ok": False, "detectors": [], "error": str(e), "returncode": -1}
            wrapper = {}
            if dest.exists():
                try:
                    wrapper = json.loads(dest.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    wrapper = {}
            if not wrapper.get("sol_path"):
                dest.write_text(
                    json.dumps(
                        {
                            "success": r.get("ok"),
                            "detectors": r.get("detectors") or [],
                            "sol_path": str(job["sol_path"]),
                            "solidifi": job["source"],
                            "family": job["family"],
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            else:
                wrapper["solidifi"] = job["source"]
                wrapper["family"] = job["family"]
                dest.write_text(json.dumps(wrapper, ensure_ascii=False, indent=2), encoding="utf-8")
            summary.append(
                {
                    "source": job["source"],
                    "ok": bool(r.get("ok")),
                    "n_alerts": len(r.get("detectors") or []),
                    "solc": r.get("solc"),
                }
            )
        write_json(slither_dir / "summary.json", summary)
        print(f"Slither {sum(1 for s in summary if s['ok'])}/{len(summary)} ok")

    by_src = {j["source"]: j for j in jobs}
    rows = []
    for path in sorted(slither_dir.glob("*.slither.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        alerts = payload.get("detectors")
        if alerts is None:
            alerts = normalize_detectors(payload)
        src = payload.get("solidifi") or ""
        job = by_src.get(src)
        if job is None:
            # recover from filename folder_buggy_N
            name = path.name.replace(".slither.json", "")
            for folder, fam in FOLDER_TO_FAMILY.items():
                key = folder.replace(" ", "_") + "_"
                if name.startswith(key):
                    stem = name[len(key) :]
                    for j in jobs:
                        if j["folder"] == folder and j["sol_path"].stem == stem:
                            job = j
                            break
                    break
        if job is None:
            continue
        humans = load_buglog(job["log_path"], family=job["family"], category=job["category"]) if job["log_path"] else []
        sol_file = Path(payload["sol_path"]) if payload.get("sol_path") else job["sol_path"]
        for i, alert in enumerate(alerts):
            det = alert.get("detector") or ""
            fam = assign_family(det, mapping, allow_other=False)
            if fam is None:
                continue
            label = match_alert(alert, humans, line_window=window, family=fam, multi_hit="keep")
            snippet = snippet_around(sol_file, list(alert.get("lines") or [])) if sol_file.exists() else ""
            rows.append(
                {
                    **alert,
                    "alert_id": make_alert_id(job["source"], det, i),
                    "family": fam,
                    "label": label,
                    "source": job["source"],
                    "source_slither": str(path),
                    "snippet": snippet,
                    "split": "solidifi",
                    "curated_categories": [job["category"]],
                }
            )

    labels_path = ROOT / "data" / "labels" / "labels_solidifi.jsonl"
    write_jsonl(labels_path, rows)
    counts: Counter = Counter((r["family"], r["label"]) for r in rows)
    lines = [
        "# SolidiFI label stats (do not mix into Curated F1)",
        "",
        f"- files: {len(jobs)}",
        f"- RAC alerts: {len(rows)}",
        "",
        "| family | keep | drop | unknown |",
        "| --- | ---: | ---: | ---: |",
    ]
    for fam in ("R", "A", "C"):
        lines.append(
            f"| {fam} | {counts[(fam, 'keep')]} | {counts[(fam, 'drop')]} | {counts[(fam, 'unknown')]} |"
        )
    if counts[("C", "drop")] < 25:
        lines += [
            "",
            f"Note: C drop = {counts[('C', 'drop')]} (injection snippets almost always match). "
            "LoRA-C will bias keep. F1 chinh van Curated.",
        ]
    stats = ROOT / "data" / "labels" / "stats_solidifi.md"
    stats.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    n_ok = sum(1 for s in summary if s.get("ok")) if summary else None
    a_keep = counts[("A", "keep")]
    cross_keep = [
        r["alert_id"]
        for r in rows
        if r.get("label") == "keep" and r.get("family") != FOLDER_TO_FAMILY.get(
            next((k for k in FOLDER_TO_FAMILY if k in str(r.get("source") or "")), "")
        )
    ]
    gates = {
        "slither_ok_pct": round(100.0 * n_ok / len(summary), 1) if summary else None,
        "A_keep": a_keep,
        "split_solidifi": all(r.get("split") == "solidifi" for r in rows) if rows else False,
        "cross_family_keep": len(cross_keep),
        "pass_slither": (n_ok / len(summary) >= 0.8) if summary else True,
        "pass_A_keep": a_keep > 14,
        "pass_family": len(cross_keep) == 0,
    }
    write_json(ROOT / "data" / "labels" / "solidifi_gates.json", gates)
    print("gates", json.dumps(gates))
    if not gates["pass_A_keep"]:
        print("WARN: A keep still <= 14 — do not treat SolidiFI as useful train aux")

    if args.export_sft:
        sft_dir = ROOT / "data" / "sft"
        for role in ("R", "A", "C"):
            n = export_sft_jsonl(labels_path, sft_dir / f"sft_solidifi_{role}.jsonl", family=role)
            print(f"SFT solidifi {role}: {n}")
        n_j = export_judge_sft(labels_path, sft_dir / "sft_solidifi_J.jsonl")
        print(f"SFT solidifi J: {n_j}")
        if not args.no_merge:
            comb = sft_dir / "combined"
            for role in ("R", "A", "C", "J"):
                n = merge_sft_jsonl(
                    [sft_dir / f"sft_{role}.jsonl", sft_dir / f"sft_solidifi_{role}.jsonl"],
                    comb / f"sft_{role}.jsonl",
                )
                print(f"SFT combined {role}: {n}")


if __name__ == "__main__":
    main()
