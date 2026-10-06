# -*- coding: utf-8 -*-
"""Parse SmartBugs Curated vulnerabilities.json."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

DASP_TO_FAMILY = {
    "reentrancy": "R",
    "access_control": "A",
    "unchecked_low_level_calls": "C",
}

PRAGMA_RE = re.compile(r"pragma\s+solidity\s+([^;]+);", re.I)


def load_curated_index(vuln_json: Path) -> list[dict[str, Any]]:
    data = json.loads(vuln_json.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("vulnerabilities.json must be a list")
    return data


def index_by_stem(entries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for e in entries:
        name = e.get("name") or ""
        stem = Path(name).stem
        out[stem] = e
        path = e.get("path") or ""
        if path:
            out[Path(path).stem] = e
    return out


def index_by_path(entries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Key by posix relative path (dataset/cat/file.sol) to avoid duplicate stems."""
    out: dict[str, dict[str, Any]] = {}
    for e in entries:
        path = (e.get("path") or "").replace("\\", "/").strip()
        if path:
            out[path] = e
            out[path.removeprefix("dataset/")] = e
    return out


def lookup_entry(
    sol_path: Path,
    dataset_root: Path,
    by_path: dict[str, dict[str, Any]],
    by_stem: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    try:
        rel = sol_path.resolve().relative_to(dataset_root.resolve()).as_posix()
    except ValueError:
        rel = f"{sol_path.parent.name}/{sol_path.name}"
    return by_path.get(rel) or by_path.get(f"dataset/{rel}") or by_stem.get(sol_path.stem)


def human_findings_for_entry(entry: dict[str, Any]) -> list[dict[str, Any]]:
    findings = []
    for v in entry.get("vulnerabilities") or []:
        cat = v.get("category") or ""
        findings.append(
            {
                "category": cat,
                "family": DASP_TO_FAMILY.get(cat),
                "type": cat,
                "lines": list(v.get("lines") or []),
            }
        )
    return findings


def parse_pragma_from_sol(sol_path: Path) -> str | None:
    try:
        text = sol_path.read_text(encoding="utf-8", errors="ignore")[:2000]
    except OSError:
        return None
    m = PRAGMA_RE.search(text)
    if not m:
        return None
    raw = m.group(1).strip()
    # ^0.4.25 or >=0.4.22 <0.6.0 → pick first x.y.z
    ver = re.search(r"(\d+\.\d+\.\d+)", raw)
    if ver:
        return ver.group(1)
    ver2 = re.search(r"(\d+\.\d+)", raw)
    if ver2:
        return ver2.group(1) + ".0"
    return None


# solc-select often lacks exact ancient patch versions; snap to a known-good compiler.
_SOLC_SNAP = {
    4: "0.4.25",
    5: "0.5.17",
    6: "0.6.12",
    7: "0.7.6",
    8: "0.8.20",
}


def snap_solc(version: str) -> str:
    m = re.match(r"(\d+)\.(\d+)", version)
    if not m:
        return "0.4.25"
    # Solidity is 0.4 / 0.5 / 0.8 — use the minor when major is 0.
    key = int(m.group(2)) if m.group(1) == "0" else int(m.group(1))
    return _SOLC_SNAP.get(key, "0.4.25")


def load_compiled_versions(csv_path: Path) -> dict[str, str]:
    """Map file stem → compiled solc from SmartBugs versions.csv."""
    import csv

    out: dict[str, str] = {}
    if not csv_path.exists():
        return out
    with csv_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            file = (row.get("file") or "").strip()
            ver = (row.get("compiled version") or "").strip()
            if file and ver:
                out[Path(file).stem] = ver
    return out


def pick_solc_version(
    entry: dict[str, Any] | None,
    sol_path: Path,
    compiled_map: dict[str, str] | None = None,
) -> str:
    if compiled_map and sol_path.stem in compiled_map:
        return compiled_map[sol_path.stem]
    raw = None
    if entry and entry.get("pragma"):
        p = str(entry["pragma"]).strip().lstrip("^>=<~ ")
        if re.match(r"\d+\.\d+\.\d+$", p):
            raw = p
        elif re.match(r"\d+\.\d+$", p):
            raw = p + ".0"
    if raw is None:
        raw = parse_pragma_from_sol(sol_path) or "0.4.25"
    return raw
