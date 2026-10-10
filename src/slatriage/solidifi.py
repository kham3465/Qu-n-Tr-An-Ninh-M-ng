"""Parse SolidiFI-benchmark injection logs → human findings (R/A/C)."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterator

# Folder name → family. Do not trust the CSV "bug type" field (garbled in Re-entrancy).
FOLDER_TO_FAMILY = {
    "Re-entrancy": "R",
    "tx.origin": "A",
    "Unchecked-Send": "C",
    "Unhandled-Exceptions": "C",
}

FOLDER_TO_CATEGORY = {
    "Re-entrancy": "reentrancy",
    "tx.origin": "access_control",
    "Unchecked-Send": "unchecked_low_level_calls",
    "Unhandled-Exceptions": "unchecked_low_level_calls",
}


def iter_solidifi_contracts(root: Path, *, folders: list[str] | None = None) -> Iterator[dict[str, Any]]:
    root = Path(root)
    names = folders or list(FOLDER_TO_FAMILY)
    for folder in names:
        fam = FOLDER_TO_FAMILY.get(folder)
        if not fam:
            continue
        d = root / folder
        if not d.is_dir():
            continue
        for sol in sorted(d.glob("buggy_*.sol")):
            stem = sol.stem  # buggy_1
            num = stem.split("_", 1)[-1]
            log = d / f"BugLog_{num}.csv"
            yield {
                "sol_path": sol,
                "log_path": log if log.exists() else None,
                "folder": folder,
                "family": fam,
                "category": FOLDER_TO_CATEGORY[folder],
                "source": f"solidifi/{folder}/{sol.name}",
            }


def load_buglog(log_path: Path, *, family: str, category: str) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    if not log_path.exists():
        return findings
    with log_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                loc = int(str(row.get("loc") or "0").strip())
                length = int(str(row.get("length") or "1").strip())
            except ValueError:
                continue
            if loc <= 0:
                continue
            length = max(1, length)
            lines = list(range(loc, loc + length))
            findings.append(
                {
                    "family": family,
                    "category": category,
                    "type": category,
                    "lines": lines,
                    "line": loc,
                }
            )
    return findings


def injection_stats(root: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    for item in iter_solidifi_contracts(root):
        n = len(load_buglog(item["log_path"], family=item["family"], category=item["category"])) if item["log_path"] else 0
        out[item["folder"]] = out.get(item["folder"], 0) + n
    return out
