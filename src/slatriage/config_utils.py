from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def detector_to_family(detector: str, mapping: dict[str, Any]) -> str | None:
    """Return 'R' | 'A' | 'C' or None if out of scope."""
    families = mapping.get("families", {})
    for fam, meta in families.items():
        if detector in meta.get("detectors", []):
            return fam
    return None
