# -*- coding: utf-8 -*-
"""Gán họ R/A/C/O theo tên detector (không để LLM tự chọn)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from .config_utils import load_yaml

FamilyId = Literal["R", "A", "C", "O"]


def load_mapping(path: str | Path) -> dict[str, Any]:
    return load_yaml(path)


def detector_to_family(detector: str, mapping: dict[str, Any]) -> FamilyId | None:
    """R/A/C nếu nằm bảng chính; None = ngoài thang (không train F1 chính)."""
    families = mapping.get("families", {})
    for fam in ("R", "A", "C"):
        meta = families.get(fam) or {}
        if detector in meta.get("detectors", []):
            return fam  # type: ignore[return-value]
    return None


def assign_family(detector: str, mapping: dict[str, Any], *, allow_other: bool = True) -> FamilyId | None:
    """
    Một alert một họ.
    - R/A/C nếu detector trong map.
    - O nếu allow_other và detector nằm other_detectors (có tín hiệu nhưng chưa đủ lập họ).
    - None nếu informational / ngoài danh sách đóng.
    """
    fam = detector_to_family(detector, mapping)
    if fam:
        return fam
    other = mapping.get("other_detectors") or []
    if allow_other and detector in other:
        return "O"
    return None


def family_to_dasp(family: str | None, closed: dict[str, Any]) -> str | None:
    for cat in closed.get("categories", []):
        if cat.get("family_default") == family:
            return cat.get("id")
    return None
