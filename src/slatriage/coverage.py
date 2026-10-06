# -*- coding: utf-8 -*-
"""Ma trận đóng: mỗi (contract, hạng DASP) → KEEP | DROP | ABSENT | UNKNOWN."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

from .config_utils import load_yaml
from .families import family_to_dasp

Cell = str  # KEEP | DROP | ABSENT | UNKNOWN


def coverage_cell(
    *,
    has_keep: bool,
    has_drop: bool,
    has_unknown: bool,
    has_alert: bool,
) -> Cell:
    if has_keep:
        return "KEEP"
    if has_unknown and not has_drop:
        return "UNKNOWN"
    if has_drop:
        return "DROP"
    if has_alert:
        return "UNKNOWN"
    return "ABSENT"


def build_matrix(
    alerts: Iterable[dict[str, Any]],
    closed: dict[str, Any],
    *,
    source_key: str = "source",
) -> list[dict[str, Any]]:
    """
    alerts: cần family, label hoặc decision, source.
    Không có alert cho một hạng → ABSENT (Coverage Officer, cấm KEEP).
    """
    cats = [c["id"] for c in closed.get("categories", [])]
    by_src: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for a in alerts:
        src = str(a.get(source_key) or a.get("source_slither") or "unknown")
        by_src[src].append(a)

    rows: list[dict[str, Any]] = []
    for src, items in sorted(by_src.items()):
        by_cat: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for a in items:
            fam = a.get("family")
            cid = family_to_dasp(fam, closed)
            if cid:
                by_cat[cid].append(a)
        for cid in cats:
            bucket = by_cat.get(cid, [])
            labels = [(x.get("decision") or x.get("label") or "").lower() for x in bucket]
            cell = coverage_cell(
                has_keep=any(x == "keep" for x in labels),
                has_drop=any(x == "drop" for x in labels),
                has_unknown=any(x == "unknown" for x in labels),
                has_alert=bool(bucket),
            )
            rows.append(
                {
                    "source": src,
                    "category": cid,
                    "cell": cell,
                    "n_alerts": len(bucket),
                }
            )
    return rows


def matrix_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    from collections import Counter

    by_cat: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        by_cat[r["category"]][r["cell"]] += 1
    return {k: dict(v) for k, v in sorted(by_cat.items())}


def load_closed(path) -> dict[str, Any]:
    return load_yaml(path)
