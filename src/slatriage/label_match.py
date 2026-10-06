"""Match Slither alerts to human labels → keep | drop | unknown (P1)."""

from __future__ import annotations

from typing import Any, Iterable


def match_alert(
    alert: dict[str, Any],
    human_findings: Iterable[dict[str, Any]],
    *,
    line_window: int = 5,
    family: str | None = None,
) -> str:
    """
    human_finding keys expected: type/family, lines (list[int]) or line (int).
    Returns keep | drop | unknown.
    """
    alert_lines = set(alert.get("lines") or [])
    if not alert_lines:
        return "unknown"

    from .curated import DASP_TO_FAMILY

    candidates = []
    for h in human_findings:
        h_fam = h.get("family")
        if not h_fam:
            raw = str(h.get("type") or h.get("category") or "")
            h_fam = raw if raw in ("R", "A", "C") else DASP_TO_FAMILY.get(raw)
        if family and h_fam and str(h_fam).upper() != str(family).upper():
            continue
        h_lines = set(h.get("lines") or [])
        if "line" in h:
            h_lines.add(int(h["line"]))
        if not h_lines:
            continue
        # any line within window
        hit = any(abs(a - b) <= line_window for a in alert_lines for b in h_lines)
        if hit:
            candidates.append(h)

    if len(candidates) == 1:
        return "keep"
    if len(candidates) > 1:
        return "unknown"
    return "drop"
