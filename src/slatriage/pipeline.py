"""Judge grounding check (the dataclasses live in schema.py)."""

from __future__ import annotations

from typing import Any


def invent_count(expert_findings: list[dict[str, Any]], judge_findings: list[dict[str, Any]]) -> int:
    """Count Judge findings not grounded in expert keep-list (by alert_id or line+type)."""
    allowed_ids = {f.get("alert_id") for f in expert_findings if f.get("decision") == "keep"}
    allowed_lines = {
        (f.get("detector") or f.get("type"), f.get("line"))
        for f in expert_findings
        if f.get("decision") == "keep"
    }
    n = 0
    for j in judge_findings:
        aid = j.get("alert_id")
        key = (j.get("detector") or j.get("type"), j.get("line"))
        if aid in allowed_ids or key in allowed_lines:
            continue
        n += 1
    return n
