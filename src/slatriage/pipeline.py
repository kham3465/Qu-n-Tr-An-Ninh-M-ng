"""End-to-end flow: .sol → Slither → family agents → Judge (P2 wires models later)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentDecision:
    family: str
    alert_id: str
    decision: str  # keep | drop
    line: int | None = None
    reason: str = ""


@dataclass
class JudgeOutput:
    findings: list[dict[str, Any]] = field(default_factory=list)
    invented_count: int = 0


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
