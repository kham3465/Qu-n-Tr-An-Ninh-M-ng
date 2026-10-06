from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Decision = Literal["keep", "drop"]
Label = Literal["keep", "drop", "unknown"]
Family = Literal["R", "A", "C", "O"]
ConfigId = Literal["A", "B", "C", "D", "E", "G"]


@dataclass
class Alert:
    alert_id: str
    detector: str
    family: Family | None
    lines: list[int] = field(default_factory=list)
    source: str = ""
    description: str = ""
    snippet: str = ""
    impact: str | None = None
    confidence: str | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Alert:
        lines = d.get("lines") or []
        if isinstance(lines, int):
            lines = [lines]
        fam = d.get("family")
        return cls(
            alert_id=str(d.get("alert_id") or d.get("id") or ""),
            detector=str(d.get("detector") or d.get("check") or ""),
            family=fam if fam in ("R", "A", "C", "O") else None,
            lines=[int(x) for x in lines],
            source=str(d.get("source") or d.get("file") or ""),
            description=str(d.get("description") or ""),
            snippet=str(d.get("snippet") or ""),
            impact=d.get("impact"),
            confidence=d.get("confidence"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExpertDecision:
    alert_id: str
    family: Family
    decision: Decision
    line: int | None = None
    reason: str = ""
    detector: str | None = None
    raw: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class JudgeFinding:
    alert_id: str
    detector: str | None = None
    line: int | None = None
    decision: Decision = "keep"
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class JudgeOutput:
    findings: list[JudgeFinding] = field(default_factory=list)
    invented_count: int = 0
    raw: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "findings": [f.to_dict() for f in self.findings],
            "invented_count": self.invented_count,
            "raw": self.raw,
        }


@dataclass
class ContractReport:
    sol_path: str
    config: ConfigId
    alerts: list[Alert]
    expert_decisions: list[ExpertDecision] = field(default_factory=list)
    judge: JudgeOutput | None = None
    predictions: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sol_path": self.sol_path,
            "config": self.config,
            "alerts": [a.to_dict() for a in self.alerts],
            "expert_decisions": [e.to_dict() for e in self.expert_decisions],
            "judge": self.judge.to_dict() if self.judge else None,
            "predictions": self.predictions,
        }
