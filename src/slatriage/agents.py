from __future__ import annotations

from typing import Any

from .llm import LLMBackend, extract_json_object
from .prompts import (
    expert_system,
    expert_user,
    judge_system,
    judge_user,
    role_prompt_council_system,
    single_prompt_system,
    single_prompt_user,
)
from .schema import Alert, ExpertDecision, Family, JudgeFinding, JudgeOutput


class ExpertAgent:
    def __init__(self, family: Family, llm: LLMBackend, *, prompt_only: bool = False):
        self.family = family
        self.llm = llm
        self.prompt_only = prompt_only

    def decide(self, alert: Alert) -> ExpertDecision:
        system = (
            role_prompt_council_system(self.family)
            if self.prompt_only
            else expert_system(self.family, trained_hint=True)
        )
        raw = self.llm.generate(system, expert_user(alert), temperature=0.0)
        try:
            data = extract_json_object(raw)
        except ValueError:
            return ExpertDecision(
                alert_id=alert.alert_id,
                family=self.family,
                decision="drop",
                line=alert.lines[0] if alert.lines else None,
                reason="parse_fail",
                detector=alert.detector,
                raw=raw,
            )
        decision = data.get("decision", "drop")
        if decision not in ("keep", "drop"):
            decision = "drop"
        return ExpertDecision(
            alert_id=str(data.get("alert_id") or alert.alert_id),
            family=self.family,
            decision=decision,
            line=data.get("line"),
            reason=str(data.get("reason") or ""),
            detector=alert.detector,
            raw=raw,
        )


class JudgeAgent:
    def __init__(self, llm: LLMBackend, *, forbid_invent: bool = True, prompt_only: bool = False):
        self.llm = llm
        self.forbid_invent = forbid_invent
        self.prompt_only = prompt_only

    def merge(self, experts: list[ExpertDecision], alerts: list[Alert]) -> JudgeOutput:
        system = judge_system(forbid_invent=self.forbid_invent)
        if self.prompt_only:
            system = role_prompt_council_system("J")
        raw = self.llm.generate(system, judge_user(experts, alerts), temperature=0.0)
        try:
            data = extract_json_object(raw)
        except ValueError:
            # fallback: keep expert keeps only
            findings = [
                JudgeFinding(
                    alert_id=e.alert_id,
                    detector=e.detector,
                    line=e.line,
                    decision="keep",
                    reason="fallback_no_parse",
                )
                for e in experts
                if e.decision == "keep"
            ]
            return JudgeOutput(findings=findings, invented_count=0, raw=raw)

        findings_raw = data.get("findings") or []
        findings = [
            JudgeFinding(
                alert_id=str(f.get("alert_id")),
                detector=f.get("detector"),
                line=f.get("line"),
                decision=f.get("decision") or "keep",
                reason=str(f.get("reason") or ""),
            )
            for f in findings_raw
            if isinstance(f, dict)
        ]
        invented = int(data.get("invented_count") or 0)
        return JudgeOutput(findings=findings, invented_count=invented, raw=raw)


def single_shot_triage(llm: LLMBackend, alerts: list[Alert]) -> list[dict[str, Any]]:
    raw = llm.generate(single_prompt_system(), single_prompt_user(alerts), temperature=0.0)
    try:
        data = extract_json_object(raw)
    except ValueError:
        return [
            {"alert_id": a.alert_id, "decision": "drop", "line": None, "reason": "parse_fail", "raw": raw}
            for a in alerts
        ]
    if isinstance(data, dict):
        data = [data]
    out = []
    for item in data:
        if not isinstance(item, dict):
            continue
        d = item.get("decision", "drop")
        if d not in ("keep", "drop"):
            d = "drop"
        out.append(
            {
                "alert_id": item.get("alert_id"),
                "decision": d,
                "line": item.get("line"),
                "reason": item.get("reason"),
            }
        )
    return out
