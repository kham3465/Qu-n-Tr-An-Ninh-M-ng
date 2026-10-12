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
        raw = self.llm.generate(system, expert_user(alert), temperature=0.0, max_new_tokens=160)
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


INFER_ALERT_CHUNK = 4
INFER_JUDGE_CHUNK = 6


class JudgeAgent:
    def __init__(self, llm: LLMBackend, *, forbid_invent: bool = True, prompt_only: bool = False):
        self.llm = llm
        self.forbid_invent = forbid_invent
        self.prompt_only = prompt_only

    def merge(self, experts: list[ExpertDecision], alerts: list[Alert]) -> JudgeOutput:
        if not alerts:
            return JudgeOutput(findings=[], invented_count=0, raw="")
        if len(alerts) <= INFER_JUDGE_CHUNK:
            return self._merge_chunk(experts, alerts)
        findings: list[JudgeFinding] = []
        invented = 0
        raws: list[str] = []
        for i in range(0, len(alerts), INFER_JUDGE_CHUNK):
            part = alerts[i : i + INFER_JUDGE_CHUNK]
            ids = {a.alert_id for a in part}
            sub = self._merge_chunk([e for e in experts if e.alert_id in ids], part)
            findings.extend(sub.findings)
            invented += int(sub.invented_count or 0)
            if sub.raw:
                raws.append(sub.raw)
        return JudgeOutput(findings=findings, invented_count=invented, raw="\n".join(raws))

    def _merge_chunk(self, experts: list[ExpertDecision], alerts: list[Alert]) -> JudgeOutput:
        system = judge_system(forbid_invent=self.forbid_invent)
        if self.prompt_only:
            system = role_prompt_council_system("J")
        raw = self.llm.generate(system, judge_user(experts, alerts), temperature=0.0, max_new_tokens=256)
        try:
            data = extract_json_object(raw)
        except ValueError:
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

        if isinstance(data, list):
            findings_raw = data
            invented = 0
        elif isinstance(data, dict):
            findings_raw = data.get("findings")
            if not isinstance(findings_raw, list):
                findings_raw = [data] if data.get("alert_id") else []
            invented = int(data.get("invented_count") or 0)
        else:
            findings_raw = []
            invented = 0
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
        return JudgeOutput(findings=findings, invented_count=invented, raw=raw)


def _parse_single_shot(raw: str, alerts: list[Alert]) -> list[dict[str, Any]]:
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


def single_shot_triage(llm: LLMBackend, alerts: list[Alert]) -> list[dict[str, Any]]:
    if not alerts:
        return []
    if len(alerts) <= INFER_ALERT_CHUNK:
        raw = llm.generate(
            single_prompt_system(),
            single_prompt_user(alerts),
            temperature=0.0,
            max_new_tokens=256,
        )
        return _parse_single_shot(raw, alerts)
    out: list[dict[str, Any]] = []
    for i in range(0, len(alerts), INFER_ALERT_CHUNK):
        out.extend(single_shot_triage(llm, alerts[i : i + INFER_ALERT_CHUNK]))
    return out
