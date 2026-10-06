"""Prompt templates for configs B/C/D/E/G — temperature 0, JSON-only answers."""

from __future__ import annotations

import json
from typing import Any

from .schema import Alert, ExpertDecision, Family

FAMILY_DESC = {
    "R": "Reentrancy and unsafe ether/value transfer flows. Do NOT judge tx.origin or access-control-only issues.",
    "A": "Access control and tx.origin authentication issues. Do NOT judge reentrancy.",
    "C": "Unchecked low-level calls and dangerous delegatecall. Do NOT judge other families.",
}


def _alert_block(alert: Alert) -> str:
    return json.dumps(
        {
            "alert_id": alert.alert_id,
            "detector": alert.detector,
            "family": alert.family,
            "lines": alert.lines,
            "source": alert.source,
            "description": alert.description,
            "snippet": alert.snippet,
        },
        ensure_ascii=False,
        indent=2,
    )


def expert_system(family: Family, trained_hint: bool = True) -> str:
    scope = FAMILY_DESC[family]
    extra = (
        "You are a specialist adjudicator for static-analysis alerts."
        if trained_hint
        else "You are a specialist role (prompt-only, not fine-tuned)."
    )
    return (
        f"{extra} Your family is {family}: {scope}\n"
        "Decide keep (true positive / real issue) or drop (false positive / out of scope noise).\n"
        "Reply with ONLY one JSON object:\n"
        '{"alert_id":"...","decision":"keep"|"drop","line":<int|null>,"reason":"1-2 sentences"}'
    )


def expert_user(alert: Alert) -> str:
    return (
        "Adjudicate this Slither alert.\n"
        f"{_alert_block(alert)}\n"
        "Output JSON only."
    )


def single_prompt_system() -> str:
    """Config B / G: one model sees all alerts."""
    return (
        "You triage Slither static-analysis alerts for Solidity.\n"
        "For EACH alert decide keep or drop. Do not invent new vulnerability types or lines "
        "that are not supported by the alert list.\n"
        "Reply with ONLY a JSON array:\n"
        '[{"alert_id":"...","decision":"keep"|"drop","line":<int|null>,"reason":"..."}]'
    )


def single_prompt_user(alerts: list[Alert]) -> str:
    payload = [_alert_block(a) for a in alerts]
    return "Alerts:\n" + "\n".join(payload) + "\nOutput JSON array only."


def judge_system(forbid_invent: bool = True) -> str:
    ban = (
        "You MUST NOT invent findings. You may only keep alerts that experts marked keep, "
        "and only with a line that appears in the expert output or source mapping."
        if forbid_invent
        else "Merge expert opinions into a final list."
    )
    return (
        "You are the Judge for a multi-agent Slither alert council.\n"
        f"{ban}\n"
        "Reply with ONLY JSON:\n"
        '{"findings":[{"alert_id":"...","detector":"...","line":<int|null>,"decision":"keep"}],'
        '"invented_count":0}'
    )


def judge_user(expert_decisions: list[ExpertDecision], alerts: list[Alert]) -> str:
    return (
        "Expert decisions:\n"
        + json.dumps([e.to_dict() for e in expert_decisions], ensure_ascii=False, indent=2)
        + "\n\nOriginal alerts:\n"
        + json.dumps([a.to_dict() for a in alerts], ensure_ascii=False, indent=2)
        + "\nOutput Judge JSON only."
    )


def role_prompt_council_system(role: str) -> str:
    """Config C: prompt-only four roles (not org titles from LLM-SmartAudit)."""
    tips = {
        "R": FAMILY_DESC["R"],
        "A": FAMILY_DESC["A"],
        "C": FAMILY_DESC["C"],
        "J": "Merge and forbid inventing new findings.",
    }
    return expert_system(role, trained_hint=False) if role in "RAC" else judge_system(True) + f"\nRole note: {tips.get(role, '')}"


def training_record(alert: dict[str, Any], label: str) -> dict[str, str]:
    """SFT chat sample for LoRA R/A/C."""
    fam = alert.get("family") or "R"
    a = Alert.from_dict(alert)
    line = (a.lines[0] if a.lines else None)
    assistant = json.dumps(
        {
            "alert_id": a.alert_id,
            "decision": label,
            "line": line,
            "reason": f"Labeled {label} for family {fam}.",
        },
        ensure_ascii=False,
    )
    return {
        "system": expert_system(fam if fam in ("R", "A", "C") else "R"),
        "user": expert_user(a),
        "assistant": assistant,
    }
