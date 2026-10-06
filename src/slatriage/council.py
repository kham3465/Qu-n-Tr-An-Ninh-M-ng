"""Run comparison configs A–E–G on a list of alerts for one contract / batch."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .agents import ExpertAgent, JudgeAgent, single_shot_triage
from .families import assign_family
from .llm import LLMBackend, MockLLM
from .pipeline import invent_count
from .schema import Alert, ConfigId, ContractReport, ExpertDecision, Family, JudgeFinding, JudgeOutput


def assign_families(alerts: list[Alert], map_path: str | Path) -> list[Alert]:
    from .config_utils import load_yaml

    mapping = load_yaml(map_path)
    out = []
    for a in alerts:
        fam = a.family or assign_family(a.detector, mapping, allow_other=False)
        a.family = fam if fam in ("R", "A", "C") else None
        out.append(a)
    return out


def config_A(alerts: list[Alert], sol_path: str = "") -> ContractReport:
    """Slither raw: every in-scope alert = keep."""
    preds = []
    for a in alerts:
        if a.family is None:
            continue
        preds.append(
            {
                "alert_id": a.alert_id,
                "decision": "keep",
                "line": a.lines[0] if a.lines else None,
                "reason": "slither_raw",
                "family": a.family,
            }
        )
    return ContractReport(sol_path=sol_path, config="A", alerts=alerts, predictions=preds)


def config_B_or_G(
    alerts: list[Alert],
    llm: LLMBackend,
    *,
    config: ConfigId = "B",
    sol_path: str = "",
) -> ContractReport:
    scoped = [a for a in alerts if a.family]
    preds = single_shot_triage(llm, scoped)
    for p in preds:
        # attach family if known
        for a in scoped:
            if a.alert_id == p.get("alert_id"):
                p["family"] = a.family
                break
    return ContractReport(sol_path=sol_path, config=config, alerts=alerts, predictions=preds)


def _run_experts(
    alerts: list[Alert],
    backends: dict[Family, LLMBackend],
    *,
    prompt_only: bool,
) -> list[ExpertDecision]:
    decisions: list[ExpertDecision] = []
    for fam in ("R", "A", "C"):
        fam_alerts = [a for a in alerts if a.family == fam]
        if not fam_alerts:
            continue
        agent = ExpertAgent(fam, backends[fam], prompt_only=prompt_only)
        for a in fam_alerts:
            decisions.append(agent.decide(a))
    return decisions


def _predictions_from_judge(judge: JudgeOutput, alerts: list[Alert]) -> list[dict[str, Any]]:
    keep_ids = {f.alert_id for f in judge.findings if f.decision == "keep"}
    preds = []
    for a in alerts:
        if a.family is None:
            continue
        decision = "keep" if a.alert_id in keep_ids else "drop"
        line = next((f.line for f in judge.findings if f.alert_id == a.alert_id), None)
        preds.append(
            {
                "alert_id": a.alert_id,
                "decision": decision,
                "line": line if line is not None else (a.lines[0] if a.lines else None),
                "family": a.family,
                "reason": "judge_final",
            }
        )
    return preds


def config_C(
    alerts: list[Alert],
    llm: LLMBackend,
    *,
    sol_path: str = "",
) -> ContractReport:
    """Four roles prompt-only, shared base LLM (no LoRA)."""
    backends = {"R": llm, "A": llm, "C": llm}
    experts = _run_experts(alerts, backends, prompt_only=True)
    judge = JudgeAgent(llm, forbid_invent=True, prompt_only=True).merge(experts, alerts)
    expert_keeps = [e.to_dict() for e in experts if e.decision == "keep"]
    judge.invented_count = invent_count(expert_keeps, [f.to_dict() for f in judge.findings])
    return ContractReport(
        sol_path=sol_path,
        config="C",
        alerts=alerts,
        expert_decisions=experts,
        judge=judge,
        predictions=_predictions_from_judge(judge, alerts),
    )


def config_D_or_E(
    alerts: list[Alert],
    expert_backends: dict[Family, LLMBackend],
    judge_llm: LLMBackend,
    *,
    config: ConfigId = "D",
    judge_trained: bool = False,
    sol_path: str = "",
) -> ContractReport:
    experts = _run_experts(alerts, expert_backends, prompt_only=False)
    judge = JudgeAgent(judge_llm, forbid_invent=True, prompt_only=not judge_trained).merge(experts, alerts)
    expert_keeps = [e.to_dict() for e in experts if e.decision == "keep"]
    judge.invented_count = invent_count(expert_keeps, [f.to_dict() for f in judge.findings])
    return ContractReport(
        sol_path=sol_path,
        config=config,
        alerts=alerts,
        expert_decisions=experts,
        judge=judge,
        predictions=_predictions_from_judge(judge, alerts),
    )


def run_config(
    config: ConfigId,
    alerts: list[Alert],
    *,
    map_path: str | Path,
    base_llm: LLMBackend | None = None,
    expert_llms: dict[Family, LLMBackend] | None = None,
    judge_llm: LLMBackend | None = None,
    g_llm: LLMBackend | None = None,
    sol_path: str = "",
) -> ContractReport:
    alerts = assign_families(list(alerts), map_path)
    base_llm = base_llm or MockLLM()
    if config == "A":
        return config_A(alerts, sol_path=sol_path)
    if config == "B":
        return config_B_or_G(alerts, base_llm, config="B", sol_path=sol_path)
    if config == "G":
        return config_B_or_G(alerts, g_llm or base_llm, config="G", sol_path=sol_path)
    if config == "C":
        return config_C(alerts, base_llm, sol_path=sol_path)
    experts = expert_llms or {"R": base_llm, "A": base_llm, "C": base_llm}
    jllm = judge_llm or base_llm
    if config == "D":
        return config_D_or_E(alerts, experts, jllm, config="D", judge_trained=False, sol_path=sol_path)
    if config == "E":
        return config_D_or_E(alerts, experts, jllm, config="E", judge_trained=True, sol_path=sol_path)
    raise ValueError(f"Unknown config {config}")
