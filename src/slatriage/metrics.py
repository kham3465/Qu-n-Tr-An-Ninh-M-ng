"""Metrics for keep/drop, invent, and per-family scores."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Iterable


def binary_prf(y_true: Iterable[str], y_pred: Iterable[str], positive: str = "keep") -> dict[str, float]:
    yt = list(y_true)
    yp = list(y_pred)
    if len(yt) != len(yp):
        raise ValueError("y_true/y_pred length mismatch")
    tp = sum(1 for t, p in zip(yt, yp) if t == positive and p == positive)
    fp = sum(1 for t, p in zip(yt, yp) if t != positive and p == positive)
    fn = sum(1 for t, p in zip(yt, yp) if t == positive and p != positive)
    tn = sum(1 for t, p in zip(yt, yp) if t != positive and p != positive)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "n": len(yt),
    }


def relative_gain(p_e: float, p_b: float) -> float | None:
    if p_b == 0:
        return None
    return (p_e - p_b) / p_b


def invent_rate(invented: int, n_findings: int) -> float:
    if n_findings <= 0:
        return 0.0 if invented <= 0 else 1.0
    return invented / n_findings


def bootstrap_f1_ci(
    y_true: list[str],
    y_pred: list[str],
    *,
    n_boot: int = 1000,
    seed: int = 0,
    positive: str = "keep",
) -> list[float] | None:
    """95% percentile bootstrap interval for F1. Families have few alerts, so F1 alone is noisy."""
    n = len(y_true)
    if n < 2:
        return None
    rng = random.Random(seed)
    scores: list[float] = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        scores.append(binary_prf([y_true[i] for i in idx], [y_pred[i] for i in idx], positive)["f1"])
    scores.sort()
    return [round(scores[int(0.025 * n_boot)], 4), round(scores[int(0.975 * n_boot) - 1], 4)]


def score_aligned(
    gold_rows: list[dict[str, Any]],
    pred_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Score predictions against gold, walking the GOLD rows.

    A gold alert (keep/drop) with no prediction counts as `drop` and is reported in `n_missing`,
    so a model cannot raise its F1 by silently skipping hard alerts. `parse_fail` counts
    predictions that fell back to drop because the model output was not valid JSON.
    """
    pred_by = {(str(p.get("source") or ""), str(p.get("alert_id") or "")): p for p in pred_rows}
    pred_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for p in pred_rows:
        pred_id[str(p.get("alert_id") or "")].append(p)

    pairs: list[tuple[dict[str, Any], dict[str, Any] | None]] = []
    for g in gold_rows:
        if g.get("label") not in ("keep", "drop"):
            continue
        key = (str(g.get("source") or ""), str(g.get("alert_id") or ""))
        p = pred_by.get(key)
        if p is None:
            hits = pred_id.get(key[1], [])
            p = hits[0] if len(hits) == 1 else None
        pairs.append((g, p))

    def _decision(p: dict[str, Any] | None) -> str:
        if p is None:
            return "drop"
        return p.get("decision") or p.get("label") or "drop"

    def _pack(items: list[tuple[dict[str, Any], dict[str, Any] | None]]) -> dict[str, Any]:
        if not items:
            return {"error": "no overlap", "n": 0}
        yt = [g["label"] for g, _ in items]
        yp = [_decision(p) for _, p in items]
        out = binary_prf(yt, yp)
        out["f1_ci"] = bootstrap_f1_ci(yt, yp)
        out["n_missing"] = sum(1 for _, p in items if p is None)
        out["parse_fail"] = sum(1 for _, p in items if p is not None and p.get("reason") == "parse_fail")
        return out

    out: dict[str, Any] = _pack(pairs)
    out["n_scored"] = len(pairs)
    by_fam: dict[str, Any] = {}
    for fam in ("R", "A", "C"):
        sub = [(g, p) for g, p in pairs if (g.get("family") or (p or {}).get("family")) == fam]
        by_fam[fam] = _pack(sub)
        by_fam[fam]["n_scored"] = len(sub)
    out["by_family"] = by_fam
    out["confusion"] = {
        "tp": out.get("tp", 0),
        "fp": out.get("fp", 0),
        "fn": out.get("fn", 0),
        "tn": out.get("tn", 0),
    }
    return out


def filter_by_sources(rows: list[dict[str, Any]], sources: set[str]) -> list[dict[str, Any]]:
    return [r for r in rows if str(r.get("source") or "") in sources]
