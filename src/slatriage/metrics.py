"""Metrics for keep/drop, invent, and per-family scores."""

from __future__ import annotations

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


def score_aligned(
    gold_rows: list[dict[str, Any]],
    pred_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    gold_by = {(str(r.get("source") or ""), str(r.get("alert_id") or "")): r for r in gold_rows}
    gold_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in gold_rows:
        gold_id[str(r.get("alert_id") or "")].append(r)

    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for p in pred_rows:
        key = (str(p.get("source") or ""), str(p.get("alert_id") or ""))
        g = gold_by.get(key)
        if g is None:
            hits = gold_id.get(str(p.get("alert_id") or ""), [])
            g = hits[0] if len(hits) == 1 else None
        if not g or g.get("label") == "unknown":
            continue
        pairs.append((g, p))

    def _pack(items: list[tuple[dict[str, Any], dict[str, Any]]]) -> dict[str, Any]:
        yt = [g["label"] for g, _ in items]
        yp = [p.get("decision") or p.get("label") for _, p in items]
        return binary_prf(yt, yp) if items else {"error": "no overlap", "n": 0}

    out: dict[str, Any] = _pack(pairs)
    out["n_scored"] = len(pairs)
    by_fam: dict[str, Any] = {}
    for fam in ("R", "A", "C"):
        sub = [(g, p) for g, p in pairs if (g.get("family") or p.get("family")) == fam]
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
