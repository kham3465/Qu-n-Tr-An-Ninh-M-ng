"""Metrics for keep/drop and invent rate (P3)."""

from __future__ import annotations

from typing import Iterable


def binary_prf(y_true: Iterable[str], y_pred: Iterable[str], positive: str = "keep") -> dict[str, float]:
    yt = list(y_true)
    yp = list(y_pred)
    assert len(yt) == len(yp)
    tp = sum(1 for t, p in zip(yt, yp) if t == positive and p == positive)
    fp = sum(1 for t, p in zip(yt, yp) if t != positive and p == positive)
    fn = sum(1 for t, p in zip(yt, yp) if t == positive and p != positive)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def relative_gain(p_e: float, p_b: float) -> float | None:
    """(P_E - P_B) / P_B ; None if P_B == 0."""
    if p_b == 0:
        return None
    return (p_e - p_b) / p_b
