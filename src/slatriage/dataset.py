"""Build SFT datasets for LoRA R/A/C/J from labels.jsonl."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

from .io_utils import iter_jsonl, write_jsonl
from .prompts import training_record


def filter_labels(
    path: str | Path,
    *,
    family: str | None = None,
    skip_unknown: bool = True,
) -> Iterator[dict[str, Any]]:
    for row in iter_jsonl(path):
        label = row.get("label")
        if skip_unknown and label == "unknown":
            continue
        if label not in ("keep", "drop"):
            continue
        if family and row.get("family") != family:
            continue
        yield row


def export_sft_jsonl(
    labels_path: str | Path,
    out_path: str | Path,
    *,
    family: str,
) -> int:
    rows = []
    for row in filter_labels(labels_path, family=family):
        rec = training_record(row, row["label"])
        # chat-style for TRL
        rows.append(
            {
                "messages": [
                    {"role": "system", "content": rec["system"]},
                    {"role": "user", "content": rec["user"]},
                    {"role": "assistant", "content": rec["assistant"]},
                ],
                "family": family,
                "alert_id": row.get("alert_id"),
                "label": row["label"],
            }
        )
    return write_jsonl(out_path, rows)


def export_judge_sft(
    labels_path: str | Path,
    out_path: str | Path,
) -> int:
    """Positive Judge samples: only keep-labeled alerts as allowed findings."""
    import json

    by_source: dict[str, list[dict[str, Any]]] = {}
    for row in filter_labels(labels_path):
        key = str(row.get("source") or row.get("source_slither") or "unknown")
        by_source.setdefault(key, []).append(row)

    rows = []
    for source, items in by_source.items():
        keeps = [r for r in items if r["label"] == "keep"]
        findings = [
            {
                "alert_id": r.get("alert_id"),
                "detector": r.get("detector"),
                "line": (r.get("lines") or [None])[0],
                "decision": "keep",
            }
            for r in keeps
        ]
        assistant = json.dumps({"findings": findings, "invented_count": 0}, ensure_ascii=False)
        user = json.dumps(
            [{"alert_id": r.get("alert_id"), "detector": r.get("detector"), "label": r.get("label")} for r in items],
            ensure_ascii=False,
        )
        rows.append(
            {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are Judge. Do not invent findings. Output JSON findings + invented_count.",
                    },
                    {"role": "user", "content": f"Source={source}\nAlerts:\n{user}"},
                    {"role": "assistant", "content": assistant},
                ],
                "family": "J",
            }
        )
    return write_jsonl(out_path, rows)
