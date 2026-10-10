"""Build SFT datasets for LoRA R/A/C/J from labels.jsonl."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

from .io_utils import iter_jsonl, write_jsonl
from .prompts import training_record

MAX_DESC_CHARS = 800
MAX_SNIPPET_CHARS = 1200


def compact_alert(row: dict[str, Any]) -> dict[str, Any]:
    """Shorten Slither text so QLoRA max_seq=2048 fits on 16GB."""
    out = dict(row)
    desc = str(out.get("description") or "")
    if len(desc) > MAX_DESC_CHARS:
        out["description"] = desc[:MAX_DESC_CHARS] + "\n..."
    snip = str(out.get("snippet") or "")
    if len(snip) > MAX_SNIPPET_CHARS:
        out["snippet"] = snip[:MAX_SNIPPET_CHARS] + "\n..."
    return out


def validate_sft_row(row: dict[str, Any]) -> None:
    msgs = row.get("messages")
    if not isinstance(msgs, list) or len(msgs) < 2:
        raise ValueError("SFT row needs messages[]")
    roles = [m.get("role") for m in msgs if isinstance(m, dict)]
    if "user" not in roles or "assistant" not in roles:
        raise ValueError("SFT row needs user + assistant")
    if not str(msgs[-1].get("content") or "").strip():
        raise ValueError("empty assistant")


def load_sft_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    for row in iter_jsonl(path):
        validate_sft_row(row)
        rows.append(row)
    if not rows:
        raise ValueError(f"empty SFT file: {path}")
    return rows


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
        rec = training_record(compact_alert(row), row["label"])
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
                "split": row.get("split") or "curated",
            }
        )
    return write_jsonl(out_path, rows)


MAX_JUDGE_ITEMS = 16


def export_judge_sft(
    labels_path: str | Path,
    out_path: str | Path,
    *,
    max_items: int = MAX_JUDGE_ITEMS,
) -> int:
    """Judge samples: keep-only findings. Chunk per source so rows fit max_seq=2048."""
    import json

    by_source: dict[str, list[dict[str, Any]]] = {}
    for row in filter_labels(labels_path):
        key = str(row.get("source") or row.get("source_slither") or "unknown")
        by_source.setdefault(key, []).append(row)

    rows = []
    for source, items in by_source.items():
        size = max(1, int(max_items))
        for off in range(0, len(items), size):
            chunk = items[off : off + size]
            keeps = [r for r in chunk if r["label"] == "keep"]
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
                [{"alert_id": r.get("alert_id"), "detector": r.get("detector"), "label": r.get("label")} for r in chunk],
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
                    "split": chunk[0].get("split") or "curated",
                    "source": source,
                }
            )
    return write_jsonl(out_path, rows)


def merge_sft_jsonl(parts: list[str | Path], out_path: str | Path) -> int:
    """Concatenate SFT files. Curated first, then auxiliary (SolidiFI). Dedup by alert_id+split."""
    rows: list[dict[str, Any]] = []
    seen: set[tuple] = set()
    for part in parts:
        path = Path(part)
        if not path.exists() or path.stat().st_size < 2:
            continue
        default_split = "solidifi" if "solidifi" in path.name else "curated"
        for row in iter_jsonl(path):
            validate_sft_row(row)
            if not row.get("split"):
                row["split"] = default_split
            key = (row.get("alert_id"), row.get("family"), row.get("split"), row.get("label"))
            if row.get("alert_id"):
                if key in seen:
                    continue
                seen.add(key)
            rows.append(row)
    return write_jsonl(out_path, rows)
