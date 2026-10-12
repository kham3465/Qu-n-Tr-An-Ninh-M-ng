"""Build SFT datasets for LoRA R/A/C/J from labels.jsonl."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import json
import random

from .agents import INFER_JUDGE_CHUNK
from .io_utils import iter_jsonl, write_jsonl
from .prompts import judge_system, judge_user, training_record
from .schema import Alert, ExpertDecision

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
    exclude_sources: set[str] | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield keep/drop rows. `exclude_sources` drops whole files (hold-out) so train never sees test."""
    for row in iter_jsonl(path):
        if exclude_sources and str(row.get("source") or "") in exclude_sources:
            continue
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
    exclude_sources: set[str] | None = None,
) -> int:
    rows = []
    for row in filter_labels(labels_path, family=family, exclude_sources=exclude_sources):
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
                "source": row.get("source"),
            }
        )
    return write_jsonl(out_path, rows)


# Same chunk size as inference (agents.INFER_JUDGE_CHUNK) so train and test prompts look alike.
MAX_JUDGE_ITEMS = INFER_JUDGE_CHUNK
EXPERT_NOISE = 0.1


def _simulated_expert(row: dict[str, Any], *, noise: float, seed: int) -> ExpertDecision:
    """Stand-in for a family expert: gold label, flipped with probability `noise`.

    The Judge must NOT see the gold label at train time (it never does at inference).
    It sees expert decisions, some of them wrong, and learns the one rule that matters:
    keep exactly what the experts kept, invent nothing.
    """
    gold = row["label"]
    flip = random.Random(f"{seed}:{row.get('alert_id')}").random() < noise
    decision = ("drop" if gold == "keep" else "keep") if flip else gold
    return ExpertDecision(
        alert_id=str(row.get("alert_id")),
        family=row.get("family") or "R",
        decision=decision,  # type: ignore[arg-type]
        line=(row.get("lines") or [None])[0],
        reason="",
        detector=row.get("detector"),
    )


def export_judge_sft(
    labels_path: str | Path,
    out_path: str | Path,
    *,
    max_items: int = MAX_JUDGE_ITEMS,
    exclude_sources: set[str] | None = None,
    noise: float = EXPERT_NOISE,
    seed: int = 0,
) -> int:
    """Judge samples in the exact inference format (`judge_system` / `judge_user`).

    Input = simulated expert decisions + alerts. Target = findings for expert keeps only.
    Chunked per source so rows fit max_seq=2048.
    """
    by_source: dict[str, list[dict[str, Any]]] = {}
    for row in filter_labels(labels_path, exclude_sources=exclude_sources):
        key = str(row.get("source") or row.get("source_slither") or "unknown")
        by_source.setdefault(key, []).append(row)
    return write_jsonl(out_path, _judge_rows(by_source, max_items=max_items, noise=noise, seed=seed))


def _judge_rows(
    by_source: dict[str, list[dict[str, Any]]], *, max_items: int, noise: float, seed: int
) -> list[dict[str, Any]]:
    rows = []
    for source, items in by_source.items():
        size = max(1, int(max_items))
        for off in range(0, len(items), size):
            chunk = items[off : off + size]
            experts = [_simulated_expert(r, noise=noise, seed=seed) for r in chunk]
            alerts = [Alert.from_dict(r) for r in chunk]
            findings = [
                {"alert_id": e.alert_id, "detector": e.detector, "line": e.line, "decision": "keep"}
                for e in experts
                if e.decision == "keep"
            ]
            assistant = json.dumps({"findings": findings, "invented_count": 0}, ensure_ascii=False)
            rows.append(
                {
                    "messages": [
                        {"role": "system", "content": judge_system(forbid_invent=True)},
                        {"role": "user", "content": judge_user(experts, alerts)},
                        {"role": "assistant", "content": assistant},
                    ],
                    "family": "J",
                    "split": chunk[0].get("split") or "curated",
                    "source": source,
                }
            )
    return rows


def rewrite_legacy_judge_sft(
    old_path: str | Path,
    out_path: str | Path,
    mapping: dict[str, Any],
    *,
    max_items: int = MAX_JUDGE_ITEMS,
    noise: float = EXPERT_NOISE,
    seed: int = 0,
    split: str = "solidifi",
) -> int:
    """Convert old Judge SFT (user prompt carried the gold `label`) to the expert-decision format.

    Used for SolidiFI, whose label file is not in the repo. Lines are only known for old `keep`
    rows (they sat in the assistant answer); the rest get no line.
    """
    from .families import assign_family

    by_source: dict[str, list[dict[str, Any]]] = {}
    for old in iter_jsonl(old_path):
        msgs = old["messages"]
        user = next(m["content"] for m in msgs if m["role"] == "user")
        source = str(old.get("source") or user.split("\n", 1)[0].removeprefix("Source="))
        items = json.loads(user.split("Alerts:\n", 1)[1])
        answer = json.loads(msgs[-1]["content"])
        line_of = {f.get("alert_id"): f.get("line") for f in answer.get("findings") or []}
        for it in items:
            line = line_of.get(it.get("alert_id"))
            by_source.setdefault(source, []).append(
                {
                    "alert_id": it.get("alert_id"),
                    "detector": it.get("detector"),
                    "family": assign_family(it.get("detector") or "", mapping, allow_other=False),
                    "label": it["label"],
                    "lines": [line] if line is not None else [],
                    "split": old.get("split") or split,
                }
            )
    return write_jsonl(out_path, _judge_rows(by_source, max_items=max_items, noise=noise, seed=seed))


def merge_sft_jsonl(
    parts: list[str | Path],
    out_path: str | Path,
    *,
    exclude_sources: set[str] | None = None,
) -> int:
    """Concatenate SFT files. Curated first, then auxiliary (SolidiFI). Dedup by alert_id+split.

    `exclude_sources` is a second guard against hold-out leakage: rows whose `source` is listed are dropped.
    """
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
            if exclude_sources and str(row.get("source") or "") in exclude_sources:
                continue
            key = (row.get("alert_id"), row.get("family"), row.get("split"), row.get("label"))
            if row.get("alert_id"):
                if key in seen:
                    continue
                seen.add(key)
            rows.append(row)
    return write_jsonl(out_path, rows)
