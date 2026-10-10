#!/usr/bin/env python3
"""Hard checks on SFT used for train. Exit 1 if a logic error would break QLoRA."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from slatriage.dataset import load_sft_jsonl, validate_sft_row
from slatriage.train_qlora import default_sft_dir, find_sft

# chat template + alert JSON must fit QLoRA max_seq=2048 (~4 chars/token worst case)
MAX_ROW_CHARS = 12000
MAX_JUDGE_CHARS = 16000


def _assistant_ok(role: str, content: str) -> str | None:
    try:
        obj = json.loads(content)
    except json.JSONDecodeError:
        return "assistant is not JSON"
    if role == "J":
        if not isinstance(obj, dict) or "findings" not in obj or "invented_count" not in obj:
            return "Judge assistant missing findings/invented_count"
        if not isinstance(obj["findings"], list):
            return "Judge findings not a list"
        return None
    if not isinstance(obj, dict):
        return "expert assistant not an object"
    if obj.get("decision") not in ("keep", "drop"):
        return f"bad decision={obj.get('decision')}"
    if "alert_id" not in obj:
        return "expert assistant missing alert_id"
    return None


def audit_role(path: Path, role: str) -> dict:
    rows = load_sft_jsonl(path)
    errors: list[str] = []
    warns: list[str] = []
    splits: Counter = Counter()
    labels: Counter = Counter()
    fams: Counter = Counter()
    lengths: list[int] = []

    for i, row in enumerate(rows):
        try:
            validate_sft_row(row)
        except ValueError as e:
            errors.append(f"{path.name}#{i}: {e}")
            continue
        fam = row.get("family")
        fams[fam] += 1
        if role != "J" and fam != role:
            errors.append(f"{path.name}#{i}: family={fam} expected {role}")
        if role == "J" and fam not in (None, "J"):
            errors.append(f"{path.name}#{i}: Judge family={fam}")
        split = row.get("split") or "curated"
        splits[split] += 1
        lab = row.get("label")
        if lab:
            labels[lab] += 1
            if lab not in ("keep", "drop"):
                errors.append(f"{path.name}#{i}: label={lab}")
        nchar = sum(len(str(m.get("content") or "")) for m in row["messages"])
        lengths.append(nchar)
        cap = MAX_JUDGE_CHARS if role == "J" else MAX_ROW_CHARS
        if nchar > cap:
            errors.append(f"{path.name}#{i}: {nchar} chars > {cap} (will blow max_seq=2048)")
        err = _assistant_ok(role, str(row["messages"][-1].get("content") or ""))
        if err:
            errors.append(f"{path.name}#{i}: {err}")

    if role != "J":
        if labels.get("keep", 0) == 0:
            errors.append(f"{role}: zero keep")
        if labels.get("drop", 0) == 0:
            warns.append(f"{role}: zero drop (adapter will always keep)")
        elif labels.get("drop", 0) < 25:
            warns.append(f"{role}: only {labels['drop']} drop (LoRA will bias keep)")

    return {
        "file": path.name,
        "n": len(rows),
        "splits": dict(splits),
        "labels": dict(labels),
        "families": dict(fams),
        "max_chars": max(lengths) if lengths else 0,
        "p95_chars": sorted(lengths)[int(0.95 * (len(lengths) - 1))] if lengths else 0,
        "errors": errors,
        "warns": warns,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "sft_audit.json")
    args = ap.parse_args()
    sft_dir = args.sft_dir or default_sft_dir(ROOT)
    report: dict = {"sft_dir": str(sft_dir), "ok": True, "roles": {}, "errors": [], "warns": []}

    if sft_dir.resolve() != (ROOT / "data" / "sft" / "combined").resolve():
        report["warns"].append(f"not using combined: {sft_dir}")

    for role in ("R", "A", "C", "J"):
        try:
            path = find_sft(sft_dir, role)
        except FileNotFoundError as e:
            report["ok"] = False
            report["errors"].append(str(e))
            continue
        info = audit_role(path, role)
        report["roles"][role] = {k: v for k, v in info.items() if k not in ("errors", "warns")}
        report["errors"].extend(info["errors"][:20])
        report["warns"].extend(info["warns"])
        if info["errors"]:
            report["ok"] = False
            report["roles"][role]["n_errors"] = len(info["errors"])

    curated = ROOT / "data" / "sft"
    combined = ROOT / "data" / "sft" / "combined"
    if combined.exists() and (curated / "sft_R.jsonl").exists():
        cr = sum(1 for _ in (curated / "sft_R.jsonl").open(encoding="utf-8") if _.strip())
        co = report["roles"].get("R", {}).get("n", 0)
        if co < cr:
            report["ok"] = False
            report["errors"].append(f"combined R ({co}) < curated R ({cr})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["ok"]:
        print("SFT AUDIT FAIL")
        return 1
    print("SFT AUDIT OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
