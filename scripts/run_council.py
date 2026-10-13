#!/usr/bin/env python3
"""Run council configs A–E–G (default backend=mock for smoke)."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from slatriage.council import run_config
from slatriage.io_utils import read_jsonl, write_json, write_jsonl
from slatriage.llm import get_backend
from slatriage.schema import Alert
from slatriage.split import read_holdout


def load_alerts(path: Path) -> list[Alert]:
    rows = read_jsonl(path)
    return [Alert.from_dict(r) for r in rows]


def report_path_for(out: Path) -> Path:
    return out.with_suffix(".report.json") if out.suffix == ".jsonl" else Path(str(out) + ".report.json")


def load_resume(out: Path) -> tuple[list[dict], set[str], list[dict], int]:
    preds = read_jsonl(out) if out.exists() else []
    done = {str(p.get("source")) for p in preds if p.get("source")}
    per_source: list[dict] = []
    invent = 0
    rp = report_path_for(out)
    if rp.exists():
        try:
            old = json.loads(rp.read_text(encoding="utf-8"))
            per_source = list(old.get("per_source") or [])
            invent = int(old.get("invented_count") or 0)
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    return preds, done, per_source, invent


def flush_progress(
    out: Path,
    preds: list[dict],
    *,
    config: str,
    invent_total: int,
    per_source: list[dict],
    n_sources: int,
) -> None:
    parse_fail = sum(1 for p in preds if p.get("reason") == "parse_fail")
    write_jsonl(out, preds)
    write_json(
        report_path_for(out),
        {
            "config": config,
            "invented_count": invent_total,
            "parse_fail": parse_fail,
            "judge": {"invented_count": invent_total},
            "n_sources": n_sources,
            "n_preds": len(preds),
            "per_source": per_source,
        },
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="SlaTriage council runner")
    ap.add_argument("--alerts", type=Path, required=True, help="JSONL alerts or labels")
    ap.add_argument("--config", choices=["A", "B", "C", "D", "E", "G"], required=True)
    ap.add_argument("--map", type=Path, default=ROOT / "configs" / "detectors_map.yaml")
    ap.add_argument("--backend", choices=["mock", "hf"], default="mock")
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    ap.add_argument("--g-model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")
    ap.add_argument("--adapter-r", type=Path, default=None)
    ap.add_argument("--adapter-a", type=Path, default=None)
    ap.add_argument("--adapter-c", type=Path, default=None)
    ap.add_argument("--adapter-j", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sol-path", default="")
    ap.add_argument(
        "--holdout",
        type=Path,
        default=None,
        help="holdout_sources.json: run only on those files (report F1 is still hold-out even without this)",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="skip sources already in --out and flush after each file (Colab/Kaggle safe)",
    )
    args = ap.parse_args()

    alerts = load_alerts(args.alerts)
    if args.holdout:
        held = read_holdout(args.holdout)
        alerts = [a for a in alerts if a.source in held]
        print(f"holdout filter: {len(held)} files -> {len(alerts)} alerts")
    if args.backend == "mock":
        base = get_backend("mock")
        g_llm = base
        experts = None
        judge = base
    else:
        # One HF process loads several 7B copies if all adapters are set — needs >16GB.
        # On T4: train one role per session; infer E on a bigger GPU or sequentially later.
        # One 4-bit 7B + swap LoRA (T4 16GB). Config G is a separate 14B process.
        base = get_backend("hf", model_id=args.model, adapter_path=None, adapter_name="base")
        g_llm = (
            get_backend("hf", model_id=args.g_model, adapter_path=None, adapter_name="base")
            if args.config == "G"
            else base
        )
        experts = {
            "R": get_backend(
                "hf",
                model_id=args.model,
                adapter_path=str(args.adapter_r) if args.adapter_r else None,
                adapter_name="R",
            ),
            "A": get_backend(
                "hf",
                model_id=args.model,
                adapter_path=str(args.adapter_a) if args.adapter_a else None,
                adapter_name="A",
            ),
            "C": get_backend(
                "hf",
                model_id=args.model,
                adapter_path=str(args.adapter_c) if args.adapter_c else None,
                adapter_name="C",
            ),
        }
        judge = get_backend(
            "hf",
            model_id=args.model,
            adapter_path=str(args.adapter_j) if args.adapter_j else None,
            adapter_name="J",
        )

    by_src: dict[str, list[Alert]] = defaultdict(list)
    for a in alerts:
        by_src[a.source or args.sol_path or "_"].append(a)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    preds: list[dict] = []
    invent_total = 0
    per_source: list[dict] = []
    done: set[str] = set()
    if args.resume:
        preds, done, per_source, invent_total = load_resume(args.out)
        if done:
            print(f"RESUME {len(preds)} preds from {len(done)} sources in {args.out.name}", flush=True)

    src_items = list(by_src.items())
    n_total = len(src_items)
    for i, (src, group) in enumerate(src_items, 1):
        if src in done:
            print(f"council {args.config} {i}/{n_total} SKIP {src}", flush=True)
            continue
        print(f"council {args.config} {i}/{n_total} n={len(group)} {src}", flush=True)
        report = run_config(
            args.config,  # type: ignore[arg-type]
            group,
            map_path=args.map,
            base_llm=base,
            expert_llms=experts,
            judge_llm=judge,
            g_llm=g_llm,
            sol_path=src,
        )
        for p in report.predictions:
            p.setdefault("source", src)
            preds.append(p)
        if report.judge:
            invent_total += int(report.judge.invented_count or 0)
        per_source.append(report.to_dict())
        flush_progress(
            args.out,
            preds,
            config=args.config,
            invent_total=invent_total,
            per_source=per_source,
            n_sources=n_total,
        )
        print(
            f"  saved {len(preds)}/{sum(len(g) for _, g in src_items)} preds sources_done={len({p.get('source') for p in preds})}/{n_total}",
            flush=True,
        )

    flush_progress(
        args.out,
        preds,
        config=args.config,
        invent_total=invent_total,
        per_source=per_source,
        n_sources=n_total,
    )
    parse_fail = sum(1 for p in preds if p.get("reason") == "parse_fail")
    n = len(preds)
    print(
        f"config={args.config} predictions={n} invented={invent_total} parse_fail={parse_fail} sources={len({p.get('source') for p in preds})}/{n_total} out={args.out.name}"
    )


if __name__ == "__main__":
    main()
