#!/usr/bin/env python3
"""Run council configs A–E–G (default backend=mock for smoke)."""

from __future__ import annotations

import argparse
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
        help="holdout_sources.json: run only on those files (the only set F1 is reported on)",
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

    preds = []
    invent_total = 0
    parse_fail = 0
    per_source = []
    src_items = list(by_src.items())
    for i, (src, group) in enumerate(src_items, 1):
        print(f"council {args.config} {i}/{len(src_items)} n={len(group)} {src}", flush=True)
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
            parse_fail += p.get("reason") == "parse_fail"
            preds.append(p)
        if report.judge:
            invent_total += int(report.judge.invented_count or 0)
        per_source.append(report.to_dict())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        args.out.with_suffix(".report.json")
        if args.out.suffix == ".jsonl"
        else Path(str(args.out) + ".report.json"),
        {
            "config": args.config,
            "invented_count": invent_total,
            "parse_fail": parse_fail,
            "judge": {"invented_count": invent_total},
            "n_sources": len(by_src),
            "per_source": per_source,
        },
    )
    n = write_jsonl(args.out, preds)
    print(f"config={args.config} predictions={n} invented={invent_total} parse_fail={parse_fail} sources={len(by_src)} out={args.out.name}")


if __name__ == "__main__":
    main()
