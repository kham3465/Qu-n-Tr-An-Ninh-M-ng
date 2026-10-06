#!/usr/bin/env python3
"""Run council configs A–E–G (default backend=mock for smoke)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from slatriage.council import run_config
from slatriage.io_utils import read_jsonl, write_json, write_jsonl
from slatriage.llm import get_backend
from slatriage.schema import Alert


def load_alerts(path: Path) -> list[Alert]:
    rows = read_jsonl(path)
    return [Alert.from_dict(r) for r in rows]


def main() -> None:
    ap = argparse.ArgumentParser(description="SlaTriage council runner")
    ap.add_argument("--alerts", type=Path, required=True, help="JSONL alerts or labels")
    ap.add_argument("--config", choices=list("ABCDEG"), required=True)
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
    args = ap.parse_args()

    alerts = load_alerts(args.alerts)
    if args.backend == "mock":
        base = get_backend("mock")
        g_llm = base
        experts = None
        judge = base
    else:
        base = get_backend("hf", model_id=args.model, adapter_path=None)
        g_llm = get_backend("hf", model_id=args.g_model, adapter_path=None) if args.config == "G" else base
        experts = {
            "R": get_backend("hf", model_id=args.model, adapter_path=str(args.adapter_r) if args.adapter_r else None),
            "A": get_backend("hf", model_id=args.model, adapter_path=str(args.adapter_a) if args.adapter_a else None),
            "C": get_backend("hf", model_id=args.model, adapter_path=str(args.adapter_c) if args.adapter_c else None),
        }
        judge = get_backend(
            "hf",
            model_id=args.model,
            adapter_path=str(args.adapter_j) if args.adapter_j else None,
        )

    report = run_config(
        args.config,  # type: ignore[arg-type]
        alerts,
        map_path=args.map,
        base_llm=base,
        expert_llms=experts,
        judge_llm=judge,
        g_llm=g_llm,
        sol_path=args.sol_path,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out.with_suffix(".report.json"), report.to_dict())
    n = write_jsonl(args.out, report.predictions)
    invent = report.judge.invented_count if report.judge else None
    print(f"config={args.config} predictions={n} invented={invent} → {args.out}")


if __name__ == "__main__":
    main()
