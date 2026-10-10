"""Assemble paper_stats + markdown + figures from labels, SFT, preds, adapters."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .charts import FIGURES, render_all
from .coverage import build_matrix, load_closed, matrix_summary
from .curated import load_curated_index
from .dataset import load_sft_jsonl
from .io_utils import read_jsonl, write_json
from .metrics import filter_by_sources, relative_gain, score_aligned
from .split import load_or_make_holdout
from .train_qlora import default_sft_dir, find_sft

ROOT_HINT = Path(__file__).resolve().parents[2]

NEED_FOR_FULL = {
    "fig01_labels_curated": "labels_v1",
    "fig02_labels_solidifi": "labels_solidifi",
    "fig03_sft_train": "data/sft/combined",
    "fig04_keep_detectors": "labels_v1",
    "fig05_coverage": "labels_v1",
    "fig06_prf_configs": "pred_A..G.jsonl after council",
    "fig07_f1_by_family": "pred_*.jsonl",
    "fig08_invent": "pred_C/D/E.report.json",
    "fig09_gain": "pred_B + pred_E",
    "fig10_ablation": "pred_E + pred_E_noR",
    "fig11_train_loss": "adapters/*/train_meta.json",
    "fig12_tp_fp_fn": "pred_*.jsonl",
    "fig13_keep_rate": "labels curated+solidifi",
    "fig14_confusion": "pred_*.jsonl",
    "fig15_absent_share": "coverage matrix",
    "fig16_holdout": "holdout_sources.json",
    "fig17_eval_loss": "adapters/*/train_meta.json eval_loss",
    "fig18_holdout_f1": "pred + hold-out files",
}


def _fam_table(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    c: Counter = Counter((r.get("family"), r.get("label")) for r in rows)
    out: dict[str, dict[str, int]] = {}
    for fam in ("R", "A", "C"):
        out[fam] = {
            "keep": c[(fam, "keep")],
            "drop": c[(fam, "drop")],
            "unknown": c[(fam, "unknown")],
        }
    return out


def discover_preds(pred_dir: Path) -> list[Path]:
    found = []
    for p in sorted(Path(pred_dir).glob("pred_*.jsonl")):
        if p.name.startswith("pred__") or p.name.startswith("_"):
            continue
        found.append(p)
    return found


def pred_name(path: Path) -> str:
    name = path.stem
    if name.startswith("pred_"):
        name = name[5:]
    return name


def load_train_curves(adapters: Path) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    loss: dict[str, list[float]] = {}
    ev: dict[str, list[float]] = {}
    if not adapters.exists():
        return loss, ev
    for role in ("R", "A", "C", "J"):
        meta = adapters / role / "train_meta.json"
        if not meta.exists():
            continue
        try:
            payload = json.loads(meta.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        ys = payload.get("loss") or [x.get("loss") for x in (payload.get("logs") or []) if "loss" in x]
        loss[role] = [float(v) for v in ys if v is not None]
        eys = payload.get("eval_loss") or [x.get("eval_loss") for x in (payload.get("logs") or []) if "eval_loss" in x]
        ev[role] = [float(v) for v in eys if v is not None]
    return loss, ev


def evaluate_preds(gold: list[dict[str, Any]], pred_files: list[Path]) -> tuple[dict, dict]:
    evals: dict[str, Any] = {}
    invent: dict[str, Any] = {}
    for pred_path in pred_files:
        name = pred_name(pred_path)
        evals[name] = score_aligned(gold, read_jsonl(pred_path))
        report = pred_path.with_suffix(".report.json")
        alt = pred_path.parent / f"{pred_path.stem}.report.json"
        for cand in (report, alt):
            if cand.exists():
                try:
                    rep = json.loads(cand.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    continue
                invent[name] = (rep.get("judge") or {}).get("invented_count", rep.get("invented_count"))
                break
    return evals, invent


def build_payload(
    *,
    root: Path,
    labels_path: Path,
    solidifi_path: Path,
    sft_dir: Path | None,
    closed_path: Path,
    curated_root: Path,
    pred_files: list[Path],
    adapters: Path,
) -> dict[str, Any]:
    sft_dir = sft_dir or default_sft_dir(root)
    labels = read_jsonl(labels_path) if labels_path.exists() else []
    solidifi = read_jsonl(solidifi_path) if solidifi_path.exists() else []
    fam_label: Counter = Counter()
    det_keep: Counter = Counter()
    for r in labels:
        fam_label[(r.get("family"), r.get("label"))] += 1
        if r.get("label") == "keep":
            det_keep[str(r.get("detector"))] += 1

    sft_counts: dict[str, int] = {}
    for role in ("R", "A", "C", "J"):
        try:
            sft_counts[role] = len(load_sft_jsonl(find_sft(sft_dir, role)))
        except Exception:
            sft_counts[role] = 0
    curated_sft: dict[str, int] = {}
    for role in ("R", "A", "C", "J"):
        try:
            curated_sft[role] = len(load_sft_jsonl(find_sft(root / "data" / "sft", role)))
        except Exception:
            curated_sft[role] = 0

    all_sources = None
    vuln = curated_root / "vulnerabilities.json"
    if vuln.exists():
        all_sources = [e.get("path") for e in load_curated_index(vuln) if e.get("path")]
    closed = load_closed(closed_path)
    cov = matrix_summary(build_matrix(labels, closed, all_sources=all_sources) if labels else [])

    evals, invent = evaluate_preds(labels, pred_files)
    hold_src = load_or_make_holdout(root / "data" / "labels" / "holdout_sources.json", labels)
    hold_set = set(hold_src)
    train_src = sorted({str(r.get("source") or "") for r in labels if r.get("source") and r.get("source") not in hold_set})
    gold_hold = filter_by_sources(labels, hold_set)
    eval_holdout: dict[str, Any] = {}
    for pred_path in pred_files:
        name = pred_name(pred_path)
        eval_holdout[name] = score_aligned(gold_hold, filter_by_sources(read_jsonl(pred_path), hold_set))
    gain = None
    if "E" in evals and "B" in evals and "f1" in evals["E"] and "f1" in evals["B"]:
        gain = {
            "precision": relative_gain(float(evals["E"].get("precision") or 0), float(evals["B"].get("precision") or 0)),
            "f1": relative_gain(float(evals["E"].get("f1") or 0), float(evals["B"].get("f1") or 0)),
        }
    train_loss, eval_loss = load_train_curves(adapters)

    return {
        "n_labels": len(labels),
        "unique_alert_id": len({r.get("alert_id") for r in labels}),
        "unique_source": len({r.get("source") for r in labels}),
        "by_family_label": {f"{a}/{b}": n for (a, b), n in sorted(fam_label.items())},
        "labels_curated_fam": _fam_table(labels),
        "labels_solidifi_fam": _fam_table(solidifi),
        "keep_by_detector": dict(det_keep.most_common()),
        "sft": sft_counts,
        "sft_curated": curated_sft,
        "sft_dir": str(sft_dir),
        "solidifi": {
            "n_labels": len(solidifi),
            "by_family_label": {
                f"{a}/{b}": n
                for (a, b), n in sorted(Counter((r.get("family"), r.get("label")) for r in solidifi).items())
            },
        },
        "coverage": cov,
        "eval": evals,
        "invented_count_by_config": invent,
        "relative_gain_E_vs_B": gain,
        "train_loss": train_loss,
        "eval_loss": eval_loss,
        "holdout": {
            "seed": 0,
            "frac": 0.3,
            "n_test_src": len(hold_src),
            "n_train_src": len(train_src),
            "n_test_alerts": len(gold_hold),
        },
        "eval_holdout": eval_holdout,
        "figure_waiting": NEED_FOR_FULL,
        "pred_files": [p.name for p in pred_files],
    }


def _md_prf(name: str, m: dict) -> str:
    if "error" in m and "f1" not in m:
        return f"| {name} | - | - | - | {m.get('n', 0)} |"
    return (
        f"| {name} | {m.get('precision', 0):.3f} | {m.get('recall', 0):.3f} | "
        f"{m.get('f1', 0):.3f} | {m.get('n_scored', m.get('n', 0))} |"
    )


def write_markdown(payload: dict[str, Any], figures: dict[str, str | None], out_md: Path) -> None:
    fam = payload.get("labels_curated_fam") or {}
    sol = payload.get("labels_solidifi_fam") or {}
    curated_sft = payload.get("sft_curated") or {}
    sft = payload.get("sft") or {}
    evals = payload.get("eval") or {}
    lines = [
        "# Chi so SlaTriage",
        "",
        "## 1. Nhan Curated (F1 chinh)",
        "",
        f"- Alert RAC: **{payload.get('n_labels')}** / contract: **{payload.get('unique_source')}**",
        "",
        "| family | keep | drop | unknown | SFT curated |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for role in ("R", "A", "C"):
        d = fam.get(role) or {}
        lines.append(
            f"| {role} | {d.get('keep', 0)} | {d.get('drop', 0)} | {d.get('unknown', 0)} | {curated_sft.get(role, 0)} |"
        )
    lines += [
        f"| J | - | - | - | {curated_sft.get('J', 0)} |",
        "",
        f"SFT train: R={sft.get('R', 0)} A={sft.get('A', 0)} C={sft.get('C', 0)} J={sft.get('J', 0)}.",
        "",
        "## 1b. SolidiFI (hoc phu)",
        "",
        f"- Alert RAC: **{(payload.get('solidifi') or {}).get('n_labels', 0)}**",
        "",
        "| family | keep | drop | unknown |",
        "| --- | ---: | ---: | ---: |",
    ]
    for role in ("R", "A", "C"):
        d = sol.get(role) or {}
        lines.append(f"| {role} | {d.get('keep', 0)} | {d.get('drop', 0)} | {d.get('unknown', 0)} |")
    lines += ["", "## 2. Coverage DASP", "", "| hang | KEEP | DROP | ABSENT | UNKNOWN |", "| --- | ---: | ---: | ---: | ---: |"]
    for cat, cells in (payload.get("coverage") or {}).items():
        lines.append(
            f"| {cat} | {cells.get('KEEP', 0)} | {cells.get('DROP', 0)} | "
            f"{cells.get('ABSENT', 0)} | {cells.get('UNKNOWN', 0)} |"
        )
    lines += ["", "## 3. Hoi dong (sau pred_*.jsonl)", "", "| config | precision | recall | f1 | n |", "| --- | ---: | ---: | ---: | ---: |"]
    if evals:
        for name, m in evals.items():
            if not isinstance(m, dict) or name in ("relative_gain_E_vs_B",):
                continue
            lines.append(_md_prf(name, m))
            for fam_n, fm in (m.get("by_family") or {}).items():
                lines.append(_md_prf(f"{name}/{fam_n}", fm))
    else:
        lines.append("| A/B/C/D/E/G | (chua chay council) |  |  |  |")
    invent = payload.get("invented_count_by_config") or {}
    if invent:
        lines += ["", "Invent:", ""]
        for k, v in invent.items():
            lines.append(f"- {k}: {v}")
    gain = payload.get("relative_gain_E_vs_B")
    if gain:
        lines += ["", f"Gain E vs B: P={gain.get('precision')} F1={gain.get('f1')}", ""]
    lines += ["", "## 4. Bieu do", ""]
    missing = []
    for key in FIGURES:
        fn = figures.get(key)
        if fn:
            lines += [f"![{key}](figures/{fn})", ""]
        else:
            missing.append(f"- `{key}` cho `{NEED_FOR_FULL[key]}`")
    pending = payload.get("figures_missing") or missing
    if pending:
        lines += ["### Cho so that (khung da ve)", ""] + [f"- `{k}`: {NEED_FOR_FULL.get(k, k)}" for k in pending] + [""]
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(
    *,
    root: Path | None = None,
    labels: Path | None = None,
    solidifi: Path | None = None,
    sft_dir: Path | None = None,
    closed: Path | None = None,
    curated_root: Path | None = None,
    pred_dir: Path | None = None,
    pred_files: list[Path] | None = None,
    adapters: Path | None = None,
    out_json: Path | None = None,
    out_md: Path | None = None,
    fig_dir: Path | None = None,
) -> dict[str, Any]:
    root = root or ROOT_HINT
    labels = labels or (root / "data" / "labels" / "labels_v1.jsonl")
    solidifi = solidifi or (root / "data" / "labels" / "labels_solidifi.jsonl")
    closed = closed or (root / "configs" / "closed_list.yaml")
    curated_root = curated_root or (root / "data" / "raw" / "smartbugs-curated")
    pred_dir = pred_dir or (root / "reports")
    adapters = adapters or (root / "adapters")
    out_json = out_json or (root / "reports" / "paper_stats.json")
    out_md = out_md or (root / "reports" / "BAO-CAO-CHI-SO.md")
    fig_dir = fig_dir or (root / "reports" / "figures")
    files = pred_files if pred_files is not None else discover_preds(pred_dir)
    payload = build_payload(
        root=root,
        labels_path=labels,
        solidifi_path=solidifi,
        sft_dir=sft_dir,
        closed_path=closed,
        curated_root=curated_root,
        pred_files=files,
        adapters=adapters,
    )
    figures, pending = render_all(fig_dir, payload)
    payload["figures"] = figures
    payload["figures_missing"] = pending
    write_json(out_json, payload)
    write_json(fig_dir / "manifest.json", {"written": figures, "pending": pending, "need": NEED_FOR_FULL})
    write_markdown(payload, figures, out_md)
    return payload
