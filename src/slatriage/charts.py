"""Paper figures. Always write PNG when matplotlib is available."""

from __future__ import annotations

from pathlib import Path
from typing import Any

FIGURES = [
    "fig01_labels_curated",
    "fig02_labels_solidifi",
    "fig03_sft_train",
    "fig04_keep_detectors",
    "fig05_coverage",
    "fig06_prf_configs",
    "fig07_f1_by_family",
    "fig08_invent",
    "fig09_gain",
    "fig10_ablation",
    "fig11_train_loss",
    "fig12_tp_fp_fn",
    "fig13_keep_rate",
    "fig14_confusion",
    "fig15_absent_share",
    "fig16_holdout",
    "fig17_eval_loss",
    "fig18_holdout_f1",
]


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "figure.dpi": 140,
            "savefig.bbox": "tight",
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )
    return plt


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt = _plt()
    plt.close(fig)
    return path


def fig_stacked_keep_drop(path: Path, title: str, by_family: dict[str, dict[str, int]]) -> Path | None:
    if not by_family:
        return None
    plt = _plt()
    fams = [f for f in ("R", "A", "C") if f in by_family]
    keep = [by_family[f].get("keep", 0) for f in fams]
    drop = [by_family[f].get("drop", 0) for f in fams]
    unk = [by_family[f].get("unknown", 0) for f in fams]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    x = range(len(fams))
    ax.bar(x, keep, label="keep")
    ax.bar(x, drop, bottom=keep, label="drop")
    ax.bar(x, unk, bottom=[k + d for k, d in zip(keep, drop)], label="unknown")
    ax.set_xticks(list(x), fams)
    ax.set_ylabel("alerts")
    ax.set_title(title)
    ax.legend()
    return _save(fig, path)


def fig_sft(path: Path, curated: dict[str, int], train: dict[str, int]) -> Path | None:
    plt = _plt()
    roles = ["R", "A", "C", "J"]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    x = range(len(roles))
    c = [curated.get(r, 0) for r in roles]
    t = [train.get(r, 0) for r in roles]
    aux = [max(0, tt - cc) for tt, cc in zip(t, c)]
    ax.bar(x, c, label="Curated")
    ax.bar(x, aux, bottom=c, label="SolidiFI aux")
    ax.set_xticks(list(x), roles)
    ax.set_ylabel("SFT rows")
    ax.set_title("Train set = Curated + SolidiFI (F1 stays on Curated)")
    ax.legend()
    return _save(fig, path)


def fig_keep_detectors(path: Path, keep_by: dict[str, int], top: int = 12) -> Path | None:
    if not keep_by:
        return None
    plt = _plt()
    items = list(keep_by.items())[:top]
    items = list(reversed(items))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.barh([k for k, _ in items], [v for _, v in items])
    ax.set_xlabel("keep (Curated)")
    ax.set_title("Keep alerts by Slither detector")
    return _save(fig, path)


def fig_coverage(path: Path, cov: dict[str, dict[str, int]]) -> Path | None:
    if not cov:
        return None
    plt = _plt()
    cats = list(cov.keys())
    cells = ["KEEP", "DROP", "ABSENT", "UNKNOWN"]
    grid = [[int(cov[c].get(k, 0)) for k in cells] for c in cats]
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    im = ax.imshow(grid, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(cells)), cells, rotation=20)
    ax.set_yticks(range(len(cats)), cats)
    for i, row in enumerate(grid):
        for j, val in enumerate(row):
            ax.text(j, i, str(val), ha="center", va="center", fontsize=8)
    ax.set_title("Closed-list coverage (contract x DASP)")
    fig.colorbar(im, ax=ax, fraction=0.03)
    return _save(fig, path)


def fig_prf(path: Path, evals: dict[str, dict[str, Any]]) -> Path | None:
    names = [k for k, m in evals.items() if isinstance(m, dict) and "f1" in m]
    if not names:
        return None
    plt = _plt()
    order = [n for n in ("A", "B", "C", "D", "E", "G", "E_noR") if n in names]
    order += [n for n in names if n not in order]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    import numpy as np

    x = np.arange(len(order))
    w = 0.25
    for i, key in enumerate(("precision", "recall", "f1")):
        ax.bar(x + (i - 1) * w, [float(evals[n].get(key) or 0) for n in order], w, label=key)
    ax.set_xticks(x, order)
    ax.set_ylim(0, 1.05)
    ax.set_title("Precision / Recall / F1 by config (Curated keep)")
    ax.legend()
    return _save(fig, path)


def fig_f1_family(path: Path, evals: dict[str, dict[str, Any]]) -> Path | None:
    names = [k for k, m in evals.items() if isinstance(m, dict) and m.get("by_family")]
    if not names:
        return None
    plt = _plt()
    import numpy as np

    order = [n for n in ("A", "B", "C", "D", "E", "G", "E_noR") if n in names]
    order += [n for n in names if n not in order]
    x = np.arange(len(order))
    w = 0.25
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for i, fam in enumerate(("R", "A", "C")):
        vals = [float((evals[n].get("by_family") or {}).get(fam, {}).get("f1") or 0) for n in order]
        ax.bar(x + (i - 1) * w, vals, w, label=f"family {fam}")
    ax.set_xticks(x, order)
    ax.set_ylim(0, 1.05)
    ax.set_title("F1 by family and config")
    ax.legend()
    return _save(fig, path)


def fig_invent(path: Path, invent: dict[str, Any]) -> Path | None:
    if not invent:
        return None
    plt = _plt()
    keys = [k for k in ("C", "D", "E") if k in invent]
    keys += [k for k in invent if k not in keys]
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.bar(keys, [int(invent[k] or 0) for k in keys])
    ax.set_ylabel("invented_count")
    ax.set_title("Judge invent (lower is better)")
    return _save(fig, path)


def fig_gain(path: Path, gain: dict[str, Any] | None) -> Path | None:
    if not gain:
        return None
    plt = _plt()
    keys = [k for k in ("precision", "f1") if k in gain and gain[k] is not None]
    if not keys:
        return None
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    vals = [float(gain[k]) * 100 for k in keys]
    ax.bar(keys, vals)
    ax.axhline(15, color="red", linestyle="--", label="15% check")
    ax.set_ylabel("relative gain %")
    ax.set_title("E vs B relative gain")
    ax.legend()
    return _save(fig, path)


def fig_ablation(path: Path, evals: dict[str, dict[str, Any]]) -> Path | None:
    e = evals.get("E") or {}
    nor = evals.get("E_noR") or evals.get("EnoR") or {}
    if "f1" not in e or "f1" not in nor:
        return None
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    labels = ["F1 overall", "F1 family R"]
    e_vals = [float(e.get("f1") or 0), float((e.get("by_family") or {}).get("R", {}).get("f1") or 0)]
    n_vals = [float(nor.get("f1") or 0), float((nor.get("by_family") or {}).get("R", {}).get("f1") or 0)]
    import numpy as np

    x = np.arange(2)
    ax.bar(x - 0.18, e_vals, 0.36, label="E (all LoRA)")
    ax.bar(x + 0.18, n_vals, 0.36, label="E without LoRA-R")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.05)
    ax.set_title("Ablation: drop LoRA-R")
    ax.legend()
    return _save(fig, path)


def fig_train_loss(path: Path, losses: dict[str, list[float]]) -> Path | None:
    series = {k: v for k, v in losses.items() if v}
    if not series:
        return None
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for role, ys in series.items():
        ax.plot(range(1, len(ys) + 1), ys, label=role)
    ax.set_xlabel("log step")
    ax.set_ylabel("loss")
    ax.set_title("QLoRA train loss")
    ax.legend()
    return _save(fig, path)


def fig_waiting(path: Path, title: str, reason: str) -> Path:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.axis("off")
    ax.text(0.5, 0.62, title, ha="center", va="center", fontsize=13, fontweight="bold")
    ax.text(0.5, 0.38, f"Waiting: {reason}", ha="center", va="center", fontsize=10, color="#555555")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    return _save(fig, path)


def fig_keep_rate(path: Path, curated: dict[str, dict[str, int]], solidifi: dict[str, dict[str, int]]) -> Path | None:
    if not curated:
        return None
    plt = _plt()
    import numpy as np

    fams = [f for f in ("R", "A", "C") if f in curated]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    x = np.arange(len(fams))

    def rate(block, fam):
        k = (block.get(fam) or {}).get("keep", 0)
        d = (block.get(fam) or {}).get("drop", 0)
        return k / (k + d) if (k + d) else 0.0

    ax.bar(x - 0.18, [rate(curated, f) for f in fams], 0.36, label="Curated")
    if solidifi:
        ax.bar(x + 0.18, [rate(solidifi, f) for f in fams], 0.36, label="SolidiFI")
    ax.set_xticks(x, fams)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("keep / (keep+drop)")
    ax.set_title("Positive rate by family")
    ax.legend()
    return _save(fig, path)


def fig_confusion(path: Path, evals: dict[str, dict[str, Any]]) -> Path | None:
    names = [k for k, m in evals.items() if isinstance(m, dict) and "tp" in m]
    if not names:
        return None
    plt = _plt()
    order = [n for n in ("A", "B", "C", "D", "E", "G") if n in names] or names[:4]
    n = len(order)
    fig, axes = plt.subplots(1, n, figsize=(3.1 * n, 3.0))
    if n == 1:
        axes = [axes]
    for ax, name in zip(axes, order):
        m = evals[name]
        grid = [[int(m.get("tn") or 0), int(m.get("fp") or 0)], [int(m.get("fn") or 0), int(m.get("tp") or 0)]]
        ax.imshow(grid, cmap="Oranges")
        ax.set_xticks([0, 1], ["pred drop", "pred keep"])
        ax.set_yticks([0, 1], ["gold drop", "gold keep"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(grid[i][j]), ha="center", va="center")
        ax.set_title(name)
    fig.suptitle("Confusion (keep = positive)")
    fig.tight_layout()
    return _save(fig, path)


def fig_absent_share(path: Path, cov: dict[str, dict[str, int]]) -> Path | None:
    if not cov:
        return None
    plt = _plt()
    cats = list(cov.keys())
    shares = []
    for c in cats:
        tot = sum(int(cov[c].get(k, 0)) for k in ("KEEP", "DROP", "ABSENT", "UNKNOWN"))
        shares.append((int(cov[c].get("ABSENT", 0)) / tot) if tot else 0.0)
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.barh(cats, shares)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("ABSENT share")
    ax.set_title("Closed-list holes (no Slither alert)")
    return _save(fig, path)


def fig_holdout(path: Path, holdout: dict[str, Any] | None) -> Path | None:
    if not holdout:
        return None
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.bar(["train files", "hold-out files"], [int(holdout.get("n_train_src") or 0), int(holdout.get("n_test_src") or 0)])
    ax.set_title(f"Curated file split (seed={holdout.get('seed', 0)}, {holdout.get('frac', 0.3):.0%} hold-out)")
    ax.set_ylabel("contracts")
    return _save(fig, path)


def fig_eval_loss(path: Path, losses: dict[str, list[float]]) -> Path | None:
    series = {k: v for k, v in losses.items() if v}
    if not series:
        return None
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for role, ys in series.items():
        ax.plot(range(1, len(ys) + 1), ys, label=role)
    ax.set_xlabel("eval log")
    ax.set_ylabel("eval_loss")
    ax.set_title("QLoRA validation loss")
    ax.legend()
    return _save(fig, path)


def fig_holdout_f1(path: Path, evals: dict[str, dict[str, Any]]) -> Path | None:
    names = [k for k, m in evals.items() if isinstance(m, dict) and "f1" in m]
    if not names:
        return None
    plt = _plt()
    order = [n for n in ("A", "B", "C", "D", "E", "G") if n in names]
    order += [n for n in names if n not in order]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.bar(order, [float(evals[n].get("f1") or 0) for n in order])
    ax.set_ylim(0, 1.05)
    ax.set_title("F1 on Curated hold-out files")
    return _save(fig, path)


def fig_counts(path: Path, evals: dict[str, dict[str, Any]]) -> Path | None:
    names = [k for k, m in evals.items() if isinstance(m, dict) and "tp" in m]
    if not names:
        return None
    plt = _plt()
    import numpy as np

    order = [n for n in ("A", "B", "C", "D", "E", "G") if n in names]
    order += [n for n in names if n not in order]
    x = np.arange(len(order))
    w = 0.25
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for i, key in enumerate(("tp", "fp", "fn")):
        ax.bar(x + (i - 1) * w, [int(evals[n].get(key) or 0) for n in order], w, label=key)
    ax.set_xticks(x, order)
    ax.set_title("TP / FP / FN by config")
    ax.legend()
    return _save(fig, path)


def render_all(out_dir: Path, payload: dict[str, Any]) -> tuple[dict[str, str | None], list[str]]:
    """Draw every figure. Missing inputs get a waiting PNG so the set is complete."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    evals = payload.get("eval") or {}
    written: dict[str, str | None] = {}
    pending: list[str] = []

    def put(name: str, path: Path | None) -> None:
        written[name] = path.name if path else None

    put(
        "fig01_labels_curated",
        fig_stacked_keep_drop(
            out_dir / "fig01_labels_curated.png",
            "Curated labels (F1 gold)",
            payload.get("labels_curated_fam") or {},
        ),
    )
    put(
        "fig02_labels_solidifi",
        fig_stacked_keep_drop(
            out_dir / "fig02_labels_solidifi.png",
            "SolidiFI aux labels (not in main F1)",
            payload.get("labels_solidifi_fam") or {},
        ),
    )
    put(
        "fig03_sft_train",
        fig_sft(out_dir / "fig03_sft_train.png", payload.get("sft_curated") or {}, payload.get("sft") or {}),
    )
    put("fig04_keep_detectors", fig_keep_detectors(out_dir / "fig04_keep_detectors.png", payload.get("keep_by_detector") or {}))
    put("fig05_coverage", fig_coverage(out_dir / "fig05_coverage.png", payload.get("coverage") or {}))
    put("fig06_prf_configs", fig_prf(out_dir / "fig06_prf_configs.png", evals))
    put("fig07_f1_by_family", fig_f1_family(out_dir / "fig07_f1_by_family.png", evals))
    put("fig08_invent", fig_invent(out_dir / "fig08_invent.png", payload.get("invented_count_by_config") or {}))
    put("fig09_gain", fig_gain(out_dir / "fig09_gain.png", payload.get("relative_gain_E_vs_B")))
    put("fig10_ablation", fig_ablation(out_dir / "fig10_ablation.png", evals))
    put("fig11_train_loss", fig_train_loss(out_dir / "fig11_train_loss.png", payload.get("train_loss") or {}))
    put("fig12_tp_fp_fn", fig_counts(out_dir / "fig12_tp_fp_fn.png", evals))
    put(
        "fig13_keep_rate",
        fig_keep_rate(
            out_dir / "fig13_keep_rate.png",
            payload.get("labels_curated_fam") or {},
            payload.get("labels_solidifi_fam") or {},
        ),
    )
    put("fig14_confusion", fig_confusion(out_dir / "fig14_confusion.png", evals))
    put("fig15_absent_share", fig_absent_share(out_dir / "fig15_absent_share.png", payload.get("coverage") or {}))
    put("fig16_holdout", fig_holdout(out_dir / "fig16_holdout.png", payload.get("holdout")))
    put("fig17_eval_loss", fig_eval_loss(out_dir / "fig17_eval_loss.png", payload.get("eval_loss") or {}))
    put("fig18_holdout_f1", fig_holdout_f1(out_dir / "fig18_holdout_f1.png", payload.get("eval_holdout") or {}))

    waiting = payload.get("figure_waiting") or {}
    for name in FIGURES:
        if not written.get(name):
            reason = waiting.get(name) or "missing input"
            fig_waiting(out_dir / f"{name}.png", name, reason)
            written[name] = f"{name}.png"
            pending.append(name)
    return written, pending
