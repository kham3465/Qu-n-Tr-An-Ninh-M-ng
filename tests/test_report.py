from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_render_dataset_and_eval_figures(tmp_path):
    from slatriage.charts import render_all

    payload = {
        "labels_curated_fam": {"R": {"keep": 36, "drop": 123, "unknown": 8}, "A": {"keep": 14, "drop": 53, "unknown": 11}, "C": {"keep": 61, "drop": 13, "unknown": 24}},
        "labels_solidifi_fam": {"R": {"keep": 10, "drop": 2, "unknown": 0}, "A": {"keep": 5, "drop": 5, "unknown": 0}, "C": {"keep": 3, "drop": 1, "unknown": 0}},
        "sft_curated": {"R": 159, "A": 67, "C": 74, "J": 111},
        "sft": {"R": 1847, "A": 4403, "C": 1005, "J": 633},
        "keep_by_detector": {"reentrancy-eth": 27, "tx-origin": 2},
        "coverage": {"reentrancy": {"KEEP": 35, "DROP": 24, "ABSENT": 83, "UNKNOWN": 1}},
        "eval": {
            "A": {"precision": 0.37, "recall": 1.0, "f1": 0.54, "tp": 111, "fp": 189, "fn": 0, "by_family": {"R": {"f1": 0.37}, "A": {"f1": 0.35}, "C": {"f1": 0.90}}},
            "B": {"precision": 0.30, "recall": 0.40, "f1": 0.34, "tp": 40, "fp": 90, "fn": 70, "by_family": {"R": {"f1": 0.2}, "A": {"f1": 0.2}, "C": {"f1": 0.3}}},
            "E": {"precision": 0.45, "recall": 0.50, "f1": 0.47, "tp": 55, "fp": 67, "fn": 56, "by_family": {"R": {"f1": 0.5}, "A": {"f1": 0.4}, "C": {"f1": 0.6}}},
            "E_noR": {"precision": 0.40, "recall": 0.42, "f1": 0.41, "tp": 46, "fp": 70, "fn": 65, "by_family": {"R": {"f1": 0.2}, "A": {"f1": 0.4}, "C": {"f1": 0.6}}},
        },
        "invented_count_by_config": {"C": 4, "D": 2, "E": 1},
        "relative_gain_E_vs_B": {"precision": 0.5, "f1": 0.38},
        "train_loss": {"R": [1.2, 1.0, 0.8], "A": [1.1, 0.9]},
        "eval_loss": {"R": [1.4, 1.1], "A": [1.3, 1.0]},
        "holdout": {"seed": 0, "frac": 0.3, "n_train_src": 78, "n_test_src": 34},
        "eval_holdout": {
            "A": {"f1": 0.5, "precision": 0.4, "recall": 1.0},
            "E": {"f1": 0.55, "precision": 0.5, "recall": 0.6},
        },
    }
    written, pending = render_all(tmp_path, payload)
    assert all(written.values()), written
    assert pending == []
    for name, fn in written.items():
        assert (tmp_path / fn).exists(), name


def test_write_report_makes_curated_figures(tmp_path):
    from slatriage.report import write_report

    payload = write_report(
        root=ROOT,
        pred_files=[ROOT / "reports" / "pred_A.jsonl"] if (ROOT / "reports" / "pred_A.jsonl").exists() else [],
        out_json=tmp_path / "paper_stats.json",
        out_md=tmp_path / "BAO-CAO.md",
        fig_dir=tmp_path / "figures",
    )
    figs = payload["figures"]
    for key in ("fig01_labels_curated", "fig03_sft_train", "fig05_coverage"):
        assert figs.get(key), key
    assert (tmp_path / "BAO-CAO.md").exists()
    md = (tmp_path / "BAO-CAO.md").read_text(encoding="utf-8")
    assert "fig01" in md
