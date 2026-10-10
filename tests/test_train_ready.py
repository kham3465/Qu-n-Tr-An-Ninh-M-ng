"""Guardrails so GPU train cannot start on broken / wrong SFT."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_default_sft_dir_is_combined():
    from slatriage.train_qlora import default_sft_dir, find_sft

    sft_dir = default_sft_dir(ROOT)
    assert sft_dir == ROOT / "data" / "sft" / "combined"
    for role in ("R", "A", "C", "J"):
        assert find_sft(sft_dir, role).parent == sft_dir


def test_combined_sft_is_train_safe():
    from slatriage.dataset import load_sft_jsonl
    from slatriage.train_qlora import default_sft_dir, find_sft

    sft_dir = default_sft_dir(ROOT)
    for role in ("R", "A", "C", "J"):
        rows = load_sft_jsonl(find_sft(sft_dir, role))
        assert len(rows) >= 50
        splits = {r.get("split") for r in rows}
        assert "curated" in splits and "solidifi" in splits
        for r in rows:
            if role != "J":
                assert r.get("family") == role
                assert r.get("label") in ("keep", "drop")
            else:
                assert r.get("family") == "J"
            ast = json.loads(r["messages"][-1]["content"])
            if role == "J":
                assert "findings" in ast and "invented_count" in ast
            else:
                assert ast["decision"] in ("keep", "drop")
                assert ast.get("alert_id")
            nchar = sum(len(str(m.get("content") or "")) for m in r["messages"])
            assert nchar < 9000, nchar


def test_combined_has_both_labels_for_experts():
    from slatriage.dataset import load_sft_jsonl
    from slatriage.train_qlora import default_sft_dir, find_sft

    sft_dir = default_sft_dir(ROOT)
    for role in ("R", "A", "C"):
        rows = load_sft_jsonl(find_sft(sft_dir, role))
        labs = {r["label"] for r in rows}
        assert labs == {"keep", "drop"}


def test_curated_eval_not_mixed_with_solidifi():
    from slatriage.io_utils import read_jsonl

    gold = read_jsonl(ROOT / "data" / "labels" / "labels_v1.jsonl")
    assert gold
    assert all(r.get("split") != "solidifi" for r in gold)
    ids = [r["alert_id"] for r in gold]
    assert len(ids) == len(set(ids))


def test_judge_export_chunks(tmp_path):
    from slatriage.dataset import export_judge_sft

    labels = tmp_path / "labels.jsonl"
    rows = []
    for i in range(40):
        rows.append(
            {
                "alert_id": f"a{i}",
                "detector": "reentrancy-eth",
                "family": "R",
                "label": "keep" if i % 2 == 0 else "drop",
                "lines": [i + 1],
                "source": "one.sol",
                "split": "solidifi",
            }
        )
    labels.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    out = tmp_path / "sft_J.jsonl"
    n = export_judge_sft(labels, out, max_items=16)
    assert n == 3
    got = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert all(len(json.loads(g["messages"][1]["content"].split("Alerts:\n", 1)[1])) <= 16 for g in got)


def test_messages_to_text_fallback():
    from slatriage.train_qlora import _messages_to_text

    class Tok:
        chat_template = None

    text = _messages_to_text(
        Tok(),
        [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": '{"decision":"keep"}'},
        ],
    )
    assert "keep" in text and "user" in text


def test_lora_cfg_targets():
    from slatriage.train_qlora import load_lora_cfg

    cfg = load_lora_cfg(ROOT / "configs" / "models.yaml")
    assert cfg.get("r") == 16
    assert "q_proj" in cfg.get("target_modules", [])
    assert "v_proj" in cfg.get("target_modules", [])


def test_train_cli_all_prepare_uses_combined():
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "train_lora.py"), "--all", "--prepare-only"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    out = r.stdout + r.stderr
    assert "combined" in out.replace("\\", "/")
    assert "prepare-only done" in out
    assert "TRAIN R" not in out


def test_train_cli_without_do_train_does_not_require_cuda():
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "train_lora.py"), "--role", "R", "--prepare-only"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert "TRAIN R" not in r.stdout


def test_mock_config_c_keeps_reentrancy():
    from slatriage.council import run_config
    from slatriage.llm import MockLLM
    from slatriage.schema import Alert

    alerts = [
        Alert(alert_id="r1", detector="reentrancy-eth", family="R", lines=[10], source="a.sol"),
        Alert(alert_id="c1", detector="unchecked-send", family="C", lines=[20], source="a.sol"),
    ]
    report = run_config(
        "C",
        alerts,
        map_path=ROOT / "configs" / "detectors_map.yaml",
        base_llm=MockLLM(),
    )
    by_id = {p["alert_id"]: p["decision"] for p in report.predictions}
    assert by_id["r1"] == "keep"
    assert by_id["c1"] == "drop"


def test_eval_config_a_overlaps_curated_gold(tmp_path):
    pred = tmp_path / "pred_A.jsonl"
    r = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_council.py"),
            "--alerts",
            str(ROOT / "data" / "labels" / "labels_v1.jsonl"),
            "--config",
            "A",
            "--backend",
            "mock",
            "--out",
            str(pred),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    outj = tmp_path / "summary.json"
    r2 = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "eval_configs.py"),
            "--gold",
            str(ROOT / "data" / "labels" / "labels_v1.jsonl"),
            "--pred",
            str(pred),
            "--out",
            str(outj),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert r2.returncode == 0, r2.stdout + r2.stderr
    summary = json.loads(outj.read_text(encoding="utf-8"))
    assert summary["A"]["n_scored"] >= 200
    assert "f1" in summary["A"]


def test_preflight_accepts_combined_without_cuda():
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "preflight.py"), "--out", str(ROOT / "reports" / "preflight.json")],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert "PREFLIGHT OK" in r.stdout
    payload = json.loads((ROOT / "reports" / "preflight.json").read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert "combined" in payload["sft_dir"].replace("\\", "/")
    assert payload["sft"]["A"]["n"] > 1000
