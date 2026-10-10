from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_compact_and_validate_sft():
    from slatriage.dataset import compact_alert, validate_sft_row

    big = compact_alert({"description": "x" * 5000, "snippet": "y" * 5000, "label": "keep"})
    assert len(big["description"]) < 1000
    assert len(big["snippet"]) < 1400
    validate_sft_row(
        {
            "messages": [
                {"role": "system", "content": "s"},
                {"role": "user", "content": "u"},
                {"role": "assistant", "content": '{"decision":"keep"}'},
            ]
        }
    )


def test_bundled_sft_files():
    from slatriage.dataset import load_sft_jsonl
    from slatriage.train_qlora import find_sft

    sft_dir = ROOT / "data" / "sft"
    if not (sft_dir / "sft_R.jsonl").exists():
        return
    for role in ("R", "A", "C", "J"):
        path = find_sft(sft_dir, role)
        rows = load_sft_jsonl(path)
        assert rows
        longest = max(len(str(r)) for r in rows)
        assert longest < 20000, longest


def test_merge_sft_keeps_split(tmp_path):
    from slatriage.dataset import merge_sft_jsonl

    a = tmp_path / "sft_R.jsonl"
    b = tmp_path / "sft_solidifi_R.jsonl"
    out = tmp_path / "combined" / "sft_R.jsonl"
    row = {
        "messages": [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": '{"decision":"keep"}'},
        ]
    }
    a.write_text(
        __import__("json").dumps({**row, "alert_id": "c1", "family": "R", "label": "keep", "split": "curated"})
        + "\n",
        encoding="utf-8",
    )
    b.write_text(
        __import__("json").dumps({**row, "alert_id": "s1", "family": "R", "label": "keep", "split": "solidifi"})
        + "\n",
        encoding="utf-8",
    )
    n = merge_sft_jsonl([a, b], out)
    assert n == 2
    texts = out.read_text(encoding="utf-8")
    assert "curated" in texts and "solidifi" in texts


def test_train_cli_prepare_only(tmp_path):
    import subprocess
    import sys

    sft = ROOT / "data" / "sft" / "sft_R.jsonl"
    if not sft.exists():
        return
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "train_lora.py"),
        "--role",
        "R",
        "--sft-dir",
        str(ROOT / "data" / "sft"),
        "--prepare-only",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "prepare-only" in r.stdout or "existing SFT" in r.stdout