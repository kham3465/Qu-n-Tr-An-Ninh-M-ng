from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_adapter_complete_and_latest_ckpt(tmp_path):
    from slatriage.train_qlora import adapter_complete, latest_checkpoint

    d = tmp_path / "R"
    d.mkdir()
    assert adapter_complete(d) is False
    (d / "train_meta.json").write_text("{}", encoding="utf-8")
    (d / "adapter_config.json").write_text("{}", encoding="utf-8")
    assert adapter_complete(d) is True
    ck = d / "checkpoints"
    (ck / "checkpoint-10").mkdir(parents=True)
    (ck / "checkpoint-30").mkdir()
    (ck / "checkpoint-2").mkdir()
    assert latest_checkpoint(d).name == "checkpoint-30"


def test_holdout_stable():
    from slatriage.io_utils import read_jsonl
    from slatriage.split import holdout_sources

    labels = read_jsonl(ROOT / "data" / "labels" / "labels_v1.jsonl")
    if not labels:
        return
    a = holdout_sources(labels, seed=0, frac=0.3)
    b = holdout_sources(labels, seed=0, frac=0.3)
    assert a == b
    assert 0 < len(a) < len({r.get("source") for r in labels})


def test_train_cli_resume_skip(tmp_path):
    import subprocess
    import sys

    dest = tmp_path / "R"
    dest.mkdir()
    (dest / "train_meta.json").write_text('{"n":1}', encoding="utf-8")
    (dest / "adapter_config.json").write_text("{}", encoding="utf-8")
    r = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "train_lora.py"),
            "--role",
            "R",
            "--sft-dir",
            str(ROOT / "data" / "sft"),
            "--out",
            str(dest),
            "--do-train",
            "--resume",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    # skip happens before CUDA if adapter complete
    out = r.stdout + r.stderr
    assert r.returncode == 0, out
    assert "RESUME skip" in out or "already complete" in out
