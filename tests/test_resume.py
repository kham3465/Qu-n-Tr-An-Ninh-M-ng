from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_train_pins_single_gpu():
    from slatriage.train_qlora import _train_device_map

    assert _train_device_map() == {"": 0}


def test_cast_skips_when_no_bf16():
    from slatriage.train_qlora import _strip_bfloat16

    class M:
        config = type("C", (), {})()

        def named_parameters(self):
            return []

        def named_buffers(self):
            return []

    _strip_bfloat16(M())


def test_strip_bf16_to_fp32():
    import torch
    from torch import nn

    from slatriage.train_qlora import _strip_bfloat16

    m = nn.Linear(2, 2)
    m.to(torch.bfloat16)
    _strip_bfloat16(m)
    assert m.weight.dtype == torch.float32
    assert m.bias.dtype == torch.float32


def test_disable_amp_clears_flags():
    from slatriage.train_qlora import _disable_amp

    class Args:
        fp16 = True
        bf16 = True
        bf16_full_eval = True

    class Acc:
        native_amp = True
        scaler = object()
        state = type("S", (), {"mixed_precision": "bf16"})()

    class Trainer:
        args = Args()
        accelerator = Acc()
        scaler = object()

    t = Trainer()
    _disable_amp(t)
    assert t.args.fp16 is False
    assert t.args.bf16 is False
    assert t.accelerator.native_amp is False
    assert t.accelerator.scaler is None
    assert t.scaler is None


def test_pin_overwrites_mixed_precision(monkeypatch):
    import os

    monkeypatch.setenv("ACCELERATE_MIXED_PRECISION", "bf16")
    from slatriage.train_qlora import pin_single_gpu

    pin_single_gpu()
    assert os.environ["ACCELERATE_MIXED_PRECISION"] == "no"


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
