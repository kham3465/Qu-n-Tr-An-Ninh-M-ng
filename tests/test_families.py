from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from slatriage.config_utils import load_yaml
from slatriage.coverage import build_matrix, coverage_cell, load_closed
from slatriage.families import assign_family


def test_assign_family_rac():
    mapping = load_yaml(ROOT / "configs" / "detectors_map.yaml")
    assert assign_family("reentrancy-eth", mapping) == "R"
    assert assign_family("tx-origin", mapping) == "A"
    assert assign_family("arbitrary-send", mapping) == "A"
    assert assign_family("unchecked-transfer", mapping) == "C"
    assert assign_family("unchecked-lowlevel", mapping) == "C"
    assert assign_family("timestamp", mapping) == "O"
    assert assign_family("naming-convention", mapping) is None


def test_coverage_absent_when_no_alert():
    assert coverage_cell(has_keep=False, has_drop=False, has_unknown=False, has_alert=False) == "ABSENT"
    assert coverage_cell(has_keep=True, has_drop=True, has_unknown=False, has_alert=True) == "KEEP"


def test_matrix_on_demo():
    from slatriage.io_utils import read_jsonl

    closed = load_closed(ROOT / "configs" / "closed_list.yaml")
    alerts = read_jsonl(ROOT / "data" / "labels" / "demo_labels.jsonl")
    rows = build_matrix(alerts, closed)
    assert rows
    reent = [r for r in rows if r["category"] == "reentrancy"]
    assert any(r["cell"] in ("KEEP", "DROP") for r in reent)
    arith = [r for r in rows if r["category"] == "arithmetic"]
    assert arith and all(r["cell"] == "ABSENT" for r in arith)


def test_matrix_includes_empty_contract():
    closed = load_closed(ROOT / "configs" / "closed_list.yaml")
    rows = build_matrix([], closed, all_sources=["dataset/other/empty.sol"])
    assert rows
    assert all(r["source"] == "dataset/other/empty.sol" for r in rows)
    assert all(r["cell"] == "ABSENT" for r in rows)
