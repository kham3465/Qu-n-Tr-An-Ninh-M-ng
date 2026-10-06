from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from slatriage.council import run_config
from slatriage.io_utils import read_jsonl
from slatriage.llm import MockLLM, extract_json_object
from slatriage.schema import Alert


def test_extract_json():
    assert extract_json_object('{"a":1}')["a"] == 1
    assert extract_json_object("```json\n{\"a\": 2}\n```")["a"] == 2


def test_config_a_keeps_all_scoped():
    rows = read_jsonl(ROOT / "data" / "labels" / "demo_labels.jsonl")
    alerts = [Alert.from_dict(r) for r in rows]
    r = run_config("A", alerts, map_path=ROOT / "configs" / "detectors_map.yaml")
    assert len(r.predictions) == 4
    assert all(p["decision"] == "keep" for p in r.predictions)


def test_mock_council_e():
    rows = read_jsonl(ROOT / "data" / "labels" / "demo_labels.jsonl")
    alerts = [Alert.from_dict(r) for r in rows]
    llm = MockLLM()
    r = run_config(
        "E",
        alerts,
        map_path=ROOT / "configs" / "detectors_map.yaml",
        base_llm=llm,
    )
    assert r.judge is not None
    assert isinstance(r.predictions, list)


def test_assign_family_and_coverage():
    from slatriage.config_utils import load_yaml
    from slatriage.coverage import build_matrix, coverage_cell, load_closed
    from slatriage.families import assign_family

    mapping = load_yaml(ROOT / "configs" / "detectors_map.yaml")
    assert assign_family("reentrancy-eth", mapping) == "R"
    assert assign_family("tx-origin", mapping) == "A"
    assert assign_family("unchecked-lowlevel", mapping) == "C"
    assert assign_family("timestamp", mapping) == "O"
    assert assign_family("naming-convention", mapping) is None
    assert coverage_cell(has_keep=False, has_drop=False, has_unknown=False, has_alert=False) == "ABSENT"

    closed = load_closed(ROOT / "configs" / "closed_list.yaml")
    rows = read_jsonl(ROOT / "data" / "labels" / "demo_labels.jsonl")
    matrix = build_matrix(rows, closed)
    assert any(r["cell"] == "KEEP" for r in matrix)
    assert any(r["category"] == "front_running" and r["cell"] == "ABSENT" for r in matrix)
