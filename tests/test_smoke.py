from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from slatriage.council import run_config
from slatriage.io_utils import read_jsonl
from slatriage.llm import MockLLM, extract_json_object
from slatriage.schema import Alert


def test_hf_backends_share_one_pool():
    from slatriage.llm import HuggingFaceLLM, get_hf_pool, release_hf_pools

    release_hf_pools()
    a = HuggingFaceLLM("unit-test/model", adapter_path="/tmp/R", adapter_name="R")
    b = HuggingFaceLLM("unit-test/model", adapter_path="/tmp/A", adapter_name="A")
    assert a._pool is b._pool
    assert get_hf_pool("unit-test/model") is a._pool
    release_hf_pools()


def test_extract_json():
    assert extract_json_object('{"a":1}')["a"] == 1
    assert extract_json_object("```json\n{\"a\": 2}\n```")["a"] == 2


def test_config_a_keeps_all_scoped():
    rows = read_jsonl(ROOT / "data" / "labels" / "demo_labels.jsonl")
    alerts = [Alert.from_dict(r) for r in rows]
    r = run_config("A", alerts, map_path=ROOT / "configs" / "detectors_map.yaml")
    assert len(r.predictions) == 4
    assert all(p["decision"] == "keep" for p in r.predictions)


def test_single_shot_chunks_many_alerts():
    from slatriage.agents import INFER_ALERT_CHUNK, single_shot_triage
    from slatriage.llm import MockLLM
    from slatriage.schema import Alert

    alerts = [
        Alert(alert_id=f"a{i}", detector="reentrancy-eth", family="R", lines=[i], source="x.sol")
        for i in range(INFER_ALERT_CHUNK + 3)
    ]
    out = single_shot_triage(MockLLM(), alerts)
    assert {p["alert_id"] for p in out} == {a.alert_id for a in alerts}


def test_judge_accepts_list_json():
    from slatriage.agents import JudgeAgent
    from slatriage.llm import LLMBackend
    from slatriage.schema import Alert, ExpertDecision

    class ListJudge(LLMBackend):
        def generate(self, system, user, **kwargs):
            return '[{"alert_id":"a1","decision":"keep","line":1}]'

    alerts = [Alert(alert_id="a1", detector="reentrancy-eth", family="R", lines=[1], source="x.sol")]
    experts = [ExpertDecision(alert_id="a1", family="R", decision="keep", line=1, detector="reentrancy-eth")]
    out = JudgeAgent(ListJudge(), forbid_invent=True).merge(experts, alerts)
    assert out.findings and out.findings[0].alert_id == "a1"


def test_judge_chunks_many_alerts():
    from slatriage.agents import INFER_JUDGE_CHUNK, JudgeAgent
    from slatriage.llm import MockLLM
    from slatriage.schema import Alert, ExpertDecision

    alerts = [
        Alert(alert_id=f"a{i}", detector="reentrancy-eth", family="R", lines=[i], source="x.sol")
        for i in range(INFER_JUDGE_CHUNK + 4)
    ]
    experts = [
        ExpertDecision(alert_id=a.alert_id, family="R", decision="keep", line=a.lines[0], detector=a.detector)
        for a in alerts
    ]
    out = JudgeAgent(MockLLM(), forbid_invent=True).merge(experts, alerts)
    assert out.findings


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
