from slatriage.curated import DASP_TO_FAMILY, human_findings_for_entry, snap_solc
from slatriage.label_match import match_alert
from slatriage.slither_runner import normalize_detectors, slither_stem
from pathlib import Path


def test_dasp_map():
    assert DASP_TO_FAMILY["reentrancy"] == "R"
    assert DASP_TO_FAMILY["access_control"] == "A"
    assert DASP_TO_FAMILY["unchecked_low_level_calls"] == "C"


def test_keep_same_family_near_line():
    humans = human_findings_for_entry(
        {"vulnerabilities": [{"lines": [42], "category": "reentrancy"}]}
    )
    alert = {"lines": [40], "detector": "reentrancy-eth"}
    assert match_alert(alert, humans, line_window=5, family="R") == "keep"


def test_drop_when_family_differs():
    humans = human_findings_for_entry(
        {"vulnerabilities": [{"lines": [42], "category": "access_control"}]}
    )
    alert = {"lines": [42], "detector": "reentrancy-eth"}
    assert match_alert(alert, humans, line_window=5, family="R") == "drop"


def test_snap_solc_and_stem():
    assert snap_solc("0.4.22") == "0.4.25"
    assert snap_solc("0.8.4") == "0.8.20"
    assert slither_stem(Path("access_control_phishable.slither.json")) == "access_control_phishable"


def test_normalize_detectors_tx_origin():
    payload = {
        "success": True,
        "results": {
            "detectors": [
                {
                    "check": "tx-origin",
                    "impact": "High",
                    "confidence": "High",
                    "description": "tx.origin used",
                    "elements": [{"source_mapping": {"lines": [17], "filename_relative": "phishable.sol"}}],
                }
            ]
        },
    }
    alerts = normalize_detectors(payload)
    assert alerts[0]["detector"] == "tx-origin"
    assert alerts[0]["lines"] == [17]
