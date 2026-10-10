from pathlib import Path

from slatriage.label_match import match_alert
from slatriage.solidifi import FOLDER_TO_FAMILY, injection_stats, iter_solidifi_contracts, load_buglog

ROOT = Path(__file__).resolve().parents[1]
SB = ROOT / "data" / "raw" / "SolidiFI-benchmark" / "buggy_contracts"


def test_folder_map():
    assert FOLDER_TO_FAMILY["Re-entrancy"] == "R"
    assert FOLDER_TO_FAMILY["tx.origin"] == "A"
    assert FOLDER_TO_FAMILY["Unchecked-Send"] == "C"


def test_buglog_and_keep_if_present():
    if not SB.exists():
        return
    stats = injection_stats(SB)
    assert stats.get("tx.origin", 0) >= 1000
    item = next(x for x in iter_solidifi_contracts(SB) if x["folder"] == "tx.origin")
    humans = load_buglog(item["log_path"], family="A", category="access_control")
    assert humans and humans[0]["lines"]
    alert = {"lines": [humans[0]["lines"][0]], "detector": "tx-origin"}
    assert match_alert(alert, humans, line_window=5, family="A", multi_hit="keep") == "keep"


def test_garbled_csv_type_ignored():
    if not SB.exists():
        return
    item = next(x for x in iter_solidifi_contracts(SB) if x["folder"] == "Re-entrancy")
    humans = load_buglog(item["log_path"], family=item["family"], category=item["category"])
    assert humans
    assert all(h["family"] == "R" for h in humans)
