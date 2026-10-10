from slatriage.metrics import binary_prf, relative_gain, score_aligned


def test_binary_prf_keep():
    m = binary_prf(["keep", "drop", "keep"], ["keep", "keep", "drop"])
    assert m["tp"] == 1 and m["fp"] == 1 and m["fn"] == 1
    assert 0 < m["f1"] < 1


def test_relative_gain():
    assert relative_gain(0.23, 0.20) == 0.15
    assert relative_gain(0.1, 0.0) is None


def test_score_aligned_per_family():
    gold = [
        {"alert_id": "a1", "source": "s1", "family": "R", "label": "keep"},
        {"alert_id": "a2", "source": "s1", "family": "R", "label": "drop"},
        {"alert_id": "a3", "source": "s2", "family": "C", "label": "keep"},
        {"alert_id": "a4", "source": "s2", "family": "C", "label": "unknown"},
    ]
    pred = [
        {"alert_id": "a1", "source": "s1", "decision": "keep"},
        {"alert_id": "a2", "source": "s1", "decision": "drop"},
        {"alert_id": "a3", "source": "s2", "decision": "drop"},
        {"alert_id": "a4", "source": "s2", "decision": "keep"},
    ]
    m = score_aligned(gold, pred)
    assert m["n_scored"] == 3
    assert m["by_family"]["R"]["f1"] == 1.0
    assert m["by_family"]["C"]["fn"] == 1
