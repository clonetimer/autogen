"""R85 cross-family policy: synthetic TRAIN-OOF feedback only, no test access."""
import copy
import math

import pytest

from phm_design.industrial.r85_family_policy import (
    choose_next, validate_pool, SCHEMA)


def pool():
    return [
        {"backbone": backbone, "objective": "ERM", "topology": "direct",
         "representation": ("temporal_bins" if backbone in ("linear", "tree")
                            else "learned_sequence")}
        for backbone in ("linear", "tcn", "tree", "gru", "cnn1d", "transformer")
    ]


def report(index, acc=.8, recall=.6, far=.1):
    return {"candidate_index": index, "role": "SOURCE_TRAIN_OOF_ONLY",
            "metrics": {"accuracy": acc,
                        "worst_environment_class_recall": recall,
                        "healthy_false_alarm": far}}


def test_anchors_and_feedback_change_chosen_model_family():
    x = pool()
    assert validate_pool(x)["gru"] == 3
    assert choose_next(x, [])[0] == 0
    assert choose_next(x, [report(0)])[0] == 1
    a = [report(0, .97, .90, .01), report(1, .58, .30, .2)]
    b = [report(0, .54, .30, .3), report(1, .98, .95, .01)]
    assert choose_next(x, a)[0] == 2  # tree follows strong linear baseline
    idx, audit = choose_next(x, b)
    assert idx == 3  # GRU follows strong TCN baseline
    assert audit["strategy"] == "TRAIN_OOF_FAMILY_FEEDBACK"
    assert audit["parent_index"] == 1
    assert audit["schema"] == SCHEMA
    assert choose_next(x, a) == choose_next(copy.deepcopy(x), copy.deepcopy(a))
    assert choose_next(x, a, mode="predeclared_static")[0] == 2
    assert choose_next(x, b, mode="predeclared_static")[0] == 2


def test_no_duplicate_or_independence_claim():
    x = pool()
    selected = []
    seen = []
    for _ in range(len(x)):
        index, _ = choose_next(x, seen)
        assert index not in selected
        selected.append(index)
        seen.append(report(index))
    assert set(selected) == set(range(6))
    with pytest.raises(ValueError, match="EXHAUSTED_FAMILY_POOL"):
        choose_next(x, seen)


def test_failed_fit_reverts_to_declarative_order_without_truth_data():
    x = pool()
    history = [report(0), report(1)]
    history[0]["metrics"] = None
    history[1]["metrics"] = None
    idx, audit = choose_next(x, history)
    assert idx == 2
    assert audit["strategy"] == "FAILED_FITS_PREDECLARED_FALLBACK"


@pytest.mark.parametrize("bad,code", [
    (lambda h: h[0].update(role="FINAL_OOD"), "UNTRUSTED_OR_PROTECTED_FEEDBACK"),
    (lambda h: h[0].update(extra_metric=1), "UNTRUSTED_OR_PROTECTED_FEEDBACK"),
    (lambda h: h[0].update(candidate_index=True), "DUPLICATE_OR_INVALID_CANDIDATE_INDEX"),
    (lambda h: h[0]["metrics"].update(accuracy=float("nan")), "INVALID_SOURCE_METRIC"),
    (lambda h: h[0]["metrics"].update(healthy_false_alarm=-.01), "INVALID_SOURCE_METRIC"),
    (lambda h: h[0]["metrics"].update(worst_environment_class_recall=True), "INVALID_SOURCE_METRIC"),
])
def test_refuse_poisoned_or_protected_feedback(bad, code):
    x = pool()
    h = [report(0), report(1)]
    bad(h)
    with pytest.raises(ValueError, match=code):
        choose_next(x, h)


def test_refuse_arbitrary_family_or_unregistered_candidate_grammar():
    x = pool()
    x[2] = {**x[2], "backbone": "auto_generated_python_code"}
    with pytest.raises(ValueError, match="UNSUPPORTED_OR_DUPLICATE_FAMILY_CANDIDATE"):
        choose_next(x, [])
    x = pool()
    x[2] = dict(x[0])
    with pytest.raises(ValueError, match="UNSUPPORTED_OR_DUPLICATE_FAMILY_CANDIDATE"):
        choose_next(x, [])
    x = pool()
    x[3]["topology"] = "factorized"
    with pytest.raises(ValueError, match="UNSUPPORTED_OR_DUPLICATE_FAMILY_CANDIDATE"):
        choose_next(x, [])


def test_static_comparison_never_reads_candidate_loss():
    x = pool()
    hist = [report(0), report(1)]
    assert choose_next(x, hist, mode="predeclared_static")[0] == 2
    changed = copy.deepcopy(hist)
    changed[0]["metrics"]["accuracy"] = 0
    changed[1]["metrics"]["accuracy"] = 1
    assert choose_next(x, changed, mode="predeclared_static")[0] == 2
