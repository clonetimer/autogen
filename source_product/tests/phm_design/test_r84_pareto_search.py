"""R84 pure search-policy tests; synthetic feedback, not detector evidence."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math

import pytest

from phm_design.industrial.r84_pareto_search import propose_pareto, METHOD


A = {"width": 8, "depth": 1, "kernel": 3, "pooling": "mean", "residual": False}
B = {"width": 16, "depth": 2, "kernel": 3, "pooling": "mean_max", "residual": True}


def digest(g):
    return hashlib.sha256(json.dumps(g, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False, separators=(",", ":")
                                    ).encode("utf-8")).hexdigest()


def row(g, acc, recall, false_alarm, count):
    m = {"accuracy": acc, "worst_environment_class_recall": recall,
         "healthy_false_alarm": false_alarm}
    return {"genome": deepcopy(g), "genome_sha256": digest(g),
            "feedback_role": "SOURCE_TRAIN_OOF_ONLY", "metrics": m,
            "parameter_count": count,
            "selection_loss": .45 * (1-acc) + .40 * (1-recall) + .15 * false_alarm}


def neighbors(g):
    out = []
    for width in (8, 16, 32):
        if width != g["width"]:
            out.append({**g, "width": width})
    for depth in (1, 2, 3):
        if depth != g["depth"]:
            out.append({**g, "depth": depth})
    out.append({**g, "kernel": 5 if g["kernel"] == 3 else 3})
    out.append({**g, "pooling": "mean_max" if g["pooling"] == "mean" else "mean"})
    out.append({**g, "residual": not g["residual"]})
    return out


def feedback():
    return [
        row(A, .80, .50, .10, 200),
        row(B, .90, .80, .03, 3000),
    ]


def test_policy_is_bounded_train_only_and_changes_parent_with_feedback():
    rows = feedback()
    visited = {x["genome_sha256"] for x in rows}
    child, parent = propose_pareto(rows, visited, neighbors)
    assert METHOD.startswith("TRAIN_OOF_")
    assert parent == digest(B)
    assert child in neighbors(B)
    assert digest(child) not in visited
    assert (child, parent) == propose_pareto(deepcopy(rows), set(visited), neighbors)
    # Degrading B's observed worst-environment recall changes selection.
    bad = row(B, .65, .30, .30, 3000)
    child2, parent2 = propose_pareto([rows[0], bad], visited, neighbors)
    assert parent2 == digest(A)
    assert child2 in neighbors(A)
    assert digest(child2) not in visited


def test_strict_pareto_with_tradeoffs_not_single_accuracy_ranking():
    rows = feedback()
    rows[0] = row(A, .95, .45, .05, 200)
    rows[1] = row(B, .90, .85, .03, 3000)
    visited = {digest(A), digest(B)}
    child, parent = propose_pareto(rows, visited, neighbors)
    assert parent in visited
    assert digest(child) not in visited
    assert child in neighbors(A) + neighbors(B)


@pytest.mark.parametrize("corruption,reason", [
    (lambda r: r[0].update(feedback_role="FINAL_OOD"),
     "PROTECTED_OR_UNKNOWN_FEEDBACK"),
    (lambda r: r[0].update(genome_sha256="0"*64), "GENOME_HASH_MISMATCH"),
    (lambda r: r[0].update(selection_loss=0.0), "INCONSISTENT_TRAIN_OOF_LOSS"),
    (lambda r: r[0]["metrics"].update(accuracy=float("nan")),
     "INVALID_TRAIN_FEEDBACK_VALUE"),
    (lambda r: r[0].update(parameter_count=None), "UNKNOWN_PARAMETER_COUNT"),
    (lambda r: r[0]["metrics"].update(oracle_accuracy=1.0),
     "INVALID_TRAIN_FEEDBACK_METRICS"),
])
def test_reject_untrusted_or_leaked_feedback(corruption, reason):
    rows = feedback()
    corruption(rows)
    with pytest.raises(ValueError, match=reason):
        propose_pareto(rows, {digest(A), digest(B)}, neighbors)


def test_no_reuse_and_no_hidden_test_metrics_in_search():
    rows = feedback()
    with pytest.raises(ValueError, match="VISITED_HISTORY_MISMATCH"):
        propose_pareto(rows, {digest(A)}, neighbors)
    with pytest.raises(ValueError, match="DUPLICATED_FEEDBACK_GENOME"):
        propose_pareto([rows[0], rows[0]], {digest(A)}, neighbors)
    visited = {digest(A), digest(B)}
    with pytest.raises(ValueError, match="EXHAUSTED_PARETO_NEIGHBORHOODS"):
        propose_pareto(rows, visited, lambda g: [A, B])


def test_does_not_modify_original_records_or_rely_on_random_state():
    rows = feedback()
    snapshot = deepcopy(rows)
    result = propose_pareto(rows, {digest(A), digest(B)}, neighbors)
    assert rows == snapshot
    assert result == propose_pareto(rows, {digest(A), digest(B)}, neighbors)


def test_no_nonfinite_or_fake_boolean_metrics():
    for value in (-1, 2, True, math.inf):
        rows = feedback()
        rows[0]["metrics"]["healthy_false_alarm"] = value
        with pytest.raises(ValueError, match="INVALID_TRAIN_FEEDBACK_VALUE"):
            propose_pareto(rows, {digest(A), digest(B)}, neighbors)
