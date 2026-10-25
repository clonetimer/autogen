"""R85 integration: genuine CPU model fits on SYNTHETIC industrial equipment.

These are development records, not independently blinded/physical test evidence.
"""
from __future__ import annotations

import copy

import pytest

from phm_design.industrial.design_engine import (
    DesignAcceptance, IndustrialDesignRejected, run_design)
from phm_design.industrial.taskspec import TaskSpecRejected
from test_r81a_unified_design_engine import task, synthetic_records


OPEN_SCREEN = DesignAcceptance(
    accuracy_min=0.0, worst_environment_class_recall_min=0.0,
    healthy_false_alarm_max=1.0)


def r85_task():
    spec = task("industrial_bearing_classification_demo.json",
                compare=("linear", "tcn", "tree", "gru"))
    spec["design"]["candidate_max"] = 4
    spec["design"]["budget"]["max_fits"] = 10  # exactly 3*3 OOF + 1 TRAIN refit
    spec["design"]["budget"]["max_cpu_seconds"] = 240
    return spec


def test_actual_cross_family_training_feedback_and_replay():
    spec = r85_task()
    rows = synthetic_records(spec, environment_count=3)
    design = run_design(
        spec, rows, acceptance=OPEN_SCREEN,
        selection_rule="r83_matched_train_loss",
        family_search_mode="feedback", family_proposals=3)
    a = design.audit
    assert a["source_data_role"] == "TRAIN_ONLY"
    assert a["split"] == "LEAVE_ONE_ENTIRE_SOURCE_ENVIRONMENT_OUT"
    assert a["r85_family_search_mode"] == "feedback"
    assert a["r85_family_proposals"] == 3
    assert a["total_model_fits"] == 10
    assert len(a["attempts"]) == 3
    chosen = [entry["trial_index"] for entry in a["attempts"]]
    assert chosen[:2] == [0, 1]
    assert chosen[2] in (2, 3)
    assert len(set(chosen)) == 3
    assert all(len(entry["folds"]) == 3 for entry in a["attempts"])
    assert all(f["disjoint"] for entry in a["attempts"] for f in entry["folds"])
    trace = a["r85_family_trace"]
    assert trace["full_pool_count"] == 4
    assert trace["trained_proposal_count"] == 3
    assert trace["protected_feedback_accessed"] is False
    assert trace["independent_generalization_proven"] is False
    assert design.audit["independent_qualification"] is False
    assert design.predict(rows[0]["measurements"])[0] in spec["task"]["labels"]


def test_static_and_feedback_use_same_folds_and_model_budget(monkeypatch):
    from phm_design.industrial import design_engine as engine
    class FastEstimator:
        def __init__(self, classes):
            import numpy as np
            self.classes_ = np.arange(classes)
            self.fit_count_ = 1
        def predict_proba(self, x):
            import numpy as np
            return np.tile([1.0, 0.0, 0.0], (len(x), 1))
    def fit_stub(trial, train_x, train_y, domains, ontology=None):
        return FastEstimator(3)
    monkeypatch.setattr(engine, "_fitted", fit_stub)
    spec = r85_task()
    rows = synthetic_records(spec, environment_count=3)
    x = run_design(spec, rows, acceptance=OPEN_SCREEN,
                   selection_rule="r83_matched_train_loss",
                   family_search_mode="feedback", family_proposals=3)
    y = run_design(spec, rows, acceptance=OPEN_SCREEN,
                   selection_rule="r83_matched_train_loss",
                   family_search_mode="predeclared_static", family_proposals=3)
    assert x.audit["total_model_fits"] == y.audit["total_model_fits"] == 10
    assert [e["trial_index"] for e in y.audit["attempts"]] == [0, 1, 2]
    assert x.audit["training_observations_sha256"] == y.audit["training_observations_sha256"]
    assert x.audit["training_manifest_sha256"] == y.audit["training_manifest_sha256"]
    assert x.audit["search_budget"] == y.audit["search_budget"]
    assert x.audit["final_evaluation_performed"] is False
    assert y.audit["final_evaluation_performed"] is False


def test_reject_final_or_oracle_channels_before_family_training(monkeypatch):
    from phm_design.industrial import design_engine as engine
    def must_not_fit(*args, **kwargs):
        raise AssertionError("TRAINER_SHOULD_NEVER_RUN")
    monkeypatch.setattr(engine, "_fitted", must_not_fit)
    spec = r85_task()
    rows = synthetic_records(spec, environment_count=3)
    protected = copy.deepcopy(rows)
    protected[0]["metadata"]["role"] = "FINAL_OOD"
    with pytest.raises(TaskSpecRejected, match="DESIGN_WORKER_PROTECTED_ROLE"):
        run_design(spec, protected, selection_rule="r83_matched_train_loss",
                   family_search_mode="feedback")
    poisoned = copy.deepcopy(rows)
    poisoned[0]["measurements"][0]["fault_injection"] = 1
    with pytest.raises(TaskSpecRejected, match="UNDECLARED_OR_PROTECTED_OBSERVATION"):
        run_design(spec, poisoned, selection_rule="r83_matched_train_loss",
                   family_search_mode="feedback")


def test_invalid_family_search_preflight():
    spec = r85_task()
    rows = synthetic_records(spec, environment_count=3)
    with pytest.raises(TaskSpecRejected, match="R85_REQUIRES_COMMON_SOURCE_RISK"):
        run_design(spec, rows, family_search_mode="feedback")
    with pytest.raises(TaskSpecRejected, match="INVALID_R85_CANDIDATE_SLOTS"):
        run_design(spec, rows, selection_rule="r83_matched_train_loss",
                   family_search_mode="feedback", family_proposals=7)
    spec["design"]["compare"] = ["linear", "tree", "tcn"]
    spec["design"]["candidate_max"] = 3
    with pytest.raises(ValueError, match="FOUR_TO_SIX|FOUR_REFERENCE_FAMILIES"):
        run_design(spec, rows, selection_rule="r83_matched_train_loss",
                   family_search_mode="feedback")
