"""R86 actual neuro-fuzzy training on SYNTHETIC data, not physical test evidence."""
from __future__ import annotations

import copy

import numpy as np
import pytest

from phm_design.industrial.r86_neuro_fuzzy import (
    FuzzyRuleClassifier, FuzzyRulePolicy, sequence_summary)
from phm_design.industrial.r86_fuzzy_source_study import run_fuzzy_source_study
from phm_design.industrial.taskspec import TaskSpecRejected
from test_r81a_unified_design_engine import synthetic_records, task


def fake_sequences():
    rng = np.random.default_rng(17)
    samples = []
    labels = []
    for y in range(3):
        for i in range(9):
            t = np.linspace(0, 1, 48)
            samples.append(np.column_stack((
                np.sin((y+1)*t*3) + .2*y + rng.normal(0, .05, 48),
                np.cos(t*(i % 3+1)) + .3*y + rng.normal(0, .05, 48),
                (y+.5)*t + rng.normal(0, .05, 48))))
            labels.append(y)
    return np.asarray(samples, dtype=float), np.asarray(labels, dtype=int)


def test_real_gaussian_tsk_fits_both_control_arms_same_count():
    seq, labels = fake_sequences()
    policy = FuzzyRulePolicy(max_rules=4)
    fixed = FuzzyRuleClassifier(policy, mode="fixed").fit(seq, labels)
    adaptive = FuzzyRuleClassifier(policy, mode="adaptive").fit(seq, labels)
    for model in (fixed, adaptive):
        pred = model.predict_proba(seq[:2])
        assert pred.shape == (2, 3)
        assert np.isfinite(pred).all()
        assert np.allclose(pred.sum(axis=1), 1.0)
        assert len(model.predict(seq[:1])) == 1
        rule = model.explain_rules(seq[:1])[0]
        assert sum(rule["rule_firing_strengths"]) == pytest.approx(1.0)
        assert 0 <= rule["dominant_rule_index"] < model.audit["selected_rule_count"]
        assert model.audit["candidate_fit_count"] == 4
        assert not model.audit["independent_qualification"]
        assert not model.audit["premise_gradient_optimization_implemented"]
        assert model.audit["fold_local_preprocessing"]
        assert len(model.source_receipt_sha256()) == 64
    assert fixed.audit["selected_rule_count"] == 4
    assert 1 <= adaptive.audit["selected_rule_count"] <= 4
    assert adaptive.audit["tail_rules_pruned"] == 4 - adaptive.audit["selected_rule_count"]
    assert fixed.audit["model_family"].startswith("GAUSSIAN_MEMBERSHIP")


def test_deterministic_model_on_same_source_training_rows():
    seq, labels = fake_sequences()
    a = FuzzyRuleClassifier(FuzzyRulePolicy()).fit(seq, labels)
    b = FuzzyRuleClassifier(FuzzyRulePolicy()).fit(seq, labels)
    assert np.array_equal(a.predict_proba(seq[:4]), b.predict_proba(seq[:4]))
    assert a.source_receipt_sha256() == b.source_receipt_sha256()


def test_observation_only_single_sequence_and_invalid_data():
    seq, labels = fake_sequences()
    model = FuzzyRuleClassifier().fit(seq, labels)
    with pytest.raises(ValueError, match="INVALID_R86_OBSERVED_SEQUENCE"):
        model.predict(seq[:1] * float("nan"))
    with pytest.raises(ValueError, match="R86_OBSERVATION_CHANNEL_MISMATCH"):
        model.predict(seq[:1, :, :2])
    with pytest.raises(ValueError, match="INVALID_R86_TRAIN_LABELS"):
        model.fit(seq, labels[:-1])
    with pytest.raises(ValueError, match="R86_NONINTEGER_OR_NEGATIVE_CLASSES"):
        model.fit(seq, labels.astype(float))
    with pytest.raises(ValueError, match="INVALID_R86_FUZZY_POLICY"):
        FuzzyRulePolicy(max_rules=30)
    with pytest.raises(ValueError, match="INVALID_R86_RULE_MODE"):
        FuzzyRuleClassifier(mode="fictitious")


def test_actual_grouped_oof_fixed_and_adaptive_fuzzy_with_linear_reference():
    spec = task("industrial_bearing_classification_demo.json",
                compare=("linear",))
    data = synthetic_records(spec, environment_count=3)
    receipt = run_fuzzy_source_study(
        spec, data, policy=FuzzyRulePolicy(max_rules=3))
    assert receipt["source_data_role"] == "TRAIN_ONLY"
    assert receipt["environment_count"] == 3
    assert receipt["three_arms_complete"]
    assert receipt["same_folds_all_arms"]
    assert receipt["fixed_adaptive_equal_candidate_fit_count"]
    assert receipt["arms"]["fixed_fuzzy"]["candidate_fits"] == 9
    assert receipt["arms"]["adaptive_fuzzy"]["candidate_fits"] == 9
    assert receipt["arms"]["linear_reference"]["candidate_fits"] == 3
    assert set(receipt["arms"]) == {
        "linear_reference", "fixed_fuzzy", "adaptive_fuzzy"}
    for arm in receipt["arms"].values():
        for key, value in arm["source_train_environment_oof"].items():
            assert 0 <= value <= 1, key
    assert receipt["blind_test_performed"] is False
    assert receipt["source_oof_is_unbiased_generalization"] is False
    assert receipt["fuzzy_mechanism_generalization_benefit_proven"] is False
    assert not receipt["eligible_for_stable"]


def test_refuse_final_data_truth_and_reused_physical_group():
    spec = task("industrial_bearing_classification_demo.json",
                compare=("linear",))
    data = synthetic_records(spec, environment_count=3)
    protected = copy.deepcopy(data)
    protected[0]["metadata"]["role"] = "FINAL_OOD"
    with pytest.raises(TaskSpecRejected, match="DESIGN_WORKER_PROTECTED_ROLE"):
        run_fuzzy_source_study(spec, protected)
    truth = copy.deepcopy(data)
    truth[0]["measurements"][0]["fault_label"] = 1
    with pytest.raises(TaskSpecRejected, match="UNDECLARED_OR_PROTECTED_OBSERVATION"):
        run_fuzzy_source_study(spec, truth)
    cross = copy.deepcopy(data)
    key = spec["data"]["group_keys"][0]
    cross[2]["metadata"][key] = cross[0]["metadata"][key]
    with pytest.raises(TaskSpecRejected, match="R86_GROUP_CROSSES_ENVIRONMENT|UNSUPPORTED_TRAINING|PHYSICAL_GROUP_CROSSES_ENVIRONMENTS"):
        # Source compiler may reject duplicated logical group before LOEO.
        cross[2]["metadata"][spec["data"]["environment_axes"][0]] = "other_regime"
        run_fuzzy_source_study(spec, cross)


def test_fuzzy_train_selection_not_independent_even_if_source_scores_high():
    spec = task("industrial_bearing_classification_demo.json",
                compare=("linear",))
    data = synthetic_records(spec, environment_count=3)
    result = run_fuzzy_source_study(
        spec, data, policy=FuzzyRulePolicy(max_rules=2))
    assert result["arms"]["adaptive_fuzzy"]["candidate_fits"] == 6
    assert not result["independent_validation_performed"]
    assert not result["fuzzy_mechanism_generalization_benefit_proven"]
    assert result["source_data_role"] == "TRAIN_ONLY"
