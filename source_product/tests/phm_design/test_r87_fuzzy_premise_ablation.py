"""R87 real analytic gradient, learned Gaussian rules and 2x2 source ablation.

All data are deterministic SYNTHETIC TRAIN trajectories: not an independent
blind fault-diagnosis result or evidence of superior generalization.
"""
from __future__ import annotations

import copy

import numpy as np
import pytest

from phm_design.industrial.r87_premise_learning import (
    fuzzy_design, gaussian_firing, premise_gradient, refine_premises)
from phm_design.industrial.r86_neuro_fuzzy import (
    FuzzyRuleClassifier, FuzzyRulePolicy)
from phm_design.industrial.r86_fuzzy_source_study import run_fuzzy_source_study
from phm_design.industrial.taskspec import TaskSpecRejected
from test_r81a_unified_design_engine import task, synthetic_records


def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    q = np.exp(x)
    return q / q.sum(axis=1, keepdims=True)


def cross_entropy(x, y, centers, log_widths, coef):
    p = softmax(fuzzy_design(x, centers, log_widths) @ coef)
    return float(-np.log(np.maximum(p[np.arange(len(y)), y], 1e-12)).mean())


def test_analytic_gaussian_center_and_width_gradients_agree_with_finite_difference():
    rng = np.random.default_rng(319)
    x = rng.normal(size=(11, 4))
    centers = rng.normal(size=(2, 4))
    widths = rng.normal(scale=.12, size=(2, 4))
    coefficients = rng.normal(scale=.3, size=(2 * 5, 3))
    labels = np.asarray([i % 3 for i in range(len(x))])
    gc, gw = premise_gradient(x, labels, centers, widths, coefficients)
    step = 1e-5
    for rule in range(2):
        for dim in range(4):
            plus, minus = centers.copy(), centers.copy()
            plus[rule, dim] += step
            minus[rule, dim] -= step
            finite = (cross_entropy(x, labels, plus, widths, coefficients)
                      - cross_entropy(x, labels, minus, widths, coefficients)) / (2*step)
            assert gc[rule, dim] == pytest.approx(finite, abs=1e-6)
            plus, minus = widths.copy(), widths.copy()
            plus[rule, dim] += step
            minus[rule, dim] -= step
            finite = (cross_entropy(x, labels, centers, plus, coefficients)
                      - cross_entropy(x, labels, centers, minus, coefficients)) / (2*step)
            assert gw[rule, dim] == pytest.approx(finite, abs=1e-6)


def test_refinement_does_not_increase_train_fit_loss_and_observes_bounded_steps():
    rng = np.random.default_rng(20)
    x = rng.normal(size=(24, 3))
    y = np.asarray([0, 1] * 12, dtype=int)
    centers = x[[0, 1, 2], :].copy()
    before = centers.copy()
    out = refine_premises(
        x, y, centers, ridge=.05, steps=6, learning_rate=.1)
    assert np.array_equal(centers, before)
    assert out["training_loss"] <= out["initial_loss"] + 1e-12
    assert 0 <= out["premise_steps_accepted"] <= 6
    assert out["consequent_refits"] >= 1
    assert np.isfinite(out["centers"]).all()
    assert np.isfinite(out["log_widths"]).all()
    assert np.allclose(gaussian_firing(x, out["centers"], out["log_widths"]).sum(1), 1)


def test_real_2x2_rule_organization_by_premise_learning_on_synthetic_source_data():
    spec = task("industrial_bearing_classification_demo.json", compare=("linear",))
    train = synthetic_records(spec, environment_count=3)
    p = FuzzyRulePolicy(max_rules=3, premise_steps=3,
                        premise_learning_rate=.12)
    receipt = run_fuzzy_source_study(
        spec, train, policy=p, include_learned_premise=True)
    assert receipt["source_data_role"] == "TRAIN_ONLY"
    assert receipt["same_folds_all_arms"] is True
    assert receipt["four_cell_rule_premise_ablation"] is True
    assert receipt["fuzzy_candidate_fits_equal_across_all_cells"]
    assert receipt["ridge_refit_count_equal_across_premise_conditions"] is False
    assert receipt["environment_count"] == 3
    assert set(receipt["arms"]) == {
        "linear_reference", "fixed_fuzzy", "adaptive_fuzzy",
        "fixed_fuzzy_learned_premise", "adaptive_fuzzy_learned_premise"}
    for name, arm in receipt["arms"].items():
        assert all(0 <= value <= 1 for value in
                   arm["source_train_environment_oof"].values())
        if "fuzzy" in name:
            assert arm["candidate_fits"] == 9
            assert len(arm["fold_rule_trace"]) == 3
            assert all(not item["heldout_group_overlap"] for item in
                       arm["fold_rule_trace"])
        if "learned_premise" in name:
            assert arm["consequent_ridge_refits"] >= 9
            assert all(item["premise_tuned"] for item in arm["fold_rule_trace"])
    assert receipt["fuzzy_mechanism_generalization_benefit_proven"] is False
    assert receipt["blind_test_performed"] is False
    assert receipt["eligible_for_stable"] is False


def test_real_trainable_fuzzy_inference_and_backwards_compatibility():
    rng = np.random.default_rng(31)
    inputs = rng.normal(size=(27, 32, 2))
    labels = np.tile([0, 1, 2], 9)
    old = FuzzyRuleClassifier(FuzzyRulePolicy(max_rules=2)).fit(inputs, labels)
    learned = FuzzyRuleClassifier(
        FuzzyRulePolicy(max_rules=2, premise_steps=4), mode="adaptive"
    ).fit(inputs, labels)
    assert old.audit["premise_gradient_optimization_implemented"] is False
    assert learned.audit["premise_gradient_optimization_implemented"] is True
    assert learned.audit["learnable_width_parameter_count"] > 0
    assert learned.audit["total_consequent_ridge_refits"] >= 2
    assert learned.audit["independent_qualification"] is False
    a = learned.predict_proba(inputs[:1])
    assert a.shape == (1, 3)
    assert np.isfinite(a).all()
    assert np.allclose(a.sum(axis=1), 1.)
    assert sum(learned.explain_rules(inputs[:1])[0]["rule_firing_strengths"]) == pytest.approx(1.)
    replay = FuzzyRuleClassifier(
        FuzzyRulePolicy(max_rules=2, premise_steps=4), mode="adaptive"
    ).fit(inputs, labels)
    assert np.allclose(a, replay.predict_proba(inputs[:1]), atol=1e-12)


def test_explicit_ablation_and_protected_test_data_refused():
    spec = task("industrial_bearing_classification_demo.json", compare=("linear",))
    records = synthetic_records(spec, environment_count=3)
    with pytest.raises(ValueError, match="R87_EXPLICIT_PREMISE_TUNING_ABLATION_REQUIRED"):
        run_fuzzy_source_study(
            spec, records, policy=FuzzyRulePolicy(max_rules=2, premise_steps=2))
    with pytest.raises(ValueError, match="R87_EXPLICIT_PREMISE_TUNING_ABLATION_REQUIRED"):
        run_fuzzy_source_study(
            spec, records, policy=FuzzyRulePolicy(max_rules=2),
            include_learned_premise=True)
    bad = copy.deepcopy(records)
    bad[0]["metadata"]["role"] = "FINAL_OOD"
    with pytest.raises(TaskSpecRejected, match="DESIGN_WORKER_PROTECTED_ROLE"):
        run_fuzzy_source_study(
            spec, bad, policy=FuzzyRulePolicy(max_rules=2, premise_steps=2),
            include_learned_premise=True)


def test_gaussian_premise_numeric_refusal():
    x = np.ones((4, 2))
    centers = np.ones((2, 2))
    widths = np.zeros_like(centers)
    with pytest.raises(ValueError, match="R87_INVALID_GAUSSIAN_PREMISE"):
        gaussian_firing(x, centers, widths + np.inf)
    with pytest.raises(ValueError, match="R87_INVALID_GAUSSIAN_PREMISE"):
        gaussian_firing(x, centers, widths[:, :1])
    with pytest.raises(ValueError, match="INVALID_R86_FUZZY_POLICY"):
        FuzzyRulePolicy(premise_steps=17)
    with pytest.raises(ValueError, match="INVALID_R86_FUZZY_POLICY"):
        FuzzyRulePolicy(premise_learning_rate=.75)
