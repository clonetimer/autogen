"""R96 fold-local diagnosis PIPELINE design — synthetic fixture tests.

Actual sklearn models execute on deliberately fabricated two-sensor values.
Passing does not prove independently measured industrial generalization.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from phm_design.industrial_contracts.pipeline_design import (
    fit_preprocessor, apply_preprocessor,
    propose_pipeline_study, evaluate_pipeline_study)
from phm_design.industrial_contracts.design_engine import DesignRejected
from test_r94_torch_like_bridge import SpyTorchLike
from test_r95_diagnostic_design_engine import RealSklearnProviderContract

ROOT = Path(__file__).resolve().parents[2] / "examples/industrial_contracts"


def data(device):
    def load(name):
        return json.loads((ROOT / f"{device}_{name}.json").read_text())
    return load("equipment"), load("train_only"), load("trial"), load("train_features")


def test_bearing_diagnostic_pipeline_proposals_are_explicit_and_deterministic():
    equipment, train, trial, _ = data("bearing")
    one = propose_pipeline_study(equipment, train, trial)
    assert one == propose_pipeline_study(equipment, train, trial)
    assert one["schema"] == "phm.autogen.pipeline-design.v1"
    assert len(one["study_sha256"]) == 64
    assert one["screening_refusal"] is None
    assert len(one["folds"]) == 4
    assert one["candidate_count"] == trial["budget"]["max_trials"] == 8
    assert one["planned_source_fit_calls"] == 32
    assert one["preprocessing_selection_uses_labels"] is False
    assert not one["validation_calibration_final_used"]
    methods = {p["preprocessing"] for p in one["pipeline_designs"]}
    assert methods == {"identity", "standard_scale", "robust_scale", "signed_log1p"}
    for p in one["pipeline_designs"]:
        assert p["stages"] == ["sensor_select", "preprocess", "provider_estimator"]
        assert p["input_channels"] == ["vibration", "motor_current"]
        assert p["algorithm_family"] in ("linear", "tree")
        assert p["source_only"] is True


def test_fold_local_fit_never_uses_scored_outlier_to_compute_standard_mean():
    # Fit only the two SOURCE fold TRAIN values. Heldout 100000 would
    # catastrophically shift mean if it leaked into preprocessing.
    state = fit_preprocessor("standard_scale", ["vibration"], [[0.], [2.]])
    assert state["location"] == [1.0]
    assert state["scale"] == [1.0]
    before = state["state_sha256"]
    heldout = apply_preprocessor(state, [[100000.]])
    assert heldout == [[99999.]]
    assert state["state_sha256"] == before
    assert apply_preprocessor(state, [[0.], [2.]]) == [[-1.], [1.]]


def test_robust_scale_and_signed_log1p_are_reproducible():
    robust = fit_preprocessor("robust_scale", ["a", "b"],
                              [[-1., 0.], [0., 0.], [1., 0.]])
    assert robust["location"] == [0., 0.]
    assert robust["scale"][0] > 0
    assert robust["scale"][1] > 0   # constant sensor fallback
    assert apply_preprocessor(robust, [[0., 0.]]) == [[0., 0.]]
    signed = fit_preprocessor("signed_log1p", ["a"], [[-100.], [0.], [100.]])
    transformed = apply_preprocessor(signed, [[-3.], [0.], [3.]])
    assert transformed[0][0] < 0 and transformed[1] == [0.0]
    assert transformed[2][0] > 0


def test_frozen_preprocessing_state_rejects_tampering_and_nonfinite_values():
    state = fit_preprocessor("standard_scale", ["a"], [[1.], [2.]])
    poisoned = deepcopy(state)
    poisoned["location"] = [500.]
    with pytest.raises(DesignRejected, match="R96_MUTATED_OR_UNAUTHORIZED_PREPROCESSOR_STATE"):
        apply_preprocessor(poisoned, [[1.]])
    with pytest.raises(DesignRejected, match="R96_UNKNOWN_PREPROCESSING_TRANSFORM"):
        fit_preprocessor("fit_all_including_validation", ["a"], [[1.]])
    with pytest.raises(DesignRejected, match="R96_NONFINITE_OR_PROTECTED_NUMERIC_INPUT"):
        apply_preprocessor(state, [[float("nan")]])
    with pytest.raises(DesignRejected, match="R96_NONFINITE_OR_PROTECTED_NUMERIC_INPUT"):
        fit_preprocessor("identity", ["a"], [[True]])


def test_complete_pipeline_design_evaluates_real_provider_shaped_models():
    equipment, train, trial, observations = data("bearing")
    study = propose_pipeline_study(equipment, train, trial)
    provider = SpyTorchLike()
    report = evaluate_pipeline_study(
        study, equipment, train, trial, observations, provider=provider)
    assert report["status"] == "SOURCE_GROUP_OOF_PIPELINE_SHORTLIST_ONLY"
    assert report["source_data_independent_of_prior_design_iterations"] is False
    assert report["preprocessor_fold_local"] is True
    assert report["actual_fit_calls"] == 32 == len(provider.created)
    assert report["actual_fit_calls"] == report["planned_fit_calls"]
    assert len(report["candidates"]) == len(report["shortlist"]) == 8
    assert report["compute_matched"] is False
    assert report["independent_model_validated"] is False
    assert report["production_qualified"] is False
    for row in report["candidates"]:
        assert 0 <= row["source_worst_group_accuracy"] <= 1
        assert 0 <= row["source_group_macro_accuracy"] <= 1
        assert 0 <= row["source_healthy_false_alarm_fraction"] <= 1
        assert row["total_provider_fit_calls"] == 4
        assert len(row["preprocessor_fit_audit"]) == 4
        for fold in row["preprocessor_fit_audit"]:
            assert fold["transform_fitted_only_on_fold_train"] is True
            assert fold["preprocessor_fit_count"] == 3
            assert fold["model_fit_call_count"] == 1
            assert fold["fold_fit_ids_sha256"] != fold["fold_heldout_ids_sha256"]
            assert len(fold["preprocessor_state_sha256"]) == 64


def test_real_sklearn_runs_raw_vs_standard_scale_on_identical_source_folds():
    pytest.importorskip("sklearn")
    equipment, train, trial, observations = data("bearing")
    trial["candidate_families"] = ["linear", "tree"]
    trial["budget"]["max_trials"] = 4
    study = propose_pipeline_study(equipment, train, trial)
    assert [x["preprocessing"] for x in study["pipeline_designs"]] == [
        "identity", "identity", "standard_scale", "standard_scale"]
    provider = RealSklearnProviderContract()
    report = evaluate_pipeline_study(
        study, equipment, train, trial, observations, provider=provider)
    assert report["status"] == "SOURCE_GROUP_OOF_PIPELINE_SHORTLIST_ONLY"
    assert len(provider.used_real_algorithms) == 16
    assert {entry["algorithm_family"] for entry in report["candidates"]} == {
        "linear", "tree"}
    assert report["actual_fit_calls"] == 16
    assert all(not entry.get("generalization_proven", False)
               for entry in report["candidates"])


def test_mutating_frozen_pipeline_or_sources_rejected_before_training():
    equipment, train, trial, observations = data("bearing")
    frozen = propose_pipeline_study(equipment, train, trial)
    tampered = deepcopy(frozen)
    tampered["pipeline_designs"][1]["preprocessing"] = "robust_scale"
    with pytest.raises(DesignRejected, match="R96_PIPELINE_OR_SOURCE_PLAN_MUTATED"):
        evaluate_pipeline_study(tampered, equipment, train, trial,
                                observations, provider=SpyTorchLike())
    changed = deepcopy(trial)
    changed["budget"]["seed"] = 993
    with pytest.raises(DesignRejected, match="R96_PIPELINE_OR_SOURCE_PLAN_MUTATED"):
        evaluate_pipeline_study(frozen, equipment, train, changed,
                                observations, provider=SpyTorchLike())


def test_forbidden_protected_role_or_target_in_features_prevents_any_fit():
    equipment, train, trial, observations = data("bearing")
    frozen = propose_pipeline_study(equipment, train, trial)
    extra = deepcopy(observations)
    extra[0]["observations"]["ground_truth"] = "FAULT"
    provider = SpyTorchLike()
    with pytest.raises(DesignRejected, match="R95_SOURCE_FEATURE_SCHEMA_OR_TARGET_LEAK"):
        evaluate_pipeline_study(frozen, equipment, train, trial,
                                extra, provider=provider)
    assert provider.created == []
    complete = json.loads((ROOT/"bearing_trusted_audit.json").read_text())
    with pytest.raises(ValueError, match="R93_PROTECTED_ROLE_IN_DESIGN_WORKER"):
        propose_pipeline_study(equipment, complete, trial)


def test_no_independent_fault_evidence_means_no_anomaly_winner():
    equipment, train, trial, features = data("compressor")
    study = propose_pipeline_study(equipment, train, trial)
    assert study["screening_refusal"] == "R95_NO_SOURCE_FAULT_LABELS_FOR_ANOMALY_RANKING"
    report = evaluate_pipeline_study(
        study, equipment, train, trial, features, provider=SpyTorchLike())
    assert report["status"] == "ABSTAIN_INSUFFICIENT_SOURCE_EVIDENCE"
    assert report["actual_fit_calls"] == 0
    assert report["shortlist"] == []


def test_source_group_budget_bounded_before_estimator_creation():
    equipment, train, trial, observations = data("bearing")
    for i in range(14):
        item = deepcopy(train["records"][i%4])
        item["record_id"] = f"device_{i}_record"
        item["physical_id"] = f"device_{i}"
        item["source_sha256"] = hashlib.sha256(f"fake-{i}".encode()).hexdigest()
        train["records"].append(item)
    study = propose_pipeline_study(equipment, train, trial)
    assert study["screening_refusal"] in (
        "R95_SOURCE_FIT_BUDGET_EXCEEDS_HARD_RESEARCH_CEILING",
        "R96_PREDECLARED_PIPELINE_FIT_BUDGET_EXCEEDED")
    assert study["planned_source_fit_calls"] == 0


def test_diagnostics_stages_are_not_freeform_user_supplied_code():
    equipment, train, trial, observations = data("bearing")
    plan = propose_pipeline_study(equipment, train, trial)
    tampered = deepcopy(plan)
    tampered["pipeline_designs"][0]["stages"] = [
        "sensor_select", "eval(request)", "provider_estimator"]
    with pytest.raises(DesignRejected, match="R96_PIPELINE_OR_SOURCE_PLAN_MUTATED"):
        evaluate_pipeline_study(tampered, equipment, train, trial,
                                observations, provider=SpyTorchLike())
