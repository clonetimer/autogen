"""R95 algorithm DESIGN tests, not simulated-physics performance proof.

The fake provider verifies exact calls. A separate sklearn-backed provider
contract test trains *real* sklearn estimators, without claiming that the
private torch_like package was checked out or that data are real equipment.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import json

import pytest

from phm_design.industrial_contracts.design_engine import (
    DesignRejected, propose_diagnostic_study, execute_source_design_study)
from phm_design.industrial_contracts.torch_like_bridge import (
    ProviderRefused, TorchLikeTabularCandidate)
from test_r94_torch_like_bridge import SpyTorchLike

EX = Path(__file__).resolve().parents[2] / "examples/industrial_contracts"


def read(name):
    return json.loads((EX / name).read_text(encoding="utf-8"))


def data(device):
    return (
        read(f"{device}_equipment.json"),
        read(f"{device}_train_only.json"),
        read(f"{device}_trial.json"),
        read(f"{device}_train_features.json"),
    )


def test_designer_synthesizes_seeded_family_sensor_and_hyperparam_candidates():
    eq, train, trial, _ = data("bearing")
    first = propose_diagnostic_study(eq, train, trial)
    second = propose_diagnostic_study(eq, train, trial)
    assert first == second
    assert first["schema"] == "phm.autogen.diagnostic-design-study.v1"
    assert len(first["study_sha256"]) == 64
    assert first["training_already_run"] is False
    assert first["candidate_count"] <= trial["budget"]["max_trials"]
    assert first["source_oof_fit_calls_planned"] == 32
    assert len(first["folds"]) == 4
    assert first["screening_refusal"] is None
    assert first["unimplemented_candidate_families"] == ["constant", "statistical"]
    assert {d["algorithm_family"] for d in first["candidate_blueprints"]} == {
        "linear", "tree"}
    assert first["candidate_blueprints"][0]["input_channels"] == trial["input_channels"]
    assert first["candidate_blueprints"][1]["input_channels"] == trial["input_channels"]
    for fold in first["folds"]:
        assert set(fold["fit_ids"]).isdisjoint(fold["score_ids"])
        assert len(fold["score_ids"]) == 1
        assert fold["axis"] == "physical_id"
        assert len(fold["group_sha256"]) == 64
    assert "validation_labels" not in json.dumps(first)


def test_actual_source_train_group_oof_runs_every_generated_algorithm_design():
    eq, train, trial, observations = data("bearing")
    design = propose_diagnostic_study(eq, train, trial)
    fake = SpyTorchLike()
    report = execute_source_design_study(
        design, eq, train, trial, observations, provider=fake)
    assert report["status"] == "SOURCE_TRAIN_OOF_DESCRIPTIVE_SHORTLIST_ONLY"
    assert report["study_sha256"] == design["study_sha256"]
    assert report["source_fit_calls_actual"] == design["source_oof_fit_calls_planned"]
    assert len(fake.created) == report["source_fit_calls_actual"] == 32
    assert len(report["evaluated_candidates"]) == design["candidate_count"]
    assert len(report["provisional_source_shortlist"]) == design["candidate_count"]
    assert report["compute_budget_parity_proven"] is False
    assert report["algorithm_generalization_proven"] is False
    assert report["production_qualified"] is False
    assert not report["calibration_validation_final_accessed"]
    for x in report["evaluated_candidates"]:
        assert x["source_oof_records_scored"] == 4
        assert x["source_oof_heldout_groups"] == 4
        assert x["observed_train_fit_calls"] == 4
        assert 0 <= x["source_oof_group_macro_accuracy"] <= 1
        assert 0 <= x["source_oof_worst_group_accuracy"] <= 1
        assert 0 <= x["source_oof_healthy_false_alarm_fraction"] <= 1
        assert x["matched_cpu_gpu_compute_proven"] is False


def test_autogen_design_variants_reach_provider_native_constructor_options():
    eq, train, trial, observations = data("bearing")
    trial["budget"]["max_trials"] = 8
    study = propose_diagnostic_study(eq, train, trial)
    spy = SpyTorchLike()
    execute_source_design_study(study, eq, train, trial, observations, provider=spy)
    assert ("LogisticRegression", {"max_iter": 400,
            "random_state": 8001, "C": 1.0}) in spy.created
    assert ("DecisionTreeClassifier", {"random_state": 8001,
            "max_depth": 5}) in spy.created
    assert {v[0] for v in spy.created} == {
        "LogisticRegression", "DecisionTreeClassifier"}


@pytest.mark.parametrize("option", [
    {"random_state": 0},
    {"__class__": "UnsafeObject"},
    {"C": -1},
    {"C": True},
    {"C": 999},
])
def test_estimator_config_space_is_strictly_whitelisted(option):
    with pytest.raises(ProviderRefused, match="R95_UNREGISTERED_OR_UNSAFE_DESIGN_HYPERPARAMETER"):
        TorchLikeTabularCandidate("linear", provider=SpyTorchLike(),
                                  design_options=option)


def test_candidate_is_not_selected_when_only_healthy_anomaly_records_exist():
    eq, train, trial, features = data("compressor")
    study = propose_diagnostic_study(eq, train, trial)
    assert study["screening_refusal"] == "R95_NO_SOURCE_FAULT_LABELS_FOR_ANOMALY_RANKING"
    assert study["candidate_count"] >= 1
    assert {x["algorithm_family"] for x in study["candidate_blueprints"]} == {
        "isolation_forest"}
    report = execute_source_design_study(
        study, eq, train, trial, features, provider=SpyTorchLike())
    assert report["status"] == "ABSTAIN_SOURCE_EVIDENCE_INSUFFICIENT"
    assert report["source_fit_calls_actual"] == 0
    assert report["provisional_source_shortlist"] == []
    assert report["algorithm_generalization_proven"] is False


def test_source_plan_must_be_frozen_before_fit_and_any_label_inspection():
    eq, train, trial, features = data("bearing")
    study = propose_diagnostic_study(eq, train, trial)
    corrupted = deepcopy(study)
    corrupted["candidate_blueprints"][0]["hyperparameters"] = {"C": 0.1}
    with pytest.raises(DesignRejected, match="R95_STUDY_OR_DATA_CHANGED_AFTER_FREEZE"):
        execute_source_design_study(corrupted, eq, train, trial, features,
                                    provider=SpyTorchLike())
    copy_trial = deepcopy(trial)
    copy_trial["budget"]["seed"] += 1
    with pytest.raises(DesignRejected, match="R95_STUDY_OR_DATA_CHANGED_AFTER_FREEZE"):
        execute_source_design_study(study, eq, train, copy_trial, features,
                                    provider=SpyTorchLike())


def test_no_protected_external_validation_can_enter_design_or_evaluation():
    eq, _, trial, features = data("bearing")
    whole = read("bearing_trusted_audit.json")
    with pytest.raises(ValueError, match="R93_PROTECTED_ROLE_IN_DESIGN_WORKER"):
        propose_diagnostic_study(eq, whole, trial)
    source = read("bearing_train_only.json")
    study = propose_diagnostic_study(eq, source, trial)
    with pytest.raises(ValueError, match="R93_PROTECTED_ROLE_IN_DESIGN_WORKER"):
        execute_source_design_study(study, eq, whole, trial, features,
                                    provider=SpyTorchLike())


def test_extra_metadata_and_label_cannot_be_feature_input_to_provider():
    eq, train, trial, features = data("bearing")
    features[0]["observations"]["ground_truth"] = "FAULT"
    study = propose_diagnostic_study(eq, train, trial)
    spy = SpyTorchLike()
    with pytest.raises(DesignRejected, match="R95_SOURCE_FEATURE_SCHEMA_OR_TARGET_LEAK"):
        execute_source_design_study(study, eq, train, trial, features,
                                    provider=spy)
    assert spy.created == []


def test_reusing_physical_group_with_different_id_does_not_bypass_r93():
    eq, train, trial, _ = data("bearing")
    train["records"][2]["source_sha256"] = train["records"][0]["source_sha256"]
    with pytest.raises(ValueError, match="R93_SAME_SIGNAL_BYTES_ASSIGNED_TO_DIFFERENT_DEVICES"):
        propose_diagnostic_study(eq, train, trial)


def test_insufficient_source_physical_group_classes_abstain_without_training():
    eq, train, trial, features = data("bearing")
    # Keep two healthy groups but only one fault physical group.
    train["records"] = train["records"][:3]
    features = features[:3]
    study = propose_diagnostic_study(eq, train, trial)
    assert study["screening_refusal"] == "R95_INSUFFICIENT_INDEPENDENT_SOURCE_DEVICES_PER_CLASS"
    result = execute_source_design_study(study, eq, train, trial, features,
                                         provider=SpyTorchLike())
    assert result["status"] == "ABSTAIN_SOURCE_EVIDENCE_INSUFFICIENT"
    assert result["evaluated_candidates"] == []


def test_unseen_environment_needs_three_SOURCE_environments_to_be_a_study():
    eq, train, trial, features = data("bearing")
    trial["validation_target"] = "unseen_environment"
    plan = propose_diagnostic_study(eq, train, trial)
    assert plan["screening_refusal"] == "R95_AT_LEAST_THREE_SOURCE_ENVIRONMENTS_REQUIRED"


def test_hard_fit_count_ceiling_avoids_silent_budget_overrun():
    eq, train, trial, features = data("bearing")
    # Expand with unique physical entities/record SHA in both classes; all
    # examples here are synthetic and remain SOURCE TRAIN.
    for n in range(12):
        row = deepcopy(train["records"][n % 4])
        row["record_id"] = f"large_train_{n:02d}"
        row["physical_id"] = f"synthetic_bearing_{n:02d}"
        row["source_sha256"] = ("%x" % (n % 16))*64
        train["records"].append(row)
    plan = propose_diagnostic_study(eq, train, trial)
    assert plan["screening_refusal"] == "R95_SOURCE_FIT_BUDGET_EXCEEDS_HARD_RESEARCH_CEILING"
    assert plan["folds"] == []
    assert plan["source_oof_fit_calls_planned"] == 0


def test_default_study_never_requests_unimplemented_neural_or_fuzzy_provider():
    eq, train, trial, _ = data("bearing")
    trial["candidate_families"] = ["cnn1d", "tcn", "fuzzy_tsk"]
    plan = propose_diagnostic_study(eq, train, trial)
    assert plan["candidate_count"] == 0
    assert plan["screening_refusal"] == "R95_NO_IMPLEMENTED_PROVIDER_FOR_TASK"
    assert plan["unimplemented_candidate_families"] == [
        "cnn1d", "tcn", "fuzzy_tsk"]


class RealSklearnProviderContract:
    """Actual sklearn models routed via the existing torch_like provider API.

    This does NOT claim import of private torch_like or native equipment data.
    """
    def __init__(self):
        self.used_real_algorithms = []

    def get_estimator_info(self, name):
        return {"kind": "classification"}

    def create_estimator(self, name, **options):
        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeClassifier
        native = {"LogisticRegression": LogisticRegression,
                  "DecisionTreeClassifier": DecisionTreeClassifier}[name]
        self.used_real_algorithms.append((name, options))
        return native(**options)

    def as_runtime(self, estimator):
        class Adapter:
            def supports(self, action):
                return action in ("fit", "predict")
            def fit(self, x, y=None):
                return estimator.fit(x, y)
            def predict(self, x):
                return estimator.predict(x)
        return Adapter()


def test_actual_sklearn_algorithms_execute_grouped_train_only_design():
    pytest.importorskip("sklearn")
    eq, train, trial, features = data("bearing")
    trial["budget"]["max_trials"] = 2  # same full sensors for both families
    plan = propose_diagnostic_study(eq, train, trial)
    provider = RealSklearnProviderContract()
    result = execute_source_design_study(plan, eq, train, trial, features,
                                         provider=provider)
    assert result["status"] == "SOURCE_TRAIN_OOF_DESCRIPTIVE_SHORTLIST_ONLY"
    assert result["source_fit_calls_actual"] == 8
    assert {x[0] for x in provider.used_real_algorithms} == {
        "LogisticRegression", "DecisionTreeClassifier"}
    assert len(result["evaluated_candidates"]) == 2
    assert all(x["source_oof_record_accuracy"] in (0, .25, .5, .75, 1)
               for x in result["evaluated_candidates"])
    assert result["selected_design_not_independently_validated"] is True
