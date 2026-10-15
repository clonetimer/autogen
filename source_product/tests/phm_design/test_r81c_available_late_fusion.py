"""R81c physically observable feature availability; synthetic TRAIN-only tests."""
import copy
import json

import numpy as np
import pytest

from phm_design.industrial.artifact_io import load_trusted_artifact
from phm_design.industrial.availability_features import execute_available, atom_dimensions
from phm_design.industrial.design_engine import (
    DesignAcceptance, IndustrialDesignRejected, run_design,
)
from phm_design.industrial.taskspec import TaskSpecRejected, compile_task_spec
from test_r81a_unified_design_engine import task, synthetic_records


SCREEN = DesignAcceptance(accuracy_min=0, worst_environment_class_recall_min=0,
                          healthy_false_alarm_max=1)


def available_spec(composition="available_average"):
    spec = task("industrial_bearing_classification_demo.json", compare=("linear",))
    for name in ("vibration_valid", "current_valid"):
        spec["observations"]["channels"].append({
            "name": name, "role": "sensor", "unit": "1",
            "available_at_decision": True})
    spec["design"]["feature_programs"] = [{
        "atoms": ["temporal_bins", "random_kernels"],
        "composition": composition,
        "bindings": {
            "temporal_bins": {
                "signal": "vibration", "validity_flag": "vibration_valid"},
            "random_kernels": {
                "signal": "motor_current", "validity_flag": "current_valid"}
        }
    }]
    spec["design"]["budget"]["max_fits"] = 96
    return spec


def available_records(spec, n_env=4):
    records = synthetic_records(spec, environment_count=n_env)
    for i, rec in enumerate(records):
        # Controlled sensor-validity patterns balanced across fault classes;
        # no labels, world IDs or failure onsets are supplied as signals.
        # Every fault class stays supported in every source inner group fold;
        # one physical example per environment has a *genuine* missing branch.
        pattern = (1., 0.) if i % 6 == 1 else (1., 1.)
        for row in rec["measurements"]:
            row["vibration_valid"], row["current_valid"] = pattern
    return records


def test_compile_explicit_sensor_and_flag_bindings_without_class_metadata():
    s = available_spec()
    plan = compile_task_spec(s)
    grid = plan["candidate_grid"]
    assert len(grid) == 1
    assert grid[0]["composition"] == "available_average"
    assert len(grid[0]["availability_bindings"]) == 2
    assert plan["qualified"] is False
    assert {e["atom"] for e in grid[0]["availability_bindings"]} == {
        "random_kernels", "temporal_bins"}
    assert {e["validity_flag"] for e in grid[0]["availability_bindings"]} == {
        "vibration_valid", "current_valid"}


def test_masks_are_read_from_sensor_flags_and_absent_data_are_not_transformed():
    s = available_spec()
    r = available_records(s)
    rows = r[1]["measurements"]
    cols = ["sample_time_s", "vibration", "motor_current",
            "vibration_valid", "current_valid"]
    matrix = np.asarray([[row[c] for c in cols] for row in rows])
    plan = compile_task_spec(s)
    bs = tuple((item["atom"], item["sensor_index"], item["validity_index"])
               for item in plan["candidate_grid"][0]["availability_bindings"])
    out = execute_available(matrix, bs)
    assert out[-2:].tolist() == [0., 1.]
    assert out.size == atom_dimensions("random_kernels") + (
        atom_dimensions("temporal_bins")) + 2
    # The absent vibration may contain arbitrary finite placeholder values:
    # its feature block must remain identically zero.
    signal_index = cols.index("motor_current")
    matrix[:, signal_index] = 1e9
    np.testing.assert_allclose(out, execute_available(matrix, bs))


@pytest.mark.parametrize("mode", ["available_average", "available_stack"])
def test_truly_fitted_late_fusion_and_unknown_deployment_replay(mode, tmp_path):
    s = available_spec(mode)
    r = available_records(s)
    try:
        a = run_design(s, r, acceptance=SCREEN)
    except IndustrialDesignRejected as exc:
        pytest.fail("AVAILABILITY_TRIAL_REJECTED:" + repr(exc.audit.get("attempts")))
    assert a.trial.composition == mode
    assert a.audit["decision"] == "TRAIN_ONLY_SELECTION_REFITTED"
    assert a.audit["source_data_role"] == "TRAIN_ONLY"
    assert a.audit["fold_model_fits"] == 4 * 9  # 3 grouped base folds per atom + refit + meta
    assert a.audit["total_model_fits"] == 5 * 9
    assert 1 <= a.model.fit_count_ <= 9
    assert len(a.model.group_audit_) == 6
    assert all(fold["disjoint"] for fold in a.model.group_audit_)
    assert a.model.meta_training_audit_["new_physical_worlds"] == 0
    assert a.predict_decisions(r[0]["measurements"])[0]["label"] in s["task"]["labels"]
    # No available component: do not force a known fault label.
    missing = copy.deepcopy(r[0]["measurements"])
    for row in missing:
        row["vibration_valid"] = 0.
        row["current_valid"] = 0.
    decision = a.predict_decisions(missing)[0]
    assert decision == {"label": "__UNKNOWN__", "confidence": 0., "abstained": True}
    assert a.predict(missing) == ["__UNKNOWN__"]
    folder = tmp_path / mode
    a.export(folder)
    restored = load_trusted_artifact(folder, trust_pickle=True)
    assert restored.predict_decisions(missing) == [decision]
    assert restored.predict(r[1]["measurements"]) == a.predict(r[1]["measurements"])
    assert json.loads((folder / "audit.json").read_text())["independent_qualification"] is False


@pytest.mark.parametrize("invalid,code", [
    ("not_binary", "NONBINARY_AVAILABLE_SENSOR_FLAG"),
    ("switching", "CHANGING_AVAILABILITY_IN_TRAJECTORY_UNSUPPORTED"),
])
def test_observation_flags_must_be_binary_and_trajectory_constant(invalid, code):
    s = available_spec()
    r = available_records(s)
    r[0]["measurements"][0]["current_valid"] = (.5 if invalid == "not_binary" else 0.)
    with pytest.raises(ValueError, match=code):
        run_design(s, r, acceptance=SCREEN)


def test_no_final_data_and_no_unregistered_flags_reach_model():
    s = available_spec()
    r = available_records(s)
    r[0]["metadata"]["role"] = "FINAL_ID"
    with pytest.raises(TaskSpecRejected, match="DESIGN_WORKER_PROTECTED_ROLE"):
        run_design(s, r, acceptance=SCREEN)


@pytest.mark.parametrize("mutate,code", [
    (lambda s: s["design"]["feature_programs"][0].update(composition="available_oracle"),
     "UNSUPPORTED_TASKSPEC_VALUE"),
    (lambda s: s["design"]["feature_programs"][0]["bindings"]["temporal_bins"].update(validity_flag="label"),
     "INVALID_AVAILABILITY_CHANNEL_BINDING"),
    (lambda s: s["design"]["feature_programs"][0]["bindings"]["random_kernels"].update(validity_flag="vibration_valid"),
     "INVALID_AVAILABILITY_CHANNEL_BINDING"),
    (lambda s: s["observations"]["channels"][-1].update(unit="V"),
     "INVALID_AVAILABILITY_FLAG_CHANNEL_CONTRACT"),
    (lambda s: s["design"].update(compare=["tree"]),
     "AVAILABILITY_REQUIRES_LINEAR_DIRECT_ERM"),
    (lambda s: s["task"].update(allow_unknown=False),
     "AVAILABILITY_REQUIRES_UNKNOWN_TASK_CONTRACT"),
    (lambda s: s["design"].update(calibration={
        "method": "temperature_selective", "max_selective_error": .2,
        "minimum_coverage": .5, "min_calibration_samples": 12}),
     "UNSUPPORTED_AVAILABILITY_CALIBRATION_OR_BRANCH_COUNT"),
])
def test_invalid_or_truth_derived_availability_contracts_fail_closed(mutate, code):
    s = available_spec()
    mutate(s)
    with pytest.raises(TaskSpecRejected, match=code):
        compile_task_spec(s)


def test_insufficient_source_group_oof_class_support_refuses_delivery():
    s = available_spec("available_stack")
    records = available_records(s)
    for rec in records:
        if rec["metadata"]["label"] == s["task"]["labels"][-1]:
            for row in rec["measurements"]:
                row["vibration_valid"] = 0.
                row["current_valid"] = 0.
    with pytest.raises(IndustrialDesignRejected) as exc:
        run_design(s, records, acceptance=SCREEN)
    audit = exc.value.audit
    assert audit["model_delivered"] is False
    assert audit["decision"] == "NO_ELIGIBLE_DESIGN"
    assert "AVAILABILITY_INSUFFICIENT_OOF_SUPPORT" in (
        audit["attempts"][0]["failure"])
