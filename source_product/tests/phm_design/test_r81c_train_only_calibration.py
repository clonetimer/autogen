"""R81c synthetic TRAIN-only nested calibration tests (NOT qualification)."""
import copy
import json

import numpy as np
import pytest

from phm_design.industrial.calibration import (
    CalibrationRejected, fit_temperature_selective,
)
from phm_design.industrial.design_engine import (
    DesignAcceptance, IndustrialDesignRejected, run_design,
)
from phm_design.industrial.taskspec import TaskSpecRejected, compile_task_spec
from test_r81a_unified_design_engine import task, synthetic_records


FUNCTIONAL = DesignAcceptance(
    accuracy_min=0, worst_environment_class_recall_min=0,
    healthy_false_alarm_max=1,
)


def calibrated_spec():
    spec = task("industrial_bearing_classification_demo.json",
                compare=("linear",))
    spec["design"]["budget"]["max_fits"] = 60
    spec["design"]["calibration"] = {
        "method": "temperature_selective",
        "max_selective_error": 0.20,
        "minimum_coverage": 0.50,
        "min_calibration_samples": 12,
    }
    return spec


def test_calibration_policy_compiles_and_rejects_unsafe_or_missing_unknown():
    spec = calibrated_spec()
    plan = compile_task_spec(spec)
    assert plan["calibration_policy"] == spec["design"]["calibration"]
    assert plan["ontology"]["unknown_label"] == "__UNKNOWN__"
    invalid = copy.deepcopy(spec)
    invalid["task"]["allow_unknown"] = False
    with pytest.raises(TaskSpecRejected, match="CALIBRATION_REQUIRES_UNKNOWN"):
        compile_task_spec(invalid)
    invalid = copy.deepcopy(spec)
    invalid["design"]["calibration"]["minimum_coverage"] = -0.2
    with pytest.raises(TaskSpecRejected, match="INVALID_CALIBRATION_POLICY"):
        compile_task_spec(invalid)
    invalid = copy.deepcopy(spec)
    invalid["design"]["calibration"]["method"] = "oracle_final_tuning"
    with pytest.raises(TaskSpecRejected, match="UNSUPPORTED_TASKSPEC_VALUE"):
        compile_task_spec(invalid)


def test_calibration_refuses_unknown_confidence_without_forcing_known_class():
    truth = np.arange(30, dtype=int) % 3
    probabilities = np.full((30, 3), .01, dtype=float)
    probabilities[np.arange(30), truth] = .98
    policy = calibrated_spec()["design"]["calibration"]
    fitted = fit_temperature_selective(probabilities, truth, policy)
    assert fitted.calibration_samples == 30
    assert fitted.calibration_coverage == 1.
    assert fitted.decide(np.array([[1 / 3, 1 / 3, 1 / 3]])).tolist() == [-1]
    assert fitted.decide(np.array([[.98, .01, .01]])).tolist() == [0]
    with pytest.raises(CalibrationRejected, match="CALIBRATION_SAMPLE_SUPPORT"):
        fit_temperature_selective(probabilities[:10], truth[:10], policy)


def test_real_inner_oof_fit_count_and_calibrated_model_export(tmp_path):
    spec = calibrated_spec()
    records = synthetic_records(spec, environment_count=4)
    model = run_design(spec, records, acceptance=FUNCTIONAL)
    assert model.calibrator is not None
    assert model.audit["calibration_role"] == "TRAIN_INNER_OOF_ONLY"
    assert model.audit["decision"] == "TRAIN_ONLY_SELECTION_REFITTED"
    assert model.audit["total_model_fits"] == 21  # outer 4*(inner 3 + 1) + final (inner 4+1)
    assert model.audit["fold_model_fits"] == 16
    assert model.audit["attempts"][0]["model_fits"] == 16
    assert model.audit["final_refit_model_fits"] == 5
    assert model.audit["compute_ledger"]["flops_parity_claimed"] is False
    assert model.audit["attempts"][0]["wall_seconds"] >= 0
    assert model.audit["attempts"][0]["inference_wall_seconds"] >= 0
    trial = model.audit["attempts"][0]
    assert trial["metrics"]["coverage"] >= spec["design"]["calibration"]["minimum_coverage"]
    assert all(f["inner_calibration"]["calibration_samples"] == 18
               for f in trial["folds"])
    decisions = model.predict_decisions(records[0]["measurements"])
    assert len(decisions) == 1
    assert decisions[0]["label"] in spec["task"]["labels"] + ["__UNKNOWN__"]
    assert isinstance(decisions[0]["abstained"], bool)
    assert 0 <= decisions[0]["confidence"] <= 1
    assert model.predict(records[0]["measurements"]) == [decisions[0]["label"]]
    model.export(tmp_path / "calibrated")
    audit = json.loads((tmp_path / "calibrated" / "audit.json").read_text())
    assert audit["calibration_state"]["temperature"] == model.calibrator.temperature
    assert audit["independent_qualification"] is False


def test_calibration_fails_closed_for_missing_inner_worlds():
    spec = calibrated_spec()
    with pytest.raises(IndustrialDesignRejected) as caught:
        run_design(spec, synthetic_records(spec, environment_count=3),
                   acceptance=FUNCTIONAL)
    assert caught.value.audit["decision"] == "NO_ELIGIBLE_DESIGN"
    assert "INSUFFICIENT_INNER_CALIBRATION_ENVIRONMENTS" in (
        caught.value.audit["attempts"][0]["failure"])


def test_calibration_fit_budget_counts_inner_models_and_final_calibration():
    spec = calibrated_spec()
    spec["design"]["budget"]["max_fits"] = 20
    with pytest.raises(IndustrialDesignRejected) as caught:
        run_design(spec, synthetic_records(spec, environment_count=4),
                   acceptance=FUNCTIONAL)
    assert caught.value.audit["decision"] == "BUDGET_EXHAUSTED_BEFORE_FINAL_REFIT"
    assert caught.value.audit["fold_model_fits"] == 16
    assert caught.value.audit["model_delivered"] is False


def test_unachievable_source_train_selective_risk_rejects_without_delivery():
    spec = calibrated_spec()
    spec["design"]["calibration"]["max_selective_error"] = .01
    with pytest.raises(IndustrialDesignRejected) as caught:
        run_design(spec, synthetic_records(spec, environment_count=4),
                   acceptance=FUNCTIONAL)
    assert caught.value.audit["decision"] == "NO_ELIGIBLE_DESIGN"
    assert "CALIBRATION_RISK_OR_COVERAGE_UNMET" in (
        caught.value.audit["attempts"][0]["failure"])


def test_final_role_and_ground_truth_are_rejected_before_calibration():
    spec = calibrated_spec()
    records = synthetic_records(spec, environment_count=4)
    records[0]["metadata"]["role"] = "FINAL_ID"
    with pytest.raises(TaskSpecRejected, match="DESIGN_WORKER_PROTECTED_ROLE"):
        run_design(spec, records, acceptance=FUNCTIONAL)
    records = synthetic_records(spec, environment_count=4)
    records[0]["measurements"][0]["ground_truth"] = 1
    with pytest.raises(TaskSpecRejected, match="UNDECLARED_OR_PROTECTED_OBSERVATION"):
        run_design(spec, records, acceptance=FUNCTIONAL)
