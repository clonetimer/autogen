"""R93 P1 cross-equipment contract tests (all metadata examples are synthetic).

Source design must never read protected VAL/CAL/FINAL records and cannot
treat raw recording windows or a time slice as independent machines.
"""
from __future__ import annotations
import copy
import json
from pathlib import Path

import pytest

from phm_design.industrial_contracts.contracts import (
    ContractRejected, compile_source_plan, validate_dataset, validate_equipment,
    validate_trial,
)
from phm_design.industrial_contracts.__main__ import main

EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "industrial_contracts"


def load(name):
    return json.loads((EXAMPLES / name).read_text(encoding="utf-8"))


def cards(family):
    return (
        load(f"{family}_equipment.json"),
        load(f"{family}_train_only.json"),
        load(f"{family}_trusted_audit.json"),
        load(f"{family}_trial.json"),
    )


@pytest.mark.parametrize("family,expected", [
    ("bearing", "unseen_physical_device"),
    ("compressor", "future_same_device"),
])
def test_two_different_machine_classes_share_exact_same_train_compiler(
        family, expected):
    equipment, train, complete, trial = cards(family)
    assert validate_equipment(equipment) == equipment
    assert validate_trial(trial, equipment) == trial
    source = compile_source_plan(equipment, train, trial)
    assert source["schema"] == "phm.autogen.source-plan.v1"
    assert source["status"] == "PLAN_ONLY_NO_MODEL_TRAINED"
    assert source["proposed_validation_scope"] == expected
    assert source["calibration_validation_final_read"] is False
    assert source["protected_target_fields_available_to_model"] is False
    assert source["model_qualified"] is False
    assert source["production_approved"] is False
    assert len(source["equipment_sha256"]) == len(source["trial_sha256"]) == 64
    trusted = validate_dataset(complete, equipment, trial)
    assert trusted["role_separation_checked"] is True
    assert trusted["independent_generalization_proven"] is False
    assert trusted["source_bytes_authenticated"] is False
    assert trusted["physical_identity_independently_authenticated"] is False
    assert trusted["eligible_for_stable"] is False
    assert "VALIDATION" in trusted["role_record_counts"]
    assert "role_record_counts" not in source
    assert not any(k in source for k in ("labels", "validation_labels", "test_labels"))


@pytest.mark.parametrize("family", ["bearing", "compressor"])
def test_design_compiler_forbids_any_protected_row(family):
    equipment, _, complete, trial = cards(family)
    with pytest.raises(ContractRejected,
                       match="R93_PROTECTED_ROLE_IN_DESIGN_WORKER"):
        compile_source_plan(equipment, complete, trial)
    bad = copy.deepcopy(complete)
    bad["records"][-1]["role"] = "FINAL"
    with pytest.raises(ContractRejected,
                       match="R93_PROTECTED_ROLE_IN_DESIGN_WORKER"):
        compile_source_plan(equipment, bad, trial)


def test_bearing_windows_across_train_and_validation_cannot_share_bearing():
    equipment, _, complete, trial = cards("bearing")
    data = copy.deepcopy(complete)
    data["records"][-1]["physical_id"] = data["records"][0]["physical_id"]
    with pytest.raises(ContractRejected,
                       match="R93_PHYSICAL_DEVICE_CROSSES_DATA_ROLES"):
        validate_dataset(data, equipment, trial)


def test_bearing_same_environment_cross_split_blocked_for_unseen_environment():
    equipment, _, complete, trial = cards("bearing")
    trial = copy.deepcopy(trial)
    trial["validation_target"] = "unseen_environment"
    data = copy.deepcopy(complete)
    data["records"][-1]["environment_id"] = data["records"][0]["environment_id"]
    with pytest.raises(ContractRejected, match="R93_ENVIRONMENT_CROSSES_ROLES"):
        validate_dataset(data, equipment, trial)


def test_acquisition_fingerprint_cannot_cross_train_val_even_same_compressor():
    equipment, _, complete, trial = cards("compressor")
    data = copy.deepcopy(complete)
    data["records"][-1]["source_sha256"] = data["records"][0]["source_sha256"]
    with pytest.raises(ContractRejected,
                       match="R93_SAME_RAW_SIGNAL_CROSSES_DATA_ROLES"):
        validate_dataset(data, equipment, trial)


def test_session_identity_cannot_cross_data_roles_without_relabeling_hash():
    equipment, _, complete, trial = cards("compressor")
    data = copy.deepcopy(complete)
    data["records"][-1]["session_id"] = data["records"][0]["session_id"]
    with pytest.raises(ContractRejected,
                       match="R93_ACQUISITION_SESSION_CROSSES_DATA_ROLES"):
        validate_dataset(data, equipment, trial)


def test_same_raw_signal_cannot_be_assigned_to_new_physical_asset():
    equipment, _, complete, trial = cards("bearing")
    data = copy.deepcopy(complete)
    data["records"][-1]["source_sha256"] = data["records"][0]["source_sha256"]
    with pytest.raises(ContractRejected,
                       match="R93_SAME_SIGNAL_BYTES_ASSIGNED_TO_DIFFERENT_DEVICES"):
        validate_dataset(data, equipment, trial)


def test_same_compressor_forward_split_requires_real_time_embargo():
    equipment, _, complete, trial = cards("compressor")
    data = copy.deepcopy(complete)
    data["records"][-1]["start_utc"] = "2026-01-03T00:00:01Z"
    with pytest.raises(ContractRejected,
                       match="R93_INSUFFICIENT_TEMPORAL_EMBARGO"):
        validate_dataset(data, equipment, trial)
    data = copy.deepcopy(complete)
    data["records"][-1]["start_utc"] = "2026-01-02T23:00:00Z"
    with pytest.raises(ContractRejected,
                       match="R93_TEMPORAL_RECORD_INTERVAL_OVERLAP"):
        validate_dataset(data, equipment, trial)


def test_compressor_no_rewinding_to_train_after_validation():
    equipment, _, complete, trial = cards("compressor")
    data = copy.deepcopy(complete)
    row = copy.deepcopy(data["records"][-1])
    row["record_id"] = "later_train"
    row["role"] = "TRAIN"
    row["label"] = "NORMAL"  # Avoid fault-label refusal before role-order guard.
    row["start_utc"] = "2026-01-04T05:00:00Z"
    row["end_utc"] = "2026-01-04T07:00:00Z"
    row["source_sha256"] = "4"*64
    row["session_id"] = "session_later_train"
    data["records"].append(row)
    with pytest.raises(ContractRejected, match="R93_TEMPORAL_DATA_ROLE_REWINDS"):
        validate_dataset(data, equipment, trial)


@pytest.mark.parametrize("field", ["fault_label", "ground_truth", "failure_time",
                                    "true_rul", "role", "physical_id"])
def test_ground_truth_or_identity_never_used_as_deployable_channel(field):
    equipment, _, _, trial = cards("bearing")
    equipment = copy.deepcopy(equipment)
    equipment["channels"][0]["name"] = field
    with pytest.raises(ContractRejected,
                       match="R93_PROTECTED_OBSERVATION_CHANNEL"):
        compile_source_plan(equipment, cards("bearing")[1], trial)


def test_non_deployment_sensor_cannot_enter_trial_inputs():
    equipment, train, _, trial = cards("bearing")
    equipment = copy.deepcopy(equipment)
    equipment["channels"][0]["available_at_decision"] = False
    with pytest.raises(ContractRejected,
                       match="R93_PROTECTED_OR_UNAVAILABLE_TRIAL_INPUT"):
        compile_source_plan(equipment, train, trial)


@pytest.mark.parametrize("bad", [
    "2026-01-01T00:00:00+09:00",
    "2026-01-01 00:00:00",
    "2026-01-01T00:00:00",
])
def test_non_utc_time_is_rejected(bad):
    equipment, _, complete, trial = cards("bearing")
    complete["records"][0]["start_utc"] = bad
    with pytest.raises(ContractRejected, match="R93_UTC_TIMESTAMP_REQUIRED"):
        validate_dataset(complete, equipment, trial)


def test_duplicate_record_ids_and_bad_origin_rejected():
    equipment, _, complete, trial = cards("bearing")
    complete["records"][0]["record_id"] = complete["records"][1]["record_id"]
    with pytest.raises(ContractRejected, match="R93_DUPLICATE_record_ids"):
        validate_dataset(complete, equipment, trial)
    equipment, train, _, trial = cards("bearing")
    train["origin"] = "unknown"
    with pytest.raises(ContractRejected, match="R93_SOURCE_ORIGIN_UNVERIFIED"):
        compile_source_plan(equipment, train, trial)


def test_anomaly_normal_only_train_and_nonphysical_forward_scope():
    equipment, train, complete, trial = cards("compressor")
    train["records"][0]["label"] = "ANOMALY"
    with pytest.raises(ContractRejected,
                       match="R93_ANOMALY_NORMAL_ONLY_TRAINING_VIOLATION"):
        compile_source_plan(equipment, train, trial)
    trial["validation_target"] = "unseen_physical_device"
    with pytest.raises(ContractRejected,
                       match="R93_SPLIT_CANNOT_SUPPORT_CLAIMED_VALIDATION_TARGET"):
        validate_dataset(complete, equipment, trial)


def test_compute_budget_is_typechecked_and_bounded():
    equipment, train, _, trial = cards("bearing")
    trial["budget"]["max_trials"] = True
    with pytest.raises(ContractRejected, match="R93_INVALID_POSITIVE_max_trials"):
        compile_source_plan(equipment, train, trial)
    equipment, train, _, trial = cards("bearing")
    trial["budget"]["max_trials"] = 2
    with pytest.raises(ContractRejected,
                       match="R93_MORE_CANDIDATES_THAN_TRIAL_BUDGET"):
        compile_source_plan(equipment, train, trial)


def test_operator_cli_refuses_inadvertent_protected_source_and_overwriting(
        capsys, tmp_path):
    common = ["--equipment", str(EXAMPLES/"bearing_equipment.json"),
              "--trial", str(EXAMPLES/"bearing_trial.json"),
              "--output", str(tmp_path/"plan.json")]
    assert main(["source-plan", *common, "--dataset",
                 str(EXAMPLES/"bearing_train_only.json")]) == 0
    source = json.loads((tmp_path/"plan.json").read_text(encoding="utf-8"))
    assert source["model_qualified"] is False
    with pytest.raises(SystemExit):
        main(["source-plan", *common, "--dataset",
              str(EXAMPLES/"bearing_trusted_audit.json")])
    assert main(["audit", "--equipment",
                 str(EXAMPLES/"compressor_equipment.json"), "--trial",
                 str(EXAMPLES/"compressor_trial.json"), "--dataset",
                 str(EXAMPLES/"compressor_trusted_audit.json"), "--output",
                 str(tmp_path/"audited.json")]) == 0
    audit = json.loads((tmp_path/"audited.json").read_text(encoding="utf-8"))
    assert audit["evaluation_scope"] == "future_same_device"
