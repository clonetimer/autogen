"""R83o synthetic metadata-only adversarial tests (NOT real Basilisk evidence)."""
from __future__ import annotations

import copy
import hashlib
import json

import pytest

from phm_design.industrial import r83o_sensor_cohort_gate as gate


def sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def fixture():
    sensor = {
        "schema": "phm.r83o.sensor.v1",
        "sensor_id": "rw_encoder_synthetic_example",
        "signal_name": "rw_speed_rad_s", "unit": "rad/s",
        "provenance": "VALIDATED_SIM_SENSOR",  # asserted, never independently verified
        "evidence_sha256": sha("synthetic evidence NOT genuine calibration"),
        "available_onboard": True,
        "quantization_step_rad_s": 0.0001,
        "max_abs_error_rad_s": 0.001,
        "sampling_hz": 5.0, "max_timestamp_jitter_s": 0.0002,
        "telemetry_allowlist": ["rw_speed_rad_s", "fsw_reference_mrp", "wheel_current_a"],
    }
    cohorts = {"schema": "phm.r83o.cohort.v1", "records": []}
    for split in ("TRAIN", "CAL", "VAL", "TEST", "FINAL"):
        for kind in ("HEALTHY", "FAULTED"):
            record_id = split.lower() + "_" + kind.lower()
            cohorts["records"].append({
                "record_id": record_id,
                "split": split,
                "campaign_id": split.lower() + "-separate-campaign",
                "source_group_id": split.lower() + "-paired-base-physics",
                "pair_id": split.lower() + "-pair",
                "run_kind": kind,
                "raw_case_sha256": sha(record_id + "-raw"),
                "observation_sha256": sha(record_id + "-obs"),
            })
    features = {"feature_fields": ["rw_speed_rad_s", "fsw_reference_mrp"]}
    return sensor, cohorts, features


def test_happy_path_is_only_metadata_preflight_not_diagnostic_qualification():
    sensor, cohorts, features = fixture()
    receipt = gate.preflight(sensor, cohorts, features)
    assert receipt["status"] == "METADATA_PRECHECK_ONLY"
    assert receipt["cohorts"]["records"] == 10
    assert receipt["cohorts"]["pair_groups"] == 5
    assert receipt["cohorts"]["missing_splits"] == []
    assert not receipt["eligible_for_stable"]
    assert not receipt["trained_diagnostic_model"]
    assert not receipt["blind_test_executed"]
    assert not receipt["verified_hardware_sensor"]
    assert not receipt["qualified_diagnostic_claim"]
    assert receipt["independent_steward_review_required"]
    assert not receipt["sensor"]["evidence_independently_verified"]
    assert receipt["input_manifest_sha256"] == gate.preflight(
        copy.deepcopy(sensor), copy.deepcopy(cohorts), copy.deepcopy(features)
    )["input_manifest_sha256"]


@pytest.mark.parametrize("name", [
    "fault_label", "fault_onset_s", "injected_fault_code",
    "oracle_rw_speed_rad_s", "ground_truth", "rw_speed_true_rad_s",
    "healthy_twin_rw_speed_rad_s", "rw_u_max", "rw_u_current",
    "rw_applied_torque", "original_case_id",
])
def test_reject_oracle_fields_even_if_the_allowlist_is_poisoned(name):
    sensor, cohorts, features = fixture()
    sensor["telemetry_allowlist"].append(name)
    features["feature_fields"].append(name)
    with pytest.raises(gate.AdmissionError, match="ORACLE_OR_FAULT_TRUTH_FIELD"):
        gate.preflight(sensor, cohorts, features)



@pytest.mark.parametrize("name", [
    "split", "split_name", "campaign_id", "source_group_id", "pair_id",
    "record_id", "run_kind", "raw_case_sha256", "observation_sha256",
    "world_id", "generation_family_id", "source_case_id", "seed",
])
def test_reject_metadata_and_split_keys_even_if_allowlisted(name):
    sensor, cohorts, features = fixture()
    sensor["telemetry_allowlist"].append(name)
    features["feature_fields"].append(name)
    with pytest.raises(gate.AdmissionError, match="PROTECTED_METADATA_FIELD"):
        gate.preflight(sensor, cohorts, features)

def test_refuse_feature_not_approved_by_actual_sensor_allowlist():
    sensor, cohorts, features = fixture()
    features["feature_fields"] = ["rw_speed_rad_s", "hidden_private_register"]
    with pytest.raises(gate.AdmissionError, match="FEATURE_NOT_ON_SENSOR_ALLOWLIST"):
        gate.preflight(sensor, cohorts, features)


@pytest.mark.parametrize("key", [
    "campaign_id", "source_group_id", "pair_id",
    "raw_case_sha256", "observation_sha256"
])
def test_cross_split_identical_lineage_is_rejected(key):
    sensor, cohorts, features = fixture()
    train = cohorts["records"][0]
    cal = cohorts["records"][2]
    cal[key] = train[key]
    with pytest.raises(gate.AdmissionError, match="CROSS_SPLIT_|DUPLICATE_RAW_CASE_SHA"):
        gate.preflight(sensor, cohorts, features)


def test_reject_overlapping_healthy_fault_pairs_with_same_split():
    sensor, cohorts, features = fixture()
    cohorts["records"][1]["pair_id"] = "different-unpaired-pair"
    with pytest.raises(gate.AdmissionError, match="INCOMPLETE_OR_OVERSIZED_PAIR"):
        gate.preflight(sensor, cohorts, features)


def test_reject_copied_group_pretending_independence():
    sensor, cohorts, features = fixture()
    cohorts["records"][2]["source_group_id"] = cohorts["records"][0]["source_group_id"]
    cohorts["records"][2]["record_id"] = "new_name_to_fake_independence"
    cohorts["records"][2]["raw_case_sha256"] = sha("new-name-synthetic")
    with pytest.raises(gate.AdmissionError, match="CROSS_SPLIT_SOURCE_GROUP_ID"):
        gate.preflight(sensor, cohorts, features)


def test_pair_must_link_same_campaign_and_physics_group():
    sensor, cohorts, features = fixture()
    cohorts["records"][1]["source_group_id"] = "other-physics"
    with pytest.raises(gate.AdmissionError, match="PAIRED_PHYSICS_GROUP_MISMATCH"):
        gate.preflight(sensor, cohorts, features)


def test_reject_double_use_of_one_raw_case_even_within_train():
    sensor, cohorts, features = fixture()
    cohorts["records"][1]["raw_case_sha256"] = cohorts["records"][0]["raw_case_sha256"]
    with pytest.raises(gate.AdmissionError, match="DUPLICATE_RAW_CASE_SHA"):
        gate.preflight(sensor, cohorts, features)


def test_partial_research_dataset_cannot_pass_completeness_gate():
    sensor, cohorts, features = fixture()
    cohorts["records"] = [r for r in cohorts["records"] if r["split"] not in {"TEST", "FINAL"}]
    receipt = gate.preflight(sensor, cohorts, features)
    assert receipt["status"] == "BLOCKED"
    assert receipt["cohorts"]["missing_splits"] == ["FINAL", "TEST"]
    assert "MISSING_REQUIRED_SPLITS" in receipt["blockers"]


def test_unknown_encoder_precision_is_not_assumed_zero():
    sensor, cohorts, features = fixture()
    sensor["provenance"] = "NATIVE_TRUTH_ONLY"
    sensor["evidence_sha256"] = None
    sensor["max_abs_error_rad_s"] = None
    sensor["quantization_step_rad_s"] = None
    receipt = gate.preflight(sensor, cohorts, features)
    assert receipt["status"] == "BLOCKED"
    assert "UNVERIFIED_SENSOR_PROVENANCE" in receipt["blockers"]
    assert "INCOMPLETE_SENSOR_ERROR_AND_SAMPLING_CONTRACT" in receipt["blockers"]
    assert not receipt["qualified_diagnostic_claim"]


@pytest.mark.parametrize("field,value,reason", [
    ("quantization_step_rad_s", 0, "INVALID_QUANTIZATION"),
    ("max_abs_error_rad_s", -0.1, "INVALID_ERROR_BOUND"),
    ("sampling_hz", True, "INVALID_SAMPLING_HZ"),
    ("max_timestamp_jitter_s", float("nan"), "INVALID_TIMESTAMP_JITTER"),
])
def test_invalid_physical_sensor_assumptions_refused(field, value, reason):
    sensor, cohorts, features = fixture()
    sensor[field] = value
    with pytest.raises(gate.AdmissionError, match=reason):
        gate.preflight(sensor, cohorts, features)


def test_unit_confusion_blocks_preflight():
    sensor, cohorts, features = fixture()
    sensor["unit"] = "rpm"
    with pytest.raises(gate.AdmissionError, match="SENSOR_UNIT_MISMATCH"):
        gate.preflight(sensor, cohorts, features)


def test_cli_refuses_duplicate_json_keys_and_nan(tmp_path, capsys):
    sensor, cohorts, features = fixture()
    sensor_path = tmp_path / "sensor.json"
    cohort_path = tmp_path / "cohorts.json"
    features_path = tmp_path / "features.json"
    cohort_path.write_text(json.dumps(cohorts), encoding="utf-8")
    features_path.write_text(json.dumps(features), encoding="utf-8")
    sensor_path.write_text('{"schema":"a","schema":"b"}', encoding="utf-8")
    args = ["--sensor", str(sensor_path), "--cohorts", str(cohort_path),
            "--features", str(features_path)]
    assert gate.main(args) == 2
    assert "DUPLICATE_JSON_KEY" in capsys.readouterr().out
    sensor_path.write_text(json.dumps(sensor).replace("0.001", "NaN"), encoding="utf-8")
    assert gate.main(args) == 2
    assert "NON_FINITE_JSON_VALUE" in capsys.readouterr().out


def test_cli_blocked_returns_distinct_exit_code(tmp_path, capsys):
    sensor, cohorts, features = fixture()
    sensor["provenance"] = "UNKNOWN"
    files = []
    for name, data in (("sensor", sensor), ("cohorts", cohorts), ("features", features)):
        path = tmp_path / (name + ".json")
        path.write_text(json.dumps(data), encoding="utf-8")
        files.append(str(path))
    rc = gate.main(["--sensor", files[0], "--cohorts", files[1], "--features", files[2]])
    assert rc == 3
    assert '"status": "BLOCKED"' in capsys.readouterr().out
