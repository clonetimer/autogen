"""R83e physical observability QA tests — SYNTHETIC traces, not native physics."""
from __future__ import annotations

import pytest

from phm_design.io import read_json, write_json
from phm_design.industrial.r83e_native_paired_observability import (
    summarize_native_paired_observability, SCHEMA,
)
from test_r82_native_campaign_gate import campaign_fixture


def _fixture(tmp_path):
    src = campaign_fixture(tmp_path / "source", splits=("train", "validation"))
    fault_task = src / "cases/train-rw-fault/task_spec.json"
    d = read_json(fault_task)
    d["modifiers"] = {"faults": [{
        "fault_type": "adcs_rw_jamming",
        "parameters": {"wheel_index": 0},
        "onset_time_s": 8.0,
    }], "degradations": [], "constraints": []}
    write_json(fault_task, d)
    return src, fault_task


def test_paired_native_claim_audit_has_no_model_or_val_access(tmp_path):
    src, _ = _fixture(tmp_path)
    # No validation telemetry should be opened, even in trusted physics QA.
    (src / "cases/validation-rw-fault/trace.csv").unlink()
    (src / "cases/validation-rw-nominal/trace.csv").unlink()
    destination = tmp_path / "paired_audit.json"
    audit = summarize_native_paired_observability(src, destination)
    assert audit["schema"] == SCHEMA
    assert audit["paired_cases"] == 1
    assert audit["source_role"] == "TRUSTED_NATIVE_TRAIN_SOURCE_PHYSICS_QA_ONLY"
    assert audit["actual_basilisk_executed_here"] is False
    assert audit["truth_available_to_model_inference"] is False
    assert audit["test_final_or_validation_traces_accessed"] is False
    assert audit["model_trained_or_selected"] is False
    assert audit["independent_qualification"] is False

    row = audit["per_pair"][0]
    assert row["fault_class"] == "ADCS_RW_JAM"
    assert row["physical_wheel_index_audit_only"] == 0
    assert row["fault_onset_s_audit_only"] == 8.0
    assert row["samples_before_onset"] == 8
    assert row["samples_at_or_after_onset"] == 12
    # Synthetic fixture happens to offset every fault speed by 7 rad/s.
    # This verifies honest pre-onset reporting; the auditor must not hide
    # violations by asserting a falsely ideal twin match.
    assert row["max_pre_onset_abs_speed_delta_rad_s"] == pytest.approx(7.0)
    assert row["max_post_onset_abs_speed_delta_rad_s"] == pytest.approx(7.0)
    assert row["post_onset_speed_rms_delta_rad_s"] == pytest.approx(7.0)
    assert read_json(destination) == audit
    with pytest.raises(FileExistsError, match="REPORT_ALREADY_EXISTS"):
        summarize_native_paired_observability(src, destination)


def test_invalid_fault_event_or_native_source_refused_without_result(tmp_path):
    src, fault_path = _fixture(tmp_path)
    doc = read_json(fault_path)
    doc["modifiers"]["faults"][0]["parameters"]["wheel_index"] = 4
    write_json(fault_path, doc)
    output = tmp_path / "should_not_exist.json"
    with pytest.raises(ValueError, match="ONSET_OR_TARGET_INVALID"):
        summarize_native_paired_observability(src, output)
    assert not output.exists()

    doc["modifiers"]["faults"][0]["parameters"]["wheel_index"] = 0
    write_json(fault_path, doc)
    contract = src / "cases/train-rw-fault/astrograph/dataset_contract.json"
    body = read_json(contract)
    body["simulation"]["backend"] = "python"
    write_json(contract, body)
    with pytest.raises(ValueError, match="R82_NATIVE_RUNTIME_CANDIDATE_EVIDENCE_REQUIRED"):
        summarize_native_paired_observability(src, output)
    assert not output.exists()
