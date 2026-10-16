"""End-to-end R82 satellite-RW data CONTRACT tests only.

All numeric telemetry and all native source files used here are synthetic
fixtures. Passing these tests is NOT Basilisk execution or physics evidence.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from phm_design.io import read_json, sha_file, write_json
from phm_design.industrial.r82_rw_study import run_rw_train_study
from phm_design.industrial.taskspec import compile_task_spec
from test_r81a_unified_design_engine import synthetic_records
from test_r82_reviewed_source_intake import native_code_fixture

EXAMPLE = (Path(__file__).resolve().parents[2] / "examples" /
           "industrial" / "satellite_rw_native_offline_train.json")


def rw_fixture(tmp_path):
    spec = json.loads(EXAMPLE.read_text())
    records = synthetic_records(spec, environment_count=3)
    root = tmp_path / "prepared" / "development"
    root.mkdir(parents=True)
    (root / "observations").mkdir()
    sample_descriptors = []
    physical_map = {}
    names = [c["name"] for c in spec["observations"]["channels"]]
    for i, record in enumerate(records):
        ident = f"synthetic_train_{i:03d}"
        group = hashlib.sha256(
            record["metadata"]["rw_run_lineage_id"].encode()).hexdigest()
        source = root / "observations" / (ident + ".csv")
        with source.open("w", encoding="utf-8", newline="") as out:
            writer = csv.DictWriter(out, fieldnames=["time_s", *names])
            writer.writeheader()
            writer.writerows(record["measurements"])
        sample_descriptors.append({
            "id": ident, "group_id": group, "lineage_id": group,
            "split": "train", "label": record["metadata"]["label"],
            "path": f"observations/{ident}.csv",
            "sha256": sha_file(source),
        })
        physical_map[ident] = {
            "rw_run_lineage_id": group,
            "operating_regime": record["metadata"]["operating_regime"],
        }
    contract = {
        "device_family": "satellite.adcs.reaction_wheel.fixed_three_wheels",
        "task": "fixed_length_case_classification",
        "channels": [
            {"name": c["name"], "unit": c["unit"],
             "role": "deployment_input",
             "provenance": "SYNTHETIC_INTERFACE_ONLY"} for c in spec["observations"]["channels"]],
        "classes": spec["task"]["labels"], "sample_s": 1/24,
        "evidence_type": "simulation",
        "scope": "SYNTHETIC_TEST_FIXTURE_NOT_BASILISK_EXECUTION",
    }
    data = {
        "schema": "phm.sequence-dataset.v1",
        "dataset_id": "R82_SYNTHETIC_RW_FIXTURE",
        "contract": contract,
        "samples": sample_descriptors,
        "provenance": {"source_kind": "native_basilisk",
                       "adapter": "SYNTHETIC_CONTRACT_TEST_ONLY"},
    }
    manifest = root / "dataset.json"
    write_json(manifest, data)
    task_path = tmp_path / "task.json"
    map_path = tmp_path / "physical_map.json"
    write_json(task_path, spec)
    write_json(map_path, {
        "schema": "phm.industrial.source-physical-map.v1",
        "records": physical_map})
    binding, checkout, wheel = native_code_fixture(tmp_path)
    return dict(
        task_spec=task_path, development_manifest=manifest,
        physical_map=map_path, runtime_binding=binding,
        native_source_checkout=checkout, basilisk_wheel_factory=wheel)


def test_six_channel_native_rw_task_spec_compiles_exact_observable_fields():
    spec = read_json(EXAMPLE)
    plan = compile_task_spec(spec)
    assert plan["budget"]["max_wall_seconds"] == 1800
    assert plan["train_data_contract"]["group_keys"] == ["rw_run_lineage_id"]
    assert plan["train_data_contract"]["environment_axes"] == ["operating_regime"]
    assert plan["ontology"]["labels"] == [
        "NOMINAL", "ADCS_RW_FRICTION_INCREASE", "ADCS_RW_JAM",
        "ADCS_RW_TORQUE_AUTHORITY_LOSS"]
    assert plan["independent_qualification"] is False


def test_synthetic_native_claim_must_pass_real_file_hashes_and_shas_as_groups(tmp_path):
    args = rw_fixture(tmp_path)
    report = run_rw_train_study(**args, output=tmp_path / "review",
                                validate_only=True)
    assert report["status"] == "TRAIN_INTAKE_ACCEPTED"
    assert report["intake"]["records"] == 24
    assert report["intake"]["physical_groups"] == 24
    assert report["intake"]["environments"] == 3
    assert report["actual_basilisk_execution_attested"] is False
    assert report["model_delivered"] is False
    assert not (tmp_path / "review" / "research_model").exists()
    assert read_json(tmp_path / "review" / "research_report.json") == report


def test_synthetic_six_channel_input_exercises_real_train_worker(tmp_path):
    args = rw_fixture(tmp_path)
    report = run_rw_train_study(**args, output=tmp_path / "research",
                                validate_only=False)
    assert report["status"] == "RESEARCH_TRAIN_FIT_ONLY"
    assert report["source_oof_selection"]["independent_qualification"] is False
    assert report["model_delivered"] is True
    assert report["intake"]["source_kind_claimed"] == "native_basilisk"
    assert report["actual_basilisk_execution_attested"] is False
    assert (tmp_path / "research" / "research_model" / "model.joblib").is_file()
    assert (tmp_path / "research" / "research_model" / "audit.json").is_file()


def test_mismatched_native_wheel_source_refuses_before_any_model_training(tmp_path):
    args = rw_fixture(tmp_path)
    args["basilisk_wheel_factory"].write_text("BAD SOURCE")
    with pytest.raises(ValueError, match="R82_NATIVE_SOURCE_SHA256_MISMATCH"):
        run_rw_train_study(**args, output=tmp_path / "refused",
                           validate_only=False)
    receipt = read_json(tmp_path / "refused" / "research_report.json")
    assert receipt["status"] == "REFUSED"
    assert receipt["model_delivered"] is False
    assert not (tmp_path / "refused" / "research_model").exists()


def test_non_native_prepared_manifest_refuses_without_creating_study(tmp_path):
    args = rw_fixture(tmp_path)
    manifest = read_json(args["development_manifest"])
    manifest["provenance"]["source_kind"] = "software_synthetic"
    write_json(args["development_manifest"], manifest)
    with pytest.raises(ValueError, match="R82_REQUIRES_PREPARED_SIX_CHANNEL_NATIVE_RW"):
        run_rw_train_study(**args, output=tmp_path / "forbidden")
    assert not (tmp_path / "forbidden").exists()


def test_review_map_draft_cannot_be_submitted_as_approved_operating_regimes(tmp_path):
    from phm_design.industrial.source_intake import (
        draft_reviewed_physical_map, intake_train_source)
    args = rw_fixture(tmp_path)
    draft_path = tmp_path / "unreviewed_map.json"
    audit = draft_reviewed_physical_map(
        read_json(args["task_spec"]), args["development_manifest"], draft_path)
    assert audit["model_ready"] is False
    draft = read_json(draft_path)
    assert len(draft["records"]) == 24
    assert all(item["operating_regime"] == "REVIEW_REQUIRED"
               for item in draft["records"].values())
    with pytest.raises(ValueError, match="R82_PHYSICAL_MAP_FIELDS_INVALID"):
        intake_train_source(
            read_json(args["task_spec"]), args["development_manifest"], draft_path,
            native_binding_path=args["runtime_binding"],
            native_source_checkout_root=args["native_source_checkout"],
            native_wheel_factory_file=args["basilisk_wheel_factory"])


def test_rw_train_study_guards_against_existing_output(tmp_path):
    args = rw_fixture(tmp_path)
    output = tmp_path / "preexisting"
    output.mkdir()
    with pytest.raises(FileExistsError, match="R82_STUDY_OUTPUT_EXISTS"):
        run_rw_train_study(**args, output=output, validate_only=True)
