"""R83c reviewed validation-only manifest interface (synthetic fixtures ONLY)."""
import copy

import pytest

from phm_design.io import read_json, write_json
from phm_design.industrial.r83_frozen_validation import (
    load_reviewed_validation_manifest, run_native_r83_validation_once,
)
from test_r82_rw_study import rw_fixture


def _args(tmp_path):
    setup = rw_fixture(tmp_path)
    dataset = read_json(setup["development_manifest"])
    # Trusted synthetic fixture preparation; NEVER evidence of native physics.
    for item in dataset["samples"]:
        item["split"] = "validation"
    write_json(setup["development_manifest"], dataset)
    return setup


def test_reviewed_validation_only_native_claim_and_hash_controls(tmp_path):
    setup = _args(tmp_path)
    spec = read_json(setup["task_spec"])
    records, receipt = load_reviewed_validation_manifest(
        spec, setup["development_manifest"], setup["physical_map"],
        runtime_binding=setup["runtime_binding"],
        native_source_checkout=setup["native_source_checkout"],
        basilisk_wheel_factory=setup["basilisk_wheel_factory"],
    )
    assert len(records) == 24
    assert all(row["metadata"]["role"] == "VALIDATION" for row in records)
    assert receipt["source_role"] == "VALIDATION_ONLY"
    assert receipt["actual_basilisk_physics_executed_by_reviewer"] is False
    assert receipt["independent_physical_source_attested"] is False
    assert receipt["validated_file_shas"] is True

    mixed = read_json(setup["development_manifest"])
    mixed["samples"][0]["split"] = "train"
    write_json(setup["development_manifest"], mixed)
    with pytest.raises(ValueError, match="DATA_ROLE_VIOLATION"):
        load_reviewed_validation_manifest(
            spec, setup["development_manifest"], setup["physical_map"],
            runtime_binding=setup["runtime_binding"],
            native_source_checkout=setup["native_source_checkout"],
            basilisk_wheel_factory=setup["basilisk_wheel_factory"])

    mixed["samples"][0]["split"] = "validation"
    write_json(setup["development_manifest"], mixed)
    telemetry = setup["development_manifest"].parent / mixed["samples"][0]["path"]
    with telemetry.open("a", encoding="utf-8") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="DATA_FILE_CHANGED"):
        load_reviewed_validation_manifest(
            spec, setup["development_manifest"], setup["physical_map"],
            runtime_binding=setup["runtime_binding"],
            native_source_checkout=setup["native_source_checkout"],
            basilisk_wheel_factory=setup["basilisk_wheel_factory"])


def test_native_validation_report_written_once_with_frozen_steward(monkeypatch, tmp_path):
    import phm_design.industrial.r83_frozen_validation as evaluator

    setup = _args(tmp_path)
    train_report_path = tmp_path / "train_research_report.json"
    write_json(train_report_path, {"comparison": {"schema": "STUB_ONLY"}})
    calls = []

    def fake_evaluator(report, spec, frozen, records, *, trusted_fixed_pickle=False):
        assert trusted_fixed_pickle is True
        assert len(records) == 24
        assert all(r["metadata"]["role"] == "VALIDATION" for r in records)
        calls.append(1)
        return {
            "schema": "phm.r83.frozen-source-independent-validation.v1",
            "model_fit_or_selection_performed": False,
            "independent_qualification": False,
        }

    monkeypatch.setattr(evaluator, "evaluate_r83_frozen_validation", fake_evaluator)
    output = tmp_path / "reviewed_val"
    result = run_native_r83_validation_once(
        train_research_report=train_report_path,
        task_spec=setup["task_spec"],
        frozen_models=tmp_path / "stub_models",
        validation_manifest=setup["development_manifest"],
        validation_physical_map=setup["physical_map"],
        runtime_binding=setup["runtime_binding"],
        native_source_checkout=setup["native_source_checkout"],
        basilisk_wheel_factory=setup["basilisk_wheel_factory"],
        output=output,
        trusted_fixed_pickle=True,
    )
    assert result["final_test_performed"] is False
    assert result["independent_qualification"] is False
    assert read_json(output / "validation_report.json") == result
    assert len(calls) == 1

    with pytest.raises(FileExistsError, match="REPORT_ALREADY_EXISTS"):
        run_native_r83_validation_once(
            train_research_report=train_report_path,
            task_spec=setup["task_spec"],
            frozen_models=tmp_path / "stub_models",
            validation_manifest=setup["development_manifest"],
            validation_physical_map=setup["physical_map"],
            runtime_binding=setup["runtime_binding"],
            native_source_checkout=setup["native_source_checkout"],
            basilisk_wheel_factory=setup["basilisk_wheel_factory"],
            output=output,
            trusted_fixed_pickle=True,
        )
    assert len(calls) == 1


def test_trusted_val_staging_never_reads_train_telemetry(tmp_path):
    from phm_design.industrial.r83_frozen_validation import (
        stage_reviewer_validation_only_view)
    from phm_design.io import sha_file
    from shutil import copyfile

    setup = rw_fixture(tmp_path)
    manifest = setup["development_manifest"]
    dataset = read_json(manifest)
    original = dataset["samples"][0]
    original_path = manifest.parent / original["path"]
    heldout_path = manifest.parent / "observations/staged_validation_case.csv"
    copyfile(original_path, heldout_path)
    lines = heldout_path.read_text(encoding="utf-8").splitlines()
    values = lines[1].split(",")
    values[1] = str(float(values[1]) + 0.129)
    lines[1] = ",".join(values)
    heldout_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    added = dict(original)
    added.update(
        id="validation_reviewer_case",
        group_id="separate_reviewed_physical_case",
        lineage_id="separate_reviewed_lineage",
        split="validation",
        path="observations/staged_validation_case.csv",
        sha256=sha_file(heldout_path),
    )
    dataset["samples"].append(added)
    write_json(manifest, dataset)
    # Modify TRAIN-only data deliberately. A trusted VAL staging process
    # must neither hash nor open these bytes.
    with original_path.open("a", encoding="utf-8") as stream:
        stream.write("train_tampered_only\n")
    output = tmp_path / "isolated_val"
    receipt = stage_reviewer_validation_only_view(manifest, output)
    assert receipt["validation_cases"] == 1
    assert receipt["train_telemetry_accessed"] is False
    assert receipt["hidden_test_or_final_accessed"] is False
    exported = read_json(output / "dataset.json")
    assert len(exported["samples"]) == 1
    assert exported["samples"][0]["split"] == "validation"
    assert exported["provenance"]["train_telemetry_accessed"] is False
    assert (output / added["path"]).is_file()
    assert not (output / original["path"]).is_file()

    dataset["samples"][0]["split"] = "test"
    write_json(manifest, dataset)
    with pytest.raises(ValueError, match="DATA_ROLE_VIOLATION"):
        stage_reviewer_validation_only_view(
            manifest, tmp_path / "must_not_exist")
    assert not (tmp_path / "must_not_exist").exists()
