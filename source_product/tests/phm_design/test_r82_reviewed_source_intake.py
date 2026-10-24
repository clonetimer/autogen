"""R82 intake-contract tests. All constructed data are SYNTHETIC FIXTURES,
not physical Basilisk execution or real bearing experiment evidence.
"""
import copy
import csv
from pathlib import Path

import pytest

from phm_design.io import sha_file, write_json, read_json
from phm_design.industrial.source_intake import intake_train_source, stage_train_only_view
from phm_design.industrial.design_engine import DesignAcceptance, run_design
from test_r81a_unified_design_engine import task, synthetic_records


def make_fixture(tmp_path, *, source_kind="real_equipment_unreviewed"):
    spec = task("industrial_bearing_classification_demo.json", compare=("linear",))
    rows = synthetic_records(spec, environment_count=3)
    channels = spec["observations"]["channels"]
    time_field = spec["observations"]["time"]["field"]
    group_key = spec["data"]["group_keys"][0]
    env_keys = spec["data"]["environment_axes"]
    root = tmp_path / "dataset"
    root.mkdir()
    (root / "observations").mkdir()
    records, map_records = [], {}
    for idx, record in enumerate(rows):
        ident = f"fixture-{idx:03d}"
        source = root / "observations" / (ident + ".csv")
        fieldnames = ["time_s", *(c["name"] for c in channels)]
        with source.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            for point in record["measurements"]:
                writer.writerow({"time_s": point[time_field],
                                 **{c["name"]: point[c["name"]] for c in channels}})
        meta = record["metadata"]
        records.append({"id": ident, "group_id": meta[group_key],
                        "lineage_id": meta[group_key], "split": "train",
                        "label": meta["label"],
                        "path": "observations/" + ident + ".csv",
                        "sha256": sha_file(source)})
        map_records[ident] = {group_key: meta[group_key],
                              **{key: meta[key] for key in env_keys}}
    evidence = ("real_unreviewed" if source_kind == "real_equipment_unreviewed"
                else "simulation")
    manifest = {"schema": "phm.sequence-dataset.v1",
                "dataset_id": "SYNTHETIC_R82_INTAKE_FIXTURE",
                "contract": {"device_family": "fixture.bearing",
                             "task": "fixed_length_case_classification",
                             "channels": [
                                 {"name": c["name"], "unit": c["unit"],
                                  "role": "deployment_input",
                                  "provenance": "SYNTHETIC_TEST_ONLY"}
                                 for c in channels],
                             "classes": spec["task"]["labels"],
                             "sample_s": 1 / 24,
                             "evidence_type": evidence,
                             "scope": "SYNTHETIC_CONTRACT_TEST_NO_PHYSICAL_PROOF"},
                "samples": records,
                "provenance": {"source_kind": source_kind,
                               "adapter": "synthetic_contract_test_only"}}
    manifest_path = root / "dataset.json"
    map_path = tmp_path / "physical_map.json"
    write_json(manifest_path, manifest)
    write_json(map_path, {"schema": "phm.industrial.source-physical-map.v1",
                          "records": map_records})
    return spec, manifest_path, map_path


def test_train_only_physical_mapping_and_raw_telemetry_to_unified_design(tmp_path):
    spec, source, mapping = make_fixture(tmp_path)
    records, audit = intake_train_source(spec, source, mapping)
    assert len(records) == 18
    assert audit["role"] == "TRAIN_ONLY"
    assert audit["environments"] == 3
    assert audit["physical_groups"] == 18
    assert audit["physical_source_independently_attested"] is False
    assert audit["independent_qualification"] is False
    assert audit["model_trained"] is False
    assert all(set(row) == {"measurements", "metadata"} for row in records)
    assert all("role" not in sample["measurements"][0] and
               "label" not in sample["measurements"][0] for sample in records)
    model = run_design(spec, records, acceptance=DesignAcceptance(
        accuracy_min=0, worst_environment_class_recall_min=0,
        healthy_false_alarm_max=1))
    assert model.audit["decision"] == "TRAIN_ONLY_SELECTION_REFITTED"
    assert model.audit["source_data_role"] == "TRAIN_ONLY"
    assert model.audit["final_evaluation_performed"] is False


def test_rejects_mixed_role_manifest_before_reading_any_file(tmp_path):
    spec, source, mapping = make_fixture(tmp_path)
    raw = read_json(source)
    raw["samples"][0]["split"] = "test"
    write_json(source, raw)
    with pytest.raises(ValueError, match="R82_PROTECTED_OR_MIXED_SOURCE_MANIFEST"):
        intake_train_source(spec, source, mapping)


def test_rejects_unproven_native_source_without_binding(tmp_path):
    spec, source, mapping = make_fixture(tmp_path, source_kind="native_basilisk")
    with pytest.raises(ValueError, match="R82_NATIVE_RUNTIME_BINDING_REQUIRED"):
        intake_train_source(spec, source, mapping)


def native_code_fixture(tmp_path):
    """Synthetic code files are for hash-path tests only, never physics."""
    root = tmp_path / "code"
    paths = {
        "runtime_source": "src/sat_sim/bsk_engine/unified_native.py",
        "fault_effect_source": "src/sat_sim/bsk_engine/project_native_modules.py",
        "wheel_factory_source": "Basilisk/utilities/simIncludeRW.py",
    }
    source_claims = {}
    for key, relative in paths.items():
        file = root / relative
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("# CONTRACT FIXTURE ONLY; NO BASILISK EXECUTION\\n")
        source_claims[key] = {"reference": relative, "sha256": sha_file(file)}
    binding_path = tmp_path / "runtime_binding.json"
    write_json(binding_path, {
        "scope": "simulation_source_binding_only",
        **source_claims,
    })
    return binding_path, root, root / paths["wheel_factory_source"]


def test_native_source_file_hashes_checked_without_claiming_real_execution(tmp_path):
    spec, source, mapping = make_fixture(tmp_path, source_kind="native_basilisk")
    binding, root, wheel = native_code_fixture(tmp_path)
    with pytest.raises(ValueError, match="R82_NATIVE_SOURCE_FILES_REQUIRED"):
        intake_train_source(spec, source, mapping, native_binding_path=binding)
    _, audit = intake_train_source(
        spec, source, mapping, native_binding_path=binding,
        native_source_checkout_root=root, native_wheel_factory_file=wheel)
    assert audit["source_kind_claimed"] == "native_basilisk"
    assert audit["physical_source_independently_attested"] is False
    assert audit["native_binding_sha256"] == sha_file(binding)
    assert len(audit["native_source_file_hashes_verified"]) == 3


@pytest.mark.parametrize("key", ["runtime_source", "fault_effect_source", "wheel_factory_source"])
def test_native_source_hash_mismatch_fails_closed(tmp_path, key):
    spec, source, mapping = make_fixture(tmp_path, source_kind="native_basilisk")
    binding, root, wheel = native_code_fixture(tmp_path)
    file = (wheel if key == "wheel_factory_source" else
            root / read_json(binding)[key]["reference"])
    file.write_text("tampered source code")
    with pytest.raises(ValueError, match="R82_NATIVE_SOURCE_SHA256_MISMATCH"):
        intake_train_source(
            spec, source, mapping, native_binding_path=binding,
            native_source_checkout_root=root, native_wheel_factory_file=wheel)


def test_native_binding_rejects_unreviewed_source_path(tmp_path):
    spec, source, mapping = make_fixture(tmp_path, source_kind="native_basilisk")
    binding, root, wheel = native_code_fixture(tmp_path)
    doc = read_json(binding)
    doc["runtime_source"]["reference"] = "../native_outside.py"
    write_json(binding, doc)
    with pytest.raises(ValueError, match="R82_UNEXPECTED_NATIVE_SOURCE_REFERENCE"):
        intake_train_source(
            spec, source, mapping, native_binding_path=binding,
            native_source_checkout_root=root, native_wheel_factory_file=wheel)


def test_source_file_hash_mismatch_rejected(tmp_path):
    spec, source, mapping = make_fixture(tmp_path)
    sample = read_json(source)["samples"][0]
    with (source.parent / sample["path"]).open("a", encoding="utf-8") as stream:
        stream.write("0,0,0\n")
    with pytest.raises(ValueError, match="DATA_FILE_CHANGED"):
        intake_train_source(spec, source, mapping)


def test_physical_group_map_must_match_source_lineage(tmp_path):
    spec, source, mapping = make_fixture(tmp_path)
    raw = read_json(mapping)
    group = spec["data"]["group_keys"][0]
    first = next(iter(raw["records"]))
    raw["records"][first][group] = "invented_group"
    write_json(mapping, raw)
    with pytest.raises(ValueError, match="R82_GROUP_LINEAGE_MISMATCH"):
        intake_train_source(spec, source, mapping)


def test_unexpected_or_truth_column_is_rejected(tmp_path):
    spec, source, mapping = make_fixture(tmp_path)
    raw = read_json(source)
    sample = raw["samples"][0]
    path = source.parent / sample["path"]
    # Rehash intentionally so failure proves the *input column contract*.
    with path.open("r", newline="", encoding="utf-8") as stream:
        data = list(csv.DictReader(stream))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=[*data[0], "ground_truth"])
        writer.writeheader()
        for row in data:
            writer.writerow({**row, "ground_truth": "1"})
    sample["sha256"] = sha_file(path)
    write_json(source, raw)
    with pytest.raises(ValueError, match="R82_RAW_TELEMETRY_COLUMNS_INVALID"):
        intake_train_source(spec, source, mapping)


def test_software_synthetic_source_kind_refused_even_with_valid_files(tmp_path):
    spec, source, mapping = make_fixture(tmp_path, source_kind="fixture_software_synthetic")
    with pytest.raises(ValueError, match="R82_UNSUPPORTED_OR_UNVERIFIED_SOURCE_KIND"):
        intake_train_source(spec, source, mapping)


def test_trusted_staging_copies_train_only_without_reading_validation_trace(tmp_path):
    spec, source, mapping = make_fixture(tmp_path)
    raw = read_json(source)
    raw["samples"].append({
        "id": "reserved-val-001", "group_id": "reserved-physical",
        "lineage_id": "reserved-lineage", "split": "validation",
        "label": raw["contract"]["classes"][0],
        "path": "observations/VALIDATION_SHOULD_NOT_EXIST.csv",
        "sha256": "0" * 64})
    write_json(source, raw)
    # Design worker fails closed if supplied mixed manifest.
    with pytest.raises(ValueError, match="R82_PROTECTED_OR_MIXED_SOURCE_MANIFEST"):
        intake_train_source(spec, source, mapping)
    staged = stage_train_only_view(source, tmp_path / "train_view")
    assert staged["validation_telemetry_accessed"] is False
    data = read_json(staged["manifest"])
    assert len(data["samples"]) == 18
    assert {row["split"] for row in data["samples"]} == {"train"}
    records, audit = intake_train_source(spec, staged["manifest"], mapping)
    assert len(records) == 18
    assert audit["role"] == "TRAIN_ONLY"


def test_trusted_staging_refuses_test_role_even_without_accessing_trace(tmp_path):
    spec, source, _mapping = make_fixture(tmp_path)
    raw = read_json(source)
    raw["samples"][0]["split"] = "test"
    write_json(source, raw)
    with pytest.raises(ValueError, match="DATA_ROLE_VIOLATION"):
        stage_train_only_view(source, tmp_path / "train_view")
