"""R82 fail-closed TRAIN-only bridge from legacy PHM sequence manifests.

This is an *intake contract*, NOT physical-source attestation. The source-kind
claim, physical-group map and Basilisk source-binding hashes are self-reported;
the caller must audit source provenance independently. NEVER load FINAL or
mixed-role manifests into this research design worker.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

from ..data import Dataset
from ..io import read_json, safe_child, sha_file, digest
from .taskspec import TaskSpec, TaskSpecRejected, compile_task_spec, validate_observations


def _valid_sha(value):
    return (type(value) is str and len(value) == 64 and
            all(char in "0123456789abcdef" for char in value))


def stage_train_only_view(development_manifest, destination):
    """Trusted preparation ONLY: copy TRAIN traces to an isolated manifest.

    Unlike the design-worker intake, the trusted staging role may inspect
    development manifest metadata for TRAIN and VALIDATION, but it never
    opens, copies or hashes a validation telemetry file. TEST/FINAL roles
    are forbidden even in this preparer. The staged manifest can then be
    passed to intake_train_source; the original mixed manifest cannot.
    """
    from shutil import copyfile
    from ..io import write_json

    source = Dataset(development_manifest,
                     allowed_splits={"train", "validation"})
    training = [sample for sample in source.samples if sample["split"] == "train"]
    if not training:
        raise ValueError("R82_NO_TRAIN_CASES_TO_STAGE")
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError("R82_STAGING_DESTINATION_EXISTS")
    destination.mkdir(parents=True)
    copies = []
    for sample in training:
        original = safe_child(source.root, sample["path"])
        if not original.is_file() or sha_file(original) != sample["sha256"]:
            raise ValueError("R82_TRAIN_FILE_INTEGRITY_FAILURE")
        dest = safe_child(destination, sample["path"])
        dest.parent.mkdir(parents=True, exist_ok=True)
        copyfile(original, dest)
        if sha_file(dest) != sample["sha256"]:
            raise ValueError("R82_STAGED_TRAIN_FILE_CHANGED")
        copies.append(dict(sample))
    output = dict(source.raw)
    output["samples"] = copies
    output["dataset_id"] = source.raw["dataset_id"] + "-train-only-r82"
    output["provenance"] = {
        **source.raw["provenance"],
        "train_only_view": True,
        "source_development_manifest_sha256": sha_file(Path(development_manifest)),
        "validation_telemetry_accessed": False,
        "independent_qualification": False,
    }
    staged_manifest = destination / "dataset.json"
    write_json(staged_manifest, output)
    return {
        "schema": "phm.industrial.r82-staged-train.v0",
        "manifest": str(staged_manifest),
        "train_records": len(copies),
        "validation_telemetry_accessed": False,
        "train_only": True,
        "independent_qualification": False,
    }


def draft_reviewed_physical_map(task_spec, train_manifest, destination):
    """Prepare a NON-EXECUTABLE physical-group review worksheet.

    Copies the source's existing group identity only; every operating regime
    is REVIEW_REQUIRED, so it is impossible to train until independently
    reviewed regimes have been written. No trace bytes or fault labels used.
    """
    from ..io import write_json

    raw = (task_spec.to_dict() if isinstance(task_spec, TaskSpec)
           else TaskSpec.from_dict(task_spec).to_dict())
    plan = compile_task_spec(raw)
    if len(plan["train_data_contract"]["group_keys"]) != 1:
        raise ValueError("R82_REVIEW_MAP_REQUIRES_ONE_PHYSICAL_GROUP_KEY")
    data = Dataset(train_manifest, allowed_splits={"train"})
    if any(item["split"] != "train" for item in data.samples):
        raise ValueError("R82_PHYSICAL_MAP_DRAFT_REQUIRES_TRAIN_ONLY")
    group = plan["train_data_contract"]["group_keys"][0]
    env_keys = plan["train_data_contract"]["environment_axes"]
    worksheet = {
        "schema": "phm.industrial.source-physical-map.v1",
        "records": {
            sample["id"]: {
                group: sample["group_id"],
                **{key: "REVIEW_REQUIRED" for key in env_keys},
            }
            for sample in data.samples
        },
    }
    path = Path(destination)
    if path.exists():
        raise FileExistsError("R82_PHYSICAL_MAP_DRAFT_ALREADY_EXISTS")
    write_json(path, worksheet)
    return {"draft_path": str(path), "sample_count": len(data.samples),
            "review_required": True, "model_ready": False,
            "provenance_self_reported": True}


def verify_native_source_binding(binding, project_root, wheel_factory_file):
    """Hash actual source files; verify identity, NOT executed physics or authorship.

    Wheel factory source is the locally installed Basilisk source path, supplied
    by the reviewing operator rather than taken from an untrusted manifest.
    Never execute or import these code files during hash review.
    """
    if project_root is None or wheel_factory_file is None:
        raise ValueError("R82_NATIVE_SOURCE_FILES_REQUIRED")
    project_root = Path(project_root)
    expected = {
        "runtime_source": "src/sat_sim/bsk_engine/unified_native.py",
        "fault_effect_source": "src/sat_sim/bsk_engine/project_native_modules.py",
    }
    reviewed = {}
    for key, relative in expected.items():
        source = binding[key]
        if source["reference"] != relative:
            raise ValueError("R82_UNEXPECTED_NATIVE_SOURCE_REFERENCE:" + key)
        path = safe_child(project_root, relative)
        if not path.is_file() or sha_file(path) != source["sha256"]:
            raise ValueError("R82_NATIVE_SOURCE_SHA256_MISMATCH:" + key)
        reviewed[key] = source["sha256"]
    wheel = Path(wheel_factory_file)
    if (binding["wheel_factory_source"]["reference"] !=
            "Basilisk/utilities/simIncludeRW.py" or
            wheel.name != "simIncludeRW.py" or not wheel.is_file() or
            sha_file(wheel) != binding["wheel_factory_source"]["sha256"]):
        raise ValueError("R82_NATIVE_SOURCE_SHA256_MISMATCH:wheel_factory_source")
    reviewed["wheel_factory_source"] = binding["wheel_factory_source"]["sha256"]
    return reviewed


def intake_train_source(task_spec, manifest_path, physical_map_path, *,
                        native_binding_path=None,
                        native_source_checkout_root=None,
                        native_wheel_factory_file=None):
    """Return (TRAIN records, research source audit), never inspect other roles.

    physical_map schema:
      {"schema":"phm.industrial.source-physical-map.v1",
       "records": {"sample_id": {"<group key>":"...", "<env axis>":"..."}}}
    The map is separately reviewed metadata, not derived from truth or telemetry.
    """
    raw = (task_spec.to_dict() if isinstance(task_spec, TaskSpec)
           else TaskSpec.from_dict(task_spec).to_dict())
    plan = compile_task_spec(raw)
    if plan["decision_mode"] != "offline_trajectory":
        raise TaskSpecRejected("R82_CAUSAL_SOURCE_INTAKE_NOT_IMPLEMENTED")
    manifest_path = Path(manifest_path)
    manifest = read_json(manifest_path)
    if (manifest.get("schema") != "phm.sequence-dataset.v1" or
            not isinstance(manifest.get("samples"), list) or
            not manifest["samples"] or
            any(s.get("split") != "train" for s in manifest["samples"])):
        raise ValueError("R82_PROTECTED_OR_MIXED_SOURCE_MANIFEST")
    # Dataset schema/lineage/file path validation occurs before opening traces.
    data = Dataset(manifest_path, allowed_splits={"train"})
    data.verify_files()
    contract = data.contract
    source_kind = data.raw["provenance"].get("source_kind")
    if (source_kind == "native_basilisk" and
            contract["evidence_type"] == "simulation"):
        if native_binding_path is None:
            raise ValueError("R82_NATIVE_RUNTIME_BINDING_REQUIRED")
        native_binding_path = Path(native_binding_path)
        binding = read_json(native_binding_path)
        if (binding.get("scope") != "simulation_source_binding_only" or
                any(not isinstance(binding.get(key), dict) or
                    not _valid_sha(binding[key].get("sha256")) or
                    not isinstance(binding[key].get("reference"), str)
                    for key in ("runtime_source", "fault_effect_source",
                                "wheel_factory_source"))):
            raise ValueError("R82_NATIVE_BINDING_CONTRACT_INCOMPLETE")
        source_hash_review = verify_native_source_binding(
            binding, native_source_checkout_root, native_wheel_factory_file)
        binding_sha = sha_file(native_binding_path)
    elif (source_kind == "real_equipment_unreviewed" and
          contract["evidence_type"] == "real_unreviewed" and
          native_binding_path is None):
        binding_sha = None
        source_hash_review = None
    else:
        raise ValueError("R82_UNSUPPORTED_OR_UNVERIFIED_SOURCE_KIND")

    declared_channels = raw["observations"]["channels"]
    if (contract["classes"] != raw["task"]["labels"] or
            [c["name"] for c in contract["channels"]] !=
            [c["name"] for c in declared_channels] or
            [c["unit"] for c in contract["channels"]] !=
            [c["unit"] for c in declared_channels]):
        raise ValueError("R82_CHANNEL_UNIT_OR_ONTOLOGY_MISMATCH")
    if any(c["role"] != "deployment_input" for c in contract["channels"]):
        raise ValueError("R82_NONDEPLOYABLE_OBSERVATION")
    if raw["observations"]["time"]["sampling"] != "fixed":
        raise ValueError("R82_IRREGULAR_TIME_ADAPTER_NOT_IMPLEMENTED")

    map_data = read_json(physical_map_path)
    if (not isinstance(map_data, dict) or
            set(map_data) != {"schema", "records"} or
            map_data["schema"] != "phm.industrial.source-physical-map.v1" or
            not isinstance(map_data["records"], dict)):
        raise ValueError("R82_PHYSICAL_MAP_SCHEMA_INVALID")
    metadata_map = map_data["records"]
    ids = {sample["id"] for sample in data.samples}
    if set(metadata_map) != ids:
        raise ValueError("R82_PHYSICAL_MAP_SAMPLE_COVERAGE_MISMATCH")
    required = set(plan["train_data_contract"]["group_keys"] +
                   plan["train_data_contract"]["environment_axes"])
    time_field = plan["feature_contract"]["time_field"]
    channel_names = plan["feature_contract"]["ordered_channel_names"]
    allowed = ["time_s", *channel_names]
    records = []
    for sample in data.samples:
        assigned = metadata_map[sample["id"]]
        if (not isinstance(assigned, dict) or set(assigned) != required or
                any(type(value) is not str or not value.strip() or
                    value == "REVIEW_REQUIRED"
                    for value in assigned.values())):
            raise ValueError("R82_PHYSICAL_MAP_FIELDS_INVALID")
        # Group identity must match the already recorded physical lineage.
        # The adapter must never invent groups from sample indices or labels.
        if len(plan["train_data_contract"]["group_keys"]) != 1:
            raise ValueError("R82_REQUIRES_SINGLE_REVIEWED_PHYSICAL_GROUP_KEY")
        group_key = plan["train_data_contract"]["group_keys"][0]
        if assigned[group_key] != sample["group_id"]:
            raise ValueError("R82_GROUP_LINEAGE_MISMATCH")
        path = safe_child(data.root, sample["path"])
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if (reader.fieldnames is None or
                    len(reader.fieldnames) != len(set(reader.fieldnames)) or
                    set(reader.fieldnames) != set(allowed)):
                raise ValueError("R82_RAW_TELEMETRY_COLUMNS_INVALID")
            rows = []
            for row in reader:
                try:
                    numeric = {name: float(row[name]) for name in allowed}
                except (ValueError, TypeError, KeyError) as exc:
                    raise ValueError("R82_NONNUMERIC_TELEMETRY") from exc
                if not all(math.isfinite(value) for value in numeric.values()):
                    raise ValueError("R82_NONFINITE_TELEMETRY")
                reading = {time_field: numeric.pop("time_s"), **numeric}
                rows.append(reading)
        if len(rows) < 16:
            raise ValueError("R82_TOO_FEW_TELEMETRY_SAMPLES")
        times = [row[time_field] for row in rows]
        if any(not math.isclose(b - a, float(contract["sample_s"]),
                                rel_tol=1.e-6, abs_tol=1.e-7)
               for a, b in zip(times, times[1:])):
            raise ValueError("R82_SAMPLE_INTERVAL_MISMATCH")
        validate_observations(raw, rows)
        records.append({"measurements": rows,
                        "metadata": {"role": "TRAIN",
                                     "label": sample["label"], **assigned}})
    checked = compile_task_spec(raw, train_manifest=[
        row["metadata"] for row in records])
    manifest_audit = checked["manifest_audit"]
    if not manifest_audit["dg_supported"]:
        raise ValueError("R82_SOURCE_ENVIRONMENT_CLASS_SUPPORT_INSUFFICIENT")
    audit = {
        "schema": "phm.industrial.r82-intake.v0",
        "model_trained": False,
        "source_manifest_sha256": sha_file(manifest_path),
        "physical_map_sha256": sha_file(Path(physical_map_path)),
        "native_binding_sha256": binding_sha,
        "native_source_file_hashes_verified": source_hash_review,
        "source_kind_claimed": source_kind,
        "data_file_hashes_checked": True,
        "role": "TRAIN_ONLY",
        "records": len(records),
        "physical_groups": manifest_audit["physical_groups"],
        "environments": manifest_audit["environment_families"],
        "task_spec_sha256": checked["task_spec_sha256"],
        "source_data_fingerprint": digest([
            {"id": sample["id"], "sha256": sample["sha256"]}
            for sample in data.samples]),
        "physical_source_independently_attested": False,
        "independent_qualification": False,
    }
    return records, audit
