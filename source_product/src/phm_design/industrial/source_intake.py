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


def intake_train_source(task_spec, manifest_path, physical_map_path, *,
                        native_binding_path=None):
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
        binding_sha = sha_file(native_binding_path)
    elif (source_kind == "real_equipment_unreviewed" and
          contract["evidence_type"] == "real_unreviewed" and
          native_binding_path is None):
        binding_sha = None
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
                any(type(value) is not str or not value.strip()
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
