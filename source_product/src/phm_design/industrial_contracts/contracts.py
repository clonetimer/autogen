"""R93 P1 — industrial cross-domain data/task contracts (stdlib-only).

Contract compiler and trusted split auditor only. Does NOT train a model,
verify external data bytes, infer physical identity, or qualify deployment.
Design workers must receive a TRAIN-only manifest: protected observations,
roles and label statistics must not be exposed to the design plan.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import json
import math
import re

EQUIPMENT_SCHEMA = "phm.autogen.equipment.v1"
DATASET_SCHEMA = "phm.autogen.dataset.v1"
TRIAL_SCHEMA = "phm.autogen.trial.v1"
SUPPORTED_ROLES = ("TRAIN", "CALIBRATION", "VALIDATION", "FINAL")
SCOPE = {
    "physical_group": ("unseen_physical_device", "unseen_environment"),
    "temporal_forward": ("future_same_device",),
}
CANDIDATES = {
    "constant", "statistical", "linear", "tree", "random_forest",
    "cnn1d", "tcn", "gru", "fuzzy_tsk",
}
IDENT = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,95}$")
HASH = re.compile(r"^[0-9a-f]{64}$")
FORBIDDEN_INPUT = {
    "label", "fault_label", "class_label", "ground_truth", "failure_time",
    "future_label", "actual_rul", "true_rul", "role", "split", "environment_id",
    "physical_id", "device_id", "session_id", "onset_time", "fault_onset",
}


class ContractRejected(ValueError):
    """Explicit fail-closed contract rejection; no implicit coercions."""


def _reject(reason):
    raise ContractRejected(reason)


def _obj(raw, keys, at):
    if not isinstance(raw, Mapping) or set(raw) != set(keys):
        _reject("R93_INVALID_KEYS_" + at)
    return raw


def _id(value, field):
    if type(value) is not str or IDENT.fullmatch(value) is None:
        _reject("R93_INVALID_IDENTIFIER_" + field)
    return value


def _sha(value, field):
    if type(value) is not str or HASH.fullmatch(value) is None:
        _reject("R93_INVALID_SHA256_" + field)
    return value


def _positive(value, field, *, integer=False, allow_zero=False, upper=None):
    if integer:
        good = type(value) is int
    else:
        good = type(value) in (int, float) and math.isfinite(value)
    if not good or (value < 0 if allow_zero else value <= 0):
        _reject("R93_INVALID_POSITIVE_" + field)
    if upper is not None and value > upper:
        _reject("R93_OUT_OF_BUDGET_" + field)
    return value


def _list(items, field, *, minimum=1):
    if not isinstance(items, list) or len(items) < minimum:
        _reject("R93_INVALID_COLLECTION_" + field)
    return items


def _unique(values, field):
    if len(values) != len(set(values)):
        _reject("R93_DUPLICATE_" + field)


def _utc(value, field):
    if type(value) is not str or not value.endswith("Z"):
        _reject("R93_UTC_TIMESTAMP_REQUIRED_" + field)
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ContractRejected("R93_INVALID_TIMESTAMP_" + field) from exc
    if result.utcoffset() != timezone.utc.utcoffset(None):
        _reject("R93_UTC_TIMESTAMP_REQUIRED_" + field)
    return result


def _canonical(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False)


def _fingerprint(obj):
    return hashlib.sha256(_canonical(obj).encode("utf-8")).hexdigest()


def validate_equipment(equipment):
    _obj(equipment, ("schema", "equipment_id", "family", "time_field",
                     "channels"), "EQUIPMENT")
    if equipment["schema"] != EQUIPMENT_SCHEMA:
        _reject("R93_EQUIPMENT_SCHEMA")
    _id(equipment["equipment_id"], "equipment_id")
    if equipment["family"] not in (
            "rotating_machinery", "process_equipment",
            "spacecraft", "other"):
        _reject("R93_UNKNOWN_EQUIPMENT_FAMILY")
    time_name = _id(equipment["time_field"], "time_field")
    seen = []
    for sensor in _list(equipment["channels"], "channels"):
        _obj(sensor, ("name", "modality", "unit",
                      "sampling_hz", "available_at_decision"), "CHANNEL")
        name = _id(sensor["name"], "channel")
        if name in FORBIDDEN_INPUT:
            _reject("R93_PROTECTED_OBSERVATION_CHANNEL")
        if sensor["modality"] not in (
                "vibration", "current", "temperature", "pressure",
                "speed", "telemetry", "command", "other"):
            _reject("R93_UNKNOWN_MODALITY")
        if (type(sensor["unit"]) is not str or not sensor["unit"].strip()
                or len(sensor["unit"]) > 64):
            _reject("R93_SENSOR_UNIT_REQUIRED")
        hz = sensor["sampling_hz"]
        if hz is not None:
            _positive(hz, "sampling_hz", upper=1e8)
        if type(sensor["available_at_decision"]) is not bool:
            _reject("R93_SENSOR_DECISION_AVAILABILITY_INVALID")
        seen.append(name)
    _unique(seen + [time_name], "channel_or_time")
    return equipment


def validate_trial(trial, equipment):
    validate_equipment(equipment)
    _obj(trial, ("schema", "trial_id", "equipment_id", "task_type", "input_channels",
                 "labels", "healthy_label", "candidate_families",
                 "validation_target", "budget"), "TRIAL")
    if trial["schema"] != TRIAL_SCHEMA:
        _reject("R93_TRIAL_SCHEMA")
    _id(trial["trial_id"], "trial_id")
    if trial["equipment_id"] != equipment["equipment_id"]:
        _reject("R93_TRIAL_EQUIPMENT_MISMATCH")
    if trial["task_type"] not in ("classification", "anomaly_detection"):
        _reject("R93_TASK_TYPE_NOT_YET_SUPPORTED")
    channels = _list(trial["input_channels"], "input_channels")
    if any(type(c) is not str for c in channels):
        _reject("R93_INVALID_INPUT_CHANNEL")
    _unique(channels, "trial_inputs")
    deployable = {s["name"] for s in equipment["channels"]
                  if s["available_at_decision"]}
    if not set(channels) <= deployable or any(c in FORBIDDEN_INPUT for c in channels):
        _reject("R93_PROTECTED_OR_UNAVAILABLE_TRIAL_INPUT")
    labels = _list(trial["labels"], "labels", minimum=2)
    for label in labels:
        _id(label, "class_label")
    _unique(labels, "class_labels")
    if trial["healthy_label"] not in labels:
        _reject("R93_HEALTHY_LABEL_NOT_IN_ONTOLOGY")
    if (trial["task_type"] == "anomaly_detection" and
            len(labels) != 2):
        _reject("R93_ANOMALY_LABELS_REQUIRE_BINARY_ONTOLOGY")
    candidates = _list(trial["candidate_families"], "families")
    if any(type(c) is not str for c in candidates):
        _reject("R93_INVALID_CANDIDATE_FAMILY")
    _unique(candidates, "candidate_families")
    if not set(candidates) <= CANDIDATES:
        _reject("R93_UNREGISTERED_MODEL_FAMILY")
    if trial["validation_target"] not in (
            "unseen_physical_device", "unseen_environment", "future_same_device"):
        _reject("R93_UNSUPPORTED_VALIDATION_TARGET")
    budget = _obj(trial["budget"], ("max_trials", "max_fit_seconds",
                                    "max_memory_mb", "seed"), "BUDGET")
    _positive(budget["max_trials"], "max_trials", integer=True, upper=1000)
    _positive(budget["max_fit_seconds"], "max_fit_seconds", upper=1e7)
    _positive(budget["max_memory_mb"], "max_memory_mb", integer=True, upper=1e6)
    _positive(budget["seed"], "seed", integer=True, allow_zero=True,
              upper=2**32-1)
    if len(candidates) > budget["max_trials"]:
        _reject("R93_MORE_CANDIDATES_THAN_TRIAL_BUDGET")
    return trial


def validate_dataset(dataset, equipment, trial, *, train_only=False):
    """Validate metadata only. Source workers MUST use train_only=True.

    Full-manifest audit should be executed only by a trusted data steward;
    the design compiler never opens VALIDATION/CALIBRATION/FINAL records.
    """
    validate_trial(trial, equipment)
    _obj(dataset, ("schema", "dataset_id", "equipment_id", "origin",
                   "license", "split_protocol", "records"), "DATASET")
    if dataset["schema"] != DATASET_SCHEMA:
        _reject("R93_DATASET_SCHEMA")
    _id(dataset["dataset_id"], "dataset_id")
    if dataset["equipment_id"] != equipment["equipment_id"]:
        _reject("R93_DATASET_EQUIPMENT_MISMATCH")
    if dataset["origin"] not in (
            "measured", "native_physics_simulation", "synthetic", "unknown"):
        _reject("R93_DATASET_ORIGIN_REQUIRED")
    if type(dataset["license"]) is not str or not dataset["license"].strip():
        _reject("R93_SOURCE_LICENSE_REQUIRED")
    split = _obj(dataset["split_protocol"], ("mode", "embargo_seconds"),
                 "SPLIT_PROTOCOL")
    mode = split["mode"]
    if mode not in SCOPE:
        _reject("R93_SPLIT_MODE_NOT_SUPPORTED")
    if trial["validation_target"] not in SCOPE[mode]:
        _reject("R93_SPLIT_CANNOT_SUPPORT_CLAIMED_VALIDATION_TARGET")
    _positive(split["embargo_seconds"], "embargo_seconds", allow_zero=True,
              upper=1e9)
    if mode == "physical_group" and split["embargo_seconds"] != 0:
        _reject("R93_PHYSICAL_GROUP_HAS_NO_TEMPORAL_EMBARGO")
    if mode == "temporal_forward" and split["embargo_seconds"] < 1:
        _reject("R93_TEMPORAL_SPLIT_NEEDS_POSITIVE_EMBARGO")
    ids = []
    rows = _list(dataset["records"], "records")
    seen_hashes = {}
    hash_roles = {}
    session_roles = {}
    entities = defaultdict(list)
    roles = Counter()
    groups = defaultdict(set)
    for row in rows:
        _obj(row, ("record_id", "physical_id", "session_id",
                   "environment_id", "role", "source_sha256",
                   "start_utc", "end_utc", "label"), "RECORD")
        rid = _id(row["record_id"], "record_id")
        ids.append(rid)
        phys = _id(row["physical_id"], "physical_id")
        _id(row["session_id"], "session_id")
        _id(row["environment_id"], "environment_id")
        role = row["role"]
        if type(role) is not str or role not in SUPPORTED_ROLES:
            _reject("R93_UNKNOWN_DATA_ROLE")
        if train_only and role != "TRAIN":
            _reject("R93_PROTECTED_ROLE_IN_DESIGN_WORKER")
        sha = _sha(row["source_sha256"], "source")
        first_owner = seen_hashes.setdefault(sha, phys)
        if first_owner != phys:
            _reject("R93_SAME_SIGNAL_BYTES_ASSIGNED_TO_DIFFERENT_DEVICES")
        if sha in hash_roles and hash_roles[sha] != role:
            _reject("R93_SAME_RAW_SIGNAL_CROSSES_DATA_ROLES")
        hash_roles[sha] = role
        key = (phys, row["session_id"])
        if key in session_roles and session_roles[key] != role:
            _reject("R93_ACQUISITION_SESSION_CROSSES_DATA_ROLES")
        session_roles[key] = role
        start = _utc(row["start_utc"], "start")
        end = _utc(row["end_utc"], "end")
        if end <= start:
            _reject("R93_INVALID_RECORD_DURATION")
        label = row["label"]
        if label is not None and label not in trial["labels"]:
            _reject("R93_UNKNOWN_DATASET_LABEL")
        if trial["task_type"] == "classification" and role == "TRAIN" and label is None:
            _reject("R93_SUPERVISED_TRAIN_LABEL_REQUIRED")
        if (trial["task_type"] == "anomaly_detection" and role == "TRAIN"
                and label not in (None, trial["healthy_label"])):
            _reject("R93_ANOMALY_NORMAL_ONLY_TRAINING_VIOLATION")
        roles[role] += 1
        groups[role].add(phys)
        entities[phys].append((start, end, role, row["session_id"]))
    _unique(ids, "record_ids")
    if roles["TRAIN"] == 0:
        _reject("R93_NO_SOURCE_TRAIN_RECORDS")
    if mode == "physical_group":
        for a in SUPPORTED_ROLES:
            for b in SUPPORTED_ROLES:
                if a < b and groups[a] & groups[b]:
                    _reject("R93_PHYSICAL_DEVICE_CROSSES_DATA_ROLES")
        if trial["validation_target"] == "unseen_environment":
            by_environment = defaultdict(set)
            for row in rows:
                by_environment[row["environment_id"]].add(row["role"])
            if any(len(rr) > 1 for rr in by_environment.values()):
                _reject("R93_ENVIRONMENT_CROSSES_ROLES")
    else:
        # Forward split under the same device. No same-source time overlap,
        # no backtracking, and a fixed embargo between different roles.
        order = {role: n for n, role in enumerate(SUPPORTED_ROLES)}
        embargo = split["embargo_seconds"]
        for physical, slices in entities.items():
            chronological = sorted(slices, key=lambda x: x[0])
            previous = None
            for start, end, role, session in chronological:
                if previous is not None:
                    pstart, pend, prole, psession = previous
                    if start < pend:
                        _reject("R93_TEMPORAL_RECORD_INTERVAL_OVERLAP")
                    if order[role] < order[prole]:
                        _reject("R93_TEMPORAL_DATA_ROLE_REWINDS")
                    if role != prole and (start - pend).total_seconds() < embargo:
                        _reject("R93_INSUFFICIENT_TEMPORAL_EMBARGO")
                previous = (start, end, role, session)
    return {
        "schema": "phm.autogen.dataset-audit.v1",
        "dataset_sha256": _fingerprint(dataset),
        "source_record_count": roles["TRAIN"],
        "role_record_counts": dict(roles) if not train_only else {"TRAIN": roles["TRAIN"]},
        "source_physical_group_count": len(groups["TRAIN"]),
        "source_environment_count": len({
            row["environment_id"] for row in rows if row["role"] == "TRAIN"}),
        "role_separation_checked": True,
        "split_mode": mode,
        "evaluation_scope": trial["validation_target"],
        "source_bytes_authenticated": False,
        "physical_identity_independently_authenticated": False,
        "model_fitted": False,
        "independent_generalization_proven": False,
        "eligible_for_stable": False,
    }


def compile_source_plan(equipment, dataset_source_only, trial):
    """Return only a TRAIN-derived design plan; no protected label leakage."""
    audit = validate_dataset(dataset_source_only, equipment, trial,
                             train_only=True)
    if dataset_source_only["origin"] == "unknown":
        _reject("R93_SOURCE_ORIGIN_UNVERIFIED")
    return {
        "schema": "phm.autogen.source-plan.v1",
        "status": "PLAN_ONLY_NO_MODEL_TRAINED",
        "equipment_sha256": _fingerprint(equipment),
        "dataset_train_sha256": _fingerprint(dataset_source_only),
        "trial_sha256": _fingerprint(trial),
        "equipment_family": equipment["family"],
        "task_type": trial["task_type"],
        "input_channels": list(trial["input_channels"]),
        "candidate_families": list(trial["candidate_families"]),
        "compute_budget": dict(trial["budget"]),
        "source_group_count": audit["source_physical_group_count"],
        "source_environment_count": audit["source_environment_count"],
        "group_protocol": audit["split_mode"],
        "proposed_validation_scope": trial["validation_target"],
        "calibration_validation_final_read": False,
        "protected_target_fields_available_to_model": False,
        "model_qualified": False,
        "production_approved": False,
    }
