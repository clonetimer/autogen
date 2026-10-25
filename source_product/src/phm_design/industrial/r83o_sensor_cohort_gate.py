"""R83o metadata-only sensor/partition admission checks.

This does NOT verify an encoder calibration, read protected observations, train
a detector, or authorize a blind test. All inputs are assertions from a steward.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Mapping

SCHEMA_SENSOR = "phm.r83o.sensor.v1"
SCHEMA_COHORT = "phm.r83o.cohort.v1"
ROLES = frozenset({"TRAIN", "CAL", "VAL", "TEST", "FINAL"})
PROVENANCE = frozenset({"UNKNOWN", "NATIVE_TRUTH_ONLY", "VALIDATED_SIM_SENSOR", "CALIBRATED_ONBOARD_SENSOR"})
_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
_SHA = re.compile(r"^[0-9a-f]{64}$")
_RESERVED_METADATA_FIELDS = frozenset({
    "split", "split_name", "role", "data_role", "campaign_id",
    "source_group_id", "source_case_id", "pair_id", "record_id",
    "run_kind", "raw_case_sha256", "observation_sha256", "world_id",
    "generation_family_id", "source_hash", "data_hash", "seed",
})
_RESERVED_METADATA_PARTS = ("sha256", "split_", "_split", "campaign_",
                            "source_group", "source_case", "pair_id",
                            "record_id", "world_id", "_hash")
_FORBIDDEN = frozenset({
    "fault", "injection", "injected", "oracle", "truth", "true",
    "target", "label", "onset", "counterfactual", "capacity",
    "groundtruth", "ground", "healthy_twin", "u_max", "u_current",
    "applied_torque", "faulted", "original_case_id",
})


class AdmissionError(ValueError):
    """Explicit preflight refusal, not a neural-model evaluation result."""


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise AdmissionError(reason)


def _mapping(value, reason: str) -> Mapping:
    _require(isinstance(value, Mapping), reason)
    return value


def _id(value, reason: str) -> str:
    _require(isinstance(value, str) and bool(value.strip()) and value == value.strip(), reason)
    return value


def _sha(value, reason: str) -> str:
    _require(isinstance(value, str) and _SHA.fullmatch(value) is not None, reason)
    return value


def _number(value, *, positive: bool, code: str) -> float | None:
    if value is None:
        return None
    _require(type(value) in (int, float) and math.isfinite(value), code)
    _require(value > 0 if positive else value >= 0, code)
    return float(value)


def _field(value: str) -> str:
    _require(isinstance(value, str) and _NAME.fullmatch(value) is not None, "INVALID_TELEMETRY_FIELD")
    _require(value not in _RESERVED_METADATA_FIELDS
             and not any(x in value for x in _RESERVED_METADATA_PARTS),
             "PROTECTED_METADATA_FIELD:" + value)
    words = set(value.split("_"))
    _require(not (words & _FORBIDDEN), "ORACLE_OR_FAULT_TRUTH_FIELD:" + value)
    _require(not any(s in value for s in ("healthy_twin", "ground_truth", "u_current",
                                         "u_max", "applied_torque", "original_case_id")),
             "ORACLE_OR_FAULT_TRUTH_FIELD:" + value)
    return value


def _field_list(value, *, code: str) -> tuple[str, ...]:
    _require(isinstance(value, list) and bool(value), code)
    fields = tuple(_field(v) for v in value)
    _require(len(set(fields)) == len(fields), "DUPLICATE_TELEMETRY_FIELD")
    return fields


def audit_sensor(sensor: Mapping) -> dict:
    """Validate explicit UNKNOWN; return blockers rather than assuming ideal sensing."""
    sensor = _mapping(sensor, "INVALID_SENSOR_CONTRACT")
    _require(sensor.get("schema") == SCHEMA_SENSOR, "UNSUPPORTED_SENSOR_SCHEMA")
    _id(sensor.get("sensor_id"), "SENSOR_ID_REQUIRED")
    _require(sensor.get("signal_name") == "rw_speed_rad_s", "UNSUPPORTED_SENSOR_MEASURAND")
    _require(sensor.get("unit") == "rad/s", "SENSOR_UNIT_MISMATCH")
    provenance = sensor.get("provenance")
    _require(provenance in PROVENANCE, "INVALID_SENSOR_PROVENANCE")
    allowed = _field_list(sensor.get("telemetry_allowlist"), code="EMPTY_TELEMETRY_ALLOWLIST")
    _require("rw_speed_rad_s" in allowed, "MEASURED_RW_SPEED_NOT_ALLOWLISTED")
    quant = _number(sensor.get("quantization_step_rad_s"), positive=True, code="INVALID_QUANTIZATION")
    error = _number(sensor.get("max_abs_error_rad_s"), positive=False, code="INVALID_ERROR_BOUND")
    hz = _number(sensor.get("sampling_hz"), positive=True, code="INVALID_SAMPLING_HZ")
    jitter = _number(sensor.get("max_timestamp_jitter_s"), positive=False, code="INVALID_TIMESTAMP_JITTER")
    _require(type(sensor.get("available_onboard")) is bool, "INVALID_ONBOARD_AVAILABILITY")
    evidence = sensor.get("evidence_sha256")
    if evidence is not None:
        _sha(evidence, "INVALID_SENSOR_EVIDENCE_SHA")
    blockers = []
    if provenance not in {"VALIDATED_SIM_SENSOR", "CALIBRATED_ONBOARD_SENSOR"}:
        blockers.append("UNVERIFIED_SENSOR_PROVENANCE")
    if evidence is None:
        blockers.append("MISSING_SENSOR_EVIDENCE_HASH")
    if any(x is None for x in (quant, error, hz, jitter)):
        blockers.append("INCOMPLETE_SENSOR_ERROR_AND_SAMPLING_CONTRACT")
    if sensor["available_onboard"] is False:
        blockers.append("SENSOR_NOT_AVAILABLE_ONBOARD")
    # An asserted evidence digest cannot establish actual calibration authenticity.
    return {"schema": SCHEMA_SENSOR, "sensor_id": sensor["sensor_id"],
            "telemetry_allowlist": list(allowed), "blockers": blockers,
            "evidence_independently_verified": False,
            "sensing_assumptions_only": True}


def audit_features(features: Mapping, sensor_receipt: Mapping) -> tuple[str, ...]:
    features = _mapping(features, "INVALID_FEATURE_MANIFEST")
    selected = _field_list(features.get("feature_fields"), code="EMPTY_SELECTED_FEATURES")
    _require(set(selected).issubset(sensor_receipt["telemetry_allowlist"]),
             "FEATURE_NOT_ON_SENSOR_ALLOWLIST")
    return selected


def audit_cohorts(cohorts: Mapping) -> dict:
    """Reject cross-partition campaign/source/paired-run overlap, without raw data."""
    cohorts = _mapping(cohorts, "INVALID_COHORT_MANIFEST")
    _require(cohorts.get("schema") == SCHEMA_COHORT, "UNSUPPORTED_COHORT_SCHEMA")
    rows = cohorts.get("records")
    _require(isinstance(rows, list) and bool(rows), "EMPTY_COHORT_RECORDS")
    partitions: dict[str, set[str]] = defaultdict(set)
    owners: dict[str, dict[str, str]] = {k: {} for k in (
        "campaign_id", "source_group_id", "pair_id", "raw_case_sha256",
        "observation_sha256")}
    seen_records: set[str] = set()
    seen_raw: set[str] = set()
    pairs: dict[str, list[Mapping]] = defaultdict(list)
    for row in rows:
        row = _mapping(row, "INVALID_COHORT_RECORD")
        record_id = _id(row.get("record_id"), "RECORD_ID_REQUIRED")
        _require(record_id not in seen_records, "DUPLICATE_RECORD_ID")
        seen_records.add(record_id)
        role = row.get("split")
        _require(role in ROLES, "INVALID_DATA_SPLIT")
        for field in ("campaign_id", "source_group_id"):
            _id(row.get(field), "MISSING_" + field.upper())
        for field in ("raw_case_sha256", "observation_sha256"):
            _sha(row.get(field), "INVALID_" + field.upper())
        _require(row["raw_case_sha256"] not in seen_raw, "DUPLICATE_RAW_CASE_SHA")
        seen_raw.add(row["raw_case_sha256"])
        pair_id = row.get("pair_id")
        if pair_id is not None:
            _id(pair_id, "INVALID_PAIR_ID")
            _require(row.get("run_kind") in {"HEALTHY", "FAULTED"}, "INVALID_PAIRED_RUN_KIND")
            pairs[pair_id].append(row)
        else:
            _require(row.get("run_kind") in {None, "UNPAIRED"}, "UNLINKED_PAIRED_RUN")
        for kind, index in owners.items():
            value = row.get(kind)
            if value is not None:
                previous = index.setdefault(value, role)
                _require(previous == role, "CROSS_SPLIT_" + kind.upper())
        partitions[role].add(record_id)
    for pair_id, cases in pairs.items():
        _require(len(cases) == 2, "INCOMPLETE_OR_OVERSIZED_PAIR:" + pair_id)
        _require({c["run_kind"] for c in cases} == {"HEALTHY", "FAULTED"},
                 "INVALID_HEALTHY_FAULT_PAIR:" + pair_id)
        _require(len({c["source_group_id"] for c in cases}) == 1
                 and len({c["campaign_id"] for c in cases}) == 1,
                 "PAIRED_PHYSICS_GROUP_MISMATCH:" + pair_id)
    missing = sorted(ROLES - partitions.keys())
    # Fails closed for qualification: a manifest without held-out partitions
    # can still be audited as a partial development manifest.
    return {"schema": SCHEMA_COHORT, "records": len(rows),
            "pair_groups": len(pairs), "roles": {r: len(partitions[r]) for r in sorted(ROLES)},
            "missing_splits": missing, "cross_split_overlap_detected": False,
            "independent_physical_cohort_verified": False}


def preflight(sensor: Mapping, cohorts: Mapping, features: Mapping) -> dict:
    """Run all gates; success asserts metadata consistency only."""
    sensor_result = audit_sensor(sensor)
    selected = audit_features(features, sensor_result)
    cohort_result = audit_cohorts(cohorts)
    blockers = list(sensor_result["blockers"])
    if cohort_result["missing_splits"]:
        blockers.append("MISSING_REQUIRED_SPLITS")
    digest = hashlib.sha256(json.dumps(
        {"sensor": sensor, "cohorts": cohorts, "features": features},
        sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False).encode("utf-8")).hexdigest()
    return {"schema": "phm.r83o.preflight.v1", "input_manifest_sha256": digest,
            "sensor": sensor_result, "cohorts": cohort_result,
            "selected_feature_fields": list(selected),
            "status": "BLOCKED" if blockers else "METADATA_PRECHECK_ONLY",
            "blockers": blockers, "eligible_for_stable": False,
            "trained_diagnostic_model": False, "blind_test_executed": False,
            "verified_hardware_sensor": False,
            "qualified_diagnostic_claim": False,
            "independent_steward_review_required": True}


def _no_duplicate_keys(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise AdmissionError("DUPLICATE_JSON_KEY:" + key)
        value[key] = item
    return value


def _invalid_constant(s):
    raise AdmissionError("NON_FINITE_JSON_VALUE:" + s)


def _load(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=_no_duplicate_keys,
                      parse_constant=_invalid_constant)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="R83o METADATA-ONLY preflight; not qualification")
    parser.add_argument("--sensor", required=True)
    parser.add_argument("--cohorts", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    try:
        receipt = preflight(_load(args.sensor), _load(args.cohorts), _load(args.features))
    except (AdmissionError, OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as err:
        print(json.dumps({"status": "REJECTED", "reason": str(err)}, sort_keys=True))
        return 2
    payload = json.dumps(receipt, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if receipt["status"] == "METADATA_PRECHECK_ONLY" else 3


if __name__ == "__main__":
    raise SystemExit(main())
