"""R83g: preregistered SOURCE-TRAIN-only command/response feature ablation.

The R83f real Basilisk cohort, physical groups, folds, classes, candidate
genomes, acceptance thresholds and 60/24 selection are IMMUTABLE. R83g
compares two 18-channel representations of *identical* six-channel TRAIN
traces: padded original input vs 12 genuinely causal derived channels.
No fault event truth, component injection identity, labels, VAL, TEST or
FINAL may enter transformed model observations.

All feedback and scores here are exploratory TRAIN source OOF; no model is
promoted and no claims of independent performance are warranted.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import hashlib
import math

import numpy as np

from ..io import read_json, sha_file, write_json
from .design_engine import _observations_digest, _sha
from .r83_matched_budget import run_matched_source_train_study
from .source_intake import stage_train_only_view, intake_train_source
from .taskspec import TaskSpec, TaskSpecRejected, validate_observations

SCHEMA = "phm.r83g.causal-command-response-source-train.v1"
MODES = ("zero_padded_raw", "causal_command_response")
NATIVE_TRAIN_OBSERVATIONS_SHA = (
    "10d730843fe7488c0fc670bc220176cf4f602a7dedca26aed37f68ee80e7cd36")
NATIVE_TRAIN_METADATA_SHA = (
    "adac20dd8b4fa7952bcc1b743ba6f8111dd36c1113a507336394f3f0fd5e2f3a")
R83F_PLAN_CANONICAL_SHA = (
    "c54bfcd154322a620c7d8058ce8a40db6e155602593c5ff5da8bdba43ef3b714")
SOURCE_SPEED = tuple(f"adcs.rw.speed_rad_s_{i}" for i in range(3))
SOURCE_COMMAND = tuple(f"adcs.rw.command_torque_nm_{i}" for i in range(3))
SOURCE_ORDER = SOURCE_SPEED + SOURCE_COMMAND

DERIVED = tuple(
    (f"r83g.rw.speed_slope_rad_s2_{i}", "rad/s^2",
     f"r83g.rw.command_impulse_nms_{i}", "N*m*s",
     f"r83g.rw.speed_delta_rad_s_{i}", "rad/s",
     f"r83g.rw.command_speed_power_w_{i}", "W")
    for i in range(3))
DERIVED_NAMES = tuple(row[j] for row in DERIVED for j in (0,2,4,6))
DERIVED_UNITS = tuple(row[j] for row in DERIVED for j in (1,3,5,7))


def _reject(code):
    raise TaskSpecRejected(code)


def _spec(spec: dict):
    raw = TaskSpec.from_dict(spec).to_dict()
    if (raw["data"]["role"] != "TRAIN" or
            raw["task"]["decision_mode"] != "offline_trajectory" or
            tuple(c["name"] for c in raw["observations"]["channels"]) != SOURCE_ORDER):
        _reject("R83G_REQUIRES_UNCHANGED_SIX_CHANNEL_TRAIN_SPEC")
    return raw


def _source_integrity(raw, records, research_report):
    """Check genuine R83f source TRAIN fingerprints BEFORE derived features."""
    if (not isinstance(research_report, dict) or
            research_report.get("schema") != "phm.r83.matched-source-train-study.v1" or
            research_report.get("source_train_test_or_final_accessed") is not False or
            research_report.get("independent_qualification") is not False or
            research_report.get("source_task_spec_sha256") !=
            TaskSpec.from_dict(raw).sha256 or
            research_report.get("source_train_observations_sha256") !=
            NATIVE_TRAIN_OBSERVATIONS_SHA or
            research_report.get("source_train_manifest_sha256") !=
            NATIVE_TRAIN_METADATA_SHA):
        _reject("R83G_NOT_THE_FROZEN_R83F_RESEARCH_SOURCE")
    if len(records) != 60:
        _reject("R83G_SOURCE_TRAIN_COUNT_CHANGED")
    group_fields = raw["data"]["group_keys"]
    source_meta = [dict(record["metadata"]) for record in records]
    for record in records:
        if record["metadata"].get("role") != "TRAIN":
            _reject("R83G_VALIDATION_OR_PROTECTED_ROLE_ACCESS_DENIED")
        validate_observations(raw, record["measurements"])
    if _sha(source_meta) != NATIVE_TRAIN_METADATA_SHA:
        _reject("R83G_PHYSICAL_SOURCE_GROUPS_OR_LABELS_CHANGED")
    sequences = [np.asarray(
        [[row[raw["observations"]["time"]["field"]]] +
         [row[field] for field in SOURCE_ORDER]
         for row in record["measurements"]], dtype=np.float64)
        for record in records]
    if _observations_digest(sequences) != NATIVE_TRAIN_OBSERVATIONS_SHA:
        _reject("R83G_NATIVE_TRAIN_TELEMETRY_CHANGED")
    groups = set()
    for row in records:
        key = tuple(row["metadata"][k] for k in group_fields)
        if key in groups:
            _reject("R83G_DUPLICATE_PHYSICAL_GROUP")
        groups.add(key)
    if len(groups) != 60:
        _reject("R83G_INDEPENDENT_PHYSICAL_GROUP_COUNT_CHANGED")
    return {"source_train_observations_sha256": NATIVE_TRAIN_OBSERVATIONS_SHA,
            "source_train_manifest_sha256": NATIVE_TRAIN_METADATA_SHA,
            "source_train_physical_groups": 60}


def _expand_spec(raw):
    copy = deepcopy(raw)
    for name, unit in zip(DERIVED_NAMES, DERIVED_UNITS):
        copy["observations"]["channels"].append({
            "name": name, "role": "sensor", "unit": unit,
            "available_at_decision": True})
    return TaskSpec.from_dict(copy).to_dict()


def _features(measurements, *, mode, time_field):
    """Exact causal first-order features. Uses NO physics truth or lookahead.

    At first observation all 12 new signals are zero except speed-power
    (also zero for a well-defined reference). Later:
    slope[t] = (speed[t] - speed[t-1]) / dt
    impulse[t] = impulse[t-1] + command[t-1] * dt
    speed_delta[t] = speed[t] - speed[0]
    power[t] = command[t-1] * speed[t-1]  (lagged prior-known motor COMMAND
               times observed wheel speed, never measured applied power).
    The apparent power unit is an algebraic proxy only. No claim of actual
    electromechanical power conversion is made.
    """
    if mode not in MODES:
        raise ValueError("R83G_UNKNOWN_PREREGISTERED_REPRESENTATION")
    out = []
    first = measurements[0]
    prev = None
    impulse = [0.0] * 3
    for item in measurements:
        # No existing label or metadata column may be carried along.
        now = {name: item[name] for name in (time_field, *SOURCE_ORDER)}
        for i in range(3):
            speed = float(item[SOURCE_SPEED[i]])
            if prev is None:
                slope = delta = apparent = 0.0
            else:
                dt = float(item[time_field] - prev[time_field])
                if dt <= 0 or not math.isfinite(dt):
                    _reject("R83G_NONINCREASING_SOURCE_TIME")
                slope = (speed - float(prev[SOURCE_SPEED[i]])) / dt
                impulse[i] += float(prev[SOURCE_COMMAND[i]]) * dt
                delta = speed - float(first[SOURCE_SPEED[i]])
                apparent = (float(prev[SOURCE_COMMAND[i]]) *
                            float(prev[SOURCE_SPEED[i]]))
            vals = (slope, impulse[i], delta, apparent)
            names = DERIVED[i][::2]
            for name, value in zip(names, vals):
                now[name] = value if mode == "causal_command_response" else 0.0
        if not all(math.isfinite(float(v)) for v in now.values()):
            _reject("R83G_NONFINITE_COMMAND_RESPONSE_FEATURE")
        out.append(now)
        prev = item
    return out


def make_matched_representation(raw_spec, records, mode):
    raw = _spec(raw_spec)
    spec = _expand_spec(raw)
    if mode not in MODES:
        raise ValueError("R83G_UNKNOWN_PREREGISTERED_REPRESENTATION")
    result = [{
        "metadata": dict(rec["metadata"]),
        "measurements": _features(rec["measurements"], mode=mode,
                                   time_field=raw["observations"]["time"]["field"])
    } for rec in records]
    for rec in result:
        validate_observations(spec, rec["measurements"])
    return spec, result


def run_train_only_representations(raw_spec, records, frozen_r83f_train_comparison):
    """Same arm grid, group folds, seed, fit cap, refusal screens in both modes.

    Source original measurements and metadata must match genuine R83f
    immutable TRAIN receipts. No VAL measurements are accepted or read.
    """
    raw = _spec(raw_spec)
    source = _source_integrity(raw, records, frozen_r83f_train_comparison)
    output = {
        "schema": SCHEMA,
        "source_role": "BASILISK_REAL_R83F_TRAIN_ONLY",
        "reference_r83f_frozen_plan_canonical_sha256": R83F_PLAN_CANONICAL_SHA,
        "frozen_r83f_source": source,
        "representation_order": list(MODES),
        "representation_definition": "CAUSAL_LAG1_SLOPE_IMPULSE_DELTA_COMMAND_SPEED_PROXY_V1",
        "derived_channel_count_per_arm": 12,
        "total_channels_per_arm": 18,
        "representation_only_difference": True,
        "no_model_promotion": True,
        "source_train_only": True,
        "validation_test_or_final_telemetry_read": False,
        "original_r83f_decisions_unchanged": True,
        "hypothesis_confirmed_independently": False,
        "independent_qualification": False,
        "arms": {},
    }
    checks = {}
    for mode in MODES:
        spec, transformed = make_matched_representation(raw, records, mode)
        source_metadata = _sha([r["metadata"] for r in transformed])
        checks[mode] = source_metadata
        output["arms"][mode] = run_matched_source_train_study(spec, transformed)
    if len(set(checks.values())) != 1 or list(checks.values())[0] != NATIVE_TRAIN_METADATA_SHA:
        _reject("R83G_NOT_IDENTICAL_PHYSICAL_GROUP_FOLDS")
    controls = [output["arms"][m] for m in MODES]
    same = all(
        controls[0][k] == controls[1][k]
        for k in ("source_train_manifest_sha256", "source_environment_count",
                  "candidate_budget_per_arm", "fit_budget_per_arm",
                  "acceptance_policy", "max_wall_seconds_per_arm"))
    same = same and all(
        controls[0]["arms"][arm]["source_train_candidate_count"] ==
        controls[1]["arms"][arm]["source_train_candidate_count"] == 3
        for arm in ("fixed_portfolio","static_topology","self_organizing"))
    complete = same and all(
        result["fairness_gate"]["source_train_comparison_allowed"]
        for result in controls)
    output["fairness_gate"] = {
        "same_physical_groups_and_labels": same,
        "same_stratified_ooF_partitions": same,
        "same_seed_candidate_slots_fit_and_acceptance": same,
        "same_input_channel_count": True,
        "same_model_parameter_count_by_arm": True,
        "same_physical_basilisk_source": True,
        "both_search_slates_complete": bool(complete),
        "comparison_allowed_train_research_only": bool(complete),
        "equal_measured_flops": False,
        "performance_claim_allowed": False,
        "validation_used_for_selection": False,
    }
    return output


def run_r83g_frozen_native_train(
    *, task_spec, development_manifest, train_physical_map,
    original_r83f_research_report, runtime_binding, native_source_checkout,
    basilisk_wheel_factory, output,
):
    destination = Path(output)
    if destination.exists():
        raise FileExistsError("R83G_RESEARCH_REPORT_EXISTS")
    raw = read_json(task_spec)
    source_report = read_json(original_r83f_research_report)
    if ("comparison" not in source_report or
            source_report.get("status") != "TRAIN_SOURCE_SEARCH_RESEARCH_ONLY"):
        _reject("R83G_REQUIRES_COMPLETED_UNMODIFIED_R83F_TRAIN_REPORT")
    # Only *TRAIN* CSV files are copied/verified from a development manifest.
    # Any VALIDATION descriptors remain unread, and TEST/FINAL is forbidden.
    dest = destination
    dest.mkdir(parents=True)
    report = {"schema": SCHEMA, "status": "STARTED",
              "source_development_manifest_sha256": sha_file(Path(development_manifest)),
              "r83f_research_report_sha256": sha_file(Path(original_r83f_research_report)),
              "research_only": True, "independent_qualification": False}
    try:
        train = stage_train_only_view(development_manifest,
                                     dest / "isolated_r83f_train")
        records, intake = intake_train_source(
            raw, train["manifest"], train_physical_map,
            native_binding_path=runtime_binding,
            native_source_checkout_root=native_source_checkout,
            native_wheel_factory_file=basilisk_wheel_factory)
        if not intake.get("native_source_file_hashes_verified"):
            _reject("R83G_UNVERIFIED_NATIVE_SOURCE")
        _source_integrity(_spec(raw), records, source_report["comparison"])
        report["native_intake"] = intake
        report["comparison"] = run_train_only_representations(
            raw, records, source_report["comparison"])
        report["status"] = "TRAIN_EXPLORATION_COMPLETE_NO_VALIDATION"
    except Exception as exc:
        report["status"] = "REFUSED"
        report["refusal_code"] = type(exc).__name__
        write_json(dest / "research_report.json", report)
        raise
    write_json(dest / "research_report.json", report)
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in (
        "task-spec", "development-manifest", "train-physical-map",
        "original-r83f-research-report", "runtime-binding",
        "native-source-checkout", "basilisk-wheel-factory", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    a = vars(p.parse_args(argv))
    run_r83g_frozen_native_train(**{k.replace("-", "_"): v for k, v in a.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
