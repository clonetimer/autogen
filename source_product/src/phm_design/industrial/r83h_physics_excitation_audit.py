"""R83h immutable REAL native TRAIN physics-only excitation audit.

This is not a feature transformer, candidate selector, neural network, or
model-validation tool. It deliberately reads protected simulator applied
torque, fault onset, severity and injection wheel for offline physical QA.
NONE of these quantities is exported as deployable model input.

Inputs: ORIGINAL sealed R83f raw native source case tree and its preexisting
45-pair TRAIN physical observability report. No VAL/TEST/FINAL path is opened.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import argparse
import csv
import hashlib
import math
import statistics

from ..io import read_json, write_json

SCHEMA = "phm.r83h.native-train-physics-excitation-audit.v1"
ORIGINAL_SHA = "c54bfcd154322a620c7d8058ce8a40db6e155602593c5ff5da8bdba43ef3b714"
MECHANISMS = (
    "ADCS_RW_FRICTION_INCREASE",
    "ADCS_RW_JAM",
    "ADCS_RW_TORQUE_AUTHORITY_LOSS",
)
REGIMES = {29.0: "inclination_low", 47.0: "inclination_medium",
           65.0: "inclination_high"}
SOURCE_PREFIX = "r83d_balanced_native_rw-"


def _reject(code):
    raise ValueError(code)


def _sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(path: Path, index: int):
    required = {
        "time_s", f"adcs.rw.speed_rad_s_{index}",
        f"adcs.rw.command_torque_nm_{index}",
        f"adcs.control.applied_torque_nm_{index}",
        f"adcs.rw.effective_max_torque_nm_{index}",
    }
    with path.open(encoding="utf-8", newline="") as inp:
        reader = csv.DictReader(inp)
        if (reader.fieldnames is None or
                len(set(reader.fieldnames)) != len(reader.fieldnames) or
                not required.issubset(reader.fieldnames)):
            _reject("R83H_ORIGINAL_NATIVE_PHYSICS_CHANNELS_MISSING")
        rows = []
        for raw in reader:
            try:
                row = {k: float(raw[k]) for k in required}
            except (KeyError, ValueError, TypeError) as exc:
                raise ValueError("R83H_INVALID_SOURCE_PHYSICS_NUMBER") from exc
            if not all(math.isfinite(v) for v in row.values()):
                _reject("R83H_NONFINITE_PHYSICS_SAMPLE")
            rows.append(row)
    if len(rows) != 61 or any(abs(r["time_s"]-i)>1.e-8
                              for i,r in enumerate(rows)):
        _reject("R83H_CHANGED_OR_IRREGULAR_61_POINT_NATIVE_TRACE")
    return rows


def _median(numbers):
    return float(statistics.median(numbers))


def _summarize(rows):
    labels = (
        "post_max_abs_target_speed_delta_rad_s",
        "nominal_post_mean_abs_command_torque_nm",
        "nominal_post_mean_abs_command_to_rating",
        "nominal_post_max_abs_command_to_rating",
        "fault_post_mean_abs_command_applied_gap_nm",
        "nominal_post_mean_abs_command_applied_gap_nm",
        "post_max_abs_paired_command_difference_nm",
        "post_max_abs_paired_applied_torque_difference_nm",
    )
    return {
        "independent_original_pairs": len(rows),
        **{key: {
            "median": _median([r[key] for r in rows]),
            "minimum": min(r[key] for r in rows),
            "maximum": max(r[key] for r in rows),
        } for key in labels},
    }


def audit_native_train_excitation(
    *, source_root: str | Path, frozen_45_pair_audit: str | Path,
    frozen_plan_audit: str | Path, output: str | Path
) -> dict:
    """Read exactly 90 original native TRAIN traces, never an unseen VAL trace."""
    root = Path(source_root)
    output = Path(output)
    if output.exists():
        raise FileExistsError("R83H_PHYSICS_QA_ALREADY_EXISTS")
    plan = read_json(frozen_plan_audit)
    if (plan.get("frozen_plan_canonical_sha256") != ORIGINAL_SHA or
            plan.get("planned_cases") != 126):
        _reject("R83H_FROZEN_126_CASE_PLAN_NOT_MATCHED")
    original = read_json(frozen_45_pair_audit)
    if (original.get("schema") !=
            "phm.r83e.original-native-paired-observability.v1" or
            original.get("source_role") !=
            "TRUSTED_NATIVE_TRAIN_SOURCE_PHYSICS_QA_ONLY" or
            original.get("paired_cases") != 45 or
            original.get("test_final_or_validation_traces_accessed") is not False or
            original.get("truth_available_to_model_inference") is not False or
            original.get("model_trained_or_selected") is not False):
        _reject("R83H_REQUIRES_ORIGINAL_AUTHENTIC_45_TRAIN_PAIR_QA")
    old = original.get("per_pair", [])
    if len(old) != 45 or len({r["source_pair_id"] for r in old}) != 45:
        _reject("R83H_DUPLICATED_OR_MISSING_SOURCE_PAIRS")
    detail = []
    for previous in sorted(old, key=lambda x: x["source_pair_id"]):
        pair = previous["source_pair_id"]
        if (not isinstance(pair, str) or not pair.startswith("train-") or
                "/" in pair or "\\" in pair or ".." in pair or
                previous["fault_class"] not in MECHANISMS):
            _reject("R83H_FORBIDDEN_NONTRAIN_SOURCE_PAIR")
        twin = {}
        for role in ("fault", "nominal"):
            basedir = root / "cases" / f"{SOURCE_PREFIX}{pair}-{role}"
            contract = read_json(basedir / "astrograph/dataset_contract.json")
            task = read_json(basedir / "task_spec.json")
            if (contract["split"] != "train" or
                    contract["pairing"]["pair_id"] != pair or
                    contract["pairing"]["case_role"] != role or
                    contract["simulation"]["backend"] != "basilisk" or
                    contract["simulation"]["bsk_version"] != "2.11.0+satfix1"):
                _reject("R83H_ORIGINAL_NATIVE_SOURCE_CLAIM_MISMATCH")
            wheel = int(previous["physical_wheel_index_audit_only"])
            if not 0 <= wheel < 3:
                _reject("R83H_INVALID_FAULT_TARGET_INDEX")
            trace = basedir / "trace.csv"
            expected = previous[
                "original_fault_trace_sha256" if role == "fault"
                else "original_nominal_trace_sha256"]
            if _sha(trace) != expected:
                _reject("R83H_ORIGINAL_TRAIN_TRACE_HASH_CHANGED")
            twin[role] = (contract, task, _rows(trace, wheel))
        contract, task, faults = twin["fault"]
        _, n_task, normal = twin["nominal"]
        if (task["parameters"] != n_task["parameters"] or
                task["simulation"]["seed"] != n_task["simulation"]["seed"] or
                contract["ground_truth"]["class_id"] != previous["fault_class"] or
                abs(float(contract["ground_truth"]["onset_time_s"])-
                    float(previous["fault_onset_s_audit_only"]))>1.e-8):
            _reject("R83H_ORIGINAL_TWIN_OR_ONSET_DRIFT")
        try:
            regime = REGIMES[float(task["parameters"]["inclination_deg"])]
        except KeyError as exc:
            raise ValueError("R83H_NONPREREGISTERED_SOURCE_REGIME") from exc
        wheel = previous["physical_wheel_index_audit_only"]
        speed = f"adcs.rw.speed_rad_s_{wheel}"
        command = f"adcs.rw.command_torque_nm_{wheel}"
        applied = f"adcs.control.applied_torque_nm_{wheel}"
        rated = f"adcs.rw.effective_max_torque_nm_{wheel}"
        onset = float(previous["fault_onset_s_audit_only"])
        post = [(a,b) for a,b in zip(faults,normal) if a["time_s"]>=onset]
        pre = [(a,b) for a,b in zip(faults,normal) if a["time_s"]<onset]
        if len(pre)<2 or len(post)<2:
            _reject("R83H_SOURCE_ONSET_OUTSIDE_OBSERVED_TRACE")
        pre_speed = max(abs(a[speed]-b[speed]) for a,b in pre)
        post_speed = max(abs(a[speed]-b[speed]) for a,b in post)
        if (abs(pre_speed-previous["max_pre_onset_abs_speed_delta_rad_s"])>1.e-8 or
                abs(post_speed-previous["max_post_onset_abs_speed_delta_rad_s"])>1.e-8):
            _reject("R83H_BASELINE_SOURCE_OBSERVABILITY_RECEIPT_CHANGED")
        ratings = [b[rated] for _,b in post]
        if min(ratings) <= 0:
            _reject("R83H_NOMINAL_RATED_RW_LIMIT_NONPOSITIVE")
        absratio = [abs(b[command])/b[rated] for _,b in post]
        detail.append({
            "source_pair_id": pair,
            "fault_class": previous["fault_class"],
            "original_source_regime": regime,
            "source_target_wheel_PHYSICS_QA_ONLY": wheel,
            "fault_onset_s_PHYSICS_QA_ONLY": onset,
            "max_pre_onset_abs_target_speed_delta_rad_s": pre_speed,
            "post_max_abs_target_speed_delta_rad_s": post_speed,
            "nominal_post_mean_abs_command_torque_nm": statistics.fmean(
                abs(b[command]) for _,b in post),
            "nominal_post_mean_abs_command_to_rating": statistics.fmean(absratio),
            "nominal_post_max_abs_command_to_rating": max(absratio),
            "fault_post_mean_abs_command_applied_gap_nm": statistics.fmean(
                abs(a[command]-a[applied]) for a,_ in post),
            "nominal_post_mean_abs_command_applied_gap_nm": statistics.fmean(
                abs(b[command]-b[applied]) for _,b in post),
            "post_max_abs_paired_command_difference_nm": max(
                abs(a[command]-b[command]) for a,b in post),
            "post_max_abs_paired_applied_torque_difference_nm": max(
                abs(a[applied]-b[applied]) for a,b in post),
            "source_trace_sha256_fault": previous["original_fault_trace_sha256"],
            "source_trace_sha256_nominal": previous["original_nominal_trace_sha256"],
            "model_inference_fields_exported": False,
        })
    groups = defaultdict(list)
    by_class = defaultdict(list)
    for rec in detail:
        groups[(rec["fault_class"],rec["original_source_regime"])].append(rec)
        by_class[rec["fault_class"]].append(rec)
    if (len(detail)!=45 or len(groups)!=9 or
            any(len(v)!=5 for v in groups.values()) or
            any(len(v)!=15 for v in by_class.values())):
        _reject("R83H_ORIGINAL_FAULT_REGIME_QUOTAS_CHANGED")
    summary = {
        "schema": SCHEMA,
        "source_kind": "R83F_126_GENUINE_BASILISK_ORIGINAL_TRAIN_TRACE_REVIEW",
        "source_role": "TRUSTED_PHYSICS_QA_NOT_MODEL_INFERENCE",
        "original_plan_sha256": ORIGINAL_SHA,
        "original_train_pair_audit_sha256": _sha(Path(frozen_45_pair_audit)),
        "original_case_traces_read": 90,
        "original_source_train_independent_groups": 45,
        "source_validation_test_or_final_files_read": False,
        "original_train_data_or_original_r83g_scores_modified": False,
        "source_internal_applied_torque_used_for_PHYSICS_QA_ONLY": True,
        "new_diagnostic_model_trained": False,
        "new_diagnostic_eligibility_claim": False,
        "independent_qualification": False,
        "per_fault_mechanism": {
            name: _summarize(by_class[name]) for name in MECHANISMS
        },
        "per_fault_regime": {
            fault+"|"+regime: _summarize(groups[(fault,regime)])
            for fault in MECHANISMS for regime in REGIMES.values()
        },
        "per_original_pair": detail,
    }
    write_json(output, summary)
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("source-root", "frozen-45-pair-audit", "frozen-plan-audit", "output"):
        p.add_argument("--"+key, type=Path, required=True)
    args=vars(p.parse_args(argv))
    report=audit_native_train_excitation(**{k.replace("-","_"):v for k,v in args.items()})
    for fault,table in report["per_fault_mechanism"].items():
        print("R83H_REAL_TRAIN_PHYSICS_EXCITATION", fault,
              "median_command_to_rated",
              table["nominal_post_mean_abs_command_to_rating"]["median"],
              "median_pair_speed_effect",
              table["post_max_abs_target_speed_delta_rad_s"]["median"])
    return 0


if __name__=="__main__":
    raise SystemExit(main())
