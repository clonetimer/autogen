"""R83e: descriptive paired RW physical observability, TRAIN physics only.

Reads the pinned SatAgentSimulation **original native source** artifacts.
Uses fault event onset and target wheel index for trusted physical QA ONLY,
never for AutoDesign observation features, training, model selection or
inference. No self-organizing model is trained or validated here.
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

from ..io import read_json, safe_child, sha_file, write_json
from .native_campaign_gate import review_native_rw_train_campaign

SCHEMA = "phm.r83e.original-native-paired-observability.v1"
SENSOR_FIELDS = tuple(
    [f"adcs.rw.speed_rad_s_{i}" for i in range(3)]
    + [f"adcs.rw.command_torque_nm_{i}" for i in range(3)])
FAULTS = ("ADCS_RW_FRICTION_INCREASE", "ADCS_RW_JAM",
          "ADCS_RW_TORQUE_AUTHORITY_LOSS")


def _source_rows(case_root: Path, contract: dict):
    source = safe_child(case_root, contract["telemetry"]["source_trace_file"])
    rows = []
    with source.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"time_s", *SENSOR_FIELDS}.issubset(reader.fieldnames or []):
            raise ValueError("R83E_ORIGINAL_SENSOR_CHANNELS_INCOMPLETE")
        for data in reader:
            try:
                row = {name: float(data[name])
                       for name in ("time_s", *SENSOR_FIELDS)}
            except (TypeError, ValueError, KeyError) as exc:
                raise ValueError("R83E_NONNUMERIC_SOURCE_TRACE") from exc
            if not all(math.isfinite(v) for v in row.values()):
                raise ValueError("R83E_NONFINITE_ORIGINAL_TRACE")
            rows.append(row)
    if len(rows) < 16:
        raise ValueError("R83E_TOO_FEW_ORIGINAL_TELEMETRY_ROWS")
    return rows, sha_file(source)


def _fault_metadata(spec: dict):
    modifiers = spec.get("modifiers", {})
    events = (modifiers.get("faults") or []) + (modifiers.get("degradations") or [])
    if len(events) != 1:
        raise ValueError("R83E_EXPECTS_SINGLE_NATIVE_FAULT_EVENT")
    event = events[0]
    onset = event.get("onset_time_s")
    wheel_index = event.get("parameters", {}).get("wheel_index")
    if (not isinstance(onset, (float, int)) or not math.isfinite(onset) or
            not 0 < onset < 60.0 or
            type(wheel_index) is not int or not 0 <= wheel_index <= 2):
        raise ValueError("R83E_SOURCE_ONSET_OR_TARGET_INVALID")
    return float(onset), wheel_index


def summarize_native_paired_observability(campaign: str | Path,
                                           output: str | Path) -> dict:
    """Inspect TRAIN physics, refusing malformed twins before writing receipt.

    Event onsets and injection truth are read in this offline physics audit
    only. The result is NOT an AutoDesign input, comparison rank, feature
    registry or source for candidate selection. There are no thresholds
    optimized from test or validation labels.
    """
    output = Path(output)
    if output.exists():
        raise FileExistsError("R83E_OBSERVABILITY_REPORT_ALREADY_EXISTS")
    root = Path(campaign)
    gate = review_native_rw_train_campaign(root)
    cases = defaultdict(dict)
    for cpath in sorted((root / "cases").glob("*/astrograph/dataset_contract.json")):
        contract = read_json(cpath)
        if contract.get("split") != "train":
            continue
        pair = contract["pairing"]["pair_id"]
        role = contract["pairing"]["case_role"]
        if role in cases[pair]:
            raise ValueError("R83E_DUPLICATE_TRAIN_PAIR_ROLE")
        cases[pair][role] = (cpath.parent.parent, contract)
    if len(cases) != gate["train_pairs"]:
        raise ValueError("R83E_NATIVE_PAIR_RECEIPT_INCONSISTENT")
    summary = []
    for pair, twins in sorted(cases.items()):
        if set(twins) != {"fault", "nominal"}:
            raise ValueError("R83E_MISSING_PHYSICAL_TWIN")
        fault_root, fault = twins["fault"]
        nominal_root, nominal = twins["nominal"]
        fault_class = fault["ground_truth"]["class_id"]
        if fault_class not in FAULTS:
            raise ValueError("R83E_UNSUPPORTED_RW_FAULT_CLASS")
        fault_task = read_json(fault_root / "task_spec.json")
        nominal_task = read_json(nominal_root / "task_spec.json")
        if (fault_task["parameters"] != nominal_task["parameters"] or
                fault_task["simulation"]["seed"] != nominal_task["simulation"]["seed"]):
            raise ValueError("R83E_UNPAIRED_PHYSICAL_SOURCE_PARAMETERS")
        onset, wheel = _fault_metadata(fault_task)
        left, left_sha = _source_rows(fault_root, fault)
        right, right_sha = _source_rows(nominal_root, nominal)
        if len(left) != len(right):
            raise ValueError("R83E_DIFFERENT_PAIRED_CASE_SAMPLE_COUNT")
        for l, r in zip(left, right):
            if not math.isclose(l["time_s"], r["time_s"],
                                abs_tol=1e-7, rel_tol=1e-8):
                raise ValueError("R83E_PAIRED_SAMPLE_TIMESTAMPS_MISMATCH")
        before = [(l, r) for l, r in zip(left, right) if l["time_s"] < onset]
        after = [(l, r) for l, r in zip(left, right) if l["time_s"] >= onset]
        if len(before) < 2 or len(after) < 2:
            raise ValueError("R83E_ONSET_OUTSIDE_OBSERVED_TRAJECTORY")
        speed = f"adcs.rw.speed_rad_s_{wheel}"
        command = f"adcs.rw.command_torque_nm_{wheel}"

        def maximum_abs_diff(name, rows):
            return max(abs(a[name] - b[name]) for a, b in rows)

        delta = [a[speed] - b[speed] for a, b in after]
        summary.append({
            "source_pair_id": pair,
            "fault_class": fault_class,
            "physical_wheel_index_audit_only": wheel,
            "fault_onset_s_audit_only": onset,
            "samples_per_case": len(left),
            "samples_before_onset": len(before),
            "samples_at_or_after_onset": len(after),
            "max_pre_onset_abs_speed_delta_rad_s": maximum_abs_diff(speed, before),
            "max_post_onset_abs_speed_delta_rad_s": maximum_abs_diff(speed, after),
            "post_onset_speed_rms_delta_rad_s": math.sqrt(
                sum(x*x for x in delta) / len(delta)),
            "final_speed_delta_rad_s": left[-1][speed] - right[-1][speed],
            "max_post_onset_abs_command_delta_nm": maximum_abs_diff(command, after),
            "original_fault_trace_sha256": left_sha,
            "original_nominal_trace_sha256": right_sha,
            "independent_diagnostic_generalization_evidence": False,
        })
    result = {
        "schema": SCHEMA,
        "source_role": "TRUSTED_NATIVE_TRAIN_SOURCE_PHYSICS_QA_ONLY",
        "native_basilisk_candidate_checked": True,
        "source_bsk_version_claimed": gate["source_bsk_version_claimed"],
        "paired_cases": len(summary),
        "actual_basilisk_executed_here": False,
        "model_trained_or_selected": False,
        "truth_onset_and_wheel_used_for_physics_audit_only": True,
        "truth_available_to_model_inference": False,
        "test_final_or_validation_traces_accessed": False,
        "independent_qualification": False,
        "per_pair": summary,
    }
    write_json(output, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = summarize_native_paired_observability(args.campaign, args.output)
    for item in result["per_pair"]:
        print("R83E_NATIVE_PAIR_PHYSICS_DESCRIPTIVE_ONLY",
              item["fault_class"],
              "max_after_speed_delta_rad_s=",
              round(item["max_post_onset_abs_speed_delta_rad_s"], 6))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
