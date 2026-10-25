"""R83n: sealed Basilisk RW-speed *measurement-limit sensitivity* (PHYSICS QA).

Practical diagnosability is NOT demonstrated: comparing a fault trajectory
against its perfectly aligned healthy counterfactual is possible only in
offline physics QA. A flight detector has no such healthy twin.

No stochastic noise is injected and no diagnostic model is trained. This
only computes SUFFICIENT source-oracle separation margins under hypothetical,
independently bounded per-observation speed error, +/- epsilon rad/s,
against the ACTUAL original 54-case R83m Basilisk source verified by SHA.
The real hardware noise/quantization/cadence remains UNKNOWN.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import json
import math
import statistics

from .r83m_original_native_guidance_threshold_gate import (
    review_original_real_native_rw_source,
    SOURCE_SCHEMA, WHEELS, INITIALS, REFERENCES,
    ORIGINAL_SHA256 as ORIGINAL_R83M_SHA256
)

SCHEMA="phm.r83n.readonly-native-wheel-speed-uncertainty-envelope.v1"
# ONLY hypothesis stress levels, NOT actual RW encoder sensor specifications.
SINGLE_SAMPLE_ABS_ERROR_LEVELS_RAD_S = (0., .0005, .001, .0025, .005, .01)
FAULT_PHYSICAL_ONSET_S=2.4
ORIGINAL_SAMPLE_S=.2
CONTRAST_LIMIT=1e-8


def _deny(message):
    raise ValueError("R83N_"+message)


def evaluate_original_native_encoder_error_envelope(
    source_file: str | Path, *, write_source_samples: bool=False
) -> dict:
    # The underlying R83m verification recomputes 54 actual C++ RW cases,
    # source original SHA, paired u_max and u_current, reference timing,
    # 27 physical comparisons, all-wheel torque/speed/rate envelopes.
    prior=review_original_real_native_rw_source(source_file)
    if (prior["actual_Basilisk_cases"]!=54 or
            prior["physical_wheel_initial_condition_groups"]!=9 or
            prior["matched_physics_comparisons"]!=27 or
            prior["protected_validation_test_final_accessed"] is not False or
            prior["diagnostic_model_eligibility_confirmed"] is not False or
            prior["original_source_json_sha256"]!=ORIGINAL_R83M_SHA256):
        _deny("MISSING_PINNED_REAL_BASILISK_SOURCE_GATE")
    if write_source_samples:
        _deny("PROHIBITED_EXPORT_OF_PRIVATE_PHYSICS_COUNTERFACTUAL")

    source=json.loads(Path(source_file).read_text(encoding="utf8"))
    if source.get("schema") != SOURCE_SCHEMA:
        _deny("ORIGINAL_NATIVE_SOURCE_SCHEMA_CHANGED")
    cases=source["physical_source_cases"]
    counts=defaultdict(list)
    starts=defaultdict(list)
    for wheel in WHEELS:
        for initial in range(len(INITIALS)):
            for ref in REFERENCES:
                prefix=f"rw{wheel}_initial{initial}_{ref}"
                # Target wheel and condition are selected strictly for
                # physical QA bookkeeping; not a deployable model input.
                healthy=cases[prefix+"_healthy"]["rows"]
                fault=cases[prefix+"_loss75"]["rows"]
                if len(healthy)!=31 or len(fault)!=31:
                    _deny("NATIVE_RECORDER_SAMPLE_COUNT_CHANGED")
                speed=f"rw.native_speed_rad_s_{wheel}"
                pre=[(a,b) for a,b in zip(healthy,fault)
                     if a["time_s"]<=FAULT_PHYSICAL_ONSET_S+1e-8]
                post=[(a,b) for a,b in zip(healthy,fault)
                      if a["time_s"]>=FAULT_PHYSICAL_ONSET_S+ORIGINAL_SAMPLE_S-1e-8]
                if len(pre)<5 or len(post)<5:
                    _deny("NATIVE_EVENT_WINDOWS_CHANGED")
                if any(abs(a[speed]-b[speed])>1e-8 for a,b in pre):
                    _deny("PRE_FAULT_WHEEL_SPEED_CONTRAST_LEAK")
                absolute_post=[abs(a[speed]-b[speed]) for a,b in post]
                peak=max(absolute_post)
                if not math.isfinite(peak):
                    _deny("NONFINITE_REAL_NATIVE_WHEEL_SPEED_CONTRAST")
                # All noise comparisons occur ONLY between ORIGINAL simulator
                # true wheel-speed states and are NOT a trained detector.
                per_error={}
                for error in SINGLE_SAMPLE_ABS_ERROR_LEVELS_RAD_S:
                    # For two independently bounded +/-error readings,
                    # difference uncertainty is <=2*error. If maximum
                    # true difference is >2*error, the paired oracular
                    # difference remains nonzero at some sample for EVERY
                    # admissible error realization. This is NOT enough for
                    # a detector without a counterfactual baseline.
                    can_resolve=peak>max(CONTRAST_LIMIT,2*error)
                    first=next((float(b["time_s"])
                                for (a,b),delta in zip(post,absolute_post)
                                if delta>max(CONTRAST_LIMIT,2*error)),None)
                    per_error[str(error)]={"pair_oracle_margin_gt_2eps":can_resolve,
                                           "first_oracle_margin_timestamp_s":first}
                    counts[(ref,error)].append(can_resolve)
                    if first is not None:
                        starts[(ref,error)].append(first)
    # Every group count is 9 source design conditions, NOT 9 independently
    # sourced spacecraft or 9 independent samples of real-world noise.
    matrix={}
    for ref in REFERENCES:
        arm={}
        for error in SINGLE_SAMPLE_ABS_ERROR_LEVELS_RAD_S:
            resolved=counts[(ref,error)]
            if len(resolved)!=9:
                _deny("ORIGINAL_MATCHED_PAIR_GROUP_ACCOUNTING_CHANGED")
            t=starts[(ref,error)]
            arm[f"{error:.4f}"]={
                "physical_condition_pairs":9,
                "paired_true_speed_contrasts_above_error_difference_bound":sum(resolved),
                "guaranteed_nonzero_counterfactual_margin_only":True,
                "first_oracle_margin_timestamp_median_s":(
                    statistics.median(t) if t else None),
                "sensor_noise_profile_measured":False,
                "actual_deployable_detector_tested":False
            }
        matrix[ref]=arm
    source_physical=prior["physical_observability_matrix"]
    for ref in REFERENCES:
        label=ref+"|loss75"
        if matrix[ref]["0.0000"][
            "paired_true_speed_contrasts_above_error_difference_bound"]!=source_physical[label][
                "wheel_speed_effect_positive_gt_1e_minus_8"]:
            _deny("PHYSICAL_SOURCE_AND_ORACLE_MARGIN_RECALC_DISAGREE")
    # Explicitly preserve the originally observed significant sources:
    # weaker pulse/no pulse zero effect; 0.9-degree source positive.
    if (matrix["zero"]["0.0000"][
            "paired_true_speed_contrasts_above_error_difference_bound"]!=0 or
            matrix["pulse05"]["0.0000"][
                "paired_true_speed_contrasts_above_error_difference_bound"]!=0 or
            matrix["pulse09"]["0.0000"][
                "paired_true_speed_contrasts_above_error_difference_bound"]!=9):
        _deny("SEALED_ORIGINAL_R83M_ACTUAL_POSITIVE_NEGATIVE_RESULTS_CHANGED")
    return {
        "schema":SCHEMA,
        "original_R83M_Basilisk_source_sha256":ORIGINAL_R83M_SHA256,
        "real_native_original_runs":54,
        "physical_wheel_initial_groups":9,
        "counterfactual_pair_cases":27,
        "native_sampling_interval_s":ORIGINAL_SAMPLE_S,
        "error_model":"HYPOTHETICAL_ABSOLUTE_BOUNDED_ERROR_PER_READING_NOT_MEASURED",
        "single_reading_abs_error_sweep_rad_s":list(SINGLE_SAMPLE_ABS_ERROR_LEVELS_RAD_S),
        "two_readings_max_difference_error_rad_s":"2*epsilon",
        "sensor_candidate":"RW_INTERNAL_NATIVE_TRUE_SPEED_NOT_VERIFIED_FLIGHT_ENCODER",
        "actual_basilisk_sensor_emulator_used":False,
        "actual_hardware_encoder_precision_known":False,
        "original_counterfactual_healthy_twin_available_on_orbit":False,
        "source_fault_target_or_applied_torque_used_as_model_input":False,
        "source_private_fault_state_retained_only_in_original_physics_QA":True,
        "protected_validation_test_final_accessed":False,
        "diagnostic_model_trained":False,
        "false_alarm_rate_or_detection_delay_measured":False,
        "physical_observability_not_equal_deployable_detectability":True,
        "error_envelope_physics_oracle_matrix":matrix,
        "original_r83f_r83g_no_eligible_design_unchanged":True,
        "independent_hardware_flight_safety_certified":False,
        "stable_release_eligible":False
    }


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-file",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("R83N_EXISTING_RESEARCH_RECEIPT_REFUSED")
    result=evaluate_original_native_encoder_error_envelope(args.source_file)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n",encoding="utf8")
    for ref,byerror in result["error_envelope_physics_oracle_matrix"].items():
        print("R83N_SEALED_ORIGINAL_COUNTERFACTUAL_ONLY",ref,byerror)
    print("R83N_NOT_SENSOR_QUALIFICATION_OR_FDIR_MODEL")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
