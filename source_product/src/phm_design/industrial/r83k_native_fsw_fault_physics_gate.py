"""R83k original native Basilisk RW torque-loss × FSW pulse proof, READ ONLY.

This original-source attestation is NOT fault classification training, not
sensor feature construction, and not independent flight hardware validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

SCHEMA="phm.r83k.original-basilisk-faulted-rw-physics-evidence.v1"
ORIGINAL_NATIVE_PHYSICS_JSON_SHA256="4b6fd9b66e48b9be7ff9247eb8d83a36cd0d3977e7616df091a087671e42ce56"
ORIGINAL_NATIVE_ACTIONS="https://github.com/clonetimer/SatAgentSimulation/actions/runs/40114200932"
ORIGINAL_SOURCE_COMMIT="74e680e0f86f84707fd69116e7e14c1601ba9343"
SOURCE_SCHEMA="sat-sim.r83k.native-fsw-faulted-rw-capability-pilot.v1"
ARMS=("zero_healthy","zero_derated","step_healthy","step_derated")
CAP_NOMINAL=.02
CAP_FAULT=.0005
ONSET_S=2.4
STEP_S=.2


def _deny(message):
    raise ValueError(message)


def _num(x):
    if type(x) not in (int,float) or not math.isfinite(x):
        _deny("R83K_NONFINITE_OR_NONNUMERIC_ORIGINAL_RW_MESSAGE")
    return float(x)


def attest_original_faulted_rw_physics(path: str | Path) -> dict:
    p=Path(path)
    if not p.is_file():
        raise FileNotFoundError("R83K_ORIGINAL_GENUINE_BASILISK_EVIDENCE_MISSING")
    source_sha=hashlib.sha256(p.read_bytes()).hexdigest()
    if source_sha!=ORIGINAL_NATIVE_PHYSICS_JSON_SHA256:
        _deny("R83K_FROZEN_ORIGINAL_REAL_BASILISK_SOURCE_CHANGED")
    original=json.loads(p.read_text(encoding="utf8"))
    if (original.get("schema")!=SOURCE_SCHEMA or
            original.get("actual_engine")!="GENUINE_BASILISK_RWMOTORTORQUE_AND_REACTIONWHEELSTATEEFFECTOR" or
            original.get("source_native_trials")!=4 or
            original.get("physical_native_rw_capacity_write_readback_done") is not True or
            original.get("reference_and_fault_are_separate_native_interventions") is not True or
            original.get("reference_nonzero_change_detected") is not True or
            original.get("provisional_research_command_and_speed_bounds_passed") is not True or
            original.get("independent_hardware_safety_certified") is not False or
            original.get("healthy_vs_fault_model_trained") is not False or
            original.get("diagnostic_separability_proven") is not False or
            original.get("held_out_validation_test_final_data_used") is not False or
            original.get("stable_release_eligible") is not False):
        _deny("R83K_INVALID_REAL_SOURCE_OR_UNSUPPORTED_DIAGNOSTIC_CLAIM")
    cases=original.get("physical_source_arms")
    if not isinstance(cases,dict) or set(cases)!=set(ARMS):
        _deny("R83K_FROZEN_FOUR_NATIVE_PHYSICAL_ARMS_REQUIRED")
    required={
        "time_s",
        *(f"fsw.native_reference_mrp_{j}" for j in range(3)),
        *(f"fsw.body_request_nm_{j}" for j in range(3)),
        *(f"rw.native_allocator_request_nm_{j}" for j in range(3)),
        *(f"rw.native_speed_rad_s_{j}" for j in range(3)),
        *(f"hub.sigma_bn_{j}" for j in range(3)),
        *(f"hub.omega_bn_b_rad_s_{j}" for j in range(3)),
        "rw.native_target_applied_torque_nm_QA_ONLY",
    }
    real_rows={}
    for name in ARMS:
        arm=cases[name]
        fault=name.endswith("_derated")
        if (arm.get("schema")!=SOURCE_SCHEMA or
                arm.get("arm")!=name or
                arm.get("execution_engine")!="BASILISK_NATIVE_SC_RW_FSW_NO_PROXY" or
                arm.get("native_scheduler_was_advanced") is not True or
                arm.get("reference_is_guidance_not_wheel_motor_telecommand") is not True or
                arm.get("true_native_actuator_capacity_derated") is not fault or
                arm.get("model_inference_uses_truth") is not False or
                arm.get("fault_onset_paused_scheduler_s_QA_ONLY")!=ONSET_S or
                arm.get("original_native_rw_hardware_limit_nm")!=CAP_NOMINAL or
                arm.get("fault_native_rw_capacity_after_nm_QA_ONLY")!=
                (CAP_FAULT if fault else CAP_NOMINAL)):
            _deny("R83K_NATIVE_RW_ORIGIN_OR_PHYSICAL_FAULT_CONTRACT_CHANGED")
        events=[e for e in arm.get("reference_setpoint_event_receipts",[]) if "rw_effector_c_backed_readback_nm" in e]
        if (len(events)!=1 or
                events[0].get("fault_scheduler_pause_s")!=ONSET_S or
                events[0].get("physical_fault_target_rw_index_QA_ONLY")!=0 or
                events[0].get("fault_actual_native_rw_max_torque_before_nm")!=CAP_NOMINAL or
                events[0].get("fault_actual_native_rw_max_torque_after_nm")!=
                (CAP_FAULT if fault else CAP_NOMINAL) or
                events[0].get("rw_effector_c_backed_readback_nm")!=
                (CAP_FAULT if fault else CAP_NOMINAL) or
                events[0].get("real_native_physical_change_not_feature_edit") is not True):
            _deny("R83K_ACTUAL_C_BACKED_CAPABILITY_READBACK_CHANGED")
        rows=arm.get("rows")
        if not isinstance(rows,list) or len(rows)!=31:
            _deny("R83K_NATIVE_RW_SAMPLE_COUNT_CHANGED")
        for i,row in enumerate(rows):
            if set(row)!=required:
                _deny("R83K_NATIVE_FSW_ACTUATOR_MESSAGE_COLUMNS_CHANGED")
            for key,value in row.items():
                _num(value)
            if abs(row["time_s"]-i*STEP_S)>1e-8:
                _deny("R83K_NATIVE_SCHEDULER_SAMPLE_TIMELINE_CHANGED")
        if fault:
            post=[r for r in rows if r["time_s"]>=ONSET_S+STEP_S-1e-8]
            if max(abs(r["rw.native_target_applied_torque_nm_QA_ONLY"]) for r in post)>CAP_FAULT+1e-9:
                _deny("R83K_SOURCE_CLAIMED_RW_FAULT_NOT_PHYSICALLY_LIMITED")
        real_rows[name]=rows
    results={}
    for label in ("zero","step"):
        good,bad=real_rows[label+"_healthy"],real_rows[label+"_derated"]
        before=[(a,b) for a,b in zip(good,bad) if a["time_s"]<=ONSET_S+1e-8]
        after=[(a,b) for a,b in zip(good,bad) if a["time_s"]>=ONSET_S+STEP_S-1e-8]
        sp="rw.native_speed_rad_s_0"
        applied="rw.native_target_applied_torque_nm_QA_ONLY"
        pre=max(abs(a[sp]-b[sp]) for a,b in before)
        post_speed=max(abs(a[sp]-b[sp]) for a,b in after)
        post_applied=max(abs(a[applied]-b[applied]) for a,b in after)
        if any(any(abs(a[f"fsw.native_reference_mrp_{j}"]-
                       b[f"fsw.native_reference_mrp_{j}"])>1e-12
                   for j in range(3)) for a,b in zip(good,bad)):
            _deny("R83K_FAULT_AND_NOMINAL_SOURCE_REFERENCE_NOT_MATCHED")
        if pre>1e-8:
            _deny("R83K_FAULT_APPEARED_BEFORE_NATIVE_PHYSICAL_INJECTION")
        results[label]={
            "pre_onset_max_native_target_speed_difference_rad_s":pre,
            "post_onset_max_native_target_speed_difference_rad_s":post_speed,
            "post_onset_max_native_target_applied_torque_difference_nm":post_applied,
            "fault_native_applied_torque_max_abs_nm":max(
                abs(r[applied]) for r in bad if r["time_s"]>=ONSET_S+STEP_S-1e-8),
        }
        raw=original["physical_pair_outcomes"][label]
        for key,value in results[label].items():
            if not math.isclose(_num(raw.get(key)),value,rel_tol=1e-9,abs_tol=1e-10):
                _deny("R83K_ORIGINAL_STATED_PHYSICS_METRIC_NOT_IN_NATIVE_SAMPLES:"+key)
    max_req=max(abs(r[f"rw.native_allocator_request_nm_{j}"])
                for rows in real_rows.values() for r in rows for j in range(3))
    max_speed=max(abs(r[f"rw.native_speed_rad_s_{j}"])
                  for rows in real_rows.values() for r in rows for j in range(3))
    for source_key,expected in (
        ("max_requested_native_wheel_torque_nm",max_req),
        ("max_native_wheel_speed_rad_s",max_speed)):
        if not math.isclose(_num(original.get(source_key)),expected,rel_tol=1e-9,abs_tol=1e-10):
            _deny("R83K_ORIGINAL_ALLOCATION_OR_WHEEL_SPEED_METRIC_CHANGED")
    if max_req>CAP_NOMINAL+1e-10 or max_speed>900.:
        _deny("R83K_RESEARCH_OBSERVATION_BOUNDS_EXCEEDED")
    observed=results["step"]
    zero=results["zero"]
    if (observed["post_onset_max_native_target_applied_torque_difference_nm"]<=1e-8 or
            observed["post_onset_max_native_target_speed_difference_rad_s"]<=1e-8 or
            zero["post_onset_max_native_target_speed_difference_rad_s"]>1e-8):
        _deny("R83K_ORIGINAL_NATIVE_PHYSICAL_EXCITATION_CONTRAST_NOT_FOUND")
    return {
        "schema":SCHEMA,
        "original_upstream_actions_run":ORIGINAL_NATIVE_ACTIONS,
        "original_native_source_commit":ORIGINAL_SOURCE_COMMIT,
        "original_json_sha256":source_sha,
        "physically_executed_native_cases":4,
        "original_samples_per_case":31,
        "actual_original_native_rw_derating_before_nm":CAP_NOMINAL,
        "actual_original_native_rw_derating_after_nm":CAP_FAULT,
        "true_rw_max_torque_physical_state_verified":True,
        "paired_no_command_fault_speed_effect_rad_s":
            zero["post_onset_max_native_target_speed_difference_rad_s"],
        "paired_with_scheduled_reference_fault_speed_effect_rad_s":
            observed["post_onset_max_native_target_speed_difference_rad_s"],
        "paired_with_reference_native_actual_applied_torque_effect_nm":
            observed["post_onset_max_native_target_applied_torque_difference_nm"],
        "original_native_peak_all_rw_allocator_command_nm":max_req,
        "provisional_scheduled_guidance_lab_bounds_passed":True,
        "physics_excitation_dependent_fault_observability_demonstrated":True,
        "trained_fault_classifier_eligible":False,
        "independent_hardware_safety_certified":False,
        "heldout_validation_test_final_telemetry_accessed":False,
        "original_r83f_r83g_no_eligible_design_unchanged":True,
        "automatic_stable_promotion_allowed":False,
    }


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-file",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    a=p.parse_args(argv)
    if a.output.exists():
        raise FileExistsError("R83K_ORIGINAL_PHYSICS_READONLY_ATTESTATION_EXISTS")
    result=attest_original_faulted_rw_physics(a.source_file)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n",encoding="utf8")
    print("R83K_REAL_BASILISK_PHYSICS_VERIFIED",result[
        "paired_with_scheduled_reference_fault_speed_effect_rad_s"],
        "diagnostic_eligibility",result["trained_fault_classifier_eligible"])
    return 0


if __name__=="__main__":
    raise SystemExit(main())
