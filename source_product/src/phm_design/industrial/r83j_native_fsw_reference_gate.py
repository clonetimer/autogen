"""R83j: read-only attestation of a GENUINE Basilisk FSW reference→RW experiment.

This NEVER trains a diagnostic model, reads RW fault/VAL/FINAL telemetry,
changes the R83f/R83g refusal, or generates Python-proxy physical data.
Only exact sealed upstream native physical source bytes are accepted.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import json
import math

SCHEMA = "phm.r83j.original-native-fsw-physics-proof-gate.v1"
UPSTREAM_ORIGINAL_ARTIFACT_JSON_SHA256 = (
    "c93773efed43b8c3a8e68290df53cba966dec9b6e74bf15f6655e0b197730ae8"
)
UPSTREAM_REAL_RUN = "https://github.com/clonetimer/SatAgentSimulation/actions/runs/40111820380"
UPSTREAM_CODE_COMMIT = "4c70d2c0fe3ca9165d794c3012509d7bb2646b93"
UPSTREAM_SOURCE_SCHEMA = "sat-sim.r83j.native-fsw-bounded-reference-excitation-lab.v1"
PHYSICS_NATIVE_MODULE_CHAIN = [
    "inertial3D", "attTrackingError", "mrpFeedback",
    "rwMotorTorque", "reactionWheelStateEffector", "spacecraft"
]


def _fail(code):
    raise ValueError(code)


def _number(value,code):
    if type(value) not in (float,int) or not math.isfinite(value):
        _fail(code)
    return float(value)


def review_original_native_fsw_proof(source_file: str | Path) -> dict:
    p=Path(source_file)
    if not p.is_file():
        raise FileNotFoundError("R83J_NO_IMMUTABLE_REAL_BASILISK_EVIDENCE")
    actual=hashlib.sha256(p.read_bytes()).hexdigest()
    if actual != UPSTREAM_ORIGINAL_ARTIFACT_JSON_SHA256:
        _fail("R83J_WRONG_OR_TAMPERED_NATIVE_FSW_PHYSICS_SOURCE")
    original=json.loads(p.read_text(encoding="utf8"))
    if (original.get("schema")!=UPSTREAM_SOURCE_SCHEMA or
            original.get("source_physics_real_native") is not True or
            original.get("independent_hardware_safety_certified") is not False or
            original.get("fault_diagnosis_generalization_measured") is not False or
            original.get("fault_injections_performed") is not False or
            original.get("no_automatic_stable_promotion") is not True or
            original.get("limits_within_provisional_lab_envelope") is not True):
        _fail("R83J_UPSTREAM_PHYSICS_PROOF_OR_CLAIM_POLICY_INVALID")
    arms=original.get("arms")
    if not isinstance(arms,dict) or set(arms)!={"reference","scheduled"}:
        _fail("R83J_ORIGINAL_NATIVE_REFERENCE_ARMS_MISSING")
    original_rows={}
    for name in ("reference","scheduled"):
        arm=arms[name]
        if (arm.get("schema")!=UPSTREAM_SOURCE_SCHEMA or
                arm.get("arm")!=name or
                arm.get("execution_engine")!="BASILISK_NATIVE_SC_RW_FSW_NO_PROXY" or
                arm.get("native_module_chain")!=PHYSICS_NATIVE_MODULE_CHAIN or
                arm.get("native_scheduler_was_advanced") is not True or
                arm.get("reference_is_guidance_not_wheel_motor_telecommand") is not True or
                arm.get("assurance")!="RESEARCH_ONLY_NOT_FLIGHT_CERTIFIED"):
            _fail("R83J_NONNATIVE_OR_FABRICATED_ARM")
        rows=arm.get("rows")
        if not isinstance(rows,list) or len(rows)!=31:
            _fail("R83J_ORIGINAL_REFERENCE_SAMPLE_COUNT_CHANGED")
        required={
            "time_s",
            *(f"fsw.native_reference_mrp_{j}" for j in range(3)),
            *(f"fsw.body_request_nm_{j}" for j in range(3)),
            *(f"rw.native_allocator_request_nm_{j}" for j in range(3)),
            *(f"rw.native_speed_rad_s_{j}" for j in range(3)),
            *(f"hub.sigma_bn_{j}" for j in range(3)),
            *(f"hub.omega_bn_b_rad_s_{j}" for j in range(3)),
        }
        for i,row in enumerate(rows):
            if set(row)!=required:
                _fail("R83J_ORIGINAL_NATIVE_SAMPLE_COLUMNS_CHANGED")
            values={k:_number(v,"R83J_NONFINITE_NATIVE_MESSAGE") for k,v in row.items()}
            if abs(values["time_s"]-i*.2)>1.e-8:
                _fail("R83J_NATIVE_SCHEDULER_TIMELINE_CHANGED")
        original_rows[name]=rows
    a,b=original_rows["reference"],original_rows["scheduled"]
    speed=lambda x:x["rw.native_speed_rad_s_0"]
    motor=lambda x:x["rw.native_allocator_request_nm_0"]
    ref=lambda x:x["fsw.native_reference_mrp_0"]
    pre=[(x,y) for x,y in zip(a,b) if x["time_s"]<=1.8+1.e-8]
    post=[(x,y) for x,y in zip(a,b) if 2.2-1e-8<=x["time_s"]<=3.8+1e-8]
    if len(pre)!=10 or len(post)!=9:
        _fail("R83J_PAIRED_NATIVE_EVENT_WINDOWS_CHANGED")
    observed={
        "pre_event_speed_abs_difference":max(abs(speed(x)-speed(y)) for x,y in pre),
        "post_event_max_rw_speed_difference_rad_s":max(abs(speed(x)-speed(y)) for x,y in post),
        "post_event_max_motor_command_difference_nm":max(abs(motor(x)-motor(y)) for x,y in post),
        "post_event_max_native_reference_mrp_difference":max(abs(ref(x)-ref(y)) for x,y in post),
        "max_motor_allocator_request_nm_all_rw":max(
            abs(row[f"rw.native_allocator_request_nm_{j}"])
            for rows in original_rows.values() for row in rows for j in range(3)),
        "max_wheel_speed_rad_s_all_rw":max(
            abs(row[f"rw.native_speed_rad_s_{j}"])
            for rows in original_rows.values() for row in rows for j in range(3)),
        "max_body_rate_rad_s_all_axes":max(
            abs(row[f"hub.omega_bn_b_rad_s_{j}"])
            for rows in original_rows.values() for row in rows for j in range(3)),
    }
    for metric,value in observed.items():
        if not math.isclose(_number(original.get(metric),
                                    "R83J_NONNUMERIC_ORIGINAL_METRIC"),
                            value,rel_tol=1e-10,abs_tol=1e-10):
            _fail("R83J_RECORDED_NATIVE_METRIC_NOT_MATCHING_SOURCE:"+metric)
    if (observed["pre_event_speed_abs_difference"]>1e-8 or
            observed["post_event_max_rw_speed_difference_rad_s"]<=1e-8 or
            observed["post_event_max_motor_command_difference_nm"]<=1e-8 or
            observed["post_event_max_native_reference_mrp_difference"]<=1e-5):
        _fail("R83J_NO_NATIVE_FSW_REFERENCE_TO_WHEEL_PHYSICS_COUPLING")
    actual_start=[y["time_s"] for y in b
                  if abs(ref(y))>1e-7]
    if not actual_start or abs(actual_start[0]-2.)>1e-8:
        _fail("R83J_GUIDANCE_ACTUAL_EFFECTIVE_TIME_NOT_RECORDED")
    if observed["max_motor_allocator_request_nm_all_rw"]>0.0200000001 or \
            observed["max_wheel_speed_rad_s_all_rw"]>900.:
        _fail("R83J_PROVISIONAL_RESEARCH_OBSERVATION_BOUND_EXCEEDED")

    return {
        "schema":SCHEMA,
        "upstream_real_basilisk_actions_url":UPSTREAM_REAL_RUN,
        "upstream_producer_commit_sha":UPSTREAM_CODE_COMMIT,
        "upstream_actual_json_sha256":actual,
        "original_physically_simulated_arms":2,
        "original_native_samples_per_arm":31,
        "actual_native_fsw_reference_change_begins_s":actual_start[0],
        "native_fsw_reference_to_rw_physical_response_observed":True,
        "native_reference_guidance_not_motor_torque_telecommand":True,
        "max_actual_paired_rw_speed_difference_rad_s":observed[
            "post_event_max_rw_speed_difference_rad_s"],
        "max_native_rw_allocator_requested_torque_nm":observed[
            "max_motor_allocator_request_nm_all_rw"],
        "provisional_lab_envelope_passed":True,
        "independent_hardware_safety_certified":False,
        "original_r83f_r83g_diagnostic_refusals_unchanged":True,
        "no_fault_mechanism_tested_here":True,
        "no_training_or_model_selection_performed":True,
        "validation_test_final_telemetry_not_opened":True,
        "autogen_algorithm_eligible_from_this_evidence":False,
        "stable_release_eligible_from_this_evidence":False,
    }


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-file",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args(argv)
    if a.output.exists():
        raise FileExistsError("R83J_ATTESTATION_ALREADY_EXISTS")
    report=review_original_native_fsw_proof(a.source_file)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n",encoding="utf8")
    print("R83J_BASILISK_FSW_PHYSICS_PROOF",report["native_fsw_reference_to_rw_physical_response_observed"],
          "AND_DIAGNOSTIC_ELIGIBILITY",report["autogen_algorithm_eligible_from_this_evidence"])
    return 0


if __name__=="__main__":
    raise SystemExit(main())
