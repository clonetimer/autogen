"""R83m read-only original Basilisk C-backed RW physics evidence audit.

No model training, classifier prediction, inference truth features, protected
VAL/TEST/FINAL cohort access, surrogate simulation, or Stable promotion.
This audits exactly an upstream 54-case original source SHA256. The cases are
physically distinct initial conditions in ONE simulator family, not 54
independent spacecraft or a generalization benchmark.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics

SCHEMA = "phm.r83m.original-basilisk-native-guidance-threshold-audit.v1"
SOURCE_SCHEMA = "sat-sim.r83m.real-native-guidance-threshold-sweep.v1"
ORIGINAL_SHA256 = "635fd4e4cd36da39a15ba1289003ba84aaf4551b02de70c3d063ada878209502"
SOURCE_RUN = "https://github.com/clonetimer/SatAgentSimulation/actions/runs/40160788578"
SOURCE_COMMIT = "6e1d54bfef6655a29e0a6c98b135f478fa539a48"
WHEELS=(0,1,2)
INITIALS=(
    (0.,0.,0.),
    (.001,-.0005,.0007),
    (-.0008,.0007,-.0004),
)
REFERENCES=("zero","pulse05","pulse09")
ANGLES={"zero":0.,"pulse05":.5,"pulse09":.9}
CAPS={"healthy":.02,"loss75":.005}
TORQUE_REQUEST_LIMIT_NM=.02
SPEED_LIMIT_RAD_S=900.
BODY_RATE_LIMIT_RAD_S=.005
ONSET_S=2.4
DT=.2


def _fail(reason):
    raise ValueError("R83M_"+reason)


def _number(x):
    if type(x) not in (float,int) or not math.isfinite(x):
        _fail("ORIGINAL_NONFINITE_OR_NONNUMERIC_BASILISK_SAMPLE")
    return float(x)


def _needed_columns():
    return {
        "time_s","rw.native_target_applied_torque_nm_QA_ONLY",
        *(f"fsw.native_reference_mrp_{i}" for i in WHEELS),
        *(f"fsw.body_request_nm_{i}" for i in WHEELS),
        *(f"rw.native_allocator_request_nm_{i}" for i in WHEELS),
        *(f"rw.native_speed_rad_s_{i}" for i in WHEELS),
        *(f"hub.sigma_bn_{i}" for i in WHEELS),
        *(f"hub.omega_bn_b_rad_s_{i}" for i in WHEELS),
    }


def _original_cap_receipt(case, wheel, cap):
    receipts=case.get("reference_setpoint_event_receipts")
    if not isinstance(receipts,list):
        _fail("MISSING_NATIVE_CPP_ACTUATOR_CAP_EVENT")
    events=[e for e in receipts if "rw_effector_c_backed_readback_nm" in e]
    if len(events)!=1:
        _fail("UNVERIFIED_NATIVE_CAP_CHANGE")
    event=events[0]
    if (event.get("fault_scheduler_pause_s")!=ONSET_S or
            event.get("physical_fault_target_rw_index_QA_ONLY")!=wheel or
            event.get("fault_actual_native_rw_max_torque_before_nm")!=CAPS["healthy"] or
            not math.isclose(_number(event.get("fault_actual_native_rw_max_torque_after_nm")),cap,abs_tol=1e-12) or
            not math.isclose(_number(event.get("rw_effector_c_backed_readback_nm")),cap,abs_tol=1e-12) or
            event.get("real_native_physical_change_not_feature_edit") is not True):
        _fail("RW_CPP_PHYSICAL_DERATING_READBACK_MISMATCH")


def review_original_real_native_rw_source(path: str | Path) -> dict:
    source=Path(path)
    if not source.is_file():
        raise FileNotFoundError("R83M_ORIGINAL_BASILISK_54_CASE_ARTIFACT_UNAVAILABLE")
    actual_sha=hashlib.sha256(source.read_bytes()).hexdigest()
    if actual_sha!=ORIGINAL_SHA256:
        _fail("ORIGINAL_BASILISK_NATIVE_PHYSICS_SHA256_MISMATCH")
    doc=json.loads(source.read_text(encoding="utf8"))
    if (doc.get("schema")!=SOURCE_SCHEMA or
            doc.get("execution_engine")!="GENUINE_BASILISK_REACTIONWHEELSTATEEFFECTOR_AND_FSW" or
            doc.get("source_native_cases")!=54 or
            doc.get("independent_wheel_and_initial_condition_groups")!=9 or
            doc.get("healthy_source_cases")!=27 or
            doc.get("paired_fault_vs_nominal_comparisons")!=27 or
            doc.get("predeclared_guidance_amplitude_deg")!=ANGLES or
            doc.get("source_all_wheels_observed") is not True or
            doc.get("fault_derating_verified_by_real_cpp_state") is not True or
            doc.get("actual_applied_motor_torque_verified_from_rwOutMsgs") is not True or
            doc.get("fault_truth_only_for_offline_physics_QA") is not True or
            doc.get("no_protected_validation_test_final_cases_generated") is not True or
            doc.get("no_machine_learning_trained") is not True or
            doc.get("not_a_generalizable_diagnostic_benchmark") is not True or
            doc.get("not_an_independent_hardware_safety_qualification") is not True or
            doc.get("stable_release_eligible") is not False):
        _fail("GIVEN_CLAIM_NOT_SUPPORTED_BY_NATIVE_SOURCE")
    source_cases=doc.get("physical_source_cases")
    expected={f"rw{w}_initial{i}_{ref}_{cap}"
              for w in WHEELS for i in range(3)
              for ref in REFERENCES for cap in CAPS}
    if not isinstance(source_cases,dict) or set(source_cases)!=expected:
        _fail("ORIGINAL_54_NATIVE_SOURCE_ARM_COVERAGE_CHANGED")
    fields=_needed_columns()
    for wheel in WHEELS:
        for initial in range(3):
            for ref in REFERENCES:
                for mode,cap in CAPS.items():
                    key=f"rw{wheel}_initial{initial}_{ref}_{mode}"
                    case=source_cases[key]
                    # New ORIGINAL source fixes the R83l target-wheel exporter
                    # bug; NO corrupted source metadata is tolerated.
                    if (case.get("source_fault_target_index_QA_ONLY")!=wheel or
                            case.get("native_independent_condition_id")!=f"rw{wheel}_initial{initial}" or
                            case.get("native_reference_context_id")!=f"rw{wheel}_initial{initial}_{ref}"):
                        _fail("ORIGINAL_ACTUATOR_TARGET_METADATA_MISMATCH")
                    if (case.get("schema")!=SOURCE_SCHEMA or
                            case.get("arm")!=key or
                            case.get("execution_engine")!="BASILISK_NATIVE_SC_RW_FSW_NO_PROXY" or
                            case.get("native_scheduler_was_advanced") is not True or
                            case.get("source_initial_state_index")!=initial or
                            case.get("source_initial_attitude_mrp")!=list(INITIALS[initial]) or
                            case.get("native_reference_mode")!=ref or
                            case.get("native_fault_capacity_mode_QA_ONLY")!=mode or
                            case.get("true_native_actuator_capacity_derated") is not (mode!="healthy") or
                            case.get("original_native_rw_hardware_limit_nm")!=CAPS["healthy"] or
                            not math.isclose(_number(case.get("fault_native_rw_capacity_after_nm_QA_ONLY")),cap,abs_tol=1e-12) or
                            case.get("model_inference_uses_truth") is not False or
                            case.get("reference_is_guidance_not_wheel_motor_telecommand") is not True or
                            case.get("assurance")!="RESEARCH_ONLY_NOT_FLIGHT_CERTIFIED"):
                        _fail("ORIGINAL_NATIVE_ARM_IDENTITY_OR_ROLE_CHANGED")
                    _original_cap_receipt(case,wheel,cap)
                    rows=case.get("rows")
                    if not isinstance(rows,list) or len(rows)!=31:
                        _fail("ORIGINAL_SOURCE_SAMPLE_COUNT_CHANGED")
                    for idx,row in enumerate(rows):
                        if not isinstance(row,dict) or set(row)!=fields:
                            _fail("NATIVE_SAMPLE_OR_PRIVATE_TRUTH_FIELDS_CHANGED")
                        if any(not math.isfinite(_number(v)) for v in row.values()) or \
                                abs(row["time_s"]-idx*DT)>1e-8:
                            _fail("ORIGINAL_SCHEDULER_TIME_OR_FINITE_SAMPLE_CHANGED")
                    if mode!="healthy":
                        post=[r for r in rows if r["time_s"]>=ONSET_S+DT-1e-8]
                        if max(abs(r["rw.native_target_applied_torque_nm_QA_ONLY"]) for r in post)>cap+1e-9:
                            _fail("ORIGINAL_NATIVE_RW_PHYSICAL_CAP_NOT_APPLIED")

    paired=doc.get("native_physical_paired_outcomes")
    if not isinstance(paired,dict) or len(paired)!=27:
        _fail("PAIRING_OR_SAMPLE_SUPPORT_CHANGED")
    counts=defaultdict(list)
    results=[]
    for w in WHEELS:
        for i in range(3):
            for ref in REFERENCES:
                context=f"rw{w}_initial{i}_{ref}"
                healthy=source_cases[context+"_healthy"]["rows"]
                for mode in ("loss75",):
                    damaged=source_cases[context+"_"+mode]["rows"]
                    before=[(a,b) for a,b in zip(healthy,damaged)
                            if a["time_s"]<=ONSET_S+1e-8]
                    after=[(a,b) for a,b in zip(healthy,damaged)
                           if a["time_s"]>=ONSET_S+DT-1e-8]
                    if len(before)<5 or len(after)<5:
                        _fail("FAULT_ONSET_OUTSIDE_NATIVE_TIMEGRID")
                    if any(any(abs(a[f"fsw.native_reference_mrp_{j}"]-
                                   b[f"fsw.native_reference_mrp_{j}"])>1e-12
                               for j in WHEELS) for a,b in zip(healthy,damaged)):
                        _fail("NATIVE_GUIDANCE_DIFFERENT_BETWEEN_FAULT_TWINS")
                    if any(any(abs(a[f"rw.native_speed_rad_s_{j}"]-
                                   b[f"rw.native_speed_rad_s_{j}"])>1e-8
                               for j in WHEELS) for a,b in before):
                        _fail("PRE_FAULT_PHYSICAL_GROUP_LEAKAGE")
                    target=f"rw.native_speed_rad_s_{w}"
                    actual="rw.native_target_applied_torque_nm_QA_ONLY"
                    measured={
                        "pre_event_max_native_rw_speed_difference_rad_s":
                            max(abs(a[target]-b[target]) for a,b in before),
                        "post_event_max_native_rw_speed_difference_rad_s":
                            max(abs(a[target]-b[target]) for a,b in after),
                        "post_event_max_native_rw_applied_torque_difference_nm":
                            max(abs(a[actual]-b[actual]) for a,b in after),
                        "post_event_fault_native_rw_actual_max_abs_nm":
                            max(abs(b[actual]) for a,b in after)
                    }
                    original=paired.get(context+"_"+mode)
                    if (not isinstance(original,dict) or
                            original.get("context")!=context or
                            original.get("native_independent_condition_id")!=f"rw{w}_initial{i}" or
                            original.get("source_fault_target_index_QA_ONLY")!=w or
                            original.get("source_initial_state_index")!=i or
                            original.get("reference_mode")!=ref or
                            original.get("capacity_mode")!=mode):
                        _fail("PUBLISHED_PAIR_SOURCE_CONDITIONS_MISMATCH")
                    for k,value in measured.items():
                        if not math.isclose(_number(original.get(k)),value,rel_tol=1e-9,abs_tol=1e-10):
                            _fail("ORIGINAL_SOURCE_REPORTED_PHYSICS_METRIC_DRIFT")
                    if measured["post_event_fault_native_rw_actual_max_abs_nm"]>CAPS[mode]+1e-9:
                        _fail("PHYSICALLY_DERATED_RW_APPLIED_TORQUE_OVER_CAP")
                    counts[(ref,mode)].append(measured)
                    results.append((context,mode,measured))
    rows=[r for arm in source_cases.values() for r in arm["rows"]]
    peak_req=max(abs(r[f"rw.native_allocator_request_nm_{j}"]) for r in rows for j in WHEELS)
    peak_speed=max(abs(r[f"rw.native_speed_rad_s_{j}"]) for r in rows for j in WHEELS)
    peak_rate=max(abs(r[f"hub.omega_bn_b_rad_s_{j}"]) for r in rows for j in WHEELS)
    for name,value in (
        ("max_original_native_allocator_request_nm",peak_req),
        ("max_original_native_wheel_speed_rad_s",peak_speed),
        ("max_original_native_body_rate_rad_s",peak_rate),
    ):
        if not math.isclose(_number(doc.get(name)),value,rel_tol=1e-9,abs_tol=1e-10):
            _fail("PHYSICAL_BOUND_NOT_RECOMPUTABLE_FROM_NATIVE_SOURCE")
    if peak_req>TORQUE_REQUEST_LIMIT_NM+1e-10 or peak_speed>SPEED_LIMIT_RAD_S or \
            peak_rate>BODY_RATE_LIMIT_RAD_S or \
            doc.get("predeclared_provisional_request_speed_envelope_passed") is not True:
        _fail("PREDECLARED_PHYSICAL_OBSERVATION_LIMIT_FAILED")
    # Independently verify amplitude-specific requested motor torque and
    # body rates; downstream cannot accept forged top-level source numbers.
    amplitudes=doc.get("original_native_request_by_reference_mode")
    if not isinstance(amplitudes,dict) or set(amplitudes)!=set(REFERENCES):
        _fail("PREDECLARED_REFERENCE_AMPLITUDE_BREAKDOWN_CHANGED")
    for ref in REFERENCES:
        selection=[row for case in source_cases.values()
                   if case["native_reference_mode"]==ref for row in case["rows"]]
        maximum=max(abs(row[f"rw.native_allocator_request_nm_{j}"])
                    for row in selection for j in WHEELS)
        rate=max(abs(row[f"hub.omega_bn_b_rad_s_{j}"])
                 for row in selection for j in WHEELS)
        entry=amplitudes[ref]
        if (entry.get("target_reference_angle_deg")!=ANGLES[ref] or
                not math.isclose(_number(entry.get("peak_allocator_command_nm_all_rw")),maximum,rel_tol=1e-10,abs_tol=1e-12) or
                not math.isclose(_number(entry.get("max_body_rate_rad_s")),rate,rel_tol=1e-10,abs_tol=1e-12)):
            _fail("ORIGINAL_REFERENCE_SPECIFIC_COMMAND_OR_RATE_DRIFT")
    groups={}
    for context in sorted(counts):
        cases=counts[context]
        if len(cases)!=9:
            _fail("REPEATED_INDEPENDENT_PHYSICAL_GROUP_COUNT_WRONG")
        vals=[x["post_event_max_native_rw_speed_difference_rad_s"] for x in cases]
        tor=[x["post_event_max_native_rw_applied_torque_difference_nm"] for x in cases]
        groups["|".join(context)]={
            "matched_wheel_initial_conditions":len(cases),
            "wheel_speed_effect_positive_gt_1e_minus_8":sum(v>1e-8 for v in vals),
            "native_applied_torque_effect_positive_gt_1e_minus_8":sum(v>1e-8 for v in tor),
            "median_max_post_fault_rw_speed_diff_rad_s":statistics.median(vals),
            "minimum_max_post_fault_rw_speed_diff_rad_s":min(vals),
            "maximum_max_post_fault_rw_speed_diff_rad_s":max(vals),
            "median_max_post_fault_native_applied_torque_diff_nm":statistics.median(tor),
        }
    return {
        "schema":SCHEMA,
        "upstream_real_native_action":SOURCE_RUN,
        "upstream_native_source_commit":SOURCE_COMMIT,
        "original_source_json_sha256":actual_sha,
        "actual_Basilisk_cases":54,
        "physical_wheel_initial_condition_groups":9,
        "healthy_sources":27,
        "matched_physics_comparisons":27,
        "original_case_exporter_metadata_shadowed_count":0,
        "original_source_metadata_wheel_target_trusted":True,
        "verified_wheel_identity_from_case_id_and_native_cpp_event":True,
        "original_raw_source_never_rewritten":True,
        "physical_observability_matrix":groups,
        "original_native_wheel_allocator_request_max_nm":peak_req,
        "original_native_wheel_speed_max_rad_s":peak_speed,
        "original_native_body_rate_max_rad_s":peak_rate,
        "source_original_cpp_wheel_torque_derating_checked":True,
        "no_reference_fault_failure_preserved":True,
        "moderate_derating_wheel_speed_manifestation_not_diagnostic_prediction":True,
        "reference_command_breakdown_original_native_verified":True,
        "original_physics_readonly":True,
        "source_truth_not_a_deployable_diagnostic_feature":True,
        "independent_hardware_or_flight_safety_certified":False,
        "diagnostic_model_eligibility_confirmed":False,
        "source_role":"EXPLORATORY_REAL_BASILISK_PHYSICS_QA_ONLY",
        "protected_validation_test_final_accessed":False,
        "stable_release_eligible":False,
    }


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-file",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args(argv)
    if a.output.exists():
        raise FileExistsError("R83M_RESEARCH_ATTESTATION_ALREADY_EXISTS")
    report=review_original_real_native_rw_source(a.source_file)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n",encoding="utf8")
    for label,item in report["physical_observability_matrix"].items():
        print("R83M_SEALED_REAL_NATIVE_OBSERVABILITY",label,
              "observed",item["wheel_speed_effect_positive_gt_1e_minus_8"],
              "matched",item["matched_wheel_initial_conditions"],
              "median_speed",item["median_max_post_fault_rw_speed_diff_rad_s"])
    return 0


if __name__=="__main__":
    raise SystemExit(main())
