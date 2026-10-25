"""R83m gate anti-fabrication tests. Fixtures are SYNTHETIC; never physics evidence."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

import pytest

from phm_design.industrial import r83m_original_native_guidance_threshold_gate as gate


def fixture_source(tmp_path,monkeypatch):
    cases={}
    for wheel in gate.WHEELS:
        for initial in range(3):
            for reference in gate.REFERENCES:
                for capacity,cap in gate.CAPS.items():
                    fault=capacity!="healthy"
                    key=f"rw{wheel}_initial{initial}_{reference}_{capacity}"
                    rows=[]
                    for t_index in range(31):
                        time=round(t_index*.2,9)
                        guided=reference!="zero" and 2.-1e-8<=time<=3.8+1e-8
                        control=(.007 if reference=="pulse09" else .003) if guided else 0.
                        physical_cap=min(control,cap) if time>=2.6-1e-8 else control
                        physical_effect=max(0,control-physical_cap) if fault and time>=2.6-1e-8 else 0.
                        row={"time_s":time,"rw.native_target_applied_torque_nm_QA_ONLY":physical_cap}
                        for j in gate.WHEELS:
                            row[f"fsw.native_reference_mrp_{j}"]=(
                                __import__("math").tan(__import__("math").radians(gate.ANGLES[reference])/4.)
                                if guided and j==wheel else 0.)
                            row[f"fsw.body_request_nm_{j}"]=(-control if j==wheel else 0.)
                            row[f"rw.native_allocator_request_nm_{j}"]=(control if j==wheel else 0.)
                            row[f"rw.native_speed_rad_s_{j}"]=20.+j+initial*.3+(
                                (time-2.4)*physical_effect if j==wheel and time>=2.6 else 0.)
                            row[f"hub.sigma_bn_{j}"]=gate.INITIALS[initial][j]
                            row[f"hub.omega_bn_b_rad_s_{j}"]=0.
                        rows.append(row)
                    cases[key]={
                        "schema":gate.SOURCE_SCHEMA,
                        "arm":key,
                        "execution_engine":"BASILISK_NATIVE_SC_RW_FSW_NO_PROXY",
                        "native_scheduler_was_advanced":True,
                        "native_independent_condition_id":f"rw{wheel}_initial{initial}",
                        "native_reference_context_id":f"rw{wheel}_initial{initial}_{reference}",
                        "source_fault_target_index_QA_ONLY":wheel,
                        "source_initial_state_index":initial,
                        "source_initial_attitude_mrp":list(gate.INITIALS[initial]),
                        "native_reference_mode":reference,
                        "native_fault_capacity_mode_QA_ONLY":capacity,
                        "true_native_actuator_capacity_derated":fault,
                        "original_native_rw_hardware_limit_nm":.02,
                        "fault_native_rw_capacity_after_nm_QA_ONLY":cap,
                        "model_inference_uses_truth":False,
                        "reference_is_guidance_not_wheel_motor_telecommand":True,
                        "assurance":"RESEARCH_ONLY_NOT_FLIGHT_CERTIFIED",
                        "reference_setpoint_event_receipts":[{
                            "fault_scheduler_pause_s":2.4,
                            "physical_fault_target_rw_index_QA_ONLY":wheel,
                            "fault_actual_native_rw_max_torque_before_nm":.02,
                            "fault_actual_native_rw_max_torque_after_nm":cap,
                            "rw_effector_c_backed_readback_nm":cap,
                            "real_native_physical_change_not_feature_edit":True}],
                        "rows":rows,
                    }
    paired={}
    for wheel in gate.WHEELS:
        for initial in range(3):
            for reference in gate.REFERENCES:
                name=f"rw{wheel}_initial{initial}_{reference}"
                good=cases[name+"_healthy"]["rows"]
                for mode in ("loss75","loss95"):
                    bad=cases[name+"_"+mode]["rows"]
                    before=[(a,b) for a,b in zip(good,bad) if a["time_s"]<=2.4+1e-8]
                    post=[(a,b) for a,b in zip(good,bad) if a["time_s"]>=2.6-1e-8]
                    speed=f"rw.native_speed_rad_s_{wheel}"
                    applied="rw.native_target_applied_torque_nm_QA_ONLY"
                    paired[name+"_"+mode]={
                        "context":name,
                        "native_independent_condition_id":f"rw{wheel}_initial{initial}",
                        "source_fault_target_index_QA_ONLY":wheel,
                        "source_initial_state_index":initial,
                        "reference_mode":reference,"capacity_mode":mode,
                        "pre_event_max_native_rw_speed_difference_rad_s":
                            max(abs(a[speed]-b[speed]) for a,b in before),
                        "post_event_max_native_rw_speed_difference_rad_s":
                            max(abs(a[speed]-b[speed]) for a,b in post),
                        "post_event_max_native_rw_applied_torque_difference_nm":
                            max(abs(a[applied]-b[applied]) for a,b in post),
                        "post_event_fault_native_rw_actual_max_abs_nm":
                            max(abs(b[applied]) for a,b in post),
                    }
    rows=[r for c in cases.values() for r in c["rows"]]
    doc={
        "schema":gate.SOURCE_SCHEMA,
        "execution_engine":"GENUINE_BASILISK_REACTIONWHEELSTATEEFFECTOR_AND_FSW",
        "source_native_cases":54,
        "independent_wheel_and_initial_condition_groups":9,
        "healthy_source_cases":27,
        "paired_fault_vs_nominal_comparisons":27,
        "predeclared_guidance_amplitude_deg":gate.ANGLES,
        "source_all_wheels_observed":True,
        "fault_derating_verified_by_real_cpp_state":True,
        "actual_applied_motor_torque_verified_from_rwOutMsgs":True,
        "fault_truth_only_for_offline_physics_QA":True,
        "no_protected_validation_test_final_cases_generated":True,
        "no_machine_learning_trained":True,
        "not_a_generalizable_diagnostic_benchmark":True,
        "not_an_independent_hardware_safety_qualification":True,
        "stable_release_eligible":False,
        "predeclared_provisional_request_speed_envelope_passed":True,
        "physical_source_cases":cases,
        "native_physical_paired_outcomes":paired,
        "max_original_native_allocator_request_nm":max(
            abs(r[f"rw.native_allocator_request_nm_{i}"]) for r in rows for i in gate.WHEELS),
        "max_original_native_wheel_speed_rad_s":max(
            abs(r[f"rw.native_speed_rad_s_{i}"]) for r in rows for i in gate.WHEELS),
        "max_original_native_body_rate_rad_s":0.,
    }
    details={}
    for reference in gate.REFERENCES:
        selected=[r for c in cases.values()
                  if c["native_reference_mode"]==reference for r in c["rows"]]
        details[reference]={
            "target_reference_angle_deg":gate.ANGLES[reference],
            "peak_allocator_command_nm_all_rw":max(
                abs(r[f"rw.native_allocator_request_nm_{i}"])
                for r in selected for i in gate.WHEELS),
            "max_body_rate_rad_s":max(
                abs(r[f"hub.omega_bn_b_rad_s_{i}"])
                for r in selected for i in gate.WHEELS),
        }
    doc["original_native_request_by_reference_mode"]=details
    file=tmp_path/"SYNTHETIC_DO_NOT_USE_AS_REAL_BASILISK.json"
    file.write_text(json.dumps(doc,sort_keys=True)+"\n",encoding="utf8")
    monkeypatch.setattr(gate,"ORIGINAL_SHA256",hashlib.sha256(file.read_bytes()).hexdigest())
    return file,doc


def test_source_schema_and_repeated_group_accounting(tmp_path,monkeypatch):
    file,doc=fixture_source(tmp_path,monkeypatch)
    receipt=gate.review_original_real_native_rw_source(file)
    assert receipt["actual_Basilisk_cases"]==54
    assert receipt["physical_wheel_initial_condition_groups"]==9
    assert receipt["healthy_sources"]==27
    assert receipt["matched_physics_comparisons"]==27
    assert receipt["physical_observability_matrix"]["pulse09|loss75"]["wheel_speed_effect_positive_gt_1e_minus_8"]==9
    assert receipt["physical_observability_matrix"]["pulse05|loss75"]["wheel_speed_effect_positive_gt_1e_minus_8"]==0
    assert receipt["reference_command_breakdown_original_native_verified"]
    assert receipt["physical_observability_matrix"]["zero|loss75"]["wheel_speed_effect_positive_gt_1e_minus_8"]==0
    assert receipt["source_truth_not_a_deployable_diagnostic_feature"]
    assert receipt["diagnostic_model_eligibility_confirmed"] is False
    assert receipt["protected_validation_test_final_accessed"] is False
    assert receipt["stable_release_eligible"] is False


def test_original_source_mutation_refused(tmp_path,monkeypatch):
    file,doc=fixture_source(tmp_path,monkeypatch)
    file.write_bytes(file.read_bytes()+b"\n ")
    with pytest.raises(ValueError,match="SHA256_MISMATCH"):
        gate.review_original_real_native_rw_source(file)


def test_false_native_applied_torque_summary_refused_even_if_rehashed(tmp_path,monkeypatch):
    file,doc=fixture_source(tmp_path,monkeypatch)
    key="rw1_initial1_pulse09_loss75"
    doc["native_physical_paired_outcomes"][key]["post_event_max_native_rw_applied_torque_difference_nm"]=0.99
    file.write_text(json.dumps(doc,sort_keys=True)+"\n",encoding="utf8")
    monkeypatch.setattr(gate,"ORIGINAL_SHA256",hashlib.sha256(file.read_bytes()).hexdigest())
    with pytest.raises(ValueError,match="PHYSICS_METRIC_DRIFT"):
        gate.review_original_real_native_rw_source(file)


def test_safety_claim_cannot_be_upgraded_with_rehashed_json(tmp_path,monkeypatch):
    file,doc=fixture_source(tmp_path,monkeypatch)
    doc["not_an_independent_hardware_safety_qualification"]=False
    file.write_text(json.dumps(doc,sort_keys=True)+"\n",encoding="utf8")
    monkeypatch.setattr(gate,"ORIGINAL_SHA256",hashlib.sha256(file.read_bytes()).hexdigest())
    with pytest.raises(ValueError,match="GIVEN_CLAIM"):
        gate.review_original_real_native_rw_source(file)
