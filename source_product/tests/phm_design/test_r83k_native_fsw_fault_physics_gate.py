"""R83k immutable original native-source gate regression; SYNTHETIC fixtures."""
import hashlib
import json
from pathlib import Path

import pytest

from phm_design.industrial import r83k_native_fsw_fault_physics_gate as g


def _fixture(tmp_path,monkeypatch):
    arms={}
    for name in g.ARMS:
        step=name.startswith("step_")
        fault=name.endswith("_derated")
        rows=[]
        for i in range(31):
            t=round(i*.2,9)
            active=step and 2.0-1e-9<=t<=3.8+1e-9
            after=fault and t>=2.6-1e-9
            allocated=.003 if active else 0.
            applied=(.0005 if after else allocated) if active else 0.
            speed=10.+(.01*(t-2.4) if active and t>=2.6 else 0.)
            if after and active:
                speed-=.003*(t-2.4)
            row={"time_s":t,"rw.native_target_applied_torque_nm_QA_ONLY":applied}
            for j in range(3):
                row[f"fsw.native_reference_mrp_{j}"]=.002181665 if active and j==0 else 0.
                row[f"fsw.body_request_nm_{j}"]= -allocated if j==0 else 0.
                row[f"rw.native_allocator_request_nm_{j}"]=allocated if j==0 else 0.
                row[f"rw.native_speed_rad_s_{j}"]=speed if j==0 else 10.+j
                row[f"hub.sigma_bn_{j}"]=0.
                row[f"hub.omega_bn_b_rad_s_{j}"]=0.
            rows.append(row)
        value=g.CAP_FAULT if fault else g.CAP_NOMINAL
        arm={
            "schema":g.SOURCE_SCHEMA, "arm":name,
            "execution_engine":"BASILISK_NATIVE_SC_RW_FSW_NO_PROXY",
            "native_scheduler_was_advanced":True,
            "reference_is_guidance_not_wheel_motor_telecommand":True,
            "true_native_actuator_capacity_derated":fault,
            "model_inference_uses_truth":False,
            "fault_onset_paused_scheduler_s_QA_ONLY":g.ONSET_S,
            "original_native_rw_hardware_limit_nm":g.CAP_NOMINAL,
            "fault_native_rw_capacity_after_nm_QA_ONLY":value,
            "reference_setpoint_event_receipts":[{
                "fault_scheduler_pause_s":g.ONSET_S,
                "physical_fault_target_rw_index_QA_ONLY":0,
                "fault_actual_native_rw_max_torque_before_nm":g.CAP_NOMINAL,
                "fault_actual_native_rw_max_torque_after_nm":value,
                "rw_effector_c_backed_readback_nm":value,
                "real_native_physical_change_not_feature_edit":True}],
            "rows":rows
        }
        arms[name]=arm
    pairs={}
    for label in ("zero","step"):
        a,b=arms[label+"_healthy"]["rows"],arms[label+"_derated"]["rows"]
        sp="rw.native_speed_rad_s_0"
        tor="rw.native_target_applied_torque_nm_QA_ONLY"
        pre=[(x,y) for x,y in zip(a,b) if x["time_s"]<=2.4+1e-8]
        after=[(x,y) for x,y in zip(a,b) if x["time_s"]>=2.6-1e-8]
        pairs[label]={
            "source_true_fault_cap_nm":g.CAP_FAULT,
            "pre_onset_max_native_target_speed_difference_rad_s":
                 max(abs(x[sp]-y[sp]) for x,y in pre),
            "post_onset_max_native_target_speed_difference_rad_s":
                 max(abs(x[sp]-y[sp]) for x,y in after),
            "post_onset_max_native_target_applied_torque_difference_nm":
                 max(abs(x[tor]-y[tor]) for x,y in after),
            "fault_native_applied_torque_max_abs_nm":
                 max(abs(y[tor]) for x,y in after),
        }
    all_rows=[r for x in arms.values() for r in x["rows"]]
    source={
        "schema":g.SOURCE_SCHEMA,
        "actual_engine":"GENUINE_BASILISK_RWMOTORTORQUE_AND_REACTIONWHEELSTATEEFFECTOR",
        "source_native_trials":4,
        "physical_native_rw_capacity_write_readback_done":True,
        "reference_and_fault_are_separate_native_interventions":True,
        "reference_nonzero_change_detected":True,
        "provisional_research_command_and_speed_bounds_passed":True,
        "independent_hardware_safety_certified":False,
        "healthy_vs_fault_model_trained":False,
        "diagnostic_separability_proven":False,
        "held_out_validation_test_final_data_used":False,
        "stable_release_eligible":False,
        "physical_source_arms":arms,
        "physical_pair_outcomes":pairs,
        "max_requested_native_wheel_torque_nm":
            max(abs(r[f"rw.native_allocator_request_nm_{j}"]) for r in all_rows for j in range(3)),
        "max_native_wheel_speed_rad_s":
            max(abs(r[f"rw.native_speed_rad_s_{j}"]) for r in all_rows for j in range(3)),
    }
    path=tmp_path/"original_faked_for_contract_only.json"
    path.write_text(json.dumps(source,sort_keys=True)+"\n",encoding="utf8")
    monkeypatch.setattr(g,"ORIGINAL_NATIVE_PHYSICS_JSON_SHA256",
                        hashlib.sha256(path.read_bytes()).hexdigest())
    return path,source


def test_physics_factorial_gate_recomputes_original_rows_and_no_model_claim(tmp_path,monkeypatch):
    path,_=_fixture(tmp_path,monkeypatch)
    report=g.attest_original_faulted_rw_physics(path)
    assert report["physically_executed_native_cases"]==4
    assert report["original_samples_per_case"]==31
    assert report["physics_excitation_dependent_fault_observability_demonstrated"]
    assert report["paired_no_command_fault_speed_effect_rad_s"]==0.
    assert report["paired_with_scheduled_reference_fault_speed_effect_rad_s"]>0.
    assert report["paired_with_reference_native_actual_applied_torque_effect_nm"]>0.
    assert report["true_rw_max_torque_physical_state_verified"]
    assert report["trained_fault_classifier_eligible"] is False
    assert report["automatic_stable_promotion_allowed"] is False
    assert report["heldout_validation_test_final_telemetry_accessed"] is False


def test_any_single_original_source_byte_tamper_denied(tmp_path,monkeypatch):
    path,_=_fixture(tmp_path,monkeypatch)
    path.write_text(path.read_text()+" ",encoding="utf8")
    with pytest.raises(ValueError,match="SOURCE_CHANGED"):
        g.attest_original_faulted_rw_physics(path)


def test_fake_native_applied_torque_metric_denied_even_when_file_rehashed(tmp_path,monkeypatch):
    path,source=_fixture(tmp_path,monkeypatch)
    source["physical_pair_outcomes"]["step"]["post_onset_max_native_target_applied_torque_difference_nm"]=99.
    path.write_text(json.dumps(source,sort_keys=True)+"\n")
    monkeypatch.setattr(g,"ORIGINAL_NATIVE_PHYSICS_JSON_SHA256",
                        hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(ValueError,match="NOT_IN_NATIVE_SAMPLES"):
        g.attest_original_faulted_rw_physics(path)


def test_fault_output_cannot_be_relabelled_as_real_safety_or_ml_qualification(tmp_path,monkeypatch):
    path,source=_fixture(tmp_path,monkeypatch)
    source["diagnostic_separability_proven"]=True
    path.write_text(json.dumps(source,sort_keys=True)+"\n")
    monkeypatch.setattr(g,"ORIGINAL_NATIVE_PHYSICS_JSON_SHA256",
                        hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(ValueError,match="UNSUPPORTED_DIAGNOSTIC_CLAIM"):
        g.attest_original_faulted_rw_physics(path)
