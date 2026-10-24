"""R83j sealed FSW proof gate: SYNTHETIC fixtures, never physics."""
from __future__ import annotations

import hashlib
import json
import math

import pytest

from phm_design.industrial import r83j_native_fsw_reference_gate as g


def _synthetic(tmp_path,monkeypatch):
    rows={}
    for mode in ("reference","scheduled"):
        r=[]
        for i in range(31):
            t=round(.2*i,9)
            pulse=mode=="scheduled" and 10<=i<=19
            active=mode=="scheduled" and i>=11
            angle=math.tan(math.radians(.5)/4) if pulse else 0.
            command=.003 if active and i<=19 else 0.
            speed=10.+(max(0,i-10)*.001 if active else 0.)
            row={"time_s":t}
            for j in range(3):
                row[f"fsw.native_reference_mrp_{j}"]=angle if j==0 else 0.
                row[f"fsw.body_request_nm_{j}"]=-command if j==0 else 0.
                row[f"rw.native_allocator_request_nm_{j}"]=command if j==0 else 0.
                row[f"rw.native_speed_rad_s_{j}"]=speed if j==0 else 20.+j
                row[f"hub.sigma_bn_{j}"]=0.
                row[f"hub.omega_bn_b_rad_s_{j}"]=0.
            r.append(row)
        rows[mode]=r
    sample = {
        "schema":g.UPSTREAM_SOURCE_SCHEMA,
        "source_physics_real_native":True,
        "independent_hardware_safety_certified":False,
        "fault_diagnosis_generalization_measured":False,
        "fault_injections_performed":False,
        "no_automatic_stable_promotion":True,
        "limits_within_provisional_lab_envelope":True,
        "arms":{
            name:{"schema":g.UPSTREAM_SOURCE_SCHEMA,"arm":name,
                  "execution_engine":"BASILISK_NATIVE_SC_RW_FSW_NO_PROXY",
                  "native_module_chain":g.PHYSICS_NATIVE_MODULE_CHAIN,
                  "native_scheduler_was_advanced":True,
                  "reference_is_guidance_not_wheel_motor_telecommand":True,
                  "assurance":"RESEARCH_ONLY_NOT_FLIGHT_CERTIFIED",
                  "rows":rows[name]}
            for name in rows
        },
    }
    a,b=rows["reference"],rows["scheduled"]
    before=[(x,y) for x,y in zip(a,b) if x["time_s"]<=1.8+1e-8]
    after=[(x,y) for x,y in zip(a,b) if 2.2-1e-8<=x["time_s"]<=3.8+1e-8]
    metrics={
        "pre_event_speed_abs_difference":max(abs(x["rw.native_speed_rad_s_0"]-y["rw.native_speed_rad_s_0"]) for x,y in before),
        "post_event_max_rw_speed_difference_rad_s":max(abs(x["rw.native_speed_rad_s_0"]-y["rw.native_speed_rad_s_0"]) for x,y in after),
        "post_event_max_motor_command_difference_nm":max(abs(x["rw.native_allocator_request_nm_0"]-y["rw.native_allocator_request_nm_0"]) for x,y in after),
        "post_event_max_native_reference_mrp_difference":max(abs(x["fsw.native_reference_mrp_0"]-y["fsw.native_reference_mrp_0"]) for x,y in after),
    }
    for f in ("max_motor_allocator_request_nm_all_rw", "max_wheel_speed_rad_s_all_rw",
              "max_body_rate_rad_s_all_axes"):
        if f=="max_motor_allocator_request_nm_all_rw":
            fields=[f"rw.native_allocator_request_nm_{j}" for j in range(3)]
        elif f=="max_wheel_speed_rad_s_all_rw":
            fields=[f"rw.native_speed_rad_s_{j}" for j in range(3)]
        else:
            fields=[f"hub.omega_bn_b_rad_s_{j}" for j in range(3)]
        metrics[f]=max(abs(r[c]) for rs in rows.values() for r in rs for c in fields)
    sample.update(metrics)
    path=tmp_path/"original_native_fsw.json"
    path.write_text(json.dumps(sample,sort_keys=True)+"\n")
    monkeypatch.setattr(g,"UPSTREAM_ORIGINAL_ARTIFACT_JSON_SHA256",
                        hashlib.sha256(path.read_bytes()).hexdigest())
    return path,sample


def test_reference_response_only_not_diagnostic_evidence(tmp_path,monkeypatch):
    path,_=_synthetic(tmp_path,monkeypatch)
    result=g.review_original_native_fsw_proof(path)
    assert result["native_fsw_reference_to_rw_physical_response_observed"]
    assert result["actual_native_fsw_reference_change_begins_s"]==pytest.approx(2.)
    assert result["original_physically_simulated_arms"]==2
    assert result["original_native_samples_per_arm"]==31
    for key in ("autogen_algorithm_eligible_from_this_evidence",
                "stable_release_eligible_from_this_evidence",
                "independent_hardware_safety_certified",
                "no_training_or_model_selection_performed"):
        assert result[key] == (key=="no_training_or_model_selection_performed")
    assert result["no_fault_mechanism_tested_here"]
    assert result["validation_test_final_telemetry_not_opened"]


def test_modified_native_bytes_refused_even_when_metrics_not_changed(tmp_path,monkeypatch):
    path,_=_synthetic(tmp_path,monkeypatch)
    path.write_text(path.read_text()+" ")
    with pytest.raises(ValueError,match="TAMPERED"):
        g.review_original_native_fsw_proof(path)


def test_forged_metric_and_provisional_safety_cannot_promote_model(tmp_path,monkeypatch):
    path,data=_synthetic(tmp_path,monkeypatch)
    data["post_event_max_rw_speed_difference_rad_s"]=99.
    path.write_text(json.dumps(data)+"\n")
    monkeypatch.setattr(g,"UPSTREAM_ORIGINAL_ARTIFACT_JSON_SHA256",
                        hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(ValueError,match="NOT_MATCHING_SOURCE"):
        g.review_original_native_fsw_proof(path)
    data["post_event_max_rw_speed_difference_rad_s"]=.009
    data["independent_hardware_safety_certified"]=True
    path.write_text(json.dumps(data)+"\n")
    monkeypatch.setattr(g,"UPSTREAM_ORIGINAL_ARTIFACT_JSON_SHA256",
                        hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(ValueError,match="CLAIM_POLICY"):
        g.review_original_native_fsw_proof(path)


def test_physical_history_not_reclassed_as_direct_motor_telecommand(tmp_path,monkeypatch):
    path,data=_synthetic(tmp_path,monkeypatch)
    data["arms"]["scheduled"]["reference_is_guidance_not_wheel_motor_telecommand"]=False
    path.write_text(json.dumps(data)+"\n")
    monkeypatch.setattr(g,"UPSTREAM_ORIGINAL_ARTIFACT_JSON_SHA256",
                        hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(ValueError,match="FABRICATED_ARM"):
        g.review_original_native_fsw_proof(path)
