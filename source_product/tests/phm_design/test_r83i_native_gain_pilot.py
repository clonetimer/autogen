"""R83i: PREREGISTRATION-only unit tests. No Python physical simulation."""
from __future__ import annotations

from copy import deepcopy

import pytest

from phm_design.io import digest
from phm_design.industrial import r83i_native_gain_pilot as pilot
from phm_design.industrial.taskspec import TaskSpecRejected


class NativeRegistryStub:
    path_map={
        "parameters.values.controller_k":"controller_k",
        "parameters.values.controller_p":"controller_p",
        "parameters.values.rw_max_torque_nm":"rw_max_torque_nm",
    }


def registry(name):
    return NativeRegistryStub() if name=="subsystem.adcs_unified_native.v1" else None


def source(monkeypatch):
    cases=[]
    for split,count in (("train",15),("validation",6)):
        for fault in pilot.FAULTS:
            for i in range(count):
                pair=f"{split}-{fault.lower()}-{i:03d}"
                params={"controller_k":3.345,"controller_p":29.2,
                        "rw_max_torque_nm":.2,
                        "rw_motor_current_limit_a":10.,
                        "initial_pointing_error_deg":11.,
                        "inclination_deg":(29.,47.,65.)[i%3],
                        "simulation_seed":100+i}
                for role in ("fault","nominal"):
                    caseid=f"r83d_balanced_native_rw-{pair}-{role}"
                    cases.append({"case_id":caseid,"overrides":{
                        "parameters":deepcopy(params),
                        "simulation.seed":100+i,
                        "metadata":{
                            "astrograph_split":split,
                            "astrograph_pair_id":pair,
                            "astrograph_fault_id":fault,
                            "astrograph_case_role":role,
                            "astrograph_fault_case_id":f"r83d_balanced_native_rw-{pair}-fault",
                            "astrograph_nominal_case_id":f"r83d_balanced_native_rw-{pair}-nominal",
                        },
                        "target.mode": "nominal" if role=="nominal" else "degradation",
                        "modifiers": {"faults":[],"degradations":[{
                            "parameters":{"wheel_index":2,"torque_scale":.5},
                            "onset_time_s":25.
                        }] if role=="fault" else [],"constraints":[]},
                    }})
    result={"task_id":"r83d_balanced_native_rw","campaign":{
        "base_spec":{"capability_id":"subsystem.adcs_unified_native.v1",
                     "simulation":{"backend":"basilisk"}},
        "count":126,"max_cases":126,"cases":cases},
        "metadata":{"r83d_preregistration":{
            "schema":"phm.r83d.native-balanced-protocol.v1"}}}
    monkeypatch.setattr(pilot,"SOURCE_PLAN",digest(result))
    return result


def test_registered_controller_gain_pilot_only_changes_two_gains(monkeypatch):
    original=source(monkeypatch)
    before=deepcopy(original)
    plan,receipt=pilot.preregister_static_gain_pilot(
        original,native_gain_registry=registry)
    assert before==original
    assert receipt["source_cases"]==36
    assert receipt["source_base_pairs"]==9
    assert receipt["actual_basilisk_executed"] is False
    assert len({x["case_id"] for x in plan["campaign"]["cases"]})==36
    assert all(c["overrides"]["metadata"]["astrograph_split"]=="train"
               for c in plan["campaign"]["cases"])
    identities={}
    for c in plan["campaign"]["cases"]:
        override=c["overrides"]
        m=override["metadata"]
        pair=m["r83i_original_source_pair_id"]
        arm=m["r83i_static_gain_arm"]
        role=m["astrograph_case_role"]
        identities.setdefault((pair,arm),{})[role]=c
        assert m["r83i_no_time_varying_command"] is True
        assert m["astrograph_pair_id"]==f"{pair}-r83i-{arm}"
        assert m["astrograph_fault_case_id"].endswith(f"-r83i-{arm}-fault")
        assert m["astrograph_nominal_case_id"].endswith(f"-r83i-{arm}-nominal")
    assert len(identities)==18
    for (pair,arm), twins in identities.items():
        assert set(twins)=={"fault","nominal"}
        a,b=(x["overrides"] for x in twins.values())
        assert a["parameters"]==b["parameters"]
        assert a["simulation.seed"]==b["simulation.seed"]
        old=next(c["overrides"] for c in original["campaign"]["cases"]
                 if c["overrides"]["metadata"]["astrograph_pair_id"]==pair)
        new=a["parameters"]
        old_params=old["parameters"]
        assert new["rw_max_torque_nm"]==old_params["rw_max_torque_nm"]
        assert new["rw_motor_current_limit_a"]==old_params["rw_motor_current_limit_a"]
        assert new["initial_pointing_error_deg"]==old_params["initial_pointing_error_deg"]
        assert new["inclination_deg"]==old_params["inclination_deg"]
        assert new["controller_k"]==pytest.approx(
            round(old_params["controller_k"]*pilot.GAIN[arm],8))
        assert new["controller_p"]==pytest.approx(
            round(old_params["controller_p"]*pilot.GAIN[arm],8))
        assert a["simulation.seed"]==old["simulation.seed"]


def test_unverified_time_varying_interface_refused(monkeypatch):
    orig=source(monkeypatch)
    class Fraud:
        path_map={**NativeRegistryStub.path_map,
                  "parameters.values.motor_torque_schedule":"torque_schedule"}
    with pytest.raises(TaskSpecRejected, match="UNREVIEWED_NATIVE_TIME_PROFILE_SURFACE"):
        pilot.preregister_static_gain_pilot(orig,native_gain_registry=lambda _:Fraud())


def test_wrong_source_plan_or_twin_integrity_refused(monkeypatch):
    original=source(monkeypatch)
    tampered=deepcopy(original)
    tampered["campaign"]["cases"][0]["overrides"]["parameters"]["controller_k"]=99.
    with pytest.raises(TaskSpecRejected,match="EXACT_ORIGINAL_R83F_NATIVE_PLAN"):
        pilot.preregister_static_gain_pilot(tampered,native_gain_registry=registry)
    corrupt=deepcopy(original)
    corrupt["campaign"]["cases"][0]["overrides"]["parameters"]["controller_k"]=99.
    monkeypatch.setattr(pilot,"SOURCE_PLAN",digest(corrupt))
    with pytest.raises(TaskSpecRejected,match="ORIGINAL_PAIR_NOT_MATCHED"):
        pilot.preregister_static_gain_pilot(corrupt,native_gain_registry=registry)
