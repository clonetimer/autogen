"""R83i physics-evidence unit tests are SYNTHETIC CONTRACTS ONLY."""
from __future__ import annotations

import csv
from pathlib import Path
from copy import deepcopy

import pytest

from phm_design.io import read_json,write_json
from phm_design.industrial import r83i_native_gain_pilot as gen
from phm_design.industrial.r83i_native_gain_evidence import review_real_native_gain_pilot
from test_r83i_native_gain_pilot import source, registry


def _demo(tmp_path,monkeypatch):
    plan,_=gen.preregister_static_gain_pilot(
        source(monkeypatch),native_gain_registry=registry)
    cfg=tmp_path/"planned.json"
    write_json(cfg,plan)
    campaign=tmp_path/"native"
    write_json(campaign/"astrograph_dataset_index.json",{
        "case_count":36,"campaign_status":"complete","dry_run":False})
    fields=["time_s",*[f"adcs.rw.speed_rad_s_{i}" for i in range(3)],
            *[f"adcs.rw.command_torque_nm_{i}" for i in range(3)],
            *[f"adcs.control.applied_torque_nm_{i}" for i in range(3)],
            *[f"adcs.rw.effective_max_torque_nm_{i}" for i in range(3)]]
    for c in plan["campaign"]["cases"]:
        ident=c["case_id"]
        sourcepath=campaign/"cases"/ident
        m=c["overrides"]["metadata"]
        params=c["overrides"]["parameters"]
        write_json(sourcepath/"astrograph/dataset_contract.json",{
            "split":"train","pairing":{"pair_id":m["astrograph_pair_id"],
                "case_role":m["astrograph_case_role"]},
            "simulation":{"backend":"basilisk","bsk_version":"2.11.0+satfix1"}})
        write_json(sourcepath/"task_spec.json",{
            "parameters":params,"simulation":{"seed":c["overrides"]["simulation.seed"]},
            "modifiers":c["overrides"]["modifiers"]})
        with (sourcepath/"trace.csv").open("w",encoding="utf8",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=fields)
            writer.writeheader()
            for t in range(61):
                effect=(.5 if m["astrograph_case_role"]=="fault" else 0.) if t>=25 else 0.
                vals={"time_s":t}
                for i in range(3):
                    vals[f"adcs.rw.speed_rad_s_{i}"]=100.0+i+t*.03+effect
                    vals[f"adcs.rw.command_torque_nm_{i}"]=.005*(
                        m["r83i_controller_multiplier"])
                    vals[f"adcs.control.applied_torque_nm_{i}"]=.004
                    vals[f"adcs.rw.effective_max_torque_nm_{i}"]=.2
                writer.writerow(vals)
    return campaign,cfg


def test_actual_shape_native_contract_and_zero_validation(monkeypatch,tmp_path):
    campaign,plan=_demo(tmp_path,monkeypatch)
    report=review_real_native_gain_pilot(
        campaign=campaign,planned_spec=plan,output=tmp_path/"report.json")
    assert report["real_native_cases_verified"]==36
    assert report["paired_source_cases"]==18
    assert report["independent_physical_parent_pairs"]==9
    assert report["time_varying_wheel_command_demonstrated"] is False
    assert report["validation_test_final_files_read"] is False
    assert report["model_trained_or_selected"] is False
    assert report["safety_provisional_900_rad_s_and_command_rating_below_1"]
    assert len(report["original_pairs"])==18
    assert all(v["reference"]["original_independent_pairs"]==3
               for v in report["physical_excitation_by_mechanism"].values())
    assert read_json(tmp_path/"report.json")==report


def test_failed_case_or_changed_runtime_source_refused(monkeypatch,tmp_path):
    campaign,plan=_demo(tmp_path,monkeypatch)
    c=read_json(campaign/"astrograph_dataset_index.json")
    c["case_count"]=35
    write_json(campaign/"astrograph_dataset_index.json",c)
    with pytest.raises(Exception,match="GENUINE_36_CASE_NATIVE_RUN_REQUIRED"):
        review_real_native_gain_pilot(campaign=campaign,planned_spec=plan,
                                     output=tmp_path/"no_bad_count")
    c["case_count"]=36
    write_json(campaign/"astrograph_dataset_index.json",c)
    first=next((campaign/"cases").iterdir())
    contract=first/"astrograph/dataset_contract.json"
    receipt=read_json(contract)
    receipt["simulation"]["backend"]="python"
    write_json(contract,receipt)
    with pytest.raises(Exception,match="DIFFERENT_FROM_PREREGISTRATION"):
        review_real_native_gain_pilot(campaign=campaign,planned_spec=plan,
                                     output=tmp_path/"no_proxy")


def test_dangerous_command_kept_visible_as_failure_not_false_certification(monkeypatch,tmp_path):
    campaign,plan=_demo(tmp_path,monkeypatch)
    first=next((campaign/"cases").iterdir())
    trace=first/"trace.csv"
    contents=trace.read_text(encoding="utf8")
    names=contents.splitlines()[0].split(",")
    line=contents.splitlines()[3].split(",")
    i=names.index("adcs.rw.command_torque_nm_0")
    line[i]="1.2"
    lines=contents.splitlines()
    lines[3]=",".join(line)
    trace.write_text("\n".join(lines)+"\n",encoding="utf8")
    report=review_real_native_gain_pilot(
        campaign=campaign,planned_spec=plan,output=tmp_path/"unsafe_report.json")
    assert report["safety_provisional_900_rad_s_and_command_rating_below_1"] is False
    assert report["observed_max_command_to_rated_ratio"]>1.
