"""R83i real-native static-gain pilot physics evidence, not an ML evaluator.

Both gain arms share the original random seed, event severity and initial
conditions. Only TRAIN physical traces. Applied torque is source-internal,
never treated as a deployable input. Comparisons are observational: controller
gain can change the system response; stronger gain is NOT proven safer/better.
"""
from __future__ import annotations

from collections import defaultdict
import argparse
import csv
import math
from pathlib import Path
import statistics

from ..io import digest, read_json, write_json, sha_file
from .r83i_native_gain_pilot import ARMS, FAULTS, INDEXES, SCHEMA as PLAN_SCHEMA
from .taskspec import TaskSpecRejected

SCHEMA = "phm.r83i.real-native-static-gain-physics-review.v1"
SPEED_BOUND_RAD_S = 900.0  # Research-only provisional engineering envelope


def _reject(reason):
    raise TaskSpecRejected(reason)


def _measure(case_root, wheel, onset):
    """Use only pinned original TRAIN case CSVs, no external source folders."""
    source = case_root/"trace.csv"
    c=f"adcs.rw.command_torque_nm_{wheel}"
    s=f"adcs.rw.speed_rad_s_{wheel}"
    rating=f"adcs.rw.effective_max_torque_nm_{wheel}"
    applied=f"adcs.control.applied_torque_nm_{wheel}"
    with source.open("r",encoding="utf-8",newline="") as f:
        reader=csv.DictReader(f)
        required={"time_s",c,s,rating,applied}
        if not required.issubset(reader.fieldnames or []) or \
                len(reader.fieldnames or [])!=len(set(reader.fieldnames or [])):
            _reject("R83I_NATIVE_SOURCE_TELEMETRY_INCOMPLETE")
        rows=[]
        for row in reader:
            vals={field:float(row[field]) for field in required}
            if not all(math.isfinite(v) for v in vals.values()):
                _reject("R83I_NONFINITE_NATIVE_TELEMETRY")
            rows.append(vals)
    if len(rows)!=61 or any(abs(row["time_s"]-i)>1e-8 for i,row in enumerate(rows)):
        _reject("R83I_NATIVE_TRACE_SAMPLE_CONTRACT_CHANGED")
    if any(row[rating]<=0 for row in rows):
        _reject("R83I_NONPOSITIVE_ACTUATOR_RATING")
    post=[row for row in rows if row["time_s"]>=onset]
    if len(post)<2 or len(post)>58:
        _reject("R83I_NATIVE_FAULT_ONSET_OUTSIDE_SOURCE")
    return {
        "trace_sha256":sha_file(source),
        "post_mean_abs_command_to_rated":statistics.fmean(abs(row[c])/row[rating] for row in post),
        "post_mean_abs_command_nm":statistics.fmean(abs(row[c]) for row in post),
        "post_mean_abs_command_applied_gap_nm":statistics.fmean(
            abs(row[c]-row[applied]) for row in post),
        "max_full_case_abs_speed_rad_s":max(abs(row[s]) for row in rows),
        "max_full_case_abs_command_to_rated":max(abs(row[c])/row[rating] for row in rows),
        "max_full_case_abs_applied_to_rated":max(abs(row[applied])/row[rating] for row in rows),
        "full_case_source_rows":len(rows),
        "declared_command_and_speed_limits_not_independently_calibrated":True,
    }


def review_real_native_gain_pilot(*, campaign, planned_spec, output):
    campaign=Path(campaign)
    planned=read_json(planned_spec)
    output=Path(output)
    if output.exists():
        raise FileExistsError("R83I_PHYSICS_REVIEW_ALREADY_EXISTS")
    claim=planned.get("metadata",{}).get("r83i",{})
    if (claim.get("schema") != PLAN_SCHEMA or
            claim.get("physics_case_count")!=36 or
            claim.get("real_physics_executed") is not False):
        _reject("R83I_PREDECLARED_36_SOURCE_CASES_REQUIRED")
    expected={x["case_id"]:x for x in planned["campaign"]["cases"]}
    if len(expected)!=36 or len(planned["campaign"]["cases"])!=36:
        _reject("R83I_DUPLICATED_PREREGISTERED_SOURCE_CASE")
    index=read_json(campaign/"astrograph_dataset_index.json")
    if (index.get("dry_run") is not False or
            index.get("case_count")!=36 or
            index.get("campaign_status")!="complete"):
        _reject("R83I_GENUINE_36_CASE_NATIVE_RUN_REQUIRED")
    detail=[]
    for caseid,orig in sorted(expected.items()):
        safe=(campaign/"cases"/caseid).resolve()
        if not safe.is_relative_to((campaign/"cases").resolve()):
            _reject("R83I_CASE_ESCAPES_SOURCE_ROOT")
        contract=read_json(safe/"astrograph/dataset_contract.json")
        task=read_json(safe/"task_spec.json")
        m=orig["overrides"]["metadata"]
        actual=task.get("parameters",{})
        if (contract["simulation"]["backend"]!="basilisk" or
                contract["simulation"]["bsk_version"]!="2.11.0+satfix1" or
                contract["split"]!="train" or
                contract["pairing"]["pair_id"]!=m["astrograph_pair_id"] or
                contract["pairing"]["case_role"]!=m["astrograph_case_role"] or
                actual!=orig["overrides"]["parameters"] or
                task["simulation"]["seed"]!=orig["overrides"]["simulation.seed"]):
            _reject("R83I_REAL_NATIVE_EXECUTION_DIFFERENT_FROM_PREREGISTRATION")
        if contract["pairing"]["case_role"]=="fault":
            modifiers=task["modifiers"]
            events=modifiers.get("faults",[])+modifiers.get("degradations",[])
            if len(events)!=1:
                _reject("R83I_NO_SINGLE_ORIGINAL_FAULT_EVENT")
            evt=events[0]
            onset=float(evt["onset_time_s"])
            wheel=int(evt["parameters"]["wheel_index"])
        else:
            # Normal physical twins expose original intended fault metadata.
            # The injection wheel/event onset are obtained from partner fault
            # only during offline research physics QA below, not model input.
            onset=wheel=None
        detail.append({"case_id":caseid,"case_root":safe,"arm":m["r83i_static_gain_arm"],
                       "parent":m["r83i_original_source_pair_id"],
                       "role":m["astrograph_case_role"],
                       "fault_class":m["astrograph_fault_id"],
                       "onset":onset,"wheel":wheel,"seed":orig["overrides"]["simulation.seed"],
                       "hardware_limit":actual["rw_max_torque_nm"],"parameters":actual})
    twins=defaultdict(dict)
    for item in detail:
        key=(item["parent"],item["arm"])
        if item["role"] in twins[key]:
            _reject("R83I_DUPLICATE_NATIVE_PAIR_ROLE")
        twins[key][item["role"]]=item
    if len(twins)!=18 or any(set(c)!={"fault","nominal"} for c in twins.values()):
        _reject("R83I_NATIVE_NOMINAL_FAULT_PAIR_INCOMPLETE")
    report=[]
    for key in sorted(twins):
        parent,arm=key
        a,b=twins[key]["fault"],twins[key]["nominal"]
        if (a["seed"]!=b["seed"] or a["parameters"]!=b["parameters"] or
                a["hardware_limit"]!=b["hardware_limit"]):
            _reject("R83I_NATIVE_PHYSICAL_TWIN_CHANGED")
        onset,wheel=a["onset"],a["wheel"]
        if onset is None or wheel not in (0,1,2) or not 0<onset<60:
            _reject("R83I_INVALID_ORIGINAL_FAULT_ONSET")
        fa=_measure(a["case_root"],wheel,onset)
        no=_measure(b["case_root"],wheel,onset)
        report.append({
            "original_physical_parent":parent,"arm":arm,
            "fault_class":a["fault_class"],
            "original_fault_target_wheel_PHYSICS_QA_ONLY":wheel,
            "original_fault_onset_PHYSICS_QA_ONLY":onset,
            "original_simulation_seed":a["seed"],
            "hardware_torque_limit_nm":a["hardware_limit"],
            "fault_trace_physics_only":fa,
            "nominal_trace_physics_only":no,
        })
    # Two static controllers must be comparable on same original physical
    # source. The intervention is gain only. Safety is observed, not assumed.
    cross=defaultdict(dict)
    for row in report:
        cross[row["original_physical_parent"]][row["arm"]]=row
    if len(cross)!=9 or any(set(x)!=set(ARMS) for x in cross.values()):
        _reject("R83I_CROSS_ARM_PHYSICAL_MATCHING_INCOMPLETE")
    for parent,groups in cross.items():
        x,y=groups["reference"],groups["gain125"]
        if (x["original_simulation_seed"]!=y["original_simulation_seed"] or
                x["hardware_torque_limit_nm"]!=y["hardware_torque_limit_nm"] or
                x["original_fault_onset_PHYSICS_QA_ONLY"]!=
                y["original_fault_onset_PHYSICS_QA_ONLY"] or
                x["original_fault_target_wheel_PHYSICS_QA_ONLY"]!=
                y["original_fault_target_wheel_PHYSICS_QA_ONLY"]):
            _reject("R83I_CROSS_ARM_PHYSICAL_CASE_DRIFT")
    by_fault={}
    for fault in FAULTS:
        rows=[x for x in report if x["fault_class"]==fault]
        if len(rows)!=6:
            _reject("R83I_MECHANISM_ARM_SAMPLE_SUPPORT_INVALID")
        by_fault[fault]={}
        for arm in ARMS:
            sel=[x for x in rows if x["arm"]==arm]
            by_fault[fault][arm]={
                "original_independent_pairs":len(sel),
                "median_nominal_command_to_rating":statistics.median(
                    x["nominal_trace_physics_only"]["post_mean_abs_command_to_rated"] for x in sel),
                "median_fault_command_applied_gap_nm":statistics.median(
                    x["fault_trace_physics_only"]["post_mean_abs_command_applied_gap_nm"] for x in sel),
            }
    max_ratio=max(x[role]["max_full_case_abs_command_to_rated"]
                  for row in report for role in ("fault_trace_physics_only",
                                                 "nominal_trace_physics_only")
                  for x in (row,))
    max_speed=max(row[role]["max_full_case_abs_speed_rad_s"]
                  for row in report for role in ("fault_trace_physics_only",
                                                 "nominal_trace_physics_only"))
    return_result={
        "schema":SCHEMA,
        "source_kind":"ACTUAL_BASILISK_NATIVE_STATIC_GAIN_ENGINEERING_PILOT",
        "preregistered_plan_canonical_sha256":digest(planned),
        "original_R83F_and_R83G_results_not_changed":True,
        "real_native_cases_verified":36,
        "independent_physical_parent_pairs":9,
        "paired_source_cases":18,
        "model_trained_or_selected":False,
        "validation_test_final_files_read":False,
        "time_varying_wheel_command_demonstrated":False,
        "unqualified_research_only":True,
        "safety_provisional_900_rad_s_and_command_rating_below_1":(
            max_speed<=SPEED_BOUND_RAD_S and max_ratio<=1.00001),
        "observed_max_target_speed_rad_s":max_speed,
        "observed_max_command_to_rated_ratio":max_ratio,
        "physical_excitation_by_mechanism":by_fault,
        "original_pairs":report,
    }
    write_json(output,return_result)
    return return_result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("campaign","planned-spec","output"):
        p.add_argument("--"+name,required=True,type=Path)
    a=p.parse_args(argv)
    report=review_real_native_gain_pilot(
        campaign=a.campaign,planned_spec=a.planned_spec,output=a.output)
    print("R83I_REAL_NATIVE_GAIN_QA",report["real_native_cases_verified"],
          "max_command_to_rated",report["observed_max_command_to_rated_ratio"],
          "provisional_safety",report["safety_provisional_900_rad_s_and_command_rating_below_1"])
    return 0


if __name__=="__main__":
    raise SystemExit(main())
