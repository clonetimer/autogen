"""Select among already-frozen feature representations without test access.

Each arm must already have completed architecture design and route selection.  This
stage compares the chosen route on development validation and the separate route-
selection calibration cohort.  It does not read risk-calibration or test data.
"""
from __future__ import annotations
from pathlib import Path
import math
from .io import read_json, write_json, sha_file, seal, verify

SCHEMA = "phm.feature-arm-selection.v1"


def _metric_pair(selection: dict) -> tuple[dict, dict, str]:
    route = selection.get("chosen_route")
    dev = selection.get("development_metrics", {}).get(route)
    cal = selection.get("calibration_metrics", {}).get(route)
    if not route or not isinstance(dev, dict) or not isinstance(cal, dict):
        raise ValueError("FEATURE_ARM_SELECTION_METRICS")
    classes = tuple(dev.get("recall_by_class", {}))
    if not classes or tuple(cal.get("recall_by_class", {})) != classes:
        raise ValueError("FEATURE_ARM_SELECTION_CLASSES")
    for m in (dev, cal):
        vals=[m.get("macro_f1"), *m["recall_by_class"].values()]
        if any(type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=1 for v in vals):
            raise ValueError("FEATURE_ARM_SELECTION_METRIC_VALUE")
    return dev, cal, route


def arm_rank(selection: dict, preference_index: int) -> tuple:
    dev, cal, _ = _metric_pair(selection)
    worst_f1 = min(dev["macro_f1"], cal["macro_f1"])
    weakest = min(min(dev["recall_by_class"].values()), min(cal["recall_by_class"].values()))
    mean_f1 = (dev["macro_f1"] + cal["macro_f1"]) / 2
    return (-worst_f1, -weakest, -mean_f1, preference_index)


def _sample_signature(selection: dict) -> tuple:
    samples=selection.get("calibration_samples")
    if not isinstance(samples,list) or not samples:
        raise ValueError("FEATURE_ARM_CALIBRATION_SAMPLES")
    return tuple((s.get("id"),s.get("group_id"),s.get("split"),s.get("label")) for s in samples)


def select_feature_arm(arms: dict[str, tuple[str|Path,str|Path]], output, *, tie_preference: list[str]) -> dict:
    if not isinstance(arms,dict) or len(arms)<2 or set(tie_preference)!=set(arms) or len(tie_preference)!=len(arms):
        raise ValueError("FEATURE_ARM_SET_OR_PREFERENCE")
    output=Path(output)
    if output.exists():raise FileExistsError("FEATURE_ARM_OUTPUT_EXISTS")
    records={};sample_sig=None
    for name,(run,selection) in arms.items():
        if not isinstance(name,str) or not name:raise ValueError("FEATURE_ARM_NAME")
        run,selection=Path(run),Path(selection);verify(run);verify(selection)
        rec=read_json(selection/"selection.json")
        if rec.get("schema")!="phm.reference-route-selection.v2" or rec.get("test_used") is not False:
            raise ValueError("FEATURE_ARM_SOURCE_SELECTION")
        sig=_sample_signature(rec)
        if sample_sig is None:sample_sig=sig
        elif sig!=sample_sig:raise ValueError("FEATURE_ARMS_NOT_MATCHED_COHORTS")
        dev,cal,route=_metric_pair(rec)
        records[name]={"run_manifest_sha256":sha_file(run/"manifest.json"),
            "selection_manifest_sha256":sha_file(selection/"manifest.json"),"route":route,
            "development":dev,"calibration":cal,"rank":None}
    for i,name in enumerate(tie_preference):records[name]["rank"]=list(arm_rank(read_json(Path(arms[name][1])/"selection.json"),i))
    chosen=min(tie_preference,key=lambda n:tuple(records[n]["rank"]))
    result={"schema":SCHEMA,"chosen_arm":chosen,"tie_preference":tie_preference,"arms":records,
        "rule":"maximize worst development/calibration macro-F1; then weakest per-class recall; then mean macro-F1; tie by frozen preference",
        "risk_calibration_used":False,"test_used":False,"production_qualified":False}
    output.mkdir(parents=True);write_json(output/"selection.json",result);seal(output,["selection.json"])
    return result


def select_feature_arm_config(config_path, output) -> dict:
    config_path=Path(config_path);c=read_json(config_path)
    if not isinstance(c,dict) or set(c)!={"schema","arms","tie_preference"} or c["schema"]!="phm.feature-arm-selection-request.v1":
        raise ValueError("FEATURE_ARM_REQUEST_SCHEMA")
    if not isinstance(c["arms"],dict):raise ValueError("FEATURE_ARM_REQUEST_ARMS")
    base=config_path.parent
    arms={}
    for name,entry in c["arms"].items():
        if not isinstance(entry,dict) or set(entry)!={"run","selection"}:raise ValueError("FEATURE_ARM_REQUEST_ENTRY")
        def resolve(v):
            p=Path(v);return p if p.is_absolute() else base/p
        arms[name]=(resolve(entry["run"]),resolve(entry["selection"]))
    return select_feature_arm(arms,output,tie_preference=c["tie_preference"])
