"""Choose whether healthy commissioning is worth its extra deployment burden.

The decision uses development-validation and route-selection calibration only.
Test and risk-calibration data are not inputs. Commissioned is eligible only when
it is reference-arm noninferior for every class in both environments and strictly
improves worst-environment Macro-F1 or minimum recall. Ties stay with mechanism.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from .io import read_json, write_json, sha_file, seal, verify, digest
from .route_selection import _selection, _development

ARMS=("mechanism","commissioned")


def _selected_metrics(run, selection):
    run, dep, source, rec = _selection(run, selection)
    route=rec["chosen_route"];dev=_development(run)[route];cal=rec["calibration_metrics"][route]
    return run,dep,source,rec,dev,cal


def _check(a,b):
    ca=set(a.get("recall_by_class",{}));cb=set(b.get("recall_by_class",{}))
    if not ca or ca!=cb:raise ValueError("ARM_CLASS_METRICS_MISMATCH")
    vals=[a.get("macro_f1"),b.get("macro_f1"),*a["recall_by_class"].values(),*b["recall_by_class"].values()]
    if any(type(v) not in (int,float) or not np.isfinite(v) or not 0<=v<=1 for v in vals):raise ValueError("ARM_METRIC_INVALID")
    return sorted(ca)


def commissioned_eligible(mech_dev,mech_cal,comm_dev,comm_cal):
    classes=_check(mech_dev,comm_dev)
    if classes!=_check(mech_cal,comm_cal):raise ValueError("ARM_CLASS_ORDER_MISMATCH")
    eps=1e-12
    f_mech=min(mech_dev["macro_f1"],mech_cal["macro_f1"]);f_comm=min(comm_dev["macro_f1"],comm_cal["macro_f1"])
    recall_ok=all(comm["recall_by_class"][c]>=mech["recall_by_class"][c]-eps
                  for comm,mech in ((comm_dev,mech_dev),(comm_cal,mech_cal)) for c in classes)
    min_mech=min(min(x["recall_by_class"].values()) for x in (mech_dev,mech_cal))
    min_comm=min(min(x["recall_by_class"].values()) for x in (comm_dev,comm_cal))
    return bool(f_comm>=f_mech-eps and recall_ok and (f_comm>f_mech+eps or min_comm>min_mech+eps))


def select_arm(mechanism_run,mechanism_selection,commissioned_run,commissioned_selection,output):
    output=Path(output)
    if output.exists():raise FileExistsError("ARM_SELECTION_OUTPUT_EXISTS")
    mr,md,ms,mrec,mdev,mcal=_selected_metrics(mechanism_run,mechanism_selection)
    cr,cd,cs,crec,cdev,ccal=_selected_metrics(commissioned_run,commissioned_selection)
    if md["contract"]["classes"]!=cd["contract"]["classes"]:raise ValueError("ARM_CLASS_CONTRACT_CHANGED")
    eligible=commissioned_eligible(mdev,mcal,cdev,ccal)
    chosen="commissioned" if eligible else "mechanism"
    record={"schema":"phm.feature-arm-selection.v1","chosen_arm":chosen,"commissioned_eligible":eligible,
      "rule":"commissioned requires per-class noninferiority in development and route-selection calibration plus strict worst-environment improvement; ties prefer mechanism",
      "test_used":False,"risk_calibration_used":False,"risk_certified":False,"production_qualified":False,
      "arms":{
        "mechanism":{"run_manifest_sha256":ms,"selection_manifest_sha256":sha_file(Path(mechanism_selection)/"manifest.json"),"route":mrec["chosen_route"],"development":mdev,"calibration":mcal},
        "commissioned":{"run_manifest_sha256":cs,"selection_manifest_sha256":sha_file(Path(commissioned_selection)/"manifest.json"),"route":crec["chosen_route"],"development":cdev,"calibration":ccal},
      }}
    output.mkdir(parents=True,exist_ok=False);write_json(output/"selection.json",record);seal(output,["selection.json"]);return record


def select_hybrid_arm(mechanism_run,mechanism_selection,hybrid_run,hybrid_selection,output):
    """Same guard as select_arm, but records the augmented arm as hybrid."""
    output=Path(output)
    if output.exists():raise FileExistsError("ARM_SELECTION_OUTPUT_EXISTS")
    mr,md,ms,mrec,mdev,mcal=_selected_metrics(mechanism_run,mechanism_selection)
    hr,hd,hs,hrec,hdev,hcal=_selected_metrics(hybrid_run,hybrid_selection)
    if md["contract"]["classes"]!=hd["contract"]["classes"]:raise ValueError("ARM_CLASS_CONTRACT_CHANGED")
    eligible=commissioned_eligible(mdev,mcal,hdev,hcal);chosen="hybrid" if eligible else "mechanism"
    record={"schema":"phm.feature-arm-selection.v2","chosen_arm":chosen,"hybrid_eligible":eligible,
      "rule":"hybrid requires per-class noninferiority in development and route-selection calibration plus strict worst-environment improvement; ties prefer mechanism",
      "test_used":False,"risk_calibration_used":False,"risk_certified":False,"production_qualified":False,
      "arms":{
        "mechanism":{"run_manifest_sha256":ms,"selection_manifest_sha256":sha_file(Path(mechanism_selection)/"manifest.json"),"route":mrec["chosen_route"],"development":mdev,"calibration":mcal},
        "hybrid":{"run_manifest_sha256":hs,"selection_manifest_sha256":sha_file(Path(hybrid_selection)/"manifest.json"),"route":hrec["chosen_route"],"development":hdev,"calibration":hcal},
      }}
    output.mkdir(parents=True,exist_ok=False);write_json(output/"selection.json",record);seal(output,["selection.json"]);return record
