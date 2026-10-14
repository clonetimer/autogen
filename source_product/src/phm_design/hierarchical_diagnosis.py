"""Hierarchical PHM diagnosis reporting.

This module does NOT claim independent component localization or causal root-cause
identification. Current localization is a deterministic functional-component
mapping from the predicted mechanism class. Root-cause evidence is a transparent
summary of predeclared mechanism features available in the diagnosis window.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

SCHEMA="phm.hierarchical-diagnosis.v1"

ONTOLOGY={
 "eps":{
   "subsystem":"EPS",
   "classes":{
     "NOMINAL":{"component":None,"mechanism":"NOMINAL","evidence":[]},
     "SOLAR_EFFICIENCY_LOSS":{"component":"solar_generation_chain","mechanism":"solar_conversion_efficiency_loss",
       "evidence":[("solar_loss_running_mean","positive"),("solar_loss_fraction","positive")]},
     "EXTRA_BUS_LOAD":{"component":"load_bus_path","mechanism":"uncommanded_or_excess_bus_load",
       "evidence":[("load_innovation_w","positive")]},
   }},
 "thermal":{
   "subsystem":"THERMAL",
   "classes":{
     "NOMINAL":{"component":None,"mechanism":"NOMINAL","evidence":[]},
     "EMISSIVITY_LOSS":{"component":"thermal_radiation_surface","mechanism":"effective_emissivity_loss",
       "evidence":[("radiation_loss_slope","positive")]},
     "EXTRA_INTERNAL_HEAT":{"component":"internal_heat_source_path","mechanism":"uncommanded_internal_heat",
       "evidence":[("unexplained_heat_intercept_w","positive"),("energy_residual_running_w","positive")]},
   }},
 "data":{
   "subsystem":"DATA_MANAGEMENT",
   "classes":{
     "NOMINAL":{"component":None,"mechanism":"NOMINAL","evidence":[]},
     "SERVICE_RATE_LOSS":{"component":"service_downlink_path","mechanism":"service_rate_loss",
       "evidence":[("service_loss_fraction","positive"),("contact_innovation_running_bps","positive")]},
     "EXCESS_GENERATION":{"component":"instrument_generation_path","mechanism":"excess_data_generation",
       "evidence":[("generation_excess_running_bps","positive")]},
   }},
}


def ontology(domain:str)->dict:
    if domain not in ONTOLOGY: raise ValueError("UNKNOWN_DIAGNOSTIC_DOMAIN")
    # immutable-by-convention constant; return only through consumers
    return ONTOLOGY[domain]


def _feature_names(contract:dict)->list[str]:
    channels=contract.get("channels")
    if not isinstance(channels,list) or not channels:
        raise ValueError("DIAGNOSTIC_FEATURE_CONTRACT")
    names=[c.get("name") for c in channels]
    if any(not isinstance(x,str) or not x for x in names) or len(names)!=len(set(names)):
        raise ValueError("DIAGNOSTIC_FEATURE_CONTRACT")
    return names


def _feature_summary(seq:np.ndarray,index:int,direction:str)->dict:
    v=np.asarray(seq[:,index],dtype=np.float64)
    if v.ndim!=1 or not len(v) or not np.isfinite(v).all():
        raise ValueError("DIAGNOSTIC_FEATURE_VALUES")
    start=max(0,len(v)-max(1,len(v)//4))
    tail=v[start:]
    med=float(np.median(tail))
    if direction=="positive": consistency=float(np.mean(tail>0))
    elif direction=="negative": consistency=float(np.mean(tail<0))
    else: raise ValueError("DIAGNOSTIC_EVIDENCE_DIRECTION")
    return {"final":float(v[-1]),"tail_median":med,"max_abs":float(np.max(np.abs(v))),
            "expected_direction":direction,"tail_direction_consistency":consistency}


def diagnosis_from_scores(domain:str,scores,feature_sequence,feature_contract:dict,threshold:float|None)->dict:
    spec=ontology(domain)
    classes=list(spec["classes"])
    s=np.asarray(scores,dtype=np.float64)
    x=np.asarray(feature_sequence,dtype=np.float64)
    if s.shape!=(len(classes),) or not np.isfinite(s).all():
        raise ValueError("DIAGNOSTIC_SCORE_SHAPE")
    if x.ndim!=2 or x.shape[1]!=len(feature_contract.get("channels",[])) or not np.isfinite(x).all():
        raise ValueError("DIAGNOSTIC_FEATURE_SHAPE")
    names=_feature_names(feature_contract)
    best=int(np.argmax(s));label=classes[best];confidence=float(s[best])
    accepted=bool(threshold is not None and confidence>=threshold)
    # threshold=None means fail-closed under current selective-risk policy.
    if not accepted:
        status="ABSTAIN_RISK_POLICY"
    else:
        status="PREDICTED"
    item=spec["classes"][label]
    evidence=[]
    for name,direction in item["evidence"]:
        if name not in names: raise ValueError("DIAGNOSTIC_EVIDENCE_FEATURE_MISSING:"+name)
        rec=_feature_summary(x,names.index(name),direction)
        evidence.append({"feature":name,**rec})
    if label=="NOMINAL":
        detection="NOMINAL" if accepted else "UNKNOWN"
        location=None;mechanism="NOMINAL" if accepted else None
        hypothesis=None
    elif accepted:
        detection="FAULT"
        location=item["component"];mechanism=item["mechanism"]
        hypothesis={"component":location,"mechanism":mechanism}
    else:
        detection="UNKNOWN";location=None;mechanism=None;hypothesis=None
    return {
      "schema":SCHEMA,"domain":domain,"subsystem":spec["subsystem"],"status":status,
      "closed_set_argmax":label,"closed_set_confidence":confidence,"scores":s.tolist(),
      "detection":detection,"functional_location":location,"mechanism":mechanism,
      "root_cause_hypothesis":hypothesis,"evidence":evidence,
      "localization_basis":"DETERMINISTIC_FROM_MECHANISM_CLASS",
      "root_cause_basis":"PREDECLARED_MECHANISM_FEATURE_SIGNATURE",
      "independent_component_localization":False,
      "causal_root_cause_identified":False,
      "claim_status":"RESEARCH_ONLY",
    }


def hierarchical_metrics(domain:str,truth_labels:list[str],predicted_labels:list[str])->dict:
    spec=ontology(domain)
    valid=set(spec["classes"])
    if len(truth_labels)!=len(predicted_labels) or not truth_labels:
        raise ValueError("DIAGNOSTIC_METRIC_LENGTH")
    if any(x not in valid for x in truth_labels+predicted_labels):
        raise ValueError("DIAGNOSTIC_METRIC_CLASS")
    t_fault=np.asarray([x!="NOMINAL" for x in truth_labels],dtype=bool)
    p_fault=np.asarray([x!="NOMINAL" for x in predicted_labels],dtype=bool)
    tp=int(np.sum(t_fault&p_fault));fn=int(np.sum(t_fault&~p_fault));fp=int(np.sum(~t_fault&p_fault));tn=int(np.sum(~t_fault&~p_fault))
    fault_recall=tp/(tp+fn) if tp+fn else None
    nominal_false_alarm=fp/(fp+tn) if fp+tn else None
    fault_idx=np.flatnonzero(t_fault)
    true_loc=[spec["classes"][truth_labels[i]]["component"] for i in fault_idx]
    pred_loc=[spec["classes"][predicted_labels[i]]["component"] for i in fault_idx]
    loc_correct=sum(a==b for a,b in zip(true_loc,pred_loc))
    mech_correct=sum(truth_labels[i]==predicted_labels[i] for i in fault_idx)
    return {
      "cases":len(truth_labels),
      "binary_detection_confusion":{"tn":tn,"fp":fp,"fn":fn,"tp":tp},
      "fault_detection_recall":fault_recall,
      "nominal_window_false_alarm_rate":nominal_false_alarm,
      "functional_localization_accuracy_on_fault_cases":loc_correct/len(fault_idx) if len(fault_idx) else None,
      "mechanism_accuracy_on_fault_cases":mech_correct/len(fault_idx) if len(fault_idx) else None,
      "localization_is_independent_task":False,
      "root_cause_is_causally_identified":False,
    }


def _evaluate_loaded(domain,data,block,pred,report,output_path)->dict:
    from pathlib import Path
    from .io import write_json
    samples=pred.get("samples");truth=pred.get("truth");scores=pred.get("scores")
    if not isinstance(samples,list) or len(samples)!=len(data.samples) or len(truth)!=len(samples) or len(scores)!=len(samples):
        raise ValueError("HIERARCHY_PREDICTION_LENGTH")
    if [s["id"] for s in samples] != [s["id"] for s in block.samples]:
        raise ValueError("HIERARCHY_SAMPLE_ORDER")
    classes=data.contract["classes"]
    if any(type(y) is not int or not 0<=y<len(classes) for y in truth):
        raise ValueError("HIERARCHY_TRUTH_INDEX")
    true_labels=[classes[y] for y in truth]
    predicted=[classes[int(np.argmax(np.asarray(s,dtype=float)))] for s in scores]
    threshold=report.get("risk_threshold") if report.get("risk_status")=="QUALIFIED_THRESHOLD" else None
    reports=[]
    accepted=[]
    evidence_consistency=[]
    for i,(s,x) in enumerate(zip(scores,block.x)):
        rec=diagnosis_from_scores(domain,s,x,data.contract,threshold)
        rec["sample_id"]=samples[i]["id"];rec["truth"]=true_labels[i]
        reports.append(rec)
        accepted.append(rec["status"]=="PREDICTED")
        if true_labels[i]!="NOMINAL" and predicted[i]==true_labels[i]:
            evidence_consistency.extend(e["tail_direction_consistency"] for e in rec["evidence"])
    metrics=hierarchical_metrics(domain,true_labels,predicted)
    metrics["selective_accepted_cases"]=int(sum(accepted))
    metrics["selective_coverage"]=float(np.mean(accepted))
    metrics["mean_expected_direction_consistency_on_correct_fault_cases"]=(
        float(np.mean(evidence_consistency)) if evidence_consistency else None)
    result={"schema":"phm.hierarchical-diagnosis-evaluation.v1","domain":domain,
      "source_risk_operation":report.get("operation"),"risk_status":report.get("risk_status"),
      "risk_threshold":threshold,"metrics":metrics,"cases":reports,
      "test_used_for_model_or_threshold_selection":False,
      "independent_component_localization":False,"causal_root_cause_identified":False,
      "production_qualified":False}
    write_json(Path(output_path),result);return result


def evaluate_feature_dataset(domain:str,dataset_path,risk_evaluation_dir,output_path)->dict:
    """Public evaluator with exact window inferred from the dataset files."""
    from pathlib import Path
    from .data import Dataset
    from .io import read_json
    data=Dataset(dataset_path,allowed_splits={"test"});data.verify_files()
    pred=read_json(Path(risk_evaluation_dir)/"predictions.json")
    report=read_json(Path(risk_evaluation_dir)/"evaluation_report.json")
    # Infer the window from the prediction count and the first CSV. Dataset.load
    # still performs contract/hash validation.
    first=data.root/data.samples[0]["path"]
    with first.open(encoding="utf-8") as f:
        n=sum(1 for _ in f)-1
    block=data.load("test",n)
    return _evaluate_loaded(domain,data,block,pred,report,output_path)


def _component_slot(component:str|None)->str|None:
    if component is None:return None
    if component.endswith("_a"):return "A"
    if component.endswith("_b"):return "B"
    raise ValueError("MULTICOMPONENT_LOCATION_SLOT")


def _component_type(component:str|None)->str|None:
    if component is None:return None
    if component.endswith("_a") or component.endswith("_b"):
        return component[:-2]
    raise ValueError("MULTICOMPONENT_COMPONENT_TYPE")


def multicomponent_mapping(domain:str)->dict[str,dict]:
    from .adapters.sat_multicomponent import FAMILIES
    if domain not in FAMILIES:raise ValueError("UNKNOWN_MULTICOMPONENT_DOMAIN")
    s=FAMILIES[domain]
    return {label:{"component":_component_type(s["truth"][i][0]),
                   "slot":_component_slot(s["truth"][i][0]),
                   "component_instance":s["truth"][i][0],
                   "mechanism":s["truth"][i][1]}
            for i,label in enumerate(s["classes"])}


def decomposed_metrics(truth_labels:list[str],predicted_labels:list[str],mapping:dict[str,dict])->dict:
    """Separate detection, component localization, mechanism and joint-class accuracy."""
    if len(truth_labels)!=len(predicted_labels) or not truth_labels:
        raise ValueError("DIAGNOSTIC_METRIC_LENGTH")
    valid=set(mapping)
    if any(x not in valid for x in truth_labels+predicted_labels):
        raise ValueError("DIAGNOSTIC_METRIC_CLASS")
    tn=fp=fn=tp=0;fault_n=0;loc_ok=0;slot_ok=0;instance_ok=0;mech_ok=0;joint_ok=0
    for truth,pred in zip(truth_labels,predicted_labels):
        tf=truth!="NOMINAL";pf=pred!="NOMINAL"
        if tf and pf:tp+=1
        elif tf and not pf:fn+=1
        elif not tf and pf:fp+=1
        else:tn+=1
        if tf:
            fault_n+=1
            loc_ok+=int(mapping[truth]["component"]==mapping[pred]["component"])
            if "slot" in mapping[truth] and "slot" in mapping[pred]:
                slot_ok+=int(mapping[truth]["slot"]==mapping[pred]["slot"])
            if "component_instance" in mapping[truth] and "component_instance" in mapping[pred]:
                instance_ok+=int(mapping[truth]["component_instance"]==mapping[pred]["component_instance"])
            mech_ok+=int(mapping[truth]["mechanism"]==mapping[pred]["mechanism"])
            joint_ok+=int(truth==pred)
    return {
      "cases":len(truth_labels),"fault_cases":fault_n,
      "binary_detection_confusion":{"tn":tn,"fp":fp,"fn":fn,"tp":tp},
      "fault_detection_recall":tp/(tp+fn) if tp+fn else None,
      "nominal_window_false_alarm_rate":fp/(fp+tn) if fp+tn else None,
      "component_localization_accuracy_on_fault_cases":loc_ok/fault_n if fault_n else None,
      "component_slot_accuracy_on_fault_cases":slot_ok/fault_n if fault_n and any("slot" in v for v in mapping.values()) else None,
      "exact_component_instance_accuracy_on_fault_cases":instance_ok/fault_n if fault_n and any("component_instance" in v for v in mapping.values()) else None,
      "mechanism_accuracy_on_fault_cases":mech_ok/fault_n if fault_n else None,
      "joint_fault_class_accuracy_on_fault_cases":joint_ok/fault_n if fault_n else None,
      "component_and_mechanism_targets_are_separately_scored":True,
      "causal_root_cause_identified":False,
    }


def multicomponent_metrics(domain:str,truth_labels:list[str],predicted_labels:list[str])->dict:
    return decomposed_metrics(truth_labels,predicted_labels,multicomponent_mapping(domain))
