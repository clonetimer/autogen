"""Metrics and transparent ontology for the multi-component benchmark."""
from __future__ import annotations
import numpy as np
from .adapters import sat_multicomponent as native

ONTOLOGY={
 "eps":{
  "NOMINAL":(None,None,"NOMINAL"),
  "SOLAR_A_EFFICIENCY_LOSS":("A","solar_a","SOLAR_EFFICIENCY_LOSS"),
  "SOLAR_B_EFFICIENCY_LOSS":("B","solar_b","SOLAR_EFFICIENCY_LOSS"),
  "LOAD_A_EXTRA":("A","load_a","EXTRA_LOAD"),
  "LOAD_B_EXTRA":("B","load_b","EXTRA_LOAD")},
 "thermal":{
  "NOMINAL":(None,None,"NOMINAL"),
  "THERMAL_A_EMISSIVITY_LOSS":("A","thermal_a","EMISSIVITY_LOSS"),
  "THERMAL_B_EMISSIVITY_LOSS":("B","thermal_b","EMISSIVITY_LOSS"),
  "THERMAL_A_EXTRA_HEAT":("A","thermal_a","EXTRA_HEAT"),
  "THERMAL_B_EXTRA_HEAT":("B","thermal_b","EXTRA_HEAT")},
 "data":{
  "NOMINAL":(None,None,"NOMINAL"),
  "SERVICE_A_RATE_LOSS":("A","service_a","SERVICE_RATE_LOSS"),
  "SERVICE_B_RATE_LOSS":("B","service_b","SERVICE_RATE_LOSS"),
  "GEN_A_EXCESS":("A","instrument_a","EXCESS_GENERATION"),
  "GEN_B_EXCESS":("B","instrument_b","EXCESS_GENERATION")},
}

def validate_ontology(domain):
    if domain not in ONTOLOGY:raise ValueError("UNKNOWN_MULTICOMPONENT_DOMAIN")
    if list(ONTOLOGY[domain])!=native.FAMILIES[domain]["classes"]:
        raise ValueError("MULTICOMPONENT_ONTOLOGY_CLASS_MISMATCH")
    return ONTOLOGY[domain]

def metrics(domain,truth_labels,predicted_labels):
    o=validate_ontology(domain)
    if len(truth_labels)!=len(predicted_labels) or not truth_labels:raise ValueError("MULTICOMPONENT_METRIC_LENGTH")
    if any(x not in o for x in [*truth_labels,*predicted_labels]):raise ValueError("MULTICOMPONENT_METRIC_CLASS")
    tf=np.asarray([x!="NOMINAL" for x in truth_labels]);pf=np.asarray([x!="NOMINAL" for x in predicted_labels])
    tn=int(np.sum(~tf&~pf));fp=int(np.sum(~tf&pf));fn=int(np.sum(tf&~pf));tp=int(np.sum(tf&pf))
    idx=np.flatnonzero(tf)
    slot=sum(o[truth_labels[i]][0]==o[predicted_labels[i]][0] for i in idx)
    component=sum(o[truth_labels[i]][1]==o[predicted_labels[i]][1] for i in idx)
    mechanism=sum(o[truth_labels[i]][2]==o[predicted_labels[i]][2] for i in idx)
    exact=sum(truth_labels[i]==predicted_labels[i] for i in idx)
    return {
      "cases":len(truth_labels),"fault_cases":int(len(idx)),
      "binary_detection_confusion":{"tn":tn,"fp":fp,"fn":fn,"tp":tp},
      "fault_detection_recall":tp/(tp+fn) if tp+fn else None,
      "nominal_window_false_alarm_rate":fp/(fp+tn) if fp+tn else None,
      "slot_A_B_localization_accuracy_on_faults":slot/len(idx) if len(idx) else None,
      "functional_component_accuracy_on_faults":component/len(idx) if len(idx) else None,
      "mechanism_accuracy_on_faults":mechanism/len(idx) if len(idx) else None,
      "exact_fault_class_accuracy_on_faults":exact/len(idx) if len(idx) else None,
      "independent_component_localization_metric":True,
      "causal_root_cause_identified":False,
    }

def metrics_from_scores(domain,truth_indices,scores):
    classes=native.FAMILIES[domain]["classes"]
    y=np.asarray(truth_indices,dtype=int);s=np.asarray(scores,dtype=float)
    if s.ndim!=2 or s.shape!=(len(y),len(classes)) or not np.isfinite(s).all():
        raise ValueError("MULTICOMPONENT_SCORE_SHAPE")
    truth=[classes[int(i)] for i in y]
    pred=[classes[int(i)] for i in np.argmax(s,axis=1)]
    return metrics(domain,truth,pred)
