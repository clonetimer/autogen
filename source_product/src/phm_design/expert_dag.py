"""Cross-domain high-level expert-route DAG synthesis.

Unlike the ADCS mechanism-vector grammar in :mod:`phm_design.algorithm_dag`,
this module treats complete diagnostic pipelines as experts.  An expert may be
reference, fixed, generated, raw, mechanism, or another already-frozen route.
Search chooses a small runtime topology from *development-only* score records.
Truth is used only for offline scoring and is never an executor input.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import hashlib
import itertools
import json
import math

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score

SCHEMA = "phm.expert-dag.v1"
REPORT_SCHEMA = "phm.expert-dag-search-report.v1"


def _digest(value: Any) -> str:
    raw=json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def _scores(x, nclasses:int) -> np.ndarray:
    a=np.asarray(x,dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != nclasses or not np.isfinite(a).all() or np.any(a < 0):
        raise ValueError("EXPERT_DAG_SCORE_SHAPE")
    s=a.sum(1)
    if np.any(s <= 0): raise ValueError("EXPERT_DAG_ZERO_SCORE_SUM")
    return a/s[:,None]


@dataclass(frozen=True)
class ExpertSignals:
    expert_scores: dict[str,np.ndarray]


def _validate_candidate(c:dict)->None:
    if not isinstance(c,dict) or c.get("schema")!=SCHEMA: raise ValueError("EXPERT_DAG_SCHEMA")
    classes=c.get("classes"); experts=c.get("experts"); grammar=c.get("grammar")
    if not isinstance(classes,list) or len(classes)<2 or len(set(classes))!=len(classes): raise ValueError("EXPERT_DAG_CLASSES")
    if not isinstance(experts,list) or not experts or len(set(experts))!=len(experts): raise ValueError("EXPERT_DAG_EXPERTS")
    if not isinstance(grammar,dict): raise ValueError("EXPERT_DAG_GRAMMAR")
    kind=grammar.get("kind")
    if kind=="single_expert":
        if grammar.get("expert") not in experts: raise ValueError("EXPERT_DAG_EXPERT")
    elif kind=="max_confidence":
        subset=grammar.get("experts")
        if not isinstance(subset,list) or len(subset)<2 or any(x not in experts for x in subset): raise ValueError("EXPERT_DAG_EXPERT_SET")
    elif kind=="confidence_fallback":
        if grammar.get("primary") not in experts or grammar.get("fallback") not in experts or grammar.get("primary")==grammar.get("fallback"):
            raise ValueError("EXPERT_DAG_FALLBACK_EXPERTS")
        t=grammar.get("threshold")
        if type(t) not in (int,float) or not math.isfinite(float(t)) or not 0 <= float(t) <= 1: raise ValueError("EXPERT_DAG_THRESHOLD")
    elif kind=="disagreement_fallback":
        trio=[grammar.get("primary"),grammar.get("secondary"),grammar.get("fallback")]
        if any(x not in experts for x in trio) or len(set(trio))<3: raise ValueError("EXPERT_DAG_DISAGREEMENT_EXPERTS")
    else: raise ValueError("EXPERT_DAG_KIND")
    if c.get("truth_inputs_used_at_runtime") is not False: raise ValueError("EXPERT_DAG_RUNTIME_TRUTH_FORBIDDEN")


def execute(candidate:dict, signals:ExpertSignals)->dict:
    """Execute a frozen expert-route DAG with no truth access."""
    _validate_candidate(candidate)
    classes=candidate["classes"]; experts=candidate["experts"]; nclasses=len(classes)
    if set(signals.expert_scores)!=set(experts): raise ValueError("EXPERT_DAG_RUNTIME_EXPERT_SET")
    norm={k:_scores(v,nclasses) for k,v in signals.expert_scores.items()}
    n={len(v) for v in norm.values()}
    if len(n)!=1: raise ValueError("EXPERT_DAG_RUNTIME_CASE_COUNT")
    ncases=next(iter(n)); g=candidate["grammar"]; kind=g["kind"]
    used=np.empty(ncases,dtype=object)
    if kind=="single_expert":
        name=g["expert"]; out=norm[name]; used[:]=name
    elif kind=="max_confidence":
        names=g["experts"]
        conf=np.column_stack([np.max(norm[x],axis=1) for x in names])
        choice=np.argmax(conf,axis=1); out=np.empty((ncases,nclasses),dtype=float)
        for i,j in enumerate(choice): used[i]=names[int(j)]; out[i]=norm[used[i]][i]
    elif kind=="confidence_fallback":
        primary=g["primary"]; fallback=g["fallback"]; threshold=float(g["threshold"])
        mask=np.max(norm[primary],axis=1) < threshold
        out=norm[primary].copy(); out[mask]=norm[fallback][mask]
        used[:]=primary; used[mask]=fallback
    else:
        p,s,f=g["primary"],g["secondary"],g["fallback"]
        pp=np.argmax(norm[p],axis=1); sp=np.argmax(norm[s],axis=1); mask=pp!=sp
        out=norm[p].copy(); out[mask]=norm[f][mask]
        used[:]=p; used[mask]=f
    return {"scores":out,"class_index":np.argmax(out,axis=1),"expert_used":used.tolist()}


def make_candidate(classes:list[str], experts:list[str], grammar:dict)->dict:
    c={"schema":SCHEMA,"classes":list(classes),"experts":list(experts),"grammar":grammar,"truth_inputs_used_at_runtime":False}
    c["fingerprint"]=_digest(c); _validate_candidate(c); return c


def records_to_arrays(records:list[dict], classes:list[str], experts:list[str]):
    if not records: raise ValueError("EXPERT_DAG_EMPTY_RECORDS")
    truth=[]; scores={e:[] for e in experts}; groups=[]
    for r in records:
        if not isinstance(r,dict): raise ValueError("EXPERT_DAG_RECORD")
        y=r.get("truth")
        if y not in classes: raise ValueError("EXPERT_DAG_TRUTH")
        ex=r.get("experts")
        if not isinstance(ex,dict) or set(ex)!=set(experts): raise ValueError("EXPERT_DAG_RECORD_EXPERT_SET")
        truth.append(classes.index(y)); groups.append(r.get("group_id"))
        for e in experts: scores[e].append(ex[e])
    sig=ExpertSignals({e:_scores(np.asarray(v),len(classes)) for e,v in scores.items()})
    return sig,np.asarray(truth,dtype=int),groups


def metrics(truth:np.ndarray,pred:np.ndarray,classes:list[str])->dict:
    truth=np.asarray(truth,dtype=int); pred=np.asarray(pred,dtype=int)
    cm=confusion_matrix(truth,pred,labels=np.arange(len(classes)))
    recall={c:(float(cm[i,i]/cm[i].sum()) if cm[i].sum() else 0.0) for i,c in enumerate(classes)}
    return {"cases":int(len(truth)),"macro_f1":float(f1_score(truth,pred,labels=np.arange(len(classes)),average="macro",zero_division=0)),
            "balanced_accuracy":float(balanced_accuracy_score(truth,pred)),"recall_by_class":recall,"confusion_matrix":cm.astype(int).tolist()}


def _eligible(m:dict,constraints:dict)->tuple[bool,list[str]]:
    reasons=[]
    if m["macro_f1"] + 1e-12 < float(constraints.get("min_macro_f1",0)): reasons.append("MACRO_F1")
    if m["balanced_accuracy"] + 1e-12 < float(constraints.get("min_balanced_accuracy",0)): reasons.append("BALANCED_ACCURACY")
    if min(m["recall_by_class"].values()) + 1e-12 < float(constraints.get("min_all_class_recall",0)): reasons.append("ALL_CLASS_RECALL")
    return not reasons,reasons


def enumerate_candidates(classes:list[str],experts:list[str],config:dict)->list[dict]:
    structures=config.get("structures",["single_expert","confidence_fallback","max_confidence"])
    thresholds=[float(x) for x in config.get("confidence_thresholds",[0.5,0.6,0.7,0.8])]
    out=[]
    if "single_expert" in structures:
        out += [make_candidate(classes,experts,{"kind":"single_expert","expert":e}) for e in experts]
    if "confidence_fallback" in structures:
        for a,b,t in itertools.product(experts,experts,thresholds):
            if a!=b: out.append(make_candidate(classes,experts,{"kind":"confidence_fallback","primary":a,"fallback":b,"threshold":t}))
    if "max_confidence" in structures and len(experts)>=2:
        for k in range(2,len(experts)+1):
            for subset in itertools.combinations(experts,k): out.append(make_candidate(classes,experts,{"kind":"max_confidence","experts":list(subset)}))
    if "disagreement_fallback" in structures and len(experts)>=3:
        for p,s,f in itertools.permutations(experts,3): out.append(make_candidate(classes,experts,{"kind":"disagreement_fallback","primary":p,"secondary":s,"fallback":f}))
    # deterministic de-dup by fingerprint
    return list({c["fingerprint"]:c for c in out}.values())


def _complexity(c:dict)->tuple:
    g=c["grammar"]; kind=g["kind"]
    order={"single_expert":0,"confidence_fallback":1,"max_confidence":2,"disagreement_fallback":3}
    if kind=="single_expert": used=1
    elif kind=="max_confidence": used=len(g["experts"])
    elif kind=="confidence_fallback": used=2
    else: used=3
    return order[kind],used



def _validate_environment_evidence(environments:dict[str,list[dict]], config:dict) -> None:
    """Optional release-grade evidence checks; independent of candidate scoring.

    A repeated environment, world group, or case ID must not count twice toward
    an independent environment gate.  This guard does not inspect test data.
    """
    policy=config.get("evidence_policy") or {}
    if not isinstance(policy,dict):
        raise ValueError("EXPERT_DAG_EVIDENCE_POLICY")
    if not policy.get("require_disjoint_groups_and_cases",False) and not any(
        k in policy for k in ("min_environments","min_groups_per_environment","min_cases_per_class_per_environment")
    ):
        return
    nenv=int(policy.get("min_environments",2)); min_groups=int(policy.get("min_groups_per_environment",1))
    min_per_class=int(policy.get("min_cases_per_class_per_environment",1))
    if nenv<2 or min_groups<1 or min_per_class<1:
        raise ValueError("EXPERT_DAG_EVIDENCE_POLICY_INVALID")
    if len(environments)<nenv:
        raise ValueError("EXPERT_DAG_INSUFFICIENT_ENVIRONMENTS")
    classes=set(config["classes"]); seen_ids=set(); seen_groups={}
    for name,rows in sorted(environments.items()):
        local_ids=set(); local_groups=set(); counts={c:0 for c in classes}
        for row in rows:
            if not isinstance(row,dict):
                raise ValueError("EXPERT_DAG_EVIDENCE_RECORD")
            rid=row.get("id"); group=row.get("group_id"); label=row.get("truth")
            if not isinstance(rid,str) or not rid.strip() or not isinstance(group,str) or not group.strip():
                raise ValueError("EXPERT_DAG_EVIDENCE_IDS_REQUIRED")
            if label not in classes:raise ValueError("EXPERT_DAG_EVIDENCE_LABEL")
            if rid in local_ids:raise ValueError("EXPERT_DAG_DUPLICATED_CASE_ID")
            local_ids.add(rid);local_groups.add(group);counts[label]+=1
            if policy.get("require_disjoint_groups_and_cases",False):
                if rid in seen_ids:raise ValueError("EXPERT_DAG_ENVIRONMENT_CASE_OVERLAP")
                prior=seen_groups.get(group)
                if prior is not None and prior!=name:
                    raise ValueError("EXPERT_DAG_ENVIRONMENT_GROUP_OVERLAP")
        if len(local_groups)<min_groups or min(counts.values())<min_per_class:
            raise ValueError("EXPERT_DAG_EVIDENCE_TOO_SMALL")
        seen_ids.update(local_ids)
        for group in local_groups:seen_groups[group]=name

def search_multi_environment(environments:dict[str,list[dict]],config:dict)->dict:
    """Maximin search across development environments; fail closed if none qualifies."""
    if config.get("schema")!="phm.expert-dag-search.v1": raise ValueError("EXPERT_DAG_SEARCH_CONFIG")
    if config.get("data_role")!="development" or bool(config.get("retrospective_replay")): raise ValueError("EXPERT_DAG_FRESH_DEVELOPMENT_REQUIRED")
    if len(environments)<2: raise ValueError("EXPERT_DAG_MULTI_ENVIRONMENTS_REQUIRED")
    _validate_environment_evidence(environments,config)
    classes=list(config.get("classes") or []); experts=list(config.get("experts") or [])
    if len(classes)<2 or not experts: raise ValueError("EXPERT_DAG_SEARCH_AXES")
    candidates=enumerate_candidates(classes,experts,config)
    results=[]
    for c in candidates:
        env_metrics={}; usage={}; reasons=[]
        for name,records in sorted(environments.items()):
            sig,y,_=records_to_arrays(records,classes,experts); r=execute(c,sig); m=metrics(y,r["class_index"],classes)
            ok,why=_eligible(m,config.get("constraints",{})); env_metrics[name]=m
            usage[name]={x:r["expert_used"].count(x) for x in experts}
            if not ok: reasons += [name+":"+x for x in why]
        worst_recall=min(v for m in env_metrics.values() for v in m["recall_by_class"].values())
        worst_f1=min(m["macro_f1"] for m in env_metrics.values()); mean_f1=float(np.mean([m["macro_f1"] for m in env_metrics.values()]))
        results.append({"candidate":c,"eligible":not reasons,"ineligibility_reasons":reasons,"environment_metrics":env_metrics,
                        "expert_usage":usage,"worst_all_class_recall":worst_recall,"worst_macro_f1":worst_f1,"mean_macro_f1":mean_f1,
                        "complexity":list(_complexity(c))})
    eligible=[x for x in results if x["eligible"]]
    chosen_pool=eligible if eligible else []
    chosen=min(chosen_pool,key=lambda x:(-x["worst_all_class_recall"],-x["worst_macro_f1"],-x["mean_macro_f1"],tuple(x["complexity"]),x["candidate"]["fingerprint"])) if chosen_pool else None
    report={"schema":REPORT_SCHEMA,"status":"COMPLETED" if chosen else "FAIL_CLOSED_NO_ELIGIBLE_CANDIDATE",
            "data_role":"development","environment_count":len(environments),"environment_names":sorted(environments),
            "candidate_count":len(results),"eligible_candidate_count":len(eligible),"qualification_eligible":bool(chosen),
            "test_or_holdout_used_for_selection":False,"truth_used_for_search_scoring_only":True,"truth_inputs_used_at_runtime":False,
            "chosen_candidate":chosen["candidate"] if chosen else None,
            "chosen_worst_all_class_recall":chosen["worst_all_class_recall"] if chosen else None,
            "chosen_worst_macro_f1":chosen["worst_macro_f1"] if chosen else None,
            "chosen_mean_macro_f1":chosen["mean_macro_f1"] if chosen else None,
            "results":results}
    report["fingerprint"]=_digest(report); return report
