"""Multi-environment route and design-arm selection for dev11 research.

All environments are development or calibration roles. Risk calibration and test
data are explicitly excluded. This is an empirical robustness selector, not a
statistical nondegradation guarantee.
"""
from __future__ import annotations
from pathlib import Path
import copy
from statistics import NormalDist
import numpy as np
from .data import Dataset, transform, reject_overlap
from .engine import deterministic_cpu
from .reference_design import score_report
from .route_selection import (
    ROUTES, POLICY, _policy, _run, _scores, _development, _no_numeric_reuse
)
from .evidence_ledger import EvidenceLedger
from .io import read_json, write_json, digest, seal, verify, sha_file, run_lock

SCHEMA = "phm.multi-environment-route-selection.v1"
ARM_SCHEMA = "phm.multi-environment-arm-selection.v1"


def _metric_check(metric, classes):
    if set(metric.get("recall_by_class", {})) != set(classes):
        raise ValueError("MULTI_ENV_CLASS_METRICS_MISMATCH")
    vals=[metric.get("macro_f1"),*metric["recall_by_class"].values()]
    if any(type(v) not in (int,float) or not np.isfinite(v) or not 0<=v<=1 for v in vals):
        raise ValueError("MULTI_ENV_METRIC_INVALID")
    n=metric.get("cases")
    if type(n) is not int or n<=0: raise ValueError("MULTI_ENV_CASE_COUNT_INVALID")
    for k in ("accepted_cases","accepted_errors","accepted_correct"):
        if type(metric.get(k)) is not int or not 0<=metric[k]<=n:
            raise ValueError("MULTI_ENV_ACCEPTED_COUNTS_INVALID")


def _worst_f1(envs):
    return min(m["macro_f1"] for m in envs.values())


def _mean_f1(envs):
    return float(np.mean([m["macro_f1"] for m in envs.values()]))


def _balanced_accuracy(metric):
    """Mean per-class recall, derived from the audited class metrics.

    The formal multicomponent studies use equal support per class, but deriving
    this value from recall_by_class keeps the selector independent of any
    redundant summary field and aligned with the primary recall objective.
    """
    vals=list(metric["recall_by_class"].values())
    if not vals: raise ValueError("MULTI_ENV_CLASS_METRICS_MISMATCH")
    return float(np.mean(vals))


def _worst_balanced_accuracy(envs):
    return min(_balanced_accuracy(m) for m in envs.values())


def _mean_balanced_accuracy(envs):
    return float(np.mean([_balanced_accuracy(m) for m in envs.values()]))


def _worst_recall(envs):
    return min(v for m in envs.values() for v in m["recall_by_class"].values())


def _worst_error_rate(envs):
    return max(m["accepted_errors"]/m["cases"] for m in envs.values())


def _min_correct_rate(envs):
    return min(m["accepted_correct"]/m["cases"] for m in envs.values())


def noninferior(candidate, baseline):
    if set(candidate)!=set(baseline) or not candidate:
        raise ValueError("MULTI_ENV_SET_MISMATCH")
    classes=tuple(next(iter(baseline.values()))["recall_by_class"])
    eps=1e-12
    for name in candidate:
        _metric_check(candidate[name],classes);_metric_check(baseline[name],classes)
        if candidate[name]["macro_f1"] < baseline[name]["macro_f1"]-eps:
            return False
        for c in classes:
            if candidate[name]["recall_by_class"][c] < baseline[name]["recall_by_class"][c]-eps:
                return False
    return True


def strictly_improves(candidate, baseline):
    eps=1e-12
    return (_worst_f1(candidate)>_worst_f1(baseline)+eps or
            _worst_recall(candidate)>_worst_recall(baseline)+eps)


def environment_rank(envs, complexity=0):
    return (-_worst_f1(envs),-_mean_f1(envs),-_worst_recall(envs),
            _worst_error_rate(envs),-_min_correct_rate(envs),complexity)



def balanced_environment_rank(envs, complexity=0):
    """Lexicographic maximin ranking with no reference-anchor lock-in."""
    return (-_worst_recall(envs),-_worst_f1(envs),-_mean_f1(envs),
            _worst_error_rate(envs),-_min_correct_rate(envs),complexity)


CONFIDENCE_MAXIMIN_ALPHA=0.05

def _wilson_interval(k:int,n:int,z:float)->tuple[float,float]:
    if type(k) is not int or type(n) is not int or not 0<=k<=n or n<=0:
        raise ValueError("RECALL_INTERVAL_COUNTS")
    p=k/n;z2=z*z;den=1+z2/n
    center=(p+z2/(2*n))/den
    half=z*np.sqrt((p*(1-p)+z2/(4*n))/n)/den
    return max(0.0,float(center-half)),min(1.0,float(center+half))

def _worst_recall_interval(envs, *, family_size:int, alpha:float=CONFIDENCE_MAXIMIN_ALPHA):
    """Observed worst recall plus a Bonferroni-Wilson interval for that cell.

    This is used only to decide whether an observed maximin advantage is large
    enough to dominate secondary robustness metrics. It is not a formal
    noninferiority guarantee.
    """
    if type(family_size) is not int or family_size<=0 or not 0<alpha<1:
        raise ValueError("CONFIDENCE_MAXIMIN_POLICY")
    rows=[]
    for env_name,m in sorted(envs.items()):
        classes=list(m["recall_by_class"]);cm=m.get("confusion_matrix")
        if not isinstance(cm,list) or len(cm)!=len(classes):
            raise ValueError("CONFIDENCE_MAXIMIN_CONFUSION")
        for i,c in enumerate(classes):
            row=cm[i]
            if not isinstance(row,list) or len(row)!=len(classes):
                raise ValueError("CONFIDENCE_MAXIMIN_CONFUSION")
            n=sum(row);k=row[i]
            if type(n) is not int or type(k) is not int or n<=0 or not 0<=k<=n:
                raise ValueError("CONFIDENCE_MAXIMIN_CONFUSION")
            r=float(m["recall_by_class"][c])
            if abs(r-k/n)>1e-10:raise ValueError("CONFIDENCE_MAXIMIN_RECALL_MISMATCH")
            rows.append((r,env_name,c,k,n))
    worst=min(x[0] for x in rows)
    tied=[x for x in rows if abs(x[0]-worst)<=1e-12]
    cell=min(tied,key=lambda x:(x[4],x[1],x[2]))
    comparisons=max(1,family_size*len(rows))
    z=NormalDist().inv_cdf(1-alpha/(2*comparisons))
    lo,hi=_wilson_interval(cell[3],cell[4],z)
    return {"recall":cell[0],"lower":lo,"upper":hi,"environment":cell[1],
            "class":cell[2],"correct":cell[3],"cases":cell[4],
            "familywise_alpha":alpha,"comparison_count":comparisons}

def _confidence_aware_choice(items:dict[str,dict], *, complexity:dict[str,int], tie_preference:list[str]):
    """Choose among candidate environment blocks without one-cell lock-in.

    Candidates whose worst-recall confidence interval overlaps the best
    candidate's lower bound remain in a conservative equivalence set. Within
    that set, worst/mean Macro-F1, selective error/correct rates and complexity
    decide before observed worst recall is used as a late tie-break.
    """
    if set(items)!=set(complexity) or set(items)!=set(tie_preference):
        raise ValueError("CONFIDENCE_MAXIMIN_CANDIDATE_SET")
    family=len(items)
    intervals={n:_worst_recall_interval(envs,family_size=family) for n,envs in items.items()}
    best_obs=max(v["recall"] for v in intervals.values())
    best_lower=max(v["lower"] for v in intervals.values() if abs(v["recall"]-best_obs)<=1e-12)
    contenders={n for n,v in intervals.items() if v["upper"]+1e-12>=best_lower}
    ranks={}
    for n,envs in items.items():
        ranks[n]=(-_worst_f1(envs),-_mean_f1(envs),_worst_error_rate(envs),
                  -_min_correct_rate(envs),int(complexity[n]),-_worst_recall(envs),tie_preference.index(n))
    chosen=min(contenders,key=lambda n:ranks[n])
    return chosen,contenders,ranks,intervals

def choose_routes_confidence_aware(environment_metrics):
    if "development" not in environment_metrics or len(environment_metrics)<3:
        raise ValueError("AT_LEAST_TWO_SELECTION_ENVIRONMENTS_REQUIRED")
    routes=set.intersection(*(set(v) for v in environment_metrics.values()))
    if "reference" not in routes or not routes<=set(ROUTES):raise ValueError("MULTI_ENV_ROUTE_SET_INVALID")
    items={r:{e:environment_metrics[e][r] for e in environment_metrics} for r in routes}
    complexity={r:ROUTES.index(r) for r in routes};tie=[r for r in ROUTES if r in routes]
    chosen,contenders,ranks,intervals=_confidence_aware_choice(items,complexity=complexity,tie_preference=tie)
    return chosen,{r:r in contenders for r in routes},ranks,intervals


def _confidence_aware_conservative_choice(items:dict[str,dict], *, complexity:dict[str,int], tie_preference:list[str]):
    """Industrial-conservative confidence-aware maximin choice.

    The primary worst-class recall is treated exactly as in dev14: Bonferroni-
    Wilson overlap defines a contender set.  When candidates are statistically
    indistinguishable on that primary safety metric, prefer the lower-complexity
    frozen artifact *before* noisy secondary point estimates.  A more complex
    candidate therefore needs a clearly separated primary-recall advantage to
    override Occam preference.  This is a selector policy, not a performance
    guarantee.
    """
    if set(items)!=set(complexity) or set(items)!=set(tie_preference):
        raise ValueError("CONFIDENCE_MAXIMIN_CANDIDATE_SET")
    family=len(items)
    intervals={n:_worst_recall_interval(envs,family_size=family) for n,envs in items.items()}
    best_obs=max(v["recall"] for v in intervals.values())
    best_lower=max(v["lower"] for v in intervals.values() if abs(v["recall"]-best_obs)<=1e-12)
    contenders={n for n,v in intervals.items() if v["upper"]+1e-12>=best_lower}
    ranks={}
    for n,envs in items.items():
        ranks[n]=(int(complexity[n]),-_worst_f1(envs),-_mean_f1(envs),
                  _worst_error_rate(envs),-_min_correct_rate(envs),
                  -_worst_recall(envs),tie_preference.index(n))
    chosen=min(contenders,key=lambda n:ranks[n])
    return chosen,contenders,ranks,intervals


def choose_routes_confidence_aware_conservative(environment_metrics):
    if "development" not in environment_metrics or len(environment_metrics)<3:
        raise ValueError("AT_LEAST_TWO_SELECTION_ENVIRONMENTS_REQUIRED")
    routes=set.intersection(*(set(v) for v in environment_metrics.values()))
    if "reference" not in routes or not routes<=set(ROUTES):raise ValueError("MULTI_ENV_ROUTE_SET_INVALID")
    items={r:{e:environment_metrics[e][r] for e in environment_metrics} for r in routes}
    complexity={r:ROUTES.index(r) for r in routes};tie=[r for r in ROUTES if r in routes]
    chosen,contenders,ranks,intervals=_confidence_aware_conservative_choice(
        items,complexity=complexity,tie_preference=tie)
    return chosen,{r:r in contenders for r in routes},ranks,intervals


def _confidence_aware_balanced_choice(items:dict[str,dict], *, complexity:dict[str,int], tie_preference:list[str]):
    """Confidence-aware maximin with a recall-aligned secondary objective.

    The Bonferroni-Wilson worst-class-recall gate is unchanged.  Inside the
    resulting contender set, rank first by worst and mean balanced accuracy
    (mean class recall on the same matched environments), then by Macro-F1 and
    selective-risk summaries.  Complexity is deliberately late: simplicity is
    a tie-breaker, not a substitute for observed multi-environment recall.
    """
    if set(items)!=set(complexity) or set(items)!=set(tie_preference):
        raise ValueError("CONFIDENCE_MAXIMIN_CANDIDATE_SET")
    family=len(items)
    intervals={n:_worst_recall_interval(envs,family_size=family) for n,envs in items.items()}
    best_obs=max(v["recall"] for v in intervals.values())
    best_lower=max(v["lower"] for v in intervals.values() if abs(v["recall"]-best_obs)<=1e-12)
    contenders={n for n,v in intervals.items() if v["upper"]+1e-12>=best_lower}
    ranks={}
    for n,envs in items.items():
        ranks[n]=(-_worst_balanced_accuracy(envs),-_mean_balanced_accuracy(envs),
                  -_worst_f1(envs),-_mean_f1(envs),_worst_error_rate(envs),
                  -_min_correct_rate(envs),int(complexity[n]),-_worst_recall(envs),
                  tie_preference.index(n))
    chosen=min(contenders,key=lambda n:ranks[n])
    return chosen,contenders,ranks,intervals


def choose_routes_confidence_aware_balanced(environment_metrics):
    if "development" not in environment_metrics or len(environment_metrics)<3:
        raise ValueError("AT_LEAST_TWO_SELECTION_ENVIRONMENTS_REQUIRED")
    routes=set.intersection(*(set(v) for v in environment_metrics.values()))
    if "reference" not in routes or not routes<=set(ROUTES):raise ValueError("MULTI_ENV_ROUTE_SET_INVALID")
    items={r:{e:environment_metrics[e][r] for e in environment_metrics} for r in routes}
    complexity={r:ROUTES.index(r) for r in routes};tie=[r for r in ROUTES if r in routes]
    chosen,contenders,ranks,intervals=_confidence_aware_balanced_choice(
        items,complexity=complexity,tie_preference=tie)
    return chosen,{r:r in contenders for r in routes},ranks,intervals


def choose_routes_balanced(environment_metrics):
    """Choose the most balanced route over development + >=2 selection environments."""
    if "development" not in environment_metrics or len(environment_metrics)<3:
        raise ValueError("AT_LEAST_TWO_SELECTION_ENVIRONMENTS_REQUIRED")
    routes=set.intersection(*(set(v) for v in environment_metrics.values()))
    if "reference" not in routes or not routes<=set(ROUTES):
        raise ValueError("MULTI_ENV_ROUTE_SET_INVALID")
    ranks={}
    for route in sorted(routes,key=ROUTES.index):
        envs={e:environment_metrics[e][route] for e in environment_metrics}
        classes=tuple(next(iter(envs.values()))["recall_by_class"])
        for m in envs.values():_metric_check(m,classes)
        ranks[route]=balanced_environment_rank(envs,ROUTES.index(route))
    chosen=min(routes,key=lambda r:ranks[r])
    return chosen,{r:True for r in routes},ranks

def choose_routes(environment_metrics):
    """Choose route against reference over development + >=2 selection environments."""
    if "development" not in environment_metrics or len(environment_metrics)<3:
        raise ValueError("AT_LEAST_TWO_SELECTION_ENVIRONMENTS_REQUIRED")
    routes=set.intersection(*(set(v) for v in environment_metrics.values()))
    if "reference" not in routes or not routes<=set(ROUTES):
        raise ValueError("MULTI_ENV_ROUTE_SET_INVALID")
    ref={e:environment_metrics[e]["reference"] for e in environment_metrics}
    eligible={};ranks={}
    for route in sorted(routes,key=ROUTES.index):
        envs={e:environment_metrics[e][route] for e in environment_metrics}
        eligible[route]=(route=="reference" or (noninferior(envs,ref) and strictly_improves(envs,ref)))
        ranks[route]=environment_rank(envs,ROUTES.index(route))
    chosen=min((r for r in routes if eligible[r]),key=lambda r:ranks[r])
    return chosen,eligible,ranks


def select_route_multi(run, calibrations:dict[str,str|Path], output, *, registry, policy=None):
    """Consume multiple independent calibration environments in one frozen operation."""
    policy=_policy(policy);output=Path(output)
    if not isinstance(calibrations,dict) or len(calibrations)<2 or any(not k.isidentifier() for k in calibrations):
        raise ValueError("MULTI_ENV_CALIBRATIONS")
    with run_lock(Path(run)):
        run,dep,source=_run(run)
        if (run/"evaluation_ledger.json").exists():
            raise ValueError("SOURCE_HOLDOUT_ALREADY_OPENED")
        development_samples=read_json(run/"development_snapshot.json")["samples"]
        dev=_development(run)
        env_metrics={"development":dev};all_samples=list(development_samples);all_numeric=[]
        hashes={};score_dump={}
        seen_groups=set(s["group_id"] for s in development_samples)
        for name,path in sorted(calibrations.items()):
            data=Dataset(path,allowed_splits={"calibration"})
            if data.contract!=dep["contract"]:raise ValueError("CALIBRATION_CONTRACT_MISMATCH")
            reject_overlap(development_samples,data.samples)
            groups={s["group_id"] for s in data.samples}
            if seen_groups & groups:raise ValueError("CROSS_ENVIRONMENT_GROUP_REUSE")
            seen_groups|=groups;data.verify_files()
            block=data.load("calibration",dep["shape"]["window"])
            signatures=_no_numeric_reuse(block.x,read_json(run/"observation_hashes.json"))
            # Also reject numerical reuse across selection environments.
            prior={sig for sig,_ in all_numeric}
            if prior & set(signatures):raise ValueError("CROSS_ENVIRONMENT_NUMERIC_REUSE")
            x=transform(block.x,read_json(run/"preprocessing.json"))
            with deterministic_cpu():scores=_scores(run,x)
            env_metrics[name]={r:score_report(block.y,s,data.contract["classes"],dep["threshold"]) for r,s in scores.items()}
            score_dump[name]={"samples":data.samples,"truth":block.y.tolist(),"routes":{r:s.tolist() for r,s in scores.items()}}
            hashes[name]=data.hash;all_samples.extend(data.samples);all_numeric.extend((s,"calibration") for s in signatures)
        chosen,eligible,ranks=choose_routes(env_metrics)
        key=digest({"source":source,"calibrations":hashes,"policy":policy,"selector":sha_file(Path(__file__))})
        with EvidenceLedger(registry) as ledger:
            cached=ledger.claim(key,"selection",source,output,dep["contract"]["device_family"],all_samples,all_numeric)
            if cached is not None:
                if cached.resolve()!=output.resolve():raise ValueError("EXACT_REPLAY_USE_ORIGINAL_OUTPUT:"+str(cached))
                return read_json(cached/"selection.json")
            try:
                output.mkdir(parents=True,exist_ok=False)
                plan={"schema":"phm.multi-environment-route-selection-plan.v1","operation":key,
                  "source_run_manifest_sha256":source,"source_design_fingerprint":dep["fingerprint"],
                  "policy":policy,"calibration_dataset_hashes":hashes,
                  "selection_environment_count":len(calibrations),
                  "implementation_sha256":sha_file(Path(__file__))}
                selection_samples=[s for s in all_samples if s.get("split")=="calibration"]
                selection_signatures=[sig for sig,role in all_numeric if role=="calibration"]
                record={"schema":SCHEMA,"operation":key,"source_run_manifest_sha256":source,
                  "source_design_fingerprint":dep["fingerprint"],"contract":dep["contract"],"shape":dep["shape"],
                  "policy":policy,"calibration_dataset_hashes":hashes,
                  "calibration_samples":selection_samples,
                  "calibration_numeric_signatures":selection_signatures,
                  "selection_environment_count":len(calibrations),"environment_metrics":env_metrics,
                  "eligibility":eligible,"rank_keys":{r:list(v) for r,v in ranks.items()},
                  "chosen_route":chosen,"test_used":False,"risk_calibration_used":False,
                  "risk_certified":False,"production_qualified":False}
                write_json(output/"plan.json",plan);write_json(output/"selection.json",record);write_json(output/"scores.json",score_dump)
                seal(output,["plan.json","selection.json","scores.json"]);ledger.complete(key,output);return record
            except Exception as exc:
                ledger.fail(key,exc);raise


def _multi_selection(run,selection):
    run,dep,source=_run(run);selection=Path(selection);verify(selection)
    rec=read_json(selection/"selection.json")
    if rec.get("schema")!=SCHEMA or rec.get("chosen_route") not in ROUTES:
        raise ValueError("MULTI_ENV_SELECTION_SCHEMA")
    if rec["source_run_manifest_sha256"]!=source or rec["source_design_fingerprint"]!=dep["fingerprint"]:
        raise ValueError("MULTI_ENV_SELECTION_RUN_CHANGED")
    return run,dep,source,rec


def selected_environment_metrics(run,selection):
    _,_,_,rec=_multi_selection(run,selection);route=rec["chosen_route"]
    return {e:m[route] for e,m in rec["environment_metrics"].items()}


def select_arm_multi(arms:dict, baseline:str, output, *, tie_preference:list[str]|None=None):
    """Select representation/search arm after each arm has frozen its own route."""
    if not isinstance(arms,dict) or len(arms)<2 or baseline not in arms:
        raise ValueError("MULTI_ARM_CONFIG")
    if tie_preference is None:tie_preference=list(arms)
    if set(tie_preference)!=set(arms):raise ValueError("MULTI_ARM_TIE_PREFERENCE")
    payload={};env_names=None;classes=None
    for name,spec in arms.items():
        if set(spec)!={"run","selection","complexity"}:raise ValueError("MULTI_ARM_SPEC")
        run,dep,source,rec=_multi_selection(spec["run"],spec["selection"])
        envs={e:m[rec["chosen_route"]] for e,m in rec["environment_metrics"].items()}
        if env_names is None:env_names=set(envs);classes=dep["contract"]["classes"]
        if set(envs)!=env_names or dep["contract"]["classes"]!=classes:raise ValueError("MULTI_ARM_ENVIRONMENT_MISMATCH")
        payload[name]={"run_manifest_sha256":source,
          "selection_manifest_sha256":sha_file(Path(spec["selection"])/"manifest.json"),
          "route":rec["chosen_route"],"environments":envs,"complexity":int(spec["complexity"])}
    base=payload[baseline]["environments"];eligible={};ranks={}
    for name,item in payload.items():
        envs=item["environments"]
        eligible[name]=(name==baseline or (noninferior(envs,base) and strictly_improves(envs,base)))
        ranks[name]=(*environment_rank(envs,item["complexity"]),tie_preference.index(name))
    chosen=min((n for n in arms if eligible[n]),key=lambda n:ranks[n])
    record={"schema":ARM_SCHEMA,"baseline_arm":baseline,"chosen_arm":chosen,
      "rule":"nonbaseline arm requires per-class noninferiority in every development/selection environment plus strict worst-environment improvement",
      "eligibility":eligible,"rank_keys":{n:list(v) for n,v in ranks.items()},"arms":payload,
      "test_used":False,"risk_calibration_used":False,"risk_certified":False,"production_qualified":False}
    output=Path(output)
    if output.exists():raise FileExistsError("MULTI_ARM_OUTPUT_EXISTS")
    output.mkdir(parents=True);write_json(output/"selection.json",record);seal(output,["selection.json"]);return record


MATCHED_SCHEMA="phm.matched-multi-environment-selection.v1"


def _sample_identity(samples):
    return [(s["id"],s["group_id"],s["lineage_id"],s["split"],s["label"]) for s in samples]


def select_matched_multi(arms:dict, calibrations:dict, baseline:str, output, *, registry,
                         tie_preference:list[str]|None=None, policy=None,
                         selection_rule:str="reference_noninferiority"):
    """Compare multiple trained representations/search strategies in one evidence claim.

    ``calibrations`` is ``{environment: {arm: dataset_path}}``. Each environment
    must describe the same underlying case identities for every arm, while feature
    contracts may differ. Development metrics stay arm-local. Risk and test roles
    are forbidden.
    """
    policy=_policy(policy);output=Path(output)
    if selection_rule not in {"reference_noninferiority","maximin_class_recall","confidence_aware_maximin_class_recall","confidence_aware_conservative_maximin_class_recall","confidence_aware_balanced_maximin_class_recall"}:
        raise ValueError("MATCHED_SELECTION_RULE")
    if not isinstance(arms,dict) or len(arms)<2 or baseline not in arms:
        raise ValueError("MATCHED_ARMS")
    if tie_preference is None:tie_preference=list(arms)
    if set(tie_preference)!=set(arms):raise ValueError("MATCHED_TIE_PREFERENCE")
    if not isinstance(calibrations,dict) or len(calibrations)<2 or any(not k.isidentifier() for k in calibrations):
        raise ValueError("MATCHED_CALIBRATIONS")
    arm_info={}
    sources=[]
    for name,spec in arms.items():
        if set(spec)!={"run","complexity"}:raise ValueError("MATCHED_ARM_SPEC")
        run,dep,source=_run(spec["run"])
        if (run/"evaluation_ledger.json").exists():raise ValueError("SOURCE_HOLDOUT_ALREADY_OPENED")
        arm_info[name]={"run":run,"dep":dep,"source":source,"development":_development(run),
                        "complexity":int(spec["complexity"])}
        sources.append(source)
    classes=next(iter(arm_info.values()))["dep"]["contract"]["classes"]
    if any(x["dep"]["contract"]["classes"]!=classes for x in arm_info.values()):
        raise ValueError("MATCHED_CLASS_CONTRACT_CHANGED")

    env_metrics={name:{"development":info["development"]} for name,info in arm_info.items()}
    score_dump={};evidence_samples=[];numeric=[];hashes={}
    samples_by_arm={name:[] for name in arms}
    signatures_by_arm={name:[] for name in arms}
    seen_groups=set()
    for env_name,by_arm in sorted(calibrations.items()):
        if set(by_arm)!=set(arms):raise ValueError("MATCHED_ENV_ARM_SET")
        canonical_identity=None
        env_samples=None
        for arm_name,path in by_arm.items():
            info=arm_info[arm_name];data=Dataset(path,allowed_splits={"calibration"})
            if data.contract!=info["dep"]["contract"]:raise ValueError("CALIBRATION_CONTRACT_MISMATCH")
            development=read_json(info["run"]/"development_snapshot.json")["samples"]
            reject_overlap(development,data.samples);data.verify_files()
            identity=_sample_identity(data.samples)
            if canonical_identity is None:
                canonical_identity=identity;env_samples=data.samples
            elif identity!=canonical_identity:
                raise ValueError("MATCHED_ENV_SAMPLE_IDENTITIES_DIFFER")
            block=data.load("calibration",info["dep"]["shape"]["window"])
            signatures=_no_numeric_reuse(block.x,read_json(info["run"]/"observation_hashes.json"))
            if set(signatures) & set(signatures_by_arm[arm_name]):
                raise ValueError("CROSS_ENVIRONMENT_NUMERIC_REUSE")
            signatures_by_arm[arm_name].extend(signatures)
            samples_by_arm[arm_name].extend(data.samples)
            x=transform(block.x,read_json(info["run"]/"preprocessing.json"))
            with deterministic_cpu():scores=_scores(info["run"],x)
            env_metrics[arm_name][env_name]={r:score_report(block.y,s,classes,info["dep"]["threshold"]) for r,s in scores.items()}
            score_dump.setdefault(env_name,{})[arm_name]={"dataset_hash":data.hash,
                "truth":block.y.tolist(),"routes":{r:s.tolist() for r,s in scores.items()}}
            hashes.setdefault(env_name,{})[arm_name]=data.hash
        groups={s["group_id"] for s in env_samples}
        if seen_groups & groups:raise ValueError("CROSS_ENVIRONMENT_GROUP_REUSE")
        seen_groups|=groups;evidence_samples.extend(env_samples)
        # One canonical numeric signature per underlying evidence item; feature
        # representations may differ, so lineage is the primary matched identity.
        numeric.extend((digest([s["id"],s["sha256"]]),"calibration") for s in env_samples)

    arm_payload={};base_envs=None
    for name,info in arm_info.items():
        route_intervals=None
        if selection_rule=="confidence_aware_maximin_class_recall":
            chosen,route_eligible,route_ranks,route_intervals=choose_routes_confidence_aware(env_metrics[name])
        elif selection_rule=="confidence_aware_conservative_maximin_class_recall":
            chosen,route_eligible,route_ranks,route_intervals=choose_routes_confidence_aware_conservative(env_metrics[name])
        elif selection_rule=="confidence_aware_balanced_maximin_class_recall":
            chosen,route_eligible,route_ranks,route_intervals=choose_routes_confidence_aware_balanced(env_metrics[name])
        else:
            chooser=choose_routes_balanced if selection_rule=="maximin_class_recall" else choose_routes
            chosen,route_eligible,route_ranks=chooser(env_metrics[name])
        selected={e:env_metrics[name][e][chosen] for e in env_metrics[name]}
        arm_payload[name]={"run_manifest_sha256":info["source"],"route":chosen,
          "route_eligibility":route_eligible,"route_rank_keys":{r:list(v) for r,v in route_ranks.items()},
          "route_recall_intervals":route_intervals,
          "environments":selected,"complexity":info["complexity"],
          "proposal_strategy":read_json(info["run"]/"design_report.json").get("proposal_strategy","feedback")}
        if name==baseline:base_envs=selected
    eligible={};ranks={}
    for name,item in arm_payload.items():
        envs=item["environments"]
        if selection_rule=="maximin_class_recall":
            eligible[name]=True
            ranks[name]=(*balanced_environment_rank(envs,item["complexity"]),tie_preference.index(name))
        elif selection_rule in {"confidence_aware_maximin_class_recall","confidence_aware_conservative_maximin_class_recall","confidence_aware_balanced_maximin_class_recall"}:
            eligible[name]=True
        else:
            eligible[name]=(name==baseline or (noninferior(envs,base_envs) and strictly_improves(envs,base_envs)))
            ranks[name]=(*environment_rank(envs,item["complexity"]),tie_preference.index(name))
    arm_recall_intervals=None
    if selection_rule in {"confidence_aware_maximin_class_recall","confidence_aware_conservative_maximin_class_recall","confidence_aware_balanced_maximin_class_recall"}:
        arm_envs={n:item["environments"] for n,item in arm_payload.items()}
        arm_complexity={n:int(item["complexity"]) for n,item in arm_payload.items()}
        if selection_rule=="confidence_aware_conservative_maximin_class_recall":
            chooser=_confidence_aware_conservative_choice
        elif selection_rule=="confidence_aware_balanced_maximin_class_recall":
            chooser=_confidence_aware_balanced_choice
        else:
            chooser=_confidence_aware_choice
        chosen_arm,contenders,confidence_ranks,arm_recall_intervals=chooser(
            arm_envs,complexity=arm_complexity,tie_preference=tie_preference)
        eligible={n:n in contenders for n in arms}
        ranks=confidence_ranks
    else:
        chosen_arm=min((n for n in arms if eligible[n]),key=lambda n:ranks[n])
    chosen_info=arm_info[chosen_arm];chosen_route=arm_payload[chosen_arm]["route"]
    composite=digest({"sources":sorted(sources),"calibrations":hashes,"policy":policy,
                      "baseline":baseline,"tie_preference":tie_preference,"selection_rule":selection_rule,
                      "selector":sha_file(Path(__file__))})
    evidence_domains=sorted({info["dep"]["contract"]["device_family"] for info in arm_info.values()})
    with EvidenceLedger(registry) as ledger:
        cached=ledger.claim(composite,"selection",digest(sorted(sources)),output,evidence_domains,evidence_samples,numeric)
        if cached is not None:
            if cached.resolve()!=output.resolve():raise ValueError("EXACT_REPLAY_USE_ORIGINAL_OUTPUT:"+str(cached))
            return read_json(cached/"selection.json")
        try:
            output.mkdir(parents=True,exist_ok=False)
            if selection_rule=="maximin_class_recall":
                rule_text="maximin across all environments/classes: worst per-class recall, then worst/mean Macro-F1, error rate, correct rate, complexity"
            elif selection_rule=="confidence_aware_maximin_class_recall":
                rule_text="confidence-aware maximin: retain candidates whose Bonferroni-Wilson worst-recall interval overlaps the best lower bound; then rank worst/mean Macro-F1, error/correct rates, complexity, observed worst recall"
            elif selection_rule=="confidence_aware_conservative_maximin_class_recall":
                rule_text="confidence-aware conservative maximin: Bonferroni-Wilson overlap defines the primary-recall contender set; among statistically indistinguishable contenders prefer lower complexity before secondary point-estimate metrics"
            elif selection_rule=="confidence_aware_balanced_maximin_class_recall":
                rule_text="confidence-aware balanced maximin: Bonferroni-Wilson overlap defines the primary-recall contender set; rank contenders by worst/mean balanced accuracy, then Macro-F1, selective risk summaries, and complexity"
            else:
                rule_text="one matched evidence use; each route uses development plus >=2 selection environments; nonbaseline arm requires per-class noninferiority in every environment plus strict worst-environment improvement"
            record={"schema":MATCHED_SCHEMA,"operation":composite,"baseline_arm":baseline,
              "chosen_arm":chosen_arm,"chosen_route":chosen_route,"selection_rule":selection_rule,
              "rule":rule_text,
              "eligibility":eligible,"rank_keys":{n:list(v) for n,v in ranks.items()},"arms":arm_payload,
              "arm_recall_intervals":arm_recall_intervals,
              "selection_environment_count":len(calibrations),"test_used":False,"risk_calibration_used":False,
              "risk_certified":False,"production_qualified":False}
            write_json(output/"selection.json",record);write_json(output/"scores.json",score_dump)
            # Freeze a standard route artifact for every arm before test. This
            # permits later diagnostic comparison without allowing test-driven reselection.
            route_paths=[]
            for arm_name,info in arm_info.items():
                sub=output/"routes"/arm_name;sub.mkdir(parents=True)
                dep=info["dep"];route=arm_payload[arm_name]["route"]
                selected_record={"schema":SCHEMA,"operation":composite+":"+arm_name,
                  "source_run_manifest_sha256":info["source"],
                  "source_design_fingerprint":dep["fingerprint"],"contract":dep["contract"],"shape":dep["shape"],
                  "policy":policy,"calibration_dataset_hashes":{e:hashes[e][arm_name] for e in hashes},
                  "calibration_samples":samples_by_arm[arm_name],
                  "calibration_numeric_signatures":signatures_by_arm[arm_name],
                  "calibration_role_consumed_for_route_selection":True,
                  "probability_calibration_performed":False,
                  "selection_environment_count":len(calibrations),
                  "environment_metrics":env_metrics[arm_name],
                  "eligibility":arm_payload[arm_name]["route_eligibility"],
                  "rank_keys":arm_payload[arm_name]["route_rank_keys"],"chosen_route":route,
                  "test_used":False,"risk_calibration_used":False,"risk_certified":False,"production_qualified":False}
                write_json(sub/"plan.json",{"schema":"phm.multi-environment-route-plan.v1","operation":composite,"arm":arm_name})
                write_json(sub/"selection.json",selected_record)
                write_json(sub/"scores.json",{e:score_dump[e][arm_name] for e in score_dump})
                seal(sub,["plan.json","selection.json","scores.json"])
                route_paths += [f"routes/{arm_name}/plan.json",f"routes/{arm_name}/selection.json",
                                f"routes/{arm_name}/scores.json",f"routes/{arm_name}/manifest.json"]
            # Convenience copy of the chosen route keeps downstream CLI simple.
            import shutil
            shutil.copytree(output/"routes"/chosen_arm,output/"chosen_route")
            seal(output,["selection.json","scores.json",*route_paths,
                         "chosen_route/plan.json","chosen_route/selection.json",
                         "chosen_route/scores.json","chosen_route/manifest.json"])
            ledger.complete(composite,output);return record
        except Exception as exc:
            ledger.fail(composite,exc);raise

def bind_matched_route_evidence(run, matched, arm:str, calibrations:dict[str,str|Path], output):
    """Materialize already-consumed matched selection evidence for risk guards.

    No prediction or reselection occurs here. The function validates the exact
    calibration datasets recorded by the sealed matched selector and records the
    sample/numeric identities required by the legacy risk-calibration interface.
    """
    run,dep,source=_run(run);matched=Path(matched);verify(matched)
    top=read_json(matched/"selection.json");scores=read_json(matched/"scores.json")
    if top.get("schema")!=MATCHED_SCHEMA or arm not in top.get("arms",{}):
        raise ValueError("MATCHED_BIND_SCHEMA_OR_ARM")
    if top["arms"][arm]["run_manifest_sha256"]!=source:
        raise ValueError("MATCHED_BIND_RUN_CHANGED")
    frozen=read_json(matched/"routes"/arm/"selection.json")
    route=top["arms"][arm]["route"]
    if frozen["chosen_route"]!=route or frozen["source_run_manifest_sha256"]!=source:
        raise ValueError("MATCHED_BIND_ROUTE_CHANGED")
    if set(calibrations)!=set(scores):raise ValueError("MATCHED_BIND_ENVIRONMENTS")
    development=read_json(run/"development_snapshot.json")["samples"]
    samples=[];numeric=[];seen_groups=set();seen_numeric=set()
    for env,path in sorted(calibrations.items()):
        data=Dataset(path,allowed_splits={"calibration"})
        if data.contract!=dep["contract"]:raise ValueError("CALIBRATION_CONTRACT_MISMATCH")
        if data.hash!=scores[env][arm]["dataset_hash"]:raise ValueError("MATCHED_BIND_DATASET_CHANGED")
        reject_overlap(development,data.samples);data.verify_files()
        groups={x["group_id"] for x in data.samples}
        if seen_groups & groups:raise ValueError("CROSS_ENVIRONMENT_GROUP_REUSE")
        seen_groups|=groups
        block=data.load("calibration",dep["shape"]["window"])
        sigs=_no_numeric_reuse(block.x,read_json(run/"observation_hashes.json"))
        if seen_numeric & set(sigs):raise ValueError("CROSS_ENVIRONMENT_NUMERIC_REUSE")
        seen_numeric|=set(sigs);samples.extend(data.samples);numeric.extend(sigs)
    record=copy.deepcopy(frozen)
    record["calibration_samples"]=samples
    record["calibration_numeric_signatures"]=numeric
    record["derived_from_matched_manifest_sha256"]=sha_file(matched/"manifest.json")
    record["no_reselection_performed"]=True
    output=Path(output)
    if output.exists():raise FileExistsError("MATCHED_BIND_OUTPUT_EXISTS")
    output.mkdir(parents=True)
    write_json(output/"plan.json",{"schema":"phm.matched-route-evidence-bind.v1","arm":arm})
    write_json(output/"selection.json",record)
    write_json(output/"scores.json",{})
    seal(output,["plan.json","selection.json","scores.json"])
    return record
