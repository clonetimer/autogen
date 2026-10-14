"""Device-independent generated correction graphs with a train-only reference.

Same existing graph grammar, mutation operators and CPU training engine. The
three graph arms have shared proposals; bounded-arm validation alone supplies
mutation feedback. Selection guard is empirical, NOT a risk certificate.
"""
from __future__ import annotations
import copy
from pathlib import Path
import numpy as np
from .data import Dataset, fit_scaler, transform, reject_numeric_overlap, reject_overlap, sequence_from_csv
from .models import Shape
from .resource_planner import plan_fixed_controls_runtime
from .reference import fit_reference, reference_logits
from .family_design import make_request
from .synthesis import propose, propose_random_control, training_config, export_selection, ranking, fixed_control_resource_plan
from .engine import fit, metrics, softmax, environment, implementation_hash, MODEL_EXECUTION_FILES, deterministic_cpu, load_ensemble, ensemble_scores
from .registry import default_registry
from .io import read_json, write_json, digest, seal, verify, run_lock, sha_file


REFERENCE_DESIGN_IMPLEMENTATION_FILES = MODEL_EXECUTION_FILES + (
    "family_design.py", "reference_design.py", "synthesis.py", "literature_baselines.py", "resource_planner.py", "optuna_proposal.py"
)

def reference_design_implementation_hash() -> str:
    return implementation_hash(REFERENCE_DESIGN_IMPLEMENTATION_FILES)


def score_report(y,scores,classes,threshold):
    predictions=scores.argmax(1);accept=scores.max(1)>=threshold;correct=predictions==y
    return {**metrics(y,predictions,classes),"accepted_cases":int(accept.sum()),
        "accepted_errors":int((accept&~correct).sum()),"accepted_correct":int((accept&correct).sum()),
        "coverage":float(accept.mean()),"fixed_uncalibrated_threshold":threshold,
        "probability_calibrated":False}


def choose_route(reference,bounded):
    # Legacy dev3 behavior. Never claim high-confidence error control from a
    # development criterion.
    if bounded is None:return "reference_only"
    noninferior=(bounded["macro_f1"]>=reference["macro_f1"]-1e-12 and
        bounded["accepted_errors"]<=reference["accepted_errors"] and
        bounded["accepted_correct"]>=reference["accepted_correct"])
    improves=(bounded["macro_f1"]>reference["macro_f1"]+1e-12 or
        bounded["accepted_errors"]<reference["accepted_errors"] or
        bounded["accepted_correct"]>reference["accepted_correct"])
    return "bounded" if noninferior and improves else "reference_only"


DEVELOPMENT_ROUTE_ORDER=("reference_only","fixed","bounded","raw","unbounded")


def choose_development_route(reference,winners,strategy="legacy_bounded_noninferiority"):
    """Choose a provisional development deployment route.

    This is deliberately separate from the later multi-environment selector.
    ``legacy_bounded_noninferiority`` reproduces historical behavior exactly.
    ``best_completed_route`` lets the design stage expose the strongest route it
    actually discovered instead of silently discarding raw/unbounded/fixed
    winners.  It is still development-only and never a risk certificate.
    """
    if strategy=="legacy_bounded_noninferiority":
        bounded=winners.get("bounded")
        return choose_route(reference,bounded["ensemble_validation"] if bounded else None)
    if strategy!="best_completed_route":raise ValueError("REFERENCE_DEVELOPMENT_ROUTE_SELECTION")
    candidates={"reference_only":reference}
    for mode in ("fixed","bounded","raw","unbounded"):
        winner=winners.get(mode)
        if winner and winner.get("status")=="COMPLETED":
            candidates[mode]=winner["ensemble_validation"]
    def rank(item):
        route,metric=item
        cases=max(1,int(metric.get("cases",1)))
        return (-float(metric["macro_f1"]),
                float(metric.get("accepted_errors",0))/cases,
                -float(metric.get("accepted_correct",0))/cases,
                DEVELOPMENT_ROUTE_ORDER.index(route))
    return min(candidates.items(),key=rank)[0]


def _feedback_parent_records(records:list[dict],strategy:str)->list[dict]:
    completed=[r for r in records if r.get("kind")=="generated" and r.get("status")=="COMPLETED"]
    if strategy=="bounded_only":
        return [r for r in completed if r.get("mode")=="bounded"]
    if strategy!="best_generated_route":
        raise ValueError("REFERENCE_FEEDBACK_PARENT_SELECTION")
    by_graph={}
    for rec in completed:
        h=rec["graph_sha256"]
        prior=by_graph.get(h)
        if prior is None or ranking(rec)<ranking(prior):
            by_graph[h]=rec
    return list(by_graph.values())


def portfolio_backend(generation:int)->str:
    """Fixed-budget proposer portfolio: shared startup, then Feedback/Optuna alternating."""
    if type(generation) is not int or generation < 0:
        raise ValueError("PORTFOLIO_GENERATION")
    if generation == 0:
        return "feedback"
    return "feedback" if generation % 2 == 1 else "optuna_tpe"

def design_reference(dataset_path,binding_path,policy_path,output,*,resume=False,on_trial_complete=None,proposal_strategy="feedback"):
    if proposal_strategy not in {"feedback","random_control","optuna_tpe","portfolio_feedback_optuna"}:
        raise ValueError("REFERENCE_PROPOSAL_STRATEGY")
    data=Dataset(dataset_path,allowed_splits={"train","validation"});data.verify_files()
    binding=read_json(binding_path);policy=read_json(policy_path)
    legacy_policy={"schema","base_design","bound_logits","max_total_trials","selection_guard"}
    optional_policy={"feedback_parent_selection","development_route_selection"}
    if (not legacy_policy<=set(policy) or set(policy)-legacy_policy-optional_policy or
        policy["schema"]!="phm.reference-design.v1"):
        raise ValueError("REFERENCE_DESIGN_POLICY_SCHEMA")
    feedback_parent_selection=policy.get("feedback_parent_selection","bounded_only")
    if feedback_parent_selection not in {"bounded_only","best_generated_route"}:
        raise ValueError("REFERENCE_FEEDBACK_PARENT_SELECTION")
    development_route_selection=policy.get("development_route_selection","legacy_bounded_noninferiority")
    if development_route_selection not in {"legacy_bounded_noninferiority","best_completed_route"}:
        raise ValueError("REFERENCE_DEVELOPMENT_ROUTE_SELECTION")
    if policy["selection_guard"]!="development_noninferiority_strict_improvement":raise ValueError("REFERENCE_SELECTION_RULE")
    k=policy["bound_logits"]
    if type(k) not in (int,float) or not np.isfinite(k) or not 0<k<=10:raise ValueError("REFERENCE_CORRECTION_BOUND")
    if data.contract!=binding.get("contract"):raise ValueError("FAMILY_INPUT_BINDING_CHANGED")
    request=make_request(binding,policy["base_design"])
    shape=Shape(request["window"],len(data.contract["channels"]),len(data.contract["classes"]))
    train,val=data.load("train",shape.window),data.load("validation",shape.window);reject_numeric_overlap(train,val)
    scaler=fit_scaler(train.x);x,vx=transform(train.x,scaler),transform(val.x,scaler)
    ref=fit_reference(x,train.y,data.contract,train.samples)
    ev=request["evolution"];n_graphs=ev["initial_graphs"]+(ev["generations"]-1)*ev["offspring_per_generation"]
    planned=(3*n_graphs+len(request["controls"]))*len(request["seeds"])
    if type(policy["max_total_trials"]) is not int or planned>policy["max_total_trials"]:raise ValueError("REFERENCE_GLOBAL_TRIAL_BUDGET")
    out=Path(output)
    # Runtime resource planning is environment-specific. On resume, reuse the
    # already persisted plan instead of re-benchmarking and risking transient
    # wall-clock noise changing the effective control specification.
    persisted_plan = read_json(out/"design_plan.json") if (out/"design_plan.json").exists() else None
    if persisted_plan is not None and "fixed_control_resource_plan" in persisted_plan:
        control_plans=copy.deepcopy(persisted_plan["fixed_control_resource_plan"])
    else:
        control_plans=plan_fixed_controls_runtime(shape,request["controls"],request["budget"],len(request["seeds"]),
            request["training"],len(train.y),len(val.y))
    effective_controls=[p["effective_spec"] for p in control_plans]
    control_plan_by_spec={digest(p["effective_spec"]):p for p in control_plans}
    fingerprint=digest({"dataset":data.hash,"binding":binding,"policy":policy,"reference":ref,
                        "proposal_strategy":proposal_strategy,"code":reference_design_implementation_hash(),"environment":environment(),
                        "fixed_control_resource_plan":control_plans})
    classes=data.contract["classes"];threshold=request["abstention"]["min_uncalibrated_score"]
    ref_val=score_report(val.y,softmax(reference_logits(ref,vx)),classes,threshold)
    with run_lock(out):
        if (out/"design_plan.json").exists():
            if read_json(out/"design_plan.json")["fingerprint"]!=fingerprint:raise ValueError("REFERENCE_RESUME_CHANGED")
            if not resume:raise FileExistsError("DESIGN_EXISTS_USE_RESUME")
            if (out/"manifest.json").exists():verify(out);return read_json(out/"design_report.json")
        else:
            write_json(out/"design_plan.json",{"fingerprint":fingerprint,"request":request,"policy":policy,
                "planned_trials":planned,"fixed_control_resource_plan":control_plans,"holdout_access":False,
                "proposal_strategy":proposal_strategy,
                "proposal_feedback":(feedback_parent_selection if proposal_strategy=="feedback" else
                    "optuna_topology_level_development_validation" if proposal_strategy=="optuna_tpe" else
                    "alternating_feedback_optuna_topology_level_development_validation" if proposal_strategy=="portfolio_feedback_optuna" else "none_random_control"),
                "matched_scope":"same graph count/seeds/epochs/parameter budget/resource gates/routes; proposal backend differs only after shared generation-0"})
        write_json(out/"reference.json",ref);write_json(out/"preprocessing.json",scaler)
        write_json(out/"fixed_control_resource_plan.json",{"schema":"phm.fixed-control-resource-plans.v1","plans":control_plans})
        write_json(out/"development_snapshot.json",data.raw)
        write_json(out/"observation_hashes.json",[digest(a.tolist()) for a in np.concatenate((train.x,val.x))])
        archive=[];parent_archive=[];seen=set()
        if proposal_strategy=="feedback":
            proposer=propose
        elif proposal_strategy=="random_control":
            proposer=propose_random_control
        elif proposal_strategy=="optuna_tpe":
            from .optuna_proposal import propose_optuna_tpe
            proposer=propose_optuna_tpe
        else:
            proposer=None
        for generation in range(ev["generations"]):
            active_strategy=proposal_strategy
            if proposal_strategy=="portfolio_feedback_optuna":
                active_strategy=portfolio_backend(generation)
                if active_strategy=="feedback":
                    proposer=propose
                    proposal_history=parent_archive
                else:
                    from .optuna_proposal import propose_optuna_tpe
                    proposer=propose_optuna_tpe
                    proposal_history=archive
            else:
                proposal_history = archive if proposal_strategy=="optuna_tpe" else parent_archive
            proposals=proposer(request,data.contract,generation,proposal_history,seen)
            if proposal_strategy=="portfolio_feedback_optuna":
                proposals=copy.deepcopy(proposals)
                proposals["portfolio_backend"]=active_strategy
            specs=[];associations={}
            for p in proposals["accepted"]:
                for mode in ("raw","bounded","unbounded"):
                    spec=(p["spec"] if mode=="raw" else {"family":"reference_dag","parameters":{
                        "reference":copy.deepcopy(ref),"graph":p["spec"]["parameters"],"mode":mode,"bound_logits":k}})
                    specs.append(spec);associations[digest(spec)]=(p,mode)
            if generation==0:specs+=effective_controls
            cfg=training_config(request,specs);cfg["budget"]["max_trials"]=len(specs)*len(request["seeds"])
            folder=out/f"generation_{generation:02d}"
            if (folder/"proposals.json").exists():
                if read_json(folder/"proposals.json")!=proposals or read_json(folder/"search.json")!=cfg:raise ValueError("REFERENCE_GENERATION_CHANGED")
            else:write_json(folder/"proposals.json",proposals);write_json(folder/"search.json",cfg)
            report=fit(data.path,folder/"search.json",folder/"run",resume=resume,on_trial_complete=on_trial_complete)
            generation_generated=[]
            for candidate in report["candidate_results"]:
                association=associations.get(digest(candidate["spec"]));p,mode=association if association else (None,"fixed")
                rec={key:candidate[key] for key in ("id","spec","description","status")}
                rec.update(generation=generation,kind="generated" if p else "fixed_control",mode=mode)
                if not p:
                    rec["resource_plan"]=control_plan_by_spec.get(digest(candidate["spec"]))
                if rec["status"]=="COMPLETED":
                    rec.update({key:candidate[key] for key in ("mean_validation_macro_f1","seed_std_macro_f1","ensemble_validation")})
                    scores=np.asarray([t["validation_scores"] for t in candidate["trials"]]).mean(0)
                    rec["ensemble_validation"]=score_report(val.y,scores,classes,threshold)
                if p:
                    rec.update({key:p[key] for key in ("genes","graph_sha256","topology_sha256","parents","edit","resources")})
                    seen.add(p["graph_sha256"])
                    generation_generated.append(rec)
                archive.append(rec)
            if proposal_strategy in {"feedback","portfolio_feedback_optuna"}:
                parent_archive.extend(_feedback_parent_records(generation_generated,feedback_parent_selection))
            write_json(out/"lineage.json",archive)
        winners={}
        for mode in ("raw","bounded","unbounded","fixed"):
            options=sorted([a for a in archive if a["mode"]==mode and a["status"]=="COMPLETED"],key=ranking)
            winner=options[0] if options else None;winners[mode]=winner
            if winner:
                dest=out/("selected_"+mode)
                if dest.exists():
                    if not (dest/"manifest.json").exists():
                        import shutil;shutil.rmtree(dest)
                    else:verify(dest)
                if not dest.exists():export_selection(out,winner,fingerprint,dest)
        route=choose_development_route(ref_val,winners,development_route_selection)
        from .family_pipeline import implementation_sha as feature_implementation_sha, TRANSFORM
        from .commissioning import implementation_sha as commissioning_implementation_sha, TRANSFORM as COMMISSIONING_TRANSFORM
        from .thermal_commissioning_v2 import implementation_sha as thermal_v2_implementation_sha, TRANSFORM as THERMAL_V2_TRANSFORM
        from .commissioning_hybrid import implementation_sha as hybrid_implementation_sha, HYBRID_SCHEMA
        transform_id=binding.get("feature_pipeline")
        if transform_id==TRANSFORM:
            feature_source=feature_implementation_sha(); requires_commissioning=False; requires_thermal_health=False
        elif transform_id==COMMISSIONING_TRANSFORM:
            feature_source=commissioning_implementation_sha(); requires_commissioning=True; requires_thermal_health=False
        elif transform_id==THERMAL_V2_TRANSFORM:
            feature_source=thermal_v2_implementation_sha(); requires_commissioning=True; requires_thermal_health=True
        elif transform_id==HYBRID_SCHEMA:
            feature_source=hybrid_implementation_sha(); requires_commissioning=True; requires_thermal_health=False
        else:
            feature_source=None; requires_commissioning=False; requires_thermal_health=False
        deployment={"raw_feature_transform":transform_id,"raw_feature_transform_sha256":feature_source,"commissioning_profile_required":requires_commissioning,"thermal_health_model_required":requires_thermal_health,"schema":"phm.reference-deployment.v1","route":route,"reference":"reference.json",
            "neural":None if route=="reference_only" else "selected_"+route,"contract":data.contract,
            "shape":shape.__dict__,"threshold":threshold,"fingerprint":fingerprint,
            "rule":"development_noninferiority_strict_improvement","risk_certified":False}
        write_json(out/"deployment.json",deployment)
        result={"schema":"phm.reference-design-report.v1","fingerprint":fingerprint,
            "status":"COMPLETED" if all(a["status"]=="COMPLETED" for a in archive) else "INCOMPLETE_TRIALS",
            "reference_validation":ref_val,"winners":winners,"deployment_route":route,
            "development_route_selection":development_route_selection,
            "planned_trials":planned,"candidate_count":len(archive),"generated_topologies":len(seen),
            "train_only_reference_fits":1,"proposal_strategy":proposal_strategy,
            "test_used":False,"claim_status":"RESEARCH_ONLY",
            "limits":["reference fitted repeatedly for integrity checks, not extra architecture trials",
              "reference buffers reported separately from trainable parameters","no calibrated risk guarantee",
              "reference-only fallback is not a neural performance gain","bounds cannot correct confident reference mistakes"]}
        write_json(out/"design_report.json",result)
        seal(out,[str(p.relative_to(out)) for p in out.rglob('*') if p.is_file() and p.name not in {'.lock','manifest.json','evaluation_ledger.json'} and not p.name.endswith('.tmp')])
        return result


def deployment_scores(run,x):
    run=Path(run);verify(run);deploy=read_json(run/"deployment.json")
    if deploy["schema"]!="phm.reference-deployment.v1":raise ValueError("REFERENCE_DEPLOYMENT_SCHEMA")
    route=deploy["route"]
    if route=="reference_only":return softmax(reference_logits(read_json(run/"reference.json"),x))
    if route not in {"fixed","bounded","raw","unbounded"}:raise ValueError("REFERENCE_ROUTE_INVALID")
    expected="selected_"+route
    if deploy.get("neural")!=expected or not (run/expected).exists():raise ValueError("REFERENCE_ROUTE_INVALID")
    _,models=load_ensemble(run/expected);return ensemble_scores(models,x)


def evaluate_reference(run,holdout,output):
    from .engine import evaluate
    run,output=Path(run),Path(output)
    with run_lock(run),deterministic_cpu():
        verify(run);deploy=read_json(run/"deployment.json")
        data=Dataset(holdout,allowed_splits={"test"})
        if data.contract!=deploy["contract"]:raise ValueError("HOLDOUT_CONTRACT_MISMATCH")
        reject_overlap(read_json(run/"development_snapshot.json")["samples"],data.samples)
        key=digest({"run":deploy["fingerprint"],"holdout":data.hash})
        ledger=run/"evaluation_ledger.json"
        if ledger.exists():
            prior=read_json(ledger)
            if prior["key"]!=key or prior["status"]!="COMPLETE":raise ValueError("HOLDOUT_ALREADY_OPENED_REVIEW_REQUIRED")
            return prior["report"]
        if output.exists():raise FileExistsError("EVALUATION_OUTPUT_EXISTS")
        data.verify_files();write_json(ledger,{"key":key,"status":"OPENED"})
        block=data.load("test",deploy["shape"]["window"])
        if set(read_json(run/"observation_hashes.json")) & {digest(v.tolist()) for v in block.x}:raise ValueError("IDENTICAL_OBSERVATIONS_ACROSS_ROLES")
        x=transform(block.x,read_json(run/"preprocessing.json"));classes=data.contract["classes"];threshold=deploy["threshold"]
        scores={"reference":softmax(reference_logits(read_json(run/"reference.json"),x))}
        output.mkdir(parents=True,exist_ok=False)
        for mode in ("raw","bounded","unbounded","fixed"):
            folder=run/("selected_"+mode)
            if folder.exists():
                evaluate(folder,holdout,output/mode)
                _,models=load_ensemble(folder);scores[mode]=ensemble_scores(models,x)
        deploy_key="reference" if deploy["route"]=="reference_only" else deploy["route"]
        if deploy_key not in scores:raise ValueError("REFERENCE_ROUTE_INVALID")
        scores["deployment"]=scores[deploy_key]
        result={"schema":"phm.reference-evaluation.v1","fingerprint":deploy["fingerprint"],"holdout_hash":data.hash,
            "deployment_route":deploy["route"],"counts":data.counts(),
            "metrics":{k:score_report(block.y,v,classes,threshold) for k,v in scores.items()},
            "predictions":[{"id":s["id"],"group_id":s["group_id"],"truth":int(y),
                "scores":{k:v[i].tolist() for k,v in scores.items()}} for i,(s,y) in enumerate(zip(block.samples,block.y))],
            "test_used_for_selection":False,"risk_certified":False,"production_qualified":False}
        write_json(output/"evaluation_report.json",result)
        seal(output,[str(p.relative_to(output)) for p in output.rglob('*') if p.is_file() and p.name!='manifest.json'])
        write_json(ledger,{"key":key,"status":"COMPLETE","report":result});return result


def predict_reference(run,csv_path,passport_path,domain):
    from .family_pipeline import feature_contract,compute_features,implementation_sha as feature_implementation_sha
    from .adapters.sat_nonadcs import contract
    run=Path(run);verify(run);deploy=read_json(run/"deployment.json")
    if deploy.get("commissioning_profile_required"):
        raise ValueError("COMMISSIONING_PROFILE_REQUIRED_USE_COMMISSIONED_PREDICTOR")
    if deploy.get("raw_feature_transform") not in (None, TRANSFORM) or deploy.get("raw_feature_transform_sha256")!=feature_implementation_sha():raise ValueError("REFERENCE_FEATURE_IMPLEMENTATION_CHANGED")
    if deploy["contract"]!=feature_contract(domain,"mechanism"):raise ValueError("REFERENCE_RAW_PIPELINE_CONTRACT")
    raw=sequence_from_csv(Path(csv_path),contract(domain),deploy["shape"]["window"])
    features=compute_features(domain,raw,read_json(passport_path),"mechanism")
    with deterministic_cpu():scores=deployment_scores(run,transform(features[None],read_json(run/"preprocessing.json")))[0]
    j=int(scores.argmax());accepted=bool(scores[j]>=deploy["threshold"])
    return {"route":deploy["route"],"status":"PREDICTED" if accepted else "ABSTAIN_LOW_SCORE",
        "prediction":deploy["contract"]["classes"][j] if accepted else None,
        "closed_set_argmax":deploy["contract"]["classes"][j],"scores":scores.tolist(),
        "probability_calibrated":False,"claim_status":"RESEARCH_ONLY"}
