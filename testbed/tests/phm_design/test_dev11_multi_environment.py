from __future__ import annotations
import csv, copy, random
from pathlib import Path
import numpy as np
import pytest
from phm_design.data import Dataset
from phm_design.io import read_json,write_json,sha_file,verify
from phm_design.reference_design import design_reference
from phm_design.synthesis import propose,propose_random_control
from phm_design.family_design import make_request
from phm_design.multi_environment_selection import choose_routes,select_route_multi,select_matched_multi,bind_matched_route_evidence,noninferior,strictly_improves
from tests.phm_design.test_design_system import fixture_dataset
from tests.phm_design.test_reference_dev3 import make_policy
from tests.phm_design.test_compositional_design import request


def metric(f1,recalls,errors=0,correct=9,cases=12):
    return {"macro_f1":f1,"recall_by_class":dict(zip(["C0","C1","C2"],recalls)),
            "cases":cases,"accepted_errors":errors,"accepted_correct":correct,
            "accepted_cases":errors+correct}


def test_multi_environment_guard_requires_every_environment_and_class():
    ref={"development":metric(.70,[.7,.6,.8]),"cal_a":metric(.68,[.7,.55,.8]),"cal_b":metric(.66,[.65,.55,.78])}
    good={"development":metric(.75,[.75,.65,.85]),"cal_a":metric(.72,[.72,.60,.82]),"cal_b":metric(.69,[.68,.58,.80])}
    bad=copy.deepcopy(good);bad["cal_b"]["recall_by_class"]["C1"]=.50
    assert noninferior(good,ref) and strictly_improves(good,ref)
    assert not noninferior(bad,ref)


def test_choose_routes_uses_worst_environment_not_single_validation():
    ref={"development":metric(.70,[.7,.6,.8]),"cal_a":metric(.69,[.7,.6,.8]),"cal_b":metric(.68,[.68,.58,.8])}
    raw={"development":metric(.90,[.9,.9,.9]),"cal_a":metric(.75,[.75,.7,.8]),"cal_b":metric(.62,[.7,.45,.8])}
    fixed={"development":metric(.78,[.8,.7,.85]),"cal_a":metric(.76,[.8,.7,.82]),"cal_b":metric(.72,[.75,.62,.8])}
    env={e:{"reference":ref[e],"raw":raw[e],"fixed":fixed[e]} for e in ref}
    chosen,eligible,ranks=choose_routes(env)
    assert not eligible["raw"] and eligible["fixed"] and chosen=="fixed"


def tiny_request(contract):
    p=make_policy()
    p["base_design"]["evolution"].update(initial_graphs=1,generations=2,offspring_per_generation=1,elite_parents=1,max_total_trials=20)
    p["base_design"]["training"]["epochs"]=2
    p["base_design"]["training"]["patience"]=2
    p["base_design"]["budget"]["max_trial_seconds"]=30
    p["max_total_trials"]=14
    b={"schema":"phm.device-binding.v1","contract":contract,"groups":request()["groups"],
       "window_samples":8,"quality_mode":"complete_observation_only_missing_values_rejected"}
    return p,b


def test_random_control_matches_initial_pool_and_ignores_feedback(tmp_path):
    dev,_=fixture_dataset(tmp_path/"data")
    d=Dataset(dev,allowed_splits={"train","validation"})
    p,b=tiny_request(d.contract);req=make_request(b,p["base_design"])
    a=propose(req,d.contract,0,[],set());r=propose_random_control(req,d.contract,0,[],set())
    assert [x["graph_sha256"] for x in a["accepted"]]==[x["graph_sha256"] for x in r["accepted"]]
    assert r["feedback_role"]=="none_random_control"
    r1=propose_random_control(req,d.contract,1,[{"mean_validation_macro_f1":1.0}],set(x["graph_sha256"] for x in r["accepted"]))
    assert r1["parent_pool"]==[] and all(x["parents"]==[] for x in r1["accepted"])


def test_feedback_optuna_and_random_design_have_matched_trial_budget(tmp_path):
    pytest.importorskip("optuna")
    dev,_=fixture_dataset(tmp_path/"data_optuna")
    d=Dataset(dev,allowed_splits={"train","validation"});p,b=tiny_request(d.contract)
    write_json(tmp_path/"po.json",p);write_json(tmp_path/"bo.json",b)
    f=design_reference(dev,tmp_path/"bo.json",tmp_path/"po.json",tmp_path/"feedback_o")
    o=design_reference(dev,tmp_path/"bo.json",tmp_path/"po.json",tmp_path/"optuna_o",proposal_strategy="optuna_tpe")
    r=design_reference(dev,tmp_path/"bo.json",tmp_path/"po.json",tmp_path/"random_o",proposal_strategy="random_control")
    assert f["planned_trials"]==o["planned_trials"]==r["planned_trials"]==14
    assert f["candidate_count"]==o["candidate_count"]==r["candidate_count"]
    assert o["proposal_strategy"]=="optuna_tpe"
    p0f=read_json(tmp_path/"feedback_o/generation_00/proposals.json")
    p0o=read_json(tmp_path/"optuna_o/generation_00/proposals.json")
    p0r=read_json(tmp_path/"random_o/generation_00/proposals.json")
    hashes=lambda x:[g["graph_sha256"] for g in x["accepted"]]
    assert hashes(p0f)==hashes(p0o)==hashes(p0r)
    p1=read_json(tmp_path/"optuna_o/generation_01/proposals.json")
    assert p1["feedback_role"]=="development_validation_optuna_tpe"
    assert p1["parent_pool"] and p1["external_optimizer"]["library"]=="optuna"
    plan=read_json(tmp_path/"optuna_o/design_plan.json")
    assert plan["proposal_strategy"]=="optuna_tpe"
    assert plan["proposal_feedback"]=="optuna_topology_level_development_validation"


def test_feedback_and_random_design_have_matched_trial_budget(tmp_path):
    dev,_=fixture_dataset(tmp_path/"data")
    d=Dataset(dev,allowed_splits={"train","validation"});p,b=tiny_request(d.contract)
    write_json(tmp_path/"p.json",p);write_json(tmp_path/"b.json",b)
    f=design_reference(dev,tmp_path/"b.json",tmp_path/"p.json",tmp_path/"feedback")
    r=design_reference(dev,tmp_path/"b.json",tmp_path/"p.json",tmp_path/"random",proposal_strategy="random_control")
    assert f["planned_trials"]==r["planned_trials"]==14
    assert f["candidate_count"]==r["candidate_count"]
    assert f["proposal_strategy"]=="feedback" and r["proposal_strategy"]=="random_control"
    p0f=read_json(tmp_path/"feedback/generation_00/proposals.json")
    p0r=read_json(tmp_path/"random/generation_00/proposals.json")
    assert [x["graph_sha256"] for x in p0f["accepted"]]==[x["graph_sha256"] for x in p0r["accepted"]]
    assert read_json(tmp_path/"feedback/generation_01/proposals.json")["parent_pool"]
    assert read_json(tmp_path/"random/generation_01/proposals.json")["parent_pool"]==[]


def calibration(root:Path,contract,seed:int,prefix:str):
    rng=np.random.default_rng(seed);samples=[];root.mkdir(parents=True,exist_ok=True)
    for label in range(3):
        for j in range(2):
            ident=f"{prefix}-{label}-{j}";path=root/(ident+".csv")
            x=rng.normal(label*2,.18,(12,2))
            with path.open("w",newline="") as f:
                w=csv.writer(f);w.writerow(["time_s",*[c["name"] for c in contract["channels"]]])
                w.writerows([[i,*row] for i,row in enumerate(x)])
            samples.append({"id":ident,"lineage_id":ident,"group_id":ident,"split":"calibration",
                "label":contract["classes"][label],"path":path.name,"sha256":sha_file(path)})
    out=root/"dataset.json";write_json(out,{"schema":"phm.sequence-dataset.v1","dataset_id":prefix,
        "contract":contract,"samples":samples,"provenance":{"source":"software_fixture"}});return out


def test_multi_environment_selection_is_sealed_and_replayable(tmp_path):
    dev,_=fixture_dataset(tmp_path/"data");d=Dataset(dev,allowed_splits={"train","validation"})
    p,b=tiny_request(d.contract);write_json(tmp_path/"p.json",p);write_json(tmp_path/"b.json",b)
    run=tmp_path/"run";design_reference(dev,tmp_path/"b.json",tmp_path/"p.json",run)
    a=calibration(tmp_path/"ca",d.contract,9917,"ca");bb=calibration(tmp_path/"cb",d.contract,11917,"cb")
    out=tmp_path/"selection";db=tmp_path/"ledger.sqlite"
    rec=select_route_multi(run,{"cal_a":a,"cal_b":bb},out,registry=db)
    assert rec["selection_environment_count"]==2 and rec["test_used"] is False
    assert len(rec["calibration_samples"])==12
    assert len(rec["calibration_numeric_signatures"])==12
    assert read_json(out/"plan.json")["schema"]=="phm.multi-environment-route-selection-plan.v1"
    verify(out)
    assert select_route_multi(run,{"cal_a":a,"cal_b":bb},out,registry=db)==rec


def test_matched_multi_consumes_shared_evidence_once_and_prefers_baseline_on_tie(tmp_path):
    dev,_=fixture_dataset(tmp_path/"data2");d=Dataset(dev,allowed_splits={"train","validation"})
    p,b=tiny_request(d.contract);write_json(tmp_path/"p2.json",p);write_json(tmp_path/"b2.json",b)
    run=tmp_path/"same_run";design_reference(dev,tmp_path/"b2.json",tmp_path/"p2.json",run)
    a=calibration(tmp_path/"mca",d.contract,22117,"mca")
    bb=calibration(tmp_path/"mcb",d.contract,23117,"mcb")
    cfg_arms={"feedback":{"run":run,"complexity":0},"random":{"run":run,"complexity":0}}
    envs={"cal_a":{"feedback":a,"random":a},"cal_b":{"feedback":bb,"random":bb}}
    out=tmp_path/"matched";rec=select_matched_multi(cfg_arms,envs,"feedback",out,registry=tmp_path/"matched.sqlite",
        tie_preference=["feedback","random"])
    assert rec["chosen_arm"]=="feedback"
    assert rec["eligibility"]=={"feedback":True,"random":False}
    verify(out);verify(out/"chosen_route")
    route=read_json(out/"chosen_route/selection.json")
    assert len(route["calibration_samples"])==12
    assert len(route["calibration_numeric_signatures"])==12


def test_training_hash_scope_excludes_evidence_and_cli_churn():
    from phm_design.engine import MODEL_EXECUTION_FILES,implementation_hash,training_implementation_hash
    from phm_design.reference_design import REFERENCE_DESIGN_IMPLEMENTATION_FILES,reference_design_implementation_hash
    assert "evidence_ledger.py" not in MODEL_EXECUTION_FILES
    assert "__main__.py" not in MODEL_EXECUTION_FILES
    assert "reference_design.py" in REFERENCE_DESIGN_IMPLEMENTATION_FILES
    assert "synthesis.py" in REFERENCE_DESIGN_IMPLEMENTATION_FILES
    assert training_implementation_hash()==implementation_hash(MODEL_EXECUTION_FILES)
    assert reference_design_implementation_hash()==implementation_hash(REFERENCE_DESIGN_IMPLEMENTATION_FILES)
    assert implementation_hash()!=training_implementation_hash()
