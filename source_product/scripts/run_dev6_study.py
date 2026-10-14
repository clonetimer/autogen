#!/usr/bin/env python3
# 用途：执行冻结的多域配对研究并保存阶段检查点。
# 参数：--spec --policy --output [--resume]。
# 输出：protocol/progress、仿真数据、模型、选择与评价记录。
"""Portable, checkpointed challenge study; runs synchronously to completion.

No parameter tuning occurs after the policy freeze. Exact completed stages can
be resumed; failed native generations are preserved and must be reviewed rather
than automatically erased. Use a separate explicit output for any new study.
"""
from __future__ import annotations
import argparse
import copy
import json
import os
import sys
from pathlib import Path
from datetime import datetime, timezone
from phm_design.io import read_json, write_json, digest, sha_file, verify, seal
from phm_design.engine import implementation_hash, environment
from phm_design.family_pipeline import generate_with_passports, prepare_family_section, predict_pipeline
from phm_design.reference_design import design_reference
from phm_design.route_selection import select_route, evaluate_suite, predict_selected

DOMAINS = ("eps", "thermal", "data")


def run(spec_path, policy_path, output, resume=False):
    spec, policy = read_json(spec_path), read_json(policy_path)
    b = Path(output)
    spec_key = digest({"spec": spec, "policy": policy, "code": implementation_hash(),
                       "runner": sha_file(Path(__file__)), "environment": environment()})
    if b.exists():
        if not resume:
            raise FileExistsError("STUDY_OUTPUT_EXISTS_USE_RESUME")
        if read_json(b/'protocol.json')['key'] != spec_key:
            raise ValueError("STUDY_RESUME_CHANGED")
    else:
        b.mkdir(parents=True)
        write_json(b/'protocol.json', {"schema":"phm.dev6-paired-study.v1", "key":spec_key,
            "spec":spec, "policy":policy, "environment":environment(),
            "implementation_sha256":implementation_hash(), "runner_sha256":sha_file(Path(__file__)),
            "frozen_utc":datetime.now(timezone.utc).isoformat(),
            "test_feedback_used_for_this_policy":False,
            "calibration_consumed_for":"route_selection_not_probability_calibration",
            "stated_assumptions":"prescribed low fidelity native subsystem; virtual meters; unvalidated sensitivity ranges"})
        write_json(b/'policy.json',policy)
        write_json(b/'progress.json', {"stage":"FROZEN","completed":[]})
    def checkpoint(stage):
        rec=read_json(b/'progress.json');rec['stage']=stage
        if stage not in rec['completed']:rec['completed'].append(stage)
        rec['utc']=datetime.now(timezone.utc).isoformat();write_json(b/'progress.json',rec)
        print(json.dumps({"stage":stage}),flush=True)
    def native_stage(cfg,folder):
        cfgpath=b/'configs'/(folder+'.json')
        if cfgpath.exists() and read_json(cfgpath)!=cfg:raise ValueError('NATIVE_CONFIG_CHANGED')
        write_json(cfgpath,cfg)
        dest=b/'campaigns'/folder
        if dest.exists():
            verify(dest)
            if read_json(dest/'passport_registry.json')['config_sha256']!=digest(cfg):raise ValueError('CAMPAIGN_CHANGED')
        else:
            generate_with_passports(cfgpath,dest)
        checkpoint('GENERATED_'+folder)
        return dest
    def prepare(campaign,domain,section,dest):
        if dest.exists():verify(dest)
        else:prepare_family_section(campaign,domain,section,'mechanism',dest)
    cfg=copy.deepcopy(spec['development_config'])
    campaign=native_stage(cfg,'development')
    for d in DOMAINS:
        dev=b/'prepared'/d/'development';prepare(campaign,d,'development',dev)
        out=b/'designs'/d
        def progress(r):
            print(json.dumps({"domain":d,**{k:r[k] for k in ('candidate','seed','status')}}),flush=True)
        design_reference(dev/'dataset.json',dev/'device_binding.json',b/'policy.json',out,resume=out.exists(),on_trial_complete=progress)
        checkpoint('TRAINED_'+d)
    for d in DOMAINS:
        cal=b/'prepared'/d/'calibration';prepare(campaign,d,'calibration',cal)
        select_route(b/'designs'/d,cal/'dataset.json',b/'selections'/d,
                     registry=b/'evidence_registry.sqlite',policy=spec['route_policy'])
        checkpoint('SELECTED_'+d)
    freeze={d:{'model_manifest':sha_file(b/'designs'/d/'manifest.json'),
               'selection_manifest':sha_file(b/'selections'/d/'manifest.json')}
            for d in DOMAINS}
    if (b/'route_freeze.json').exists():
        if read_json(b/'route_freeze.json')['artifacts']!=freeze:raise ValueError('FROZEN_ROUTES_CHANGED')
    else:write_json(b/'route_freeze.json',{'artifacts':freeze,'test_generation_started':False})
    cohorts={}
    for name,ranges in spec['test_cohorts'].items():
        tc=copy.deepcopy(cfg);tc['groups_per_role']={r:int(spec['test_groups'] if r=='test' else 0) for r in cfg['groups_per_role']}
        for field,pair in ranges.items():tc['challenge'][field]['test']=pair
        cohorts[name]=native_stage(tc,name)
    results={};inference={}
    for d in DOMAINS:
        holds={}
        for name,camp in cohorts.items():
            hold=b/'prepared'/d/name;prepare(camp,d,'holdout',hold);holds[name]=hold/'dataset.json'
        rep=evaluate_suite(b/'designs'/d,b/'selections'/d,holds,b/'evaluation'/d,registry=b/'evidence_registry.sqlite')
        results[d]=rep
        # Raw-CSV, restored-pipeline prediction check; no training decisions.
        camp=cohorts['iid'];raw=read_json(camp/'native'/d/'holdout/dataset.json');s=raw['samples'][0]
        pred=predict_selected(b/'designs'/d,b/'selections'/d,
            camp/'native'/d/'holdout'/s['path'],camp/'passports'/d/(s['lineage_id']+'.json'),d)
        expected=read_json(b/'evaluation'/d/'iid_predictions.json')
        idx=next(i for i,row in enumerate(expected['samples']) if row['id']==s['id'])
        import numpy as np
        target=expected['scores'][rep['chosen_route']][idx]
        np.testing.assert_allclose(pred['scores'],target,rtol=2e-5,atol=2e-6)
        inference[d]={'id':s['id'],'route':pred['route'],'scores':pred['scores'],
                      'batch_single_max_abs_diff':float(np.max(np.abs(np.asarray(pred['scores'])-target)))}
        checkpoint('EVALUATED_'+d)
    write_json(b/'results.json',results);write_json(b/'raw_inference.json',inference)
    total_cases=sum(read_json(p/'summary.json')['cases'] for p in [campaign,*cohorts.values()])
    summary={'status':'COMPLETE','native_trajectory_executions':total_cases,
        'train_design_trials':sum(read_json(b/'designs'/d/'design_report.json')['planned_trials'] for d in DOMAINS),
        'paired_test_worlds':spec['test_groups'],'test_cohorts':list(cohorts),'results':results,
        'production_qualified':False,'risk_certified':False}
    write_json(b/'summary.json',summary);checkpoint('COMPLETE')
    return summary


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--spec',required=True);p.add_argument('--policy',required=True)
    p.add_argument('--output',required=True);p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    try:run(args.spec,args.policy,args.output,args.resume)
    except Exception as exc:
        print(json.dumps({'status':'FAILED','type':type(exc).__name__,'reason':str(exc)}),file=sys.stderr,flush=True)
        raise

if __name__=='__main__':main()
