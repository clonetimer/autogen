from __future__ import annotations
import copy
import importlib.util
from pathlib import Path
import json
import numpy as np
import pytest
from phm_design.adapters.sat_nonadcs import (PROTOCOL,ROLES,FAMILIES,check_config,world_id,parameters,
    plan_campaign,contract,binding,run_native,pair_audit,generate_nonadcs)
from phm_design.family_design import make_request
from phm_design.synthesis import propose
from phm_design.graph import analyze_graph,compositional_registry
from phm_design.models import Shape
from phm_design.data import Dataset,sequence_from_csv,reject_overlap
from phm_design.io import write_json,verify


def config(count=1):return dict(schema=PROTOCOL,seed=120201,groups_per_role=dict(train=count,validation=0,calibration=0,test=0),domains=list(FAMILIES))

def policy():
    return dict(schema="phm.family-design-policy.v1",grammar=dict(widths=[8,16],max_depth=2,max_branches=3,kernels=[3],dilations=[1],allow_sharing=False),
        evolution=dict(seed=7,initial_graphs=2,generations=2,offspring_per_generation=1,elite_parents=1,max_proposal_attempts=200,max_total_trials=8),
        controls=[dict(family="mlp",parameters=dict(widths=[8],representation="statistics"))],seeds=[11,23],
        training=dict(epochs=2,batch_size=12,learning_rate=.003,weight_decay=.001,patience=2),
        budget=dict(max_parameters_per_model=50000,max_ensemble_parameters=100000,max_trials=8,max_trial_seconds=30),
        selection="mean_seed_validation_macro_f1",abstention=dict(min_uncalibrated_score=.6))


@pytest.mark.parametrize('role',ROLES)
def test_role_stability_when_counts_change(role):
    a=config(2);a['groups_per_role']={r:2 for r in ROLES}
    b=copy.deepcopy(a);b['groups_per_role']['train']=5
    x=[(r['world_id'],r['parameters']) for r in plan_campaign(a) if r['role']==role]
    y=[(r['world_id'],r['parameters']) for r in plan_campaign(b) if r['role']==role and r['index_in_role']<2]
    assert x==y


def test_domains_share_worlds_but_roles_never_do():
    c=config();c['groups_per_role']={r:1 for r in ROLES};plan=plan_campaign(c)
    ids=[{x['world_id'] for x in plan if x['role']==r} for r in ROLES]
    assert all(len(x)==1 for x in ids);assert len(set.union(*ids))==4


@pytest.mark.parametrize('change',[{'seed':True},{'seed':-1},{'domains':['eps','eps']},{'domains':['adcs']},{'extra':1}, {'groups_per_role':{'train':1}}])
def test_invalid_configs_fail_closed(change):
    c=config();c.update(change)
    with pytest.raises(ValueError):check_config(c)


@pytest.mark.parametrize('domain',FAMILIES)
def test_families_compile_actual_generated_graphs(domain):
    b=binding(domain);request=make_request(b,policy())
    proposal=propose(request,b['contract'],0,[],set())
    shape=Shape(request['window'],len(b['contract']['channels']),len(b['contract']['classes']))
    import torch
    for item in proposal['accepted']:
        desc=analyze_graph(shape,item['spec']['parameters'])
        model=compositional_registry().build(item['spec'],shape,11,50000)
        assert model(torch.zeros(2,shape.window,shape.channels)).shape==(2,3)
        assert sum(p.numel() for p in model.parameters())==desc['parameter_count']


def test_unreviewed_cross_component_sharing_rejected():
    p=policy();p['grammar']['allow_sharing']=True
    with pytest.raises(ValueError,match='SHARING'):make_request(binding('thermal'),p)


def test_truth_role_rejected_in_binding():
    b=copy.deepcopy(binding('eps'));b['contract']['channels'][0]['role']='training_label'
    with pytest.raises(ValueError):make_request(b,policy())


def test_bad_component_partition_rejected():
    b=copy.deepcopy(binding('data'));b['groups'][0]['channels']=['queue_counter_bits']
    with pytest.raises(ValueError):make_request(b,policy())


@pytest.fixture(scope='module')
def native_campaign(tmp_path_factory):
    if importlib.util.find_spec('Basilisk') is None:pytest.skip('Basilisk optional runtime unavailable')
    root=tmp_path_factory.mktemp('nonadcs');cfg=root/'config.json';write_json(cfg,config())
    output=root/'data';report=generate_nonadcs(cfg,output)
    return output,report


def test_nine_real_native_cases(native_campaign):
    out,report=native_campaign
    assert report['cases']==9 and report['fault_pairs_with_numeric_difference']==6
    assert report['training_executed'] is False
    assert report['capability_registry_qualification'] is False
    verify(out)


@pytest.mark.parametrize('domain',FAMILIES)
def test_native_dataset_labels_truth_and_count(native_campaign,domain):
    out,_=native_campaign;d=Dataset(out/domain/'development/dataset.json',allowed_splits={'train'})
    d.verify_files();block=d.load('train',FAMILIES[domain]['window'])
    assert block.x.shape==(3,FAMILIES[domain]['window'],len(FAMILIES[domain]['channels']))
    assert len(block.groups)==1 and set(block.y)=={0,1,2}
    csvfile=out/domain/'development'/d.samples[0]['path']
    assert 'fault' not in csvfile.read_text().splitlines()[0]
    for rec in (out/'case_records').glob('*.json'):
        r=json.loads(rec.read_text())
        assert r['all_output_timestamps_verified'] and r['pair_audit']['prefault_exact_match']


@pytest.mark.parametrize('domain',FAMILIES)
def test_native_same_input_replay(native_campaign,domain):
    out,_=native_campaign
    p=parameters(world_id(120201,'train',0),domain)
    first=run_native(domain,p,1);second=run_native(domain,p,1)
    assert first['observations']==second['observations']
    assert first['truth']==second['truth']


def test_output_exists_not_overwritten(native_campaign,tmp_path):
    out,_=native_campaign;p=tmp_path/'cfg.json';write_json(p,config())
    with pytest.raises(FileExistsError):generate_nonadcs(p,out)


def test_holdout_role_cannot_enter_development(native_campaign,tmp_path):
    out,_=native_campaign;raw=json.loads((out/'eps/development/dataset.json').read_text())
    raw['samples'][0]['split']='test';p=tmp_path/'dataset.json';write_json(p,raw)
    with pytest.raises(ValueError,match='DATA_ROLE'):Dataset(p,allowed_splits={'train','validation'})


def test_group_reuse_across_roles_rejected(native_campaign,tmp_path):
    out,_=native_campaign;raw=json.loads((out/'eps/development/dataset.json').read_text())
    raw['samples'][0]['split']='validation';p=tmp_path/'dataset.json';write_json(p,raw)
    with pytest.raises(ValueError,match='CROSS_SPLIT'):Dataset(p,allowed_splits={'train','validation'})


def test_simulation_missing_inputs_fail_at_csv_contract(native_campaign,tmp_path):
    out,_=native_campaign;d=Dataset(out/'thermal/development/dataset.json',allowed_splits={'train'})
    original=out/'thermal/development'/d.samples[0]['path'];s=original.read_text().splitlines()
    cells=s[-1].split(',');cells[-1]='nan';s[-1]=','.join(cells);p=tmp_path/'broken.csv';p.write_text('\n'.join(s))
    with pytest.raises(ValueError,match='NONFINITE'):sequence_from_csv(p,contract('thermal'),128)
