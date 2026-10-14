from __future__ import annotations
import copy
from pathlib import Path
import numpy as np
import torch
import pytest
from phm_design.data import Dataset,fit_scaler,transform
from phm_design.engine import fit,load_ensemble,ensemble_scores,evaluate,predict,deterministic_cpu
from phm_design.graph import SCHEMA,analyze_graph,graph_identity
from phm_design.io import write_json,read_json,verify,digest
from phm_design.physics import (PhysicalProgram,analyze_program,fit_feature_scaler,analyze_mechanism)
from phm_design.registry import default_registry
from phm_design.synthesis import GraphBuilder
from phm_design.mechanism_design import build_matched_plan,mechanism_design
from phm_design.models import Shape
from phm_design.adapters.sat_rw_mechanism import momentum_program
from tests.phm_design.test_design_system import fixture_dataset,config

SOURCE={'reference':'analytic_test_fixture_not_device_evidence','sha256':'a'*64}

def toy_contract():
    return {'device_family':'industrial_motor.software_fixture','task':'fixed_length_case_classification',
            'channels':[{'name':'sensor.speed','unit':'rad/s','role':'deployment_input','provenance':'analytic fixture'},
                        {'name':'command.torque','unit':'N*m','role':'deployment_input','provenance':'analytic fixture'}],
            'classes':['N','F'],'sample_s':0.5,'evidence_type':'software_synthetic','scope':'analytic test only'}

def toy_program(contract=None):
    c=contract or toy_contract()
    return {'schema':'phm.physical-program.v1','inputs':c['channels'],'sample_s':c['sample_s'],
            'constants':[{'id':'k0','value':2.,'unit':'kg*m^2','role':'design_constant','source':copy.deepcopy(SOURCE)}],
            'nodes':[{'id':'p0','op':'delta','inputs':['x0']},
                     {'id':'p1','op':'mul','inputs':['k0','p0']},
                     {'id':'p2','op':'trapezoid_integral','inputs':['x1']},
                     {'id':'p3','op':'sub','inputs':['p1','p2']}],
            'outputs':[{'name':'balance','node':'p3','unit':'N*m*s','component':'motor'}],
            'assumptions':['fixed isolated axis, no friction, analytic only'],
            'source':copy.deepcopy(SOURCE),'evidence_scope':'analytic_fixture'}


def graph_for(channels,classes,feature_only=False):
    b=GraphBuilder();current='input'
    if feature_only:current=b.add('select',[current],{'indices':[channels]})
    current=b.add('linear',[current],{'width':4});current=b.add('tanh',[current])
    current=b.add('mean',[current]);current=b.add('linear',[current],{'width':classes})
    return {'schema':SCHEMA,'nodes':b.nodes,'output':current}


def test_physical_analytic_balance_and_known_fault():
    c=toy_contract();p=toy_program(c)
    t=np.arange(8)*0.5;speed=3.+0.4*t/2
    raw=torch.tensor(np.stack([speed,np.full(8,0.4)],-1)[None])
    residual=PhysicalProgram(p,c)(raw)
    torch.testing.assert_close(residual,torch.zeros_like(residual),atol=1e-14,rtol=0)
    fault=raw.clone();fault[...,0]=3.+0.2*torch.tensor(t)/2
    torch.testing.assert_close(PhysicalProgram(p,c)(fault)[0,:,0],-0.2*torch.tensor(t),atol=1e-14,rtol=0)


def test_no_future_dependence_and_gradient():
    c=toy_contract();m=PhysicalProgram(toy_program(c),c)
    x=torch.randn(2,9,2,dtype=torch.float64,requires_grad=True);changed=x.detach().clone();changed[:,5:]+=100
    torch.testing.assert_close(m(x)[:,:5],m(changed)[:,:5],atol=0,rtol=0)
    m(x).square().sum().backward();assert x.grad is not None and torch.isfinite(x.grad).all()


@pytest.mark.parametrize('issue',['unit','undeclared','cycle','code','fault_constant','truth','missing_source','wrong_output','unused','sample_period','nonfinite','missing_assumptions','clip_unit','clip_negative'])
def test_physical_invalid_contracts_blocked(issue):
    c=toy_contract();p=toy_program(c)
    if issue=='unit':p['nodes'][-1]['inputs']=['x0','x1']
    elif issue=='undeclared':p['nodes'][0]['inputs']=['x99']
    elif issue=='cycle':p['nodes'][0]['inputs']=['p3']
    elif issue=='code':p['nodes'][0]['op']='eval'
    elif issue=='fault_constant':p['constants'][0]['role']='simulator_truth'
    elif issue=='truth':c['channels'][0]['name']='truth.speed'
    elif issue=='missing_source':p['constants'][0]['source']['sha256']=''
    elif issue=='wrong_output':p['outputs'][0]['unit']='N*m'
    elif issue=='unused':p['nodes'].append({'id':'p9','op':'delta','inputs':['x0']})
    elif issue=='sample_period':p['sample_s']=2
    elif issue=='nonfinite':p['constants'][0]['value']=float('nan')
    elif issue=='missing_assumptions':p['assumptions']=[]
    elif issue=='clip_unit':p['nodes'][0]={'id':'p0','op':'symmetric_clip','inputs':['x0','k0']}
    else:
        p['constants'][0]['unit']='rad/s';p['constants'][0]['value']=-2;p['nodes'][0]={'id':'p0','op':'symmetric_clip','inputs':['x0','k0']}
    with pytest.raises(ValueError):analyze_program(p,c)


def test_coupled_rw_balance_has_correct_actual_inertia_dependency():
    c=toy_contract();c['channels']=[{'name':f'sensor.a{i}','unit':('N*m' if 3<=i<6 else 'rad/s'),'role':'deployment_input','provenance':'analytic'} for i in range(9)]
    c['sample_s']=0.1
    p=momentum_program(c,[2.,3.,4.],[0.2]*3,SOURCE)
    t=torch.arange(12,dtype=torch.float64)*0.1
    x=torch.zeros(1,12,9,dtype=torch.float64)
    for i,j in enumerate([2.,3.,4.]):
        x[0,:,6+i]=0.05*t
        x[0,:,i]=10+(0.1/j-0.05)*t
        x[0,:,3+i]=0.1
    torch.testing.assert_close(PhysicalProgram(p,c)(x),torch.zeros(1,12,3,dtype=torch.float64),atol=1e-13,rtol=0)
    wrong=copy.deepcopy(p);wrong['constants'][0]['value']=0.015
    assert PhysicalProgram(wrong,c)(x).abs().max()>0.01


def make_data(root):
    dev,test=fixture_dataset(root,channels=2,classes=3)
    for path in [dev,test]:
        d=read_json(path)
        d['contract']['channels'][0]['unit']='rad/s';d['contract']['channels'][1]['unit']='N*m'
        write_json(path,d)
    return dev,test


def make_request():
    base=config()
    return {'schema':'phm.mechanism-design.v1','window':8,'groups':[['sensor.x0','sensor.x1']],
            'control_channels':['sensor.x0'],'candidate_pairs':2,'proposal_seed':122,
            'max_proposal_attempts':100,'grammar':{'widths':[4,8],'max_depth':2,'kernels':[3],'dilations':[1]},
            'seeds':[3,5],'training':{**base['training'],'epochs':3,'patience':3},
            'budget':{**base['budget'],'max_parameters_per_model':10000,'max_ensemble_parameters':20000,'max_trials':6},
            'max_neural_macs_per_case':100000,'selection':base['selection'],'abstention':base['abstention']}


@pytest.mark.parametrize('seed',range(6))
def test_generated_mechanism_graph_pair_counts_backward_and_replay(tmp_path,seed):
    dev,_=make_data(tmp_path/'data');d=Dataset(dev,allowed_splits={'train','validation'})
    p=toy_program(d.contract);r=make_request();r['proposal_seed']=seed
    plan=build_matched_plan(d,p,r);assert plan==build_matched_plan(d,p,r)
    shape=Shape(8,2,3);registry=default_registry()
    for pair in plan['pairs']:
        specs=pair['specs'];counts=[]
        for mode in specs:
            s=specs[mode];model=registry.build(s,shape,3,10000)
            model(torch.randn(2,8,2)).sum().backward()
            assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in model.parameters())
            counts.append(sum(x.numel() for x in model.parameters()))
        assert counts[0]==counts[1]
        assert specs['physical']['parameters']['graph']==specs['observed_delta']['parameters']['graph']
        assert 'mechanisms' in [b['source'] for b in pair['genes']['branches']]


@pytest.mark.parametrize('issue',['patience','mac_budget','params','groups','unit','holdout','trial_budget'])
def test_planner_scope_budget_and_group_rejection(tmp_path,issue):
    dev,test=make_data(tmp_path/'data');d=Dataset(dev,allowed_splits={'train','validation'})
    p=toy_program(d.contract);r=make_request()
    if issue=='patience':r['training']['patience']=1
    elif issue=='mac_budget':r['max_neural_macs_per_case']=1;r['max_proposal_attempts']=3
    elif issue=='params':r['budget']['max_parameters_per_model']=1;r['max_proposal_attempts']=3
    elif issue=='groups':r['groups']=[['sensor.x0']]
    elif issue=='unit':r['groups']=[['sensor.x0'],['sensor.x1']]
    elif issue=='trial_budget':r['budget']['max_trials']=1
    else:
        with pytest.raises(ValueError,match='DATA_ROLE'):Dataset(test,allowed_splits={'train','validation'})
        return
    with pytest.raises(ValueError):build_matched_plan(d,p,r)


def test_training_scaler_binding_gate(tmp_path):
    dev,_=make_data(tmp_path/'data');d=Dataset(dev,allowed_splits={'train','validation'})
    p=toy_program(d.contract);plan=build_matched_plan(d,p,make_request())
    cfg=plan['configs']['physical'];cfg['candidates'][0]['parameters']['feature_scaler']['mean'][0]+=1
    path=tmp_path/'bad.json';write_json(path,cfg)
    with pytest.raises(ValueError,match='SCALER_NOT_TRAIN_ONLY'):fit(dev,path,tmp_path/'run')


def test_end_to_end_design_restore_evaluate_predict_resume_and_integrity(tmp_path):
    dev,test=make_data(tmp_path/'data');d=Dataset(dev,allowed_splits={'train','validation'})
    pp=tmp_path/'program.json';rp=tmp_path/'request.json';out=tmp_path/'design'
    write_json(pp,toy_program(d.contract));write_json(rp,make_request())
    report=mechanism_design(dev,pp,rp,out)
    assert report['status']=='COMPLETED' and report['same_completed_epoch_budget']
    assert report['training_trials']==8 and not report['test_used']
    hashes=verify(out)
    assert mechanism_design(dev,pp,rp,out,resume=True)==report
    assert verify(out)==hashes
    for mode in ['physical','observed_delta']:
        run=out/mode/'run';bundle,models=load_ensemble(run)
        result=evaluate(run,test,tmp_path/f'eval_{mode}')
        assert result['status']=='EVALUATED' and not result['real_device_qualified']
        sample=read_json(test)['samples'][0]
        prediction=predict(run,test.parent/sample['path'],{c['name']:c['unit'] for c in d.contract['channels']})
        assert prediction['closed_set_argmax'] in d.contract['classes']
    plan=read_json(out/'plan.json');plan['request']['proposal_seed']+=1;write_json(out/'plan.json',plan)
    with pytest.raises(ValueError,match='INTEGRITY'):mechanism_design(dev,pp,rp,out,resume=True)
