from __future__ import annotations
import copy
import csv
import random
from pathlib import Path
import numpy as np
import pytest
import torch
from phm_design.robust_ops import Observation,MomentumBank,QualityFusion,validate_mechanism
from phm_design.robust_data import clean_profile,perturb,packet_hash,packet_to_json,packet_from_json,check_profile
from phm_design.robust_model import generate_spec,mutate_spec,analyze_robust,build_model,fit_robust_scalers
from phm_design.robust_engine import (decision_metrics,validate_request,design_robust,load_robust,predict_robust,evaluate_robust)
from phm_design.io import write_json,sha_file,read_json


def contract(wheels=1):
    names=[]
    for i in range(wheels):
        names.extend([{'name':f'w{i}_{n}','unit':u,'role':'deployment_input','provenance':'analytic unit fixture not hardware'} for n,u in [('speed','rad/s'),('command','N*m'),('gyro','rad/s')]])
    return {'device_family':'analytic_fixed_axis_fixture','task':'fixed_length_case_classification','channels':names,
            'classes':['normal','fault'],'sample_s':1.,'evidence_type':'software_synthetic','scope':'unit fixture only'}


def mechanism(wheels=1):
    return {'schema':'phm.robust-momentum.v1','groups':[[3*i,3*i+1,3*i+2] for i in range(wheels)],
       'inertia_bounds_kg_m2':[[1.,1.]]*wheels,'nominal_limits_nm':[2.]*wheels,
       'source':{'reference':'fixture.json','sha256':'0'*64},'assumptions':['balanced fixed axes, zero nominal friction, analytic only'],
       'max_age_s':2.,'sync_tolerance_s':0.,'max_interval_s':1.1,
       'inertia_scope':'hypothetical_diagnostic_parameter_sensitivity_not_hardware_tolerance'}


def obs(wheels=1):
    t=torch.arange(8,dtype=torch.float64)[None]
    x=torch.stack([t,torch.ones_like(t),torch.zeros_like(t)],-1).repeat(1,1,wheels).float()
    return Observation(x,torch.ones_like(x,dtype=torch.bool),t.unsqueeze(-1).expand_as(x).clone(),t)


def spec(wheels=1):
    c=contract(wheels);m=mechanism(wheels);p=obs(wheels)
    s=fit_robust_scalers(p.values.numpy(),[p],c,m)
    return generate_spec(random.Random(12),c,m,s,8,{'widths':[4,8],'kernels':[3],'dilations':[1],'max_depth':2},
               {'min_raw_coverage':.25,'min_physical_coverage':.25,'max_branch_final_age_s':3.})[0]


def test_nominal_and_inertia_bank():
    m=mechanism();m['inertia_bounds_kg_m2']=[[1.,2.]]
    r=MomentumBank(contract(),m)(obs());assert r['computable'][0,:,0].tolist()==[False]+[True]*7
    torch.testing.assert_close(r['residual'][0,3,0],torch.tensor([0.,1.5,3.],dtype=torch.float64))


def test_gap_reset_not_cumulative_bridge():
    p=obs();p.values[:,3,0]=float('nan')
    m=mechanism();m['inertia_bounds_kg_m2']=[[2.,2.]]
    r=MomentumBank(contract(),m)(p)
    assert r['computable'][0,:,0].tolist()==[False,True,True,False,False,True,True,True]
    assert r['residual'][0,5,0,0].item()==1.


@pytest.mark.parametrize('change',['future','backwards','process_duplicate','mask_type'])
def test_invalid_observations(change):
    p=obs()
    if change=='future':p.source_s[:,2,0]=3.
    elif change=='backwards':p.source_s[:,4,0]=1.
    elif change=='process_duplicate':p.time_s[:,4]=3.
    else:p.observed=p.observed.float()
    with pytest.raises(ValueError):p.validate()


@pytest.mark.parametrize('value',[float('nan'),float('inf'),float('-inf')])
def test_nonfinite_measurement_is_missing(value):
    p=obs();p.values[:,3,0]=value
    r=MomentumBank(contract(),mechanism())(p)
    assert not r['computable'][0,3,0];assert torch.isfinite(r['residual']).all()


def test_repeated_source_stamp_not_fresh_interval():
    p=obs();p.source_s[:,3]=p.source_s[:,2]
    r=MomentumBank(contract(),mechanism())(p);assert not r['computable'][0,3,0]


def test_relative_delay_disables_physics_but_not_all_raw():
    p=obs();p.source_s[:,:,2]-=1
    s=spec();model=build_model(s,3,100000);r=model.assess(p)
    assert not r['branch_available'][0,1];assert r['branch_available'][0,0]


def test_common_delay_and_staleness():
    p=obs();p.source_s-=1
    assert MomentumBank(contract(),mechanism())(p)['computable'][0,1,0]
    p.source_s-=5
    assert not MomentumBank(contract(),mechanism())(p)['computable'].any()


def test_causal_physical_prefix_and_gradient():
    p=obs();p.values.requires_grad_();bank=MomentumBank(contract(),mechanism());a=bank(p)['residual']
    q=obs();q.values[:,-1,0]+=10;b=bank(q)['residual'];torch.testing.assert_close(a[:,:-1],b[:,:-1])
    a.sum().backward();assert torch.isfinite(p.values.grad).all()


@pytest.mark.parametrize('which',['unit','truth','source','bounds','groups','time','scope'])
def test_bad_mechanism_contract(which):
    c=contract();m=mechanism()
    if which=='unit':c['channels'][0]['unit']='rpm'
    elif which=='truth':c['channels'][0]['role']='truth'
    elif which=='source':m['source']['sha256']='missing'
    elif which=='bounds':m['inertia_bounds_kg_m2']=[[2.,1.]]
    elif which=='groups':m['groups']=[[0,0,2]]
    elif which=='time':m['max_interval_s']=float('nan')
    else:m['inertia_scope']='hardware_proven'
    with pytest.raises(ValueError):validate_mechanism(c,m)


def test_quality_fusion_all_invalid():
    g=QualityFusion();e=torch.full((2,3,4),float('nan'));a=torch.zeros(2,3,dtype=torch.bool);q=torch.zeros(2,3,2)
    fused,ok,w=g(e,a,q);assert not ok.any();assert fused.abs().sum()==0;assert w.sum()==0


def test_quality_fusion_only_available_branch_and_backward():
    g=QualityFusion();e=torch.randn(2,3,4,requires_grad=True);a=torch.tensor([[True,False,False]]*2);q=torch.tensor([[[1.,0.]]*3]*2)
    f,ok,w=g(e,a,q);torch.testing.assert_close(f,e[:,0]);f.sum().backward();assert torch.isfinite(e.grad).all()


@pytest.mark.parametrize('fusion',['quality_gate','masked_mean','masked_concat'])
@pytest.mark.parametrize('shared',[True,False])
def test_composed_resources_backward_and_all_invalid(fusion,shared):
    s=spec(2);s['fusion']=fusion;s['share_components']=shared
    model=build_model(s,11,100000);assert sum(p.numel() for p in model.parameters())==analyze_robust(s)['parameter_count']
    r=model.assess(obs(2));assert r['logits'].shape==(1,2);r['logits'].square().sum().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    p=obs(2);p.observed[:]=False;p.values[:]=float('nan')
    out=model.assess(p);assert not out['computable'].any();assert out['logits'].abs().sum()==0


def test_graph_generation_mutation_and_sharing():
    s=spec(2);g={'widths':[4,8],'max_depth':2,'kernels':[3,5],'dilations':[1,2]}
    changes=[]
    for seed in range(15):
        ch,genes=mutate_spec(s,random.Random(seed),g);analyze_robust(ch);changes.append(genes['operation'])
    assert len(set(changes))>=4
    a=copy.deepcopy(s);a['share_components']=True;b=copy.deepcopy(s);b['share_components']=False
    assert analyze_robust(a)['parameter_count']<analyze_robust(b)['parameter_count']


def test_placeholder_zero_is_not_valid_zero():
    s=spec();m=build_model(s,9,100000);p=obs();p.values[:]=0;assert m.assess(p)['computable'].item()
    p.observed[:]=False;assert not m.assess(p)['computable'].item()


def test_train_only_feature_scaler():
    p=obs();m=mechanism();s=fit_robust_scalers(p.values.numpy(),[p],contract(),m)
    p2=obs();p2.values[:,2:4]=float('nan');p2.observed[:,2:4]=False
    z=fit_robust_scalers(p.values.numpy(),[p2],contract(),m)
    assert s['raw']==z['raw'];assert z['physical']['fit_role']=='train_only';assert np.isfinite(z['physical']['scale']).all()


def test_perturbation_label_and_order_independent():
    x=np.repeat(obs().values.numpy(),2,0);samples=[{'id':'a','label':'x'},{'id':'b','label':'y'}]
    p=clean_profile(3);p['noise_std']=[.1,0,.01]
    a=perturb(x,samples,1.,p,71);b=perturb(x[::-1].copy(),list(reversed(samples)),1.,p,71)
    torch.testing.assert_close(a.values,b.values.flip(0));samples[0]['label']='never-used';assert packet_hash(a)==packet_hash(perturb(x,samples,1.,p,71))


def test_delay_preserves_source_and_no_future_fill():
    p=clean_profile(3);p['delay_steps']=[2,0,0];r=perturb(obs().values.numpy(),[{'id':'a'}],1.,p,3)
    assert not r.observed[0,:2,0].any();assert r.source_s[0,2,0]==0;assert r.values[0,2,0]==0


def test_packet_roundtrip_and_boolean_contract():
    p=obs();p.observed[:,3]=False;p.values[:,3]=float('nan');p.source_s[:,3]=float('nan')
    j=packet_to_json(p,contract());q=packet_from_json(j,contract(),8);assert packet_hash(p)==packet_hash(q)
    j['observed'][0][0][0]=1
    with pytest.raises(ValueError):packet_from_json(j,contract(),8)


def test_all_rejected_remain_in_denominator():
    r=decision_metrics(np.array([0,1]),np.array([[.9,.1],[.1,.9]]),np.array([False,False]),['a','b'],.6)
    assert r['cases']==2 and r['accuracy_all_cases']==0 and r['macro_f1_with_incomputable_as_errors']==0 and r['accepted_accuracy'] is None
    assert r['confusion_matrix']==[[0,0,1],[0,0,1]]


@pytest.mark.parametrize('field,value',[('gap_length',99),('noise_std',[-1,0,0]),('delay_steps',[9,0,0]),('id','../escape')])
def test_bad_stress_profile(field,value):
    p=clean_profile(3);p[field]=value
    with pytest.raises(ValueError):check_profile(p,3,8)


def tiny_setup(root):
    root=Path(root);c=contract();all_samples=[]
    for split,n in [('train',8),('validation',4),('test',4)]:
        samples=[]
        for i in range(n):
            rng=np.random.default_rng(10000+len(all_samples)*17+i+sum(map(ord,split)))
            label=i%2;t=np.arange(8);u=.2+rng.uniform(0,.3);rate=t*u*(1 if not label else .4)+rng.normal(0,.01,8)
            x=np.stack([rate,np.full(8,u),np.zeros(8)],1);filename=f'{split}_{i}.csv';folder=root/('holdout' if split=='test' else 'development');folder.mkdir(parents=True,exist_ok=True)
            with (folder/filename).open('w',newline='') as f:
                writer=csv.writer(f);writer.writerow(['time_s']+[a['name'] for a in c['channels']]);writer.writerows(np.column_stack([t,x]))
            samples.append({'id':f'{split}_{i}','group_id':f'g_{split}_{i}','lineage_id':f'l_{split}_{i}','label':c['classes'][label],'split':split,'path':filename,'sha256':sha_file(folder/filename)})
        all_samples.extend(samples)
    for part,ss in [('development',{'train','validation'}),('holdout',{'test'})]:
        write_json(root/part/'dataset.json',{'schema':'phm.sequence-dataset.v1','dataset_id':part,'contract':c,'samples':[s for s in all_samples if s['split'] in ss],'provenance':{'scope':'analytic software fixture'}})
    write_json(root/'binding.json',{'fixture':'analytical only'});m=mechanism();m['source']={'reference':'binding.json','sha256':sha_file(root/'binding.json')}
    clean=clean_profile(3);missing=clean_profile(3,'all_missing','safety_only');missing['drop_channels']=[0,1,2]
    r={'schema':'phm.robust-design.v1','window':8,'proposal_seed':74,'initial_candidates':1,'mutations':1,'max_proposals':50,
       'grammar':{'widths':[4],'max_depth':1,'kernels':[3],'dilations':[1]},'seeds':[11,23],
       'training':{'epochs':2,'batch_size':8,'learning_rate':.003,'weight_decay':.0001,'checkpoint_every':1},
       'budget':{'max_parameters':50000,'max_ensemble_parameters':100000,'max_trials':4,'max_trial_seconds':60,'max_neural_macs':1000000},
       'quality_policy':{'min_raw_coverage':.25,'min_physical_coverage':.25,'max_branch_final_age_s':3},'mechanism':m,
       'train_profiles':[clean],'validation_profiles':[clean,missing],'evaluation_profiles':[clean,missing],
       'train_perturbation_seed':10,'validation_perturbation_seed':20,'evaluation_perturbation_seed':30,'threshold':.6,
       'evidence_scope':'hypothetical_measurement_stress_not_hardware_specification'}
    write_json(root/'request.json',r);return r,c


def test_end_to_end_resume_predict_and_sealed_evaluate(tmp_path):
    r,c=tiny_setup(tmp_path);run=tmp_path/'run';args=(tmp_path/'development/dataset.json',tmp_path/'request.json',tmp_path/'binding.json',run)
    report=design_robust(*args);assert report['status']=='COMPLETED';assert report['training_trials']==4
    before={str(p):sha_file(p) for p in run.rglob('model.pt')}
    assert design_robust(*args,resume=True)['fingerprint']==report['fingerprint']
    assert before=={str(p):sha_file(p) for p in run.rglob('model.pt')}
    b,models=load_robust(run);p=obs();p.observed[:]=False;write_json(tmp_path/'packet.json',packet_to_json(p,c))
    pred=predict_robust(run,tmp_path/'packet.json');assert pred['records'][0]['prediction'] is None and pred['records'][0]['scores'] is None
    ev=evaluate_robust(run,tmp_path/'holdout/dataset.json',tmp_path/'evaluation');assert ev['base_case_count']==4;assert ev['profile_reports']['all_missing']['accepted_cases']==0
    assert evaluate_robust(run,tmp_path/'holdout/dataset.json',tmp_path/'unused')==ev
    with (run/b['weights'][0]).open('ab') as f:f.write(b'corrupted')
    with pytest.raises(ValueError,match='INTEGRITY'):load_robust(run)


def test_development_only_and_binding_rejection(tmp_path):
    r,c=tiny_setup(tmp_path)
    with pytest.raises(ValueError,match='DATA_ROLE'):design_robust(tmp_path/'holdout/dataset.json',tmp_path/'request.json',tmp_path/'binding.json',tmp_path/'run')
    write_json(tmp_path/'binding.json',{'fixture':'modified'})
    with pytest.raises(ValueError,match='SOURCE_HASH'):design_robust(tmp_path/'development/dataset.json',tmp_path/'request.json',tmp_path/'binding.json',tmp_path/'run')


def test_partial_trial_resume(tmp_path):
    tiny_setup(tmp_path);args=(tmp_path/'development/dataset.json',tmp_path/'request.json',tmp_path/'binding.json',tmp_path/'run')
    def stop(record):raise RuntimeError('deliberate interruption')
    with pytest.raises(RuntimeError,match='deliberate'):design_robust(*args,on_trial_complete=stop)
    p=next((tmp_path/'run').rglob('model.pt'));h=sha_file(p)
    r=design_robust(*args,resume=True);assert r['status']=='COMPLETED' and sha_file(p)==h


def test_model_rejects_changed_processing_grid():
    p=obs();p.time_s*=2;p.source_s*=2
    with pytest.raises(ValueError,match='PROCESSING_GRID'):
        build_model(spec(),3,100000).assess(p)
