import copy
import random
from pathlib import Path
import numpy as np
import pytest
import torch
from phm_design.data import Dataset, fit_scaler, transform
from phm_design.models import Shape
from phm_design.reference import fit_reference, reference_logits
from phm_design.registry import default_registry
from phm_design.engine import deterministic_cpu,fit,load_ensemble,ensemble_scores
from phm_design.io import read_json,write_json,sha_file,verify
from phm_design.synthesis import lower_genome,random_genome
from phm_design.reference_design import design_reference,evaluate_reference,choose_route,choose_development_route
from tests.phm_design.test_design_system import fixture_dataset,config
from tests.phm_design.test_compositional_design import request


def setup(tmp_path,channels=2,classes=3):
    dev,test=fixture_dataset(tmp_path/'data',channels,classes)
    d=Dataset(dev,allowed_splits={'train','validation'});t=d.load('train',8)
    x=transform(t.x,fit_scaler(t.x));ref=fit_reference(x,t.y,d.contract,t.samples)
    req=request(channels);shape=Shape(8,channels,classes)
    graph=lower_genome(random_genome(random.Random(123),req['grammar']),[[i] for i in range(channels)],shape)
    spec={'family':'reference_dag','parameters':{'reference':ref,'graph':graph,'mode':'bounded','bound_logits':.5}}
    return dev,test,d,t,x,shape,spec


@pytest.mark.parametrize('channels,classes',[(2,2),(2,3),(3,4)])
def test_reference_matches_numpy_and_zero_correction(tmp_path,channels,classes):
    _,_,_,_,x,s,spec=setup(tmp_path,channels,classes)
    with deterministic_cpu():
        registry=default_registry();model=registry.build(spec,s,11,100000)
        a,b=model.parts(torch.from_numpy(x))
        assert torch.count_nonzero(b)==0
        np.testing.assert_allclose(a.numpy(),reference_logits(spec['parameters']['reference'],x),atol=2e-6,rtol=2e-6)
        assert model.reference.mean.dtype==torch.float64
        assert len(list(model.reference.parameters()))==0
        assert sum(p.numel() for p in model.parameters())==registry.describe(spec,s)['parameter_count']


@pytest.mark.parametrize('gain',[-100.,-.5,0.,.5,100.])
def test_pointwise_bound_and_margin_protection(tmp_path,gain):
    *_,x,s,spec=setup(tmp_path)
    with deterministic_cpu():
        model=default_registry().build(spec,s,11,100000);model.gain.data.fill_(gain)
        z,c=model.parts(torch.from_numpy(x));assert c.abs().max()<=.5+1e-7
        largest=z.topk(2,dim=1).values;protected=(largest[:,0]-largest[:,1])>1.0
        assert protected.any()
        assert torch.equal(z.argmax(1)[protected],(z+c).argmax(1)[protected])


def test_reference_is_frozen_and_dag_learns(tmp_path):
    *_,x,s,spec=setup(tmp_path)
    with deterministic_cpu():
        model=default_registry().build(spec,s,11,100000);before={k:v.clone() for k,v in model.reference.state_dict().items()}
        # All graph paths train after the zero gate has opened.
        model.gain.data.fill_(.2);opt=torch.optim.AdamW(model.parameters(),lr=.01)
        for _ in range(2):
            opt.zero_grad();model(torch.from_numpy(x)).square().mean().backward();opt.step()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.residual.parameters())
        assert all(torch.equal(v,model.reference.state_dict()[k]) for k,v in before.items())


@pytest.mark.parametrize('issue',['fit_role','class_order','nan','bound','mode'])
def test_bad_reference_rejected(tmp_path,issue):
    *_,s,spec=setup(tmp_path);p=spec['parameters']
    if issue=='fit_role':p['reference']['fit_role']='validation'
    elif issue=='class_order':p['reference']['class_indices']=[2,1,0]
    elif issue=='nan':p['reference']['mean'][0]=float('nan')
    elif issue=='bound':p['bound_logits']=-1
    else:p['mode']='invented'
    with pytest.raises(ValueError):default_registry().build(spec,s,1,100000)


def test_generic_fit_checks_actual_train_reference(tmp_path):
    dev,_,_,_,_,_,spec=setup(tmp_path);spec['parameters']['reference']['coef'][0][0]+=.01
    c=config();c['candidates']=[spec];c['budget']['max_parameters_per_model']=100000;c['budget']['max_ensemble_parameters']=200000
    write_json(tmp_path/'cfg.json',c)
    with pytest.raises(ValueError,match='REFERENCE_NOT_FITTED'):fit(dev,tmp_path/'cfg.json',tmp_path/'run')


def test_selection_not_forced_to_neural():
    ref={'macro_f1':.9,'accepted_errors':1,'accepted_correct':10}
    assert choose_route(ref,ref)=='reference_only'
    assert choose_route(ref,{'macro_f1':.95,'accepted_errors':2,'accepted_correct':12})=='reference_only'
    assert choose_route(ref,{'macro_f1':.92,'accepted_errors':1,'accepted_correct':11})=='bounded'
    assert choose_route(ref,None)=='reference_only'


def test_best_completed_development_route_can_expose_raw_winner():
    ref={'macro_f1':.90,'accepted_errors':2,'accepted_correct':12,'cases':16}
    def winner(f1,errors,correct):
        return {'status':'COMPLETED','ensemble_validation':{
            'macro_f1':f1,'accepted_errors':errors,'accepted_correct':correct,'cases':16}}
    winners={
        'fixed':winner(.975,1,15),
        'bounded':winner(.900,1,14),
        'raw':winner(.987,1,15),
        'unbounded':winner(.979,1,15),
    }
    assert choose_development_route(ref,winners,'best_completed_route')=='raw'
    # Legacy behavior remains exactly bounded-vs-reference.
    assert choose_development_route(ref,winners,'legacy_bounded_noninferiority')=='bounded'


def test_best_completed_development_route_tie_is_deterministic():
    metric={'macro_f1':.95,'accepted_errors':1,'accepted_correct':15,'cases':16}
    winners={m:{'status':'COMPLETED','ensemble_validation':dict(metric)}
             for m in ('fixed','bounded','raw','unbounded')}
    # Explicit route order provides deterministic replay for exact ties.
    assert choose_development_route(metric,winners,'best_completed_route')=='reference_only'


def make_policy():
    req=request();base={k:copy.deepcopy(v) for k,v in req.items() if k not in {'window','groups','structure_reference'}}
    base['schema']='phm.family-design-policy.v1';base['grammar']['allow_sharing']=False
    base['evolution'].update(initial_graphs=1,generations=2,offspring_per_generation=1,elite_parents=1,max_total_trials=20)
    base['training']['epochs']=2;base['budget']['max_trial_seconds']=30
    return {'schema':'phm.reference-design.v1','base_design':base,'bound_logits':.5,'max_total_trials':14,
            'selection_guard':'development_noninferiority_strict_improvement'}


def test_design_resume_evaluate_restore_and_hash_tamper(tmp_path):
    dev,test,d,*_=setup(tmp_path)
    b={'schema':'phm.device-binding.v1','contract':d.contract,'groups':request()['groups'],
       'window_samples':8,'quality_mode':'complete_observation_only_missing_values_rejected'}
    write_json(tmp_path/'binding.json',b);write_json(tmp_path/'policy.json',make_policy())
    run=tmp_path/'run'
    result=design_reference(dev,tmp_path/'binding.json',tmp_path/'policy.json',run)
    assert result['status']=='COMPLETED' and result['planned_trials']==14
    assert result['generated_topologies']==2 and not result['test_used']
    # Every same-topology comparison occurs in all three modes.
    lineage=read_json(run/'lineage.json')
    generated=[a for a in lineage if a['kind']=='generated']
    assert len(generated)==6
    assert any(a['parents'] for a in generated)
    for gh in {a['graph_sha256'] for a in generated}:
        assert {a['mode'] for a in generated if a['graph_sha256']==gh}=={'raw','bounded','unbounded'}
    weights={str(p):sha_file(p) for p in run.rglob('*.pt')}
    assert design_reference(dev,tmp_path/'binding.json',tmp_path/'policy.json',run,resume=True)==result
    assert weights=={str(p):sha_file(Path(p)) for p in weights}
    report=evaluate_reference(run,test,tmp_path/'eval');assert not report['test_used_for_selection']
    assert evaluate_reference(run,test,tmp_path/'ignored')==report
    bundle,models=load_ensemble(run/'selected_bounded')
    assert models and not any(p.requires_grad for p in models[0].reference.parameters())
    (run/'reference.json').write_text('{}')
    with pytest.raises(ValueError,match='INTEGRITY'):verify(run)


def test_design_rejects_holdout_and_global_budget(tmp_path):
    dev,test,d,*_=setup(tmp_path)
    b={'schema':'phm.device-binding.v1','contract':d.contract,'groups':request()['groups'],'window_samples':8,
       'quality_mode':'complete_observation_only_missing_values_rejected'}
    write_json(tmp_path/'b.json',b);p=make_policy();p['max_total_trials']=1;write_json(tmp_path/'p.json',p)
    with pytest.raises(ValueError,match='DATA_ROLE'):design_reference(test,tmp_path/'b.json',tmp_path/'p.json',tmp_path/'x')
    with pytest.raises(ValueError,match='GLOBAL_TRIAL'):design_reference(dev,tmp_path/'b.json',tmp_path/'p.json',tmp_path/'y')


def test_registered_initialization_does_not_change_rng(tmp_path):
    *_,x,shape,spec=setup(tmp_path)
    with deterministic_cpu():
        before=torch.get_rng_state().clone();r=default_registry()
        first=r.build(spec,shape,123,100000);second=r.build(spec,shape,123,100000)
        assert torch.equal(before,torch.get_rng_state())
        assert all(torch.equal(v,second.state_dict()[k]) for k,v in first.state_dict().items())
        desc=r.describe(spec,shape)
        assert desc['frozen_reference_bytes']>0
        assert desc['frozen_reference_scalars']==sum(b.numel() for b in first.reference.buffers())


def test_confident_reference_mistake_is_not_correctness_guarantee(tmp_path):
    *_,x,shape,spec=setup(tmp_path)
    p=spec['parameters']['reference']
    p['coef']=(np.zeros((3,14))).tolist();p['intercept']=[100.,0.,0.]
    with deterministic_cpu():
        m=default_registry().build(spec,shape,1,100000);m.gain.data.fill_(100)
        predictions=m(torch.from_numpy(x)).argmax(1)
        # A large-margin error remains protected too: never call the bound an accuracy guarantee.
        assert (predictions==0).all()


def test_completed_trial_resume_after_callback_interrupt(tmp_path):
    dev,test,d,*_=setup(tmp_path)
    b={'schema':'phm.device-binding.v1','contract':d.contract,'groups':request()['groups'],'window_samples':8,
       'quality_mode':'complete_observation_only_missing_values_rejected'}
    p=make_policy();p['base_design']['evolution']['generations']=1;p['max_total_trials']=8
    write_json(tmp_path/'b.json',b);write_json(tmp_path/'p.json',p)
    def stop(record):raise RuntimeError('deliberate interruption after committed trial')
    run=tmp_path/'resume_run'
    with pytest.raises(RuntimeError,match='deliberate interruption'):
        design_reference(dev,tmp_path/'b.json',tmp_path/'p.json',run,on_trial_complete=stop)
    saved={str(f):sha_file(f) for f in run.rglob('model.pt')}
    assert len(saved)==1
    report=design_reference(dev,tmp_path/'b.json',tmp_path/'p.json',run,resume=True)
    assert report['status']=='COMPLETED'
    assert all(sha_file(Path(f))==h for f,h in saved.items())
