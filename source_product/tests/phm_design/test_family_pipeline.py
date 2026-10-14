import copy
from pathlib import Path
import numpy as np
import pytest
from phm_design.family_pipeline import (make_passport,validate_passport,compute_features,feature_binding,
    feature_contract,generate_with_passports,prepare_family_section,TRANSFORM,CONTEXT)
from phm_design.adapters.sat_nonadcs import parameters,world_id,FAMILIES
from phm_design.family_design import make_request
from phm_design.synthesis import propose
from phm_design.models import Shape
from phm_design.graph import analyze_graph,compositional_registry
from tests.phm_design.test_nonadcs import policy


def passport(domain):
    w=world_id(101,'train',0);return make_passport(w,domain,parameters(w,domain))

def raw(domain,n=32):
    t=np.arange(n)
    if domain=='eps':return np.column_stack((t%2,10+0*t,20*(t%2),20*(t%2)-10))
    if domain=='thermal':return np.column_stack((t%2,5+0*t,300+.5*t))
    return np.column_stack((t%2,100+0*t,150+0*t,1000+50*t))

@pytest.mark.parametrize('domain',FAMILIES)
@pytest.mark.parametrize('mode',['context','mechanism'])
def test_different_families_compile_with_same_designer(domain,mode):
    b=feature_binding(domain,mode);request=make_request(b,policy());out=propose(request,b['contract'],0,[],set())
    shape=Shape(b['window_samples'],len(b['contract']['channels']),3)
    import torch
    for item in out['accepted']:
        model=compositional_registry().build(item['spec'],shape,11,50000)
        assert model(torch.zeros(2,shape.window,shape.channels)).shape==(2,3)
        assert analyze_graph(shape,item['spec']['parameters'])['parameter_count']==sum(x.numel() for x in model.parameters())

@pytest.mark.parametrize('domain',FAMILIES)
def test_mode_shapes_and_causality(domain):
    p=passport(domain);x=raw(domain);x2=x.copy();x2[24:,-1]+=1
    for mode in ('context','mechanism'):
        a=compute_features(domain,x,p,mode);b=compute_features(domain,x2,p,mode)
        assert a.shape==(32,len(feature_contract(domain,mode)['channels']))
        np.testing.assert_array_equal(a[:24],b[:24]);assert np.isfinite(a).all()

@pytest.mark.parametrize('domain',FAMILIES)
def test_fault_truth_never_copied_to_passport(domain):
    w=world_id(101,'train',0);p=parameters(w,domain);a=make_passport(w,domain,p)
    p.update(onset_s=999,remaining_fraction=.001,extra_load_w=99,extra_heat_w=77)
    assert a==make_passport(w,domain,p)
    assert 'remaining_fraction' not in str(a) and 'onset_s' not in str(a)

@pytest.mark.parametrize('bad',['remaining_fraction','fault_label','actual_efficiency'])
def test_added_truth_and_unknown_keys_blocked(bad):
    p=passport('eps');p['parameters'][bad]={'value':1,'unit':'1','role':'normal_design_parameter'}
    with pytest.raises(ValueError,match='UNDECLARED'):validate_passport(p,'eps')

def test_bad_units_and_nan_blocked():
    p=passport('thermal');p['parameters']['normal_heat_capacity_j_per_k']['unit']='J'
    with pytest.raises(ValueError,match='UNIT'):validate_passport(p,'thermal')
    x=raw('eps').astype(float);x[2,2]=np.nan
    with pytest.raises(ValueError,match='VALUES'):compute_features('eps',x,passport('eps'),'mechanism')

def test_eps_analytic_balance():
    p=passport('eps');p['parameters']['normal_full_light_power_w']['value']=20.
    x=raw('eps');f=compute_features('eps',x,p,'mechanism')[:,-4:]
    np.testing.assert_allclose(f[:,:3],0,atol=1e-6)
    y=x.copy();y[:,3]-=5;assert np.allclose(compute_features('eps',y,p,'mechanism')[:,-3],5)

def test_constant_thermal_no_identifiability_claim():
    p=passport('thermal');vals=[100.,1.,1e-9]
    for name,value in zip(p['parameters'],vals):p['parameters'][name]['value']=value
    x=np.column_stack((np.ones(32),np.full(32,7.1),np.full(32,300.)))
    f=compute_features('thermal',x,p,'mechanism')[:,-4:]
    np.testing.assert_allclose(f[:,0],0,atol=1e-6)
    assert not f[:,-1].any() # constant T does not distinguish emissivity/heat by regression

def test_queue_empty_not_mechanism_healthy_evidence():
    x=raw('data');x[:,3]=0
    f=compute_features('data',x,passport('data'),'mechanism')[:,-4:]
    assert not f[:,-1].any()

def test_same_initial_topologies_match_modes():
    for domain in FAMILIES:
        proposals=[]
        for mode in ('context','mechanism'):
            b=feature_binding(domain,mode);r=make_request(b,policy())
            proposals.append([x['graph_sha256'] for x in propose(r,b['contract'],0,[],set())['accepted']])
        assert proposals[0]==proposals[1]

@pytest.fixture(scope='module')
def contextual_campaign(tmp_path_factory):
    import importlib.util
    if importlib.util.find_spec('Basilisk') is None:pytest.skip('optional native runtime')
    from phm_design.io import write_json
    root=tmp_path_factory.mktemp('passport_campaign');cfg=root/'config.json'
    write_json(cfg,{'schema':'phm.native-nonadcs.v1','seed':99107,
        'groups_per_role':{'train':1,'validation':1,'calibration':0,'test':1},'domains':list(FAMILIES)})
    output=root/'generated';generate_with_passports(cfg,output)
    return output

@pytest.mark.parametrize('domain',FAMILIES)
def test_passports_same_for_fault_variants_and_prepare_roles(contextual_campaign,tmp_path,domain):
    from phm_design.io import read_json
    from phm_design.data import Dataset
    root=contextual_campaign
    prepare_family_section(root,domain,'development','mechanism',tmp_path/'prepared')
    data=Dataset(tmp_path/'prepared/dataset.json',allowed_splits={'train','validation'})
    assert len(data.samples)==6
    a=data.load('train',FAMILIES[domain]['window']).x
    k=len(FAMILIES[domain]['channels']);length=len(CONTEXT[domain])
    for j in range(1,3):np.testing.assert_array_equal(a[0,:,k:k+length],a[j,:,k:k+length])
    assert not data.load('train',FAMILIES[domain]['window']).groups & data.load('validation',FAMILIES[domain]['window']).groups
    with pytest.raises(ValueError,match='ROLE'):
        Dataset(root/'native'/domain/'holdout/dataset.json',allowed_splits={'train','validation'})

@pytest.mark.parametrize('domain',FAMILIES)
def test_pipeline_training_export_raw_inference(contextual_campaign,tmp_path,domain):
    from phm_design.io import read_json,write_json
    from phm_design.family_design import design_family
    from phm_design.family_pipeline import attach_pipeline,predict_pipeline
    from phm_design.engine import predict
    root=contextual_campaign;prepared=tmp_path/'prepared'
    prepare_family_section(root,domain,'development','mechanism',prepared)
    holdout=tmp_path/'holdout';prepare_family_section(root,domain,'holdout','mechanism',holdout)
    p=tmp_path/'policy.json';write_json(p,policy())
    out=tmp_path/'run';design_family(prepared/'dataset.json',prepared/'device_binding.json',p,out)
    package=tmp_path/'pipeline';attach_pipeline(out/'selected_composed',domain,'mechanism',package)
    sample=read_json(root/'native'/domain/'holdout/dataset.json')['samples'][0]
    rawpath=root/'native'/domain/'holdout'/sample['path']
    passportfile=root/'passports'/domain/(sample['lineage_id']+'.json')
    result=predict_pipeline(package,rawpath,passportfile)
    processed=read_json(holdout/'dataset.json')['samples'][0]
    units={c['name']:c['unit'] for c in feature_contract(domain,'mechanism')['channels']}
    expected=predict(out/'selected_composed',holdout/processed['path'],units)
    np.testing.assert_array_equal(result['scores'],list(expected['scores'].values()))
    assert result['closed_set_argmax']==expected['closed_set_argmax']
    (package/'pipeline.json').write_text('{}')
    with pytest.raises(ValueError,match='INTEGRITY'):predict_pipeline(package,rawpath,passportfile)
