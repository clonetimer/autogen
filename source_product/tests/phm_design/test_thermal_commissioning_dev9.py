import copy
import importlib.util
from pathlib import Path
import numpy as np
import pytest

from phm_design.thermal_commissioning_v2 import (
    MODEL_SCHEMA, COEFFICIENTS, TRANSFORM, _fit_linear_health_model,
    validate_model, thermal_v2_contract, thermal_v2_binding, compute_features,
    fit_registry, prepare_section,
)
from phm_design.commissioning import PROFILE_SCHEMA, REFERENCE_NAMES, generate_commissioning
from phm_design.family_pipeline import make_passport, compute_features as base_features, generate_with_passports
from phm_design.adapters.sat_nonadcs import parameters, world_id, FAMILIES
from phm_design.family_design import make_request
from phm_design.synthesis import propose
from phm_design.models import Shape
from phm_design.graph import compositional_registry
from phm_design.io import write_json, read_json
from phm_design.data import Dataset
from tests.phm_design.test_family_pipeline import raw
from tests.phm_design.test_nonadcs import policy


def fake_profile_and_model():
    domain='thermal';world=world_id(909,'train',0);passport=make_passport(world,domain,parameters(world,domain))
    x=raw(domain,128).astype(float);f=base_features(domain,x,passport,'mechanism')[:,-4:]
    refs=[float(np.median(f[64:,i])) for i in range(3)]
    profile={"schema":PROFILE_SCHEMA,"domain":domain,"instance_lineage":world,
        "source_observation_sha256":"0"*64,"source_role":"separate_nominal_commissioning_observation",
        "reference_method":"median_of_valid_second_half_mechanism_features","valid_reference_samples":8,
        "references":dict(zip(REFERENCE_NAMES[domain],refs)),"fault_information_used":False,
        "paired_diagnosis_nominal_used":False,"simulator_truth_used":False,"scope":"fixture"}
    model={"schema":MODEL_SCHEMA,"domain":"thermal","instance_lineage":world,"source_observation_sha256":"1"*64,
        "source_role":"separate_nominal_commissioning_observation",
        "regression":"healthy_balance_residual ~ solar_input + heat_command + nominal_radiation + intercept",
        "fit_samples":64,"coefficients":dict(zip(COEFFICIENTS,[.01,-.02,.03,.2])),"fit_rms_w":.1,"condition_number":20.,
        "fault_information_used":False,"paired_diagnosis_nominal_used":False,"simulator_truth_used":False,"scope":"fixture"}
    return passport,profile,model,x


def test_linear_health_regression_recovers_known_coefficients():
    rng=np.random.default_rng(42);X=np.column_stack((rng.normal(10,2,100),rng.normal(5,1,100),rng.normal(9,1.5,100),np.ones(100)))
    truth=np.array([.03,-.04,.07,.5]);y=X@truth
    coef,rms,cond,n=_fit_linear_health_model(y,X)
    np.testing.assert_allclose(coef,truth,rtol=1e-10,atol=1e-10)
    assert rms<1e-10 and np.isfinite(cond) and n==50


def test_thermal_v2_contract_compiles_with_same_designer():
    b=thermal_v2_binding();assert b['feature_pipeline']==TRANSFORM and b['thermal_health_model_required']
    req=make_request(b,policy());proposal=propose(req,b['contract'],0,[],set());shape=Shape(b['window_samples'],len(b['contract']['channels']),3)
    import torch
    for item in proposal['accepted']:
        model=compositional_registry().build(item['spec'],shape,11,50000)
        assert model(torch.zeros(2,shape.window,shape.channels)).shape==(2,3)


def test_thermal_v2_feature_causality_and_shape():
    passport,profile,model,x=fake_profile_and_model();a=compute_features(x,passport,profile,model)
    x2=x.copy();x2[-8:,2]+=1.5;b=compute_features(x2,passport,profile,model)
    assert a.shape==(128,len(thermal_v2_contract()['channels'])) and np.isfinite(a).all()
    np.testing.assert_array_equal(a[:-8],b[:-8])


def test_thermal_health_model_rejects_forbidden_and_lineage_mismatch():
    passport,profile,model,x=fake_profile_and_model();bad=copy.deepcopy(model);bad['fault_information_used']=True
    with pytest.raises(ValueError,match='FORBIDDEN'):validate_model(bad)
    other=copy.deepcopy(model);other['instance_lineage']='other'
    with pytest.raises(ValueError,match='INSTANCE'):compute_features(x,passport,profile,other)


@pytest.fixture(scope='module')
def real_campaign(tmp_path_factory):
    if importlib.util.find_spec('Basilisk') is None:pytest.skip('optional native runtime')
    root=tmp_path_factory.mktemp('thermal_v2');cfg=root/'cfg.json'
    write_json(cfg,{"schema":"phm.native-nonadcs.challenge.v2","seed":919107,
      "groups_per_role":{"train":1,"validation":1,"calibration":0,"test":0},"domains":["thermal"],
      "challenge":{
        "fault_fraction_by_role":{r:[.08,.12] for r in ('train','validation','calibration','test')},
        "onset_fraction_by_role":{r:[.4,.5] for r in ('train','validation','calibration','test')},
        "nuisance_scale_by_role":{r:[.8,1.] for r in ('train','validation','calibration','test')},
        "boundary_cycles_by_role":{r:[2.,3.] for r in ('train','validation','calibration','test')}}})
    campaign=root/'campaign';generate_with_passports(cfg,campaign)
    health=root/'health';generate_commissioning(campaign,health)
    models=root/'thermal_models';record=fit_registry(campaign,health,models)
    return campaign,health,models,record


def test_real_healthy_regression_registry_uses_only_health_observation(real_campaign):
    campaign,health,models,record=real_campaign
    assert record['completed_models']==2 and not record['fault_information_used'] and not record['diagnosis_time_paired_nominal_used']
    for item in record['models'].values():
        m=read_json(models/item['model']);validate_model(m)
        assert m['fit_samples']>20 and m['fit_rms_w']>=0 and m['condition_number']<1e8
        text=str(m).lower()
        for forbidden in ('fault_fraction','remaining_fraction','onset_s','extra_heat_w'):
            assert forbidden not in text


def test_real_prepare_v2_development_dataset(real_campaign,tmp_path):
    campaign,health,models,_=real_campaign;out=tmp_path/'prepared'
    prepare_section(campaign,health,models,'development',out)
    d=Dataset(out/'dataset.json',allowed_splits={'train','validation'});d.verify_files()
    assert len(d.samples)==6 and d.contract==thermal_v2_contract()
    tr=d.load('train',FAMILIES['thermal']['window']);va=d.load('validation',FAMILIES['thermal']['window'])
    assert not tr.groups & va.groups
