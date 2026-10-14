import copy
import importlib.util
from pathlib import Path
import numpy as np
import pytest

from phm_design.adapters.sat_nonadcs import FAMILIES, parameters, world_id
from phm_design.commissioning import (
    PROFILE_SCHEMA, TRANSFORM, REFERENCE_NAMES, commissioned_contract, commissioned_binding,
    compute_commissioned_features, validate_profile, generate_commissioning,
    prepare_commissioned_section, _profile_from_raw,
)
from phm_design.family_pipeline import make_passport, feature_contract, compute_features
from phm_design.family_design import make_request
from phm_design.synthesis import propose
from phm_design.models import Shape
from phm_design.graph import compositional_registry
from phm_design.io import write_json, read_json, sha_file
from phm_design.data import Dataset
from tests.phm_design.test_nonadcs import policy
from tests.phm_design.test_family_pipeline import raw


def profile(domain):
    w=world_id(808,'train',0);p=make_passport(w,domain,parameters(w,domain));x=raw(domain,64 if domain!='thermal' else 128)
    # Synthetic fixtures can be too unexcited for thermal/data profile extraction;
    # construct a legal profile directly from mechanism medians for unit tests.
    f=compute_features(domain,x,p,'mechanism')[:,-4:]
    vals=[]
    for i in range(3): vals.append(float(np.median(f[len(f)//2:,i])))
    return p,{"schema":PROFILE_SCHEMA,"domain":domain,"instance_lineage":w,
        "source_observation_sha256":"0"*64,"source_role":"separate_nominal_commissioning_observation",
        "reference_method":"median_of_valid_second_half_mechanism_features","valid_reference_samples":8,
        "references":dict(zip(REFERENCE_NAMES[domain],vals)),"fault_information_used":False,
        "paired_diagnosis_nominal_used":False,"simulator_truth_used":False,
        "scope":"software fixture"}


@pytest.mark.parametrize('domain',FAMILIES)
def test_commissioned_contract_and_designer_compile(domain):
    b=commissioned_binding(domain);assert b['feature_pipeline']==TRANSFORM and b['commissioning_profile_required']
    req=make_request(b,policy());proposal=propose(req,b['contract'],0,[],set())
    shape=Shape(b['window_samples'],len(b['contract']['channels']),3)
    import torch
    for item in proposal['accepted']:
        m=compositional_registry().build(item['spec'],shape,11,50000)
        assert m(torch.zeros(2,shape.window,shape.channels)).shape==(2,3)


@pytest.mark.parametrize('domain',FAMILIES)
def test_commissioned_feature_shape_causality_and_centering(domain):
    p,h=profile(domain);n=64 if domain!='thermal' else 128;x=raw(domain,n).astype(float)
    a=compute_commissioned_features(domain,x,p,h)
    x2=x.copy();x2[-8:,-1]+=2
    b=compute_commissioned_features(domain,x2,p,h)
    assert a.shape==(n,len(commissioned_contract(domain)['channels']))
    np.testing.assert_array_equal(a[:-8],b[:-8])
    assert np.isfinite(a).all()
    # First three extra channels are mechanism values minus frozen health references.
    base=compute_features(domain,x,p,'mechanism')
    refs=np.array([h['references'][k] for k in REFERENCE_NAMES[domain]])
    np.testing.assert_allclose(a[:,-7:-4],base[:,-4:-1]-refs,rtol=1e-6,atol=1e-6)


def test_profile_rejects_forbidden_information_and_lineage_mismatch():
    p,h=profile('eps');bad=copy.deepcopy(h);bad['fault_information_used']=True
    with pytest.raises(ValueError,match='FORBIDDEN'):validate_profile(bad,'eps')
    x=raw('eps',64);other=copy.deepcopy(h);other['instance_lineage']='other'
    with pytest.raises(ValueError,match='INSTANCE'):compute_commissioned_features('eps',x,p,other)


@pytest.mark.parametrize('domain',FAMILIES)
def test_profile_has_only_observation_derived_references(domain):
    p,h=profile(domain);validate_profile(h,domain)
    text=str(h).lower()
    for forbidden in ('fault_fraction','remaining_fraction','onset_s','actual_efficiency','extra_heat_w','extra_load_w'):
        assert forbidden not in text


@pytest.fixture(scope='module')
def commissioned_campaign(tmp_path_factory):
    if importlib.util.find_spec('Basilisk') is None:pytest.skip('optional native runtime')
    from phm_design.family_pipeline import generate_with_passports
    root=tmp_path_factory.mktemp('commissioned_campaign');cfg=root/'config.json'
    write_json(cfg,{'schema':'phm.native-nonadcs.challenge.v2','seed':881107,
        'groups_per_role':{'train':1,'validation':1,'calibration':1,'test':1},'domains':list(FAMILIES),
        'challenge':{
          'fault_fraction_by_role':{r:[.08,.12] for r in ('train','validation','calibration','test')},
          'onset_fraction_by_role':{r:[.4,.5] for r in ('train','validation','calibration','test')},
          'nuisance_scale_by_role':{r:[.8,1.] for r in ('train','validation','calibration','test')},
          'boundary_cycles_by_role':{r:[2.,3.] for r in ('train','validation','calibration','test')}}})
    campaign=root/'campaign';generate_with_passports(cfg,campaign)
    health=root/'health';record=generate_commissioning(campaign,health)
    return campaign,health,record


def test_separate_native_commissioning_runs_are_created_and_sealed(commissioned_campaign):
    campaign,health,record=commissioned_campaign
    assert record['completed_profiles']==12 and record['separate_nominal_runs']==12
    assert not record['fault_information_used'] and not record['diagnosis_time_paired_nominal_used']
    for item in record['profiles'].values():
        assert (health/item['profile']).is_file() and (health/item['observation']).is_file()


@pytest.mark.parametrize('domain',FAMILIES)
def test_prepare_commissioned_roles_and_profile_provenance(commissioned_campaign,tmp_path,domain):
    campaign,health,_=commissioned_campaign
    out=tmp_path/domain;prepare_commissioned_section(campaign,health,domain,'development',out)
    d=Dataset(out/'dataset.json',allowed_splits={'train','validation'});d.verify_files()
    assert len(d.samples)==6 and d.contract==commissioned_contract(domain)
    records=read_json(out/'source_records.json')
    assert all(len(r['commissioning_profile_sha256'])==64 for r in records)
    assert not d.load('train',FAMILIES[domain]['window']).groups & d.load('validation',FAMILIES[domain]['window']).groups
