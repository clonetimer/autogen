import copy
import importlib.util
from pathlib import Path
import numpy as np
import pytest
from phm_design.adapters import sat_nonadcs as n
from phm_design.adapters import nonadcs_challenge as c
from phm_design.family_pipeline import make_passport, validate_passport
from phm_design.io import read_json, digest, write_json, verify


def config(groups=1):
    return {"schema": c.SCHEMA, "seed": 24061003,
        "domains": list(n.FAMILIES), "groups_per_role": {r: groups for r in n.ROLES},
        "challenge": {
            "fault_fraction_by_role": {r: [.1,.3] for r in n.ROLES},
            "onset_fraction_by_role": {r: [.35,.55] for r in n.ROLES},
            "nuisance_scale_by_role": {r: [0.,.7] for r in n.ROLES},
            "boundary_cycles_by_role": {r: [2.,3.2] for r in n.ROLES}}}


LEGACY = read_json(Path(__file__).parent / 'fixtures/dev3_native_replay.json')


@pytest.mark.parametrize('row', LEGACY['parameters'], ids=lambda r:r['role']+'-'+r['domain'])
def test_legacy_rng_dictionaries_remain_identical(row):
    assert n.world_id(row['seed'],row['role'],row['index']) == row['world']
    assert n.parameters(row['world'],row['domain']) == row['parameters']


@pytest.mark.parametrize('row', LEGACY['native_cases'], ids=lambda r:r['domain']+'-'+str(r['label']))
def test_legacy_native_trajectory_replay(row):
    if importlib.util.find_spec('Basilisk') is None:pytest.skip('optional native runtime')
    d=row['domain'];p=n.parameters(n.world_id(120201,'train',0),d)
    r=n.run_native(d,p,row['label'])
    assert digest(r['observations'])==row['observations']
    assert digest(r['truth'])==row['truth']
    assert digest(r['output_times_ns'])==row['times']


def test_challenge_has_only_one_plan_dispatch_and_keeps_roles_stable():
    a=config(2);b=copy.deepcopy(a);b['groups_per_role']['train']=4
    left=n.plan_campaign(a);right=n.plan_campaign(b)
    assert left==[r for r in right if r['index_in_role']<2]
    assert len({x['world_id'] for x in left})==8
    for r in left:
        p=r['parameters'];assert .1<=p['fault_fraction']<=.3
        assert 0<=p['nuisance_scale']<=.7
        assert p['excess_generation_multiplier']==1+p['fault_fraction']


@pytest.mark.parametrize('field',list(c.RANGES))
@pytest.mark.parametrize('bad',[[True,.2],[.3,.1],[float('nan'),.4],[-1.,10.]])
def test_invalid_challenge_ranges_are_rejected(field,bad):
    cfg=config();cfg['challenge'][field]['test']=bad
    with pytest.raises(ValueError):n.plan_campaign(cfg)


@pytest.mark.parametrize('field',list(c.RANGES))
def test_challenge_missing_role_rejected(field):
    cfg=config();del cfg['challenge'][field]['calibration']
    with pytest.raises(ValueError):n.check_config(cfg)


def test_factor_change_preserves_pair_identity_and_other_normal_parameters():
    cfg=config();weak=copy.deepcopy(cfg);weak['challenge']['fault_fraction_by_role']['test']=[.04,.2]
    a=n.plan_campaign(cfg);b=n.plan_campaign(weak)
    for left,right in zip(a,b):
        assert left['world_id']==right['world_id']
        for key in left['parameters']:
            if left['role']=='test' and key in {'fault_fraction','remaining_fraction','extra_heat_w','extra_load_w','excess_generation_multiplier'}:continue
            assert left['parameters'][key]==right['parameters'][key]


def test_true_nuisance_is_absent_from_nominal_passport():
    for item in n.plan_campaign(config()):
        p=make_passport(item['world_id'],item['domain'],item['parameters'])
        validate_passport(p,item['domain'])
        assert not any('gain' in k or 'bias' in k or 'fault' in k for k in p['parameters'])
        updated=copy.deepcopy(item['parameters']);updated['nuisance_scale']=3;updated['solar_gain']=.85
        assert make_passport(item['world_id'],item['domain'],updated)==p


def test_nine_native_challenge_cases(tmp_path):
    if importlib.util.find_spec('Basilisk') is None:pytest.skip('optional native runtime')
    cfg=config();cfg['groups_per_role']={r:int(r=='train') for r in n.ROLES}
    f=tmp_path/'c.json';write_json(f,cfg)
    out=tmp_path/'out';report=n.generate_nonadcs(f,out)
    assert report['cases']==9
    assert not report['hardware_validated']
    verify(out)
    for d in cfg['domains']:
        binding=read_json(out/d/'device_binding.json')
        assert binding['measurement_settings']['nuisance_not_in_normal_passport'] is True
    for p in (out/'case_records').glob('*.json'):
        assert read_json(p)['pair_audit']['prefault_exact_match']
