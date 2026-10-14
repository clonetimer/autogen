import numpy as np
import pytest
from phm_design.platform_features import FIELDS,spec,registry,balance_features
from phm_design.open_design import Program,FeatureUnavailable
from phm_design.adapters.platform_stress import fault_envelope
from phm_design.release_qualification import assess_release,LEGACY_CELLS

@pytest.mark.parametrize('domain',list(FIELDS))
@pytest.mark.parametrize('kind',['constant','smooth','near_rank_deficient'])
def test_platform_features_are_finite(domain,kind):
    n=80;t=np.arange(n,dtype=float);a=np.ones((n,len(FIELDS[domain])));a[:,0]=t
    if kind=='smooth':
        for j in range(1,a.shape[1]):a[:,j]=1+j*.1+np.sin(t/(4+j))
    elif kind=='near_rank_deficient':a[:,1:]+=np.arange(n)[:,None]*1e-14
    if domain.startswith('Thermal'):
        a[:,-2 if domain.endswith('localization') else -1]+=300
        if domain.endswith('localization'):a[:,-1]+=300
    r=registry(domain)
    for atom in r.names:
        x=r.transform(Program((atom,),'linear'),a,spec(domain),healthy_prefix_confirmed=True)
        assert x.ndim==1 and len(x)>0 and np.isfinite(x).all()

@pytest.mark.parametrize('domain',list(FIELDS))
def test_prefix_contract_not_silently_relaxed(domain):
    a=np.ones((20,len(FIELDS[domain])));a[:,0]=np.arange(20)
    with pytest.raises(FeatureUnavailable):registry(domain).transform(Program(('balance',),'linear'),a,spec(domain))
    assert registry(domain,raw_only=True).names==('temporal','kernels','difference_kernels')

@pytest.mark.parametrize('profile,t,expected',[('step',9,0),('step',10,1),('progressive',10,0),('progressive',15,.5),('progressive',25,1),('intermittent',10,1),('intermittent',16,0),('intermittent',20,1)])
def test_native_envelope(profile,t,expected):
    p={'onset_s':10,'fault_profile':profile,'ramp_s':10,'intermittent_period_s':10,'duty_cycle':.5}
    assert fault_envelope(t,p)==expected

@pytest.mark.parametrize('profile,extra',[('other',{}),('progressive',{'ramp_s':0}),('intermittent',{'intermittent_period_s':0,'duty_cycle':.5}),('intermittent',{'intermittent_period_s':5,'duty_cycle':1})])
def test_invalid_native_envelope_is_rejected(profile,extra):
    with pytest.raises(ValueError):fault_envelope(11,{'onset_s':10,'fault_profile':profile,**extra})

def test_known_limitation_kept_not_pass_and_empty_env_blocks():
    e={'capabilities':{c:{'status':'SUPPORTED'} for c in LEGACY_CELLS}}
    e['capabilities']['ADCS:weak_fault']={'status':'KNOWN_LIMITATION','weak_recall':.333}
    r=assess_release(e,accepted_limitations=['ADCS:weak_fault'])
    assert r['legacy_12_cells_retained'] and r['decision']=='HOLD_DO_NOT_PROMOTE'
    assert r['capabilities']['ADCS:weak_fault']['status']=='KNOWN_LIMITATION'
    assert not r['known_limitations_count_as_pass']

def test_cannot_expand_exceptions_after_seeing_failure():
    with pytest.raises(ValueError):assess_release({},accepted_limitations=['EPS:weak_fault'])

def test_missing_scope_and_missing_calibration_block():
    r=assess_release({'capabilities':{}},accepted_limitations=[])
    assert not r['legacy_12_cells_retained'] and any(b['gate']=='independent_calibration' for b in r['blockers'])
