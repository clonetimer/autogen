import numpy as np
import pytest
from dataclasses import replace
from phm_design.open_design import ObservationSpec,Program,generic_registry
from phm_design.open_design_fit import TrainingExample,evaluate_program,refit,predict,design
from phm_design.open_design_numerics import matrix_support
from phm_design.open_design_validation import validation_splits
from phm_design.open_design_feasibility import audit_observation_aliases


def fixture():
    spec=ObservationSpec('generic',('t','x'),'t');rs=[]
    for group in range(9):
        for label in (0,1):
            t=np.linspace(0,1,48);x=np.sin(t*(label+1))+label+group*.01
            for profile in ('identity','drift_quant','delay_noise'):
                rs.append(TrainingExample(np.c_[t,x],label,'w'+str(group),profile=profile))
    fmap={'w'+str(i):'f'+str(i%3) for i in range(9)}
    return spec,rs,fmap

@pytest.mark.parametrize('a',[np.zeros((12,4)),np.ones((12,4)),np.eye(4),np.ones((2,5)),np.diag([1.,1e-300])])
def test_support_finite_for_valid_singular_inputs(a):
    z=matrix_support(a)
    assert np.isfinite([z.capped_log_condition,z.rank_fraction]).all()
    assert 0<=z.rank_fraction<=1

@pytest.mark.parametrize('a',[[],[1,2],np.empty((0,3)),[[np.nan]],[[np.inf]],[[1,-np.inf]]])
def test_support_rejects_bad_input(a):
    with pytest.raises(ValueError,match='INVALID_DESIGN_MATRIX'):matrix_support(a)

@pytest.mark.parametrize('scale',[1.,1e-200,1e200,-3.])
def test_rank_is_scale_invariant(scale):
    a=np.array([[1.,0,2],[0,1,1],[1,1,3],[2,1,5]])
    x=matrix_support(a);y=matrix_support(a*scale)
    assert x.effective_rank==y.effective_rank==2
    assert x.rank_deficient and y.rank_deficient
    assert abs(x.capped_log_condition-y.capped_log_condition)<1e-8

def test_full_rank_not_marked_deficient():
    a=matrix_support(np.eye(4));assert not a.rank_deficient and a.rank_fraction==1.

def test_family_fold_covers_once_and_no_leakage():
    spec,rs,f=fixture();g=np.array([x.group for x in rs]);seen=np.zeros(len(g),int)
    for tr,te,a in validation_splits(g,f):
        assert a['family_disjoint'];assert set(g[tr]).isdisjoint(g[te]);seen[te]+=1
        assert {f[w] for w in g[tr]}.isdisjoint(f[w] for w in g[te])
    assert (seen==1).all()

def test_default_world_split_retained():
    _,rs,_=fixture();assert len(validation_splits([e.group for e in rs]))==3

@pytest.mark.parametrize('kind',['missing','extra','empty_family','one_family','bad_type'])
def test_bad_family_metadata_is_not_silently_guessed(kind):
    _,rs,f=fixture()
    if kind=='missing':f.pop('w1')
    elif kind=='extra':f['not-in-data']='z'
    elif kind=='empty_family':f['w0']=''
    elif kind=='one_family':f={k:'same' for k in f}
    else:f=[]
    with pytest.raises(ValueError):validation_splits([e.group for e in rs],f)

def test_family_fold_needs_enough_training_worlds():
    with pytest.raises(ValueError,match='INSUFFICIENT_TRAIN_WORLDS'):
        validation_splits(['a','b','c'],{'a':'x','b':'y','c':'y'})

def test_family_evaluation_and_prediction_contract():
    spec,rs,f=fixture();reg=generic_registry();p=Program(('temporal_bins',),'linear')
    m,rows,folds=evaluate_program(rs,spec,(0,1),0,p,reg,{},validation_families=f)
    assert len(rows)==len(rs) and m['validation_mode']=='leave_environment_family_out'
    assert len(m['family_metrics'])==3 and all(x['family_disjoint'] for x in folds)
    model=refit(rs,spec,(0,1),0,p,reg,{})
    assert 'family' not in repr(model['observation_spec'])
    assert predict(model,rs[0].observations,reg)['prediction'] in (0,1)

def test_design_api_exposes_family_validation():
    spec,rs,f=fixture();model,report=design(rs,spec,(0,1),0,max_trials=1,validation_families=f)
    assert report['attempts'][0]['metrics']['validation_mode']=='leave_environment_family_out'

def test_exact_alias_has_conditional_empirical_bound():
    spec,rs,_=fixture();rs=[e for e in rs if e.profile=='identity']
    # Same declared observable vector, different labels, three physical worlds.
    a=rs[0].observations
    rows=[TrainingExample(a.copy(),0,'a'),TrainingExample(a.copy(),1,'b'),TrainingExample(a.copy(),1,'c')]
    r=audit_observation_aliases(rows,spec,(0,1))
    assert r['minimum_empirical_errors']==1 and r['conflicting_observation_groups']==1
    assert r['minimum_empirical_error_fraction']==pytest.approx(1/3)
    assert not r['universal_infeasibility']

def test_no_exact_alias_not_feasibility_proof():
    spec,rs,_=fixture();r=audit_observation_aliases(rs,spec,(0,1))
    assert r['status']=='NO_EXACT_CONFLICT_FOUND_NOT_PROOF_OF_FEASIBILITY'
    assert r['identity_records']==18 and not r['release_qualified']

def test_signed_zero_is_not_distinct_evidence():
    spec=ObservationSpec('g',('t','x'),'t');t=np.arange(32.)
    rows=[TrainingExample(np.c_[t,np.full(32,v)],lab,g) for v,lab,g in [(0.,0,'a'),(-0.,1,'b'),(0.,1,'c')]]
    assert audit_observation_aliases(rows,spec,(0,1))['minimum_empirical_errors']==1

@pytest.mark.parametrize('role',['calibration','sealed_blind','development_scoring'])
def test_alias_audit_never_opens_reserved_role(role):
    spec,rs,_=fixture();rs[0]=replace(rs[0],role=role)
    with pytest.raises(ValueError,match='PROTECTED_ROLE'):audit_observation_aliases(rs,spec,(0,1))
