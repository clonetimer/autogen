"""R50 availability composition contracts; no protected real data in fixtures."""
import numpy as np
import pytest
from dataclasses import replace
from phm_design.open_design import Program,FeatureRegistry,FeatureUnavailable,ObservationSpec,FeedbackPlanner
from phm_design.open_design_fit import TrainingExample,feature_matrix,refit,predict,evaluate_program

SPEC=ObservationSpec('synthetic_process',('time_s','value'))

def basic(a):return np.array([np.mean(a[:,1]),np.std(a[:,1])])
def optional(a):
    if a[0,1]<-10:raise FeatureUnavailable('REFERENCE_ABSENT')
    return basic(a)
def broken(a):raise ValueError('IMPLEMENTATION_BROKEN')
def invalid(a):return np.array([np.nan])

def registry():
    r=FeatureRegistry();r.register('base',basic);r.register('optional',optional)
    r.register('prefix',basic,requires_healthy_prefix=True);return r

def examples():
    out=[];t=np.arange(30,dtype=float)
    for g in range(6):
        for y in range(3):
            a=np.c_[t,y*5+np.sin(t)*.05+g*.01]
            if g%2:a[0,1]=-20
            out.append(TrainingExample(a,y,'physical_'+str(g),healthy_prefix_confirmed=True))
    return out

@pytest.fixture(scope='module')
def fitted():
    p=Program(('base','optional'),'extra',composition='available_stack')
    return refit(examples(),SPEC,(0,1,2),0,p,registry(),{})

@pytest.mark.parametrize('comp',['available_stack','available_average'])
@pytest.mark.parametrize('learner',['linear','extra','forest','gnb','svm','knn','probability_mix'])
def test_joint_composition_and_family(comp,learner):
    p=Program(('base','optional'),learner,composition=comp)
    m=refit(examples(),SPEC,(0,1,2),0,p,registry(),{})
    out=predict(m,examples()[3].observations,registry())
    assert out['status']=='RESEARCH_PREDICTION'
    assert out['available_components']==['base']
    assert out['unavailable_components']=={'optional':'REFERENCE_ABSENT'}
    assert m['release_qualified'] is False

@pytest.mark.parametrize('comp',['available_stack','available_average'])
def test_input_corruption_not_fallback(comp):
    r=registry();p=Program(('base','optional'),'extra',composition=comp)
    m=refit(examples(),SPEC,(0,1,2),0,p,r,{})
    a=examples()[0].observations.copy();a[5,1]=np.nan
    assert predict(m,a,r)['reason']=='OBSERVATION_NONFINITE'
    a=examples()[0].observations.copy();a[3,0]=a[2,0]
    assert predict(m,a,r)['reason']=='TIME_NOT_STRICTLY_INCREASING'

@pytest.mark.parametrize('fn',[broken,invalid])
def test_arbitrary_errors_not_masked(fn):
    r=registry();r.register('bad',fn)
    with pytest.raises(ValueError):
        feature_matrix(examples(),SPEC,Program(('base','bad'),'extra',composition='available_stack'),r,{})

@pytest.mark.parametrize('comp',['available_stack','available_average'])
def test_all_unavailable_is_unknown(comp):
    r=registry();p=Program(('optional','prefix'),'extra',composition=comp)
    m=refit(examples(),SPEC,(0,1,2),0,p,r,{})
    out=predict(m,examples()[3].observations,r,healthy_prefix_confirmed=False)
    assert out['prediction']==-1 and out['reason']=='NO_AVAILABLE_COMPONENT'

def test_inner_outer_group_isolation():
    p=Program(('base','optional'),'extra',composition='available_stack')
    metrics,rows,folds=evaluate_program(examples(),SPEC,(0,1,2),0,p,registry(),{})
    for f in folds:
        assert set(f['train_groups']).isdisjoint(f['validation_groups'])
        for audit in f['inner_group_audit']:
            assert set(audit['base_train_groups']).isdisjoint(audit['meta_oof_groups'])
            assert set(audit['base_train_groups']).isdisjoint(f['validation_groups'])
    assert len(rows)==18

def test_training_mask_is_not_new_world(fitted):
    a=fitted['estimator'].meta_training_audit_
    assert a['physical_worlds']==6 and a['new_physical_worlds']==0
    assert a['augmented_meta_rows']>a['original_rows']
    assert a['per_original_row_weight_preserved']

def test_same_bases_three_modes(fitted):
    p=Program(**{**fitted['program'],'atoms':tuple(fitted['program']['atoms'])})
    X,_=feature_matrix(examples(),SPEC,p,registry(),{})
    m=fitted['estimator'];before=[id(x) for x in m.bases_]
    partial=m.predict_with_mode(X,'learned');strict=m.predict_with_mode(X,'strict_all')
    avg=m.predict_with_mode(X,'available_average')
    assert np.sum(strict==-1)==9 and np.sum(partial==-1)==0 and np.sum(avg==-1)==0
    assert [id(x) for x in m.bases_]==before

def test_zero_mask_and_nonbinary_mask(fitted):
    m=fitted['estimator'];n=sum(fitted['atom_sizes'])+2
    assert m.predict(np.zeros((1,n)))[0]==-1
    a=np.zeros((1,n));a[0,-1]=.5
    with pytest.raises(ValueError,match='MASK_NOT_BINARY'):m.predict(a)

def test_dimension_contract(fitted):
    with pytest.raises(ValueError,match='DIMENSION'):fitted['estimator'].predict(np.zeros((1,3)))

def test_extra_fields_rejected(fitted):
    rows=[{'time_s':float(i),'value':0.,'truth':0} for i in range(30)]
    assert predict(fitted,rows,registry())['reason']=='EXACT_DEPLOYMENT_FIELDS_REQUIRED'

def test_unregistered_component_not_skipped(fitted):
    r=FeatureRegistry();r.register('base',basic)
    out=predict(fitted,examples()[0].observations,r)
    assert out['prediction']==-1 and out['reason'].startswith('UNKNOWN_FEATURE_ATOM')

@pytest.mark.parametrize('role',['sealed_blind','calibration','development_scoring','reserved_R13'])
def test_role_barrier(role):
    e=examples();e[0].role=role
    with pytest.raises(ValueError,match='PROTECTED_ROLE'):
        refit(e,SPEC,(0,1,2),0,Program(('base',),'extra',composition='available_stack'),registry(),{})

def test_graph_and_key(fitted):
    p=Program(('base','optional'),'extra');q=replace(p,composition='available_stack')
    assert p.key!=q.key
    graph=fitted['graph'];assert graph['composition']=='available_stack'
    assert graph['component_availability']['all_unavailable_returns_unknown']
    assert graph['fixed_fault_threshold'] is None

def test_legacy_default_compatible():
    p=Program(('base',),'linear');assert p.composition=='concat'
    m=refit(examples(),SPEC,(0,1,2),0,p,registry(),{})
    assert predict(m,examples()[0].observations,registry())['status']=='RESEARCH_PREDICTION'

@pytest.mark.parametrize('composition',['invalid',None])
def test_invalid_composition(composition):
    with pytest.raises(ValueError,match='COMPOSITION'):Program(('base',),'extra',composition=composition)

def test_nested_group_composition_rejected():
    with pytest.raises(ValueError,match='NESTED_GROUP'):Program(('base',),'group_stack',composition='available_stack')

def test_planner_explores_composition():
    planner=FeedbackPlanner(('base','optional'),seed=50019)
    kinds=set()
    for i in range(100):
        p=planner.propose();kinds.add(p.composition);planner.observe(p,[.4,.3,.2])
    assert kinds=={'concat','available_stack','available_average'}

def test_cache_broad_legacy_catch_cannot_poison_strict():
    r=registry();r.register('bad',broken);cache={}
    with pytest.raises(ValueError):feature_matrix(examples(),SPEC,Program(('base','bad'),'linear'),r,cache)
    with pytest.raises(ValueError,match='IMPLEMENTATION_BROKEN'):
        feature_matrix(examples(),SPEC,Program(('base','bad'),'linear',composition='available_stack'),r,cache)


def test_prefix_is_component_requirement_not_unconditional():
    r=registry();p=Program(('base','prefix'),'extra',composition='available_stack')
    m=refit(examples(),SPEC,(0,1,2),0,p,r,{})
    assert not m['graph']['requires_initial_healthy_prefix']
    assert m['graph']['component_prefix_requirements']=={'base':False,'prefix':True}
    out=predict(m,examples()[0].observations,r,healthy_prefix_confirmed=False)
    assert out['status']=='RESEARCH_PREDICTION' and out['available_components']==['base']
