from dataclasses import replace
import json
import numpy as np
import pytest
from phm_design.open_design import ObservationSpec, Program, FeatureRegistry, generic_registry, FeedbackPlanner
from phm_design.open_design_fit import (TrainingExample, feature_matrix, metrics, evaluate_program,
    refit, validate_training, fit_estimator, predict)
from phm_design.open_design_ensemble import GroupProbabilityEnsemble

SPEC = ObservationSpec('unseen_domain', ('time_s','sensor_a','sensor_b'))
def obs(c=0, w=0):
    t=np.linspace(0,40,48)
    return np.c_[t,np.sin(t*.2)*(1+c)+w*.01,np.cos(t*.17)+c*.01*t]
def examples():
    return [TrainingExample(obs(c,w),c,str(w)) for w in range(6) for c in range(3)]
def tiny_registry(shift=0):
    r=FeatureRegistry();r.register('a',lambda a:np.mean(a[:,1:],axis=0)+shift)
    return r

@pytest.mark.parametrize('change',['data','reorder','mutate','spec','registry','prefix'])
def test_cache_cannot_cross_contaminate_tasks(change):
    es=examples();r=tiny_registry();p=Program(('a',),'linear');cache={}
    feature_matrix(es,SPEC,p,r,cache)
    spec=SPEC
    if change=='data':es=[replace(e, observations=e.observations+np.array([0,30,12])) for e in es]
    elif change=='reorder':es=es[::-1]
    elif change=='mutate':es[0].observations[:,1]+=42
    elif change=='spec':spec=ObservationSpec('other_domain',SPEC.fields)
    elif change=='registry':r=tiny_registry(33)
    else:es=[replace(e,healthy_prefix_confirmed=True) for e in es]
    actual=feature_matrix(es,spec,p,r,cache);expected=feature_matrix(es,spec,p,r,{})
    assert np.array_equal(actual[0],expected[0]) and np.array_equal(actual[1],expected[1])


def test_feature_cache_hit_reuses_array():
    es=examples();r=tiny_registry();cache={};p=Program(('a',),'linear')
    feature_matrix(es,SPEC,p,r,cache);cached=cache['a'][0]
    feature_matrix(es,SPEC,p,r,cache);assert cached is cache['a'][0]


def test_health_unknown_is_not_known_fault_alarm():
    m=metrics([0,0,0,1,2],[0,1,-1,1,2],['a']*5,[0,1,2],0)
    assert m['healthy_false_alarms']==1 and m['healthy_unknown']==1
    assert m['healthy_nonacceptances']==2 and m['unknown']==1
    assert m['recall'][0]==pytest.approx(1/3)

@pytest.mark.parametrize('classes',[(-1,0,1),(0,True,2),(0.,1.,2.),(0,0,1)])
def test_invalid_unknown_or_noninteger_classes(classes):
    with pytest.raises(ValueError,match='INVALID_CLASSES'):validate_training(examples(),SPEC,classes)

@pytest.mark.parametrize('api',['evaluate','refit'])
@pytest.mark.parametrize('role',['calibration','sealed_blind','development_scoring','test'])
def test_all_public_training_paths_reject_protected_roles(api,role):
    es=examples();es[0].role=role;f=evaluate_program if api=='evaluate' else refit
    with pytest.raises(ValueError,match='PROTECTED'):
        f(es,SPEC,(0,1,2),0,Program(('a',),'linear'),tiny_registry(),{})

@pytest.mark.parametrize('api',['evaluate','refit'])
def test_missing_healthy_label_rejected(api):
    f=evaluate_program if api=='evaluate' else refit
    with pytest.raises(ValueError,match='HEALTHY_LABEL'):
        f(examples(),SPEC,(0,1,2),99,Program(('a',),'linear'),tiny_registry(),{})


def test_refit_cannot_silently_drop_a_class():
    es=examples();r=FeatureRegistry()
    def atom(a):
        if a[:,1].std()>1.8:raise ValueError('unsupported')
        return a[:,1:].mean(0)
    r.register('a',atom)
    with pytest.raises(ValueError,match='MISSING_REFIT_CLASS'):
        refit(es,SPEC,(0,1,2),0,Program(('a',),'linear'),r,{})

@pytest.mark.parametrize('learner',['group_stack','group_average'])
def test_group_compositions_fit_with_no_domain_special_case(learner):
    rs=examples();p=Program(('temporal_bins',),learner)
    m,rows,folds=evaluate_program(rs,SPEC,(0,1,2),0,p,generic_registry(),{})
    assert len(rows)==len(rs)
    for outer in folds:
        train=set(outer['train_groups']);test=set(outer['validation_groups'])
        assert outer['disjoint'] and len(outer['inner_group_audit'])==3
        for inner in outer['inner_group_audit']:
            a=set(inner['base_train_groups']);b=set(inner['meta_oof_groups'])
            assert a.isdisjoint(b) and a|b<=train and (a|b).isdisjoint(test)
            assert inner['scalers_fitted_inside_base_train']


def test_ensemble_without_group_metadata_refuses_to_fit():
    with pytest.raises(ValueError,match='PHYSICAL_GROUPS'):
        fit_estimator(Program(('a',),'group_stack'),np.ones((12,2)),np.arange(12)%3,0)

@pytest.mark.parametrize('combination',['learned','average'])
def test_nonconsecutive_labels_and_roundtrip(tmp_path,combination):
    import joblib
    rng=np.random.default_rng(4);X=rng.normal(size=(36,5));y=np.tile([2,5,11],12);g=np.repeat(np.arange(12),3)
    m=GroupProbabilityEnsemble(combination=combination).fit(X,y,groups=g)
    p=m.predict_proba(X);assert np.allclose(p.sum(1),1.) and set(m.predict(X))<=set(y)
    f=tmp_path/'model';joblib.dump(m,f);assert np.array_equal(m.predict(X),joblib.load(f).predict(X))
    assert len(m.group_audit_)==3 and m.meta_training_source_=='strict_group_out_of_fold_predictions'


def test_terminal_ablation_reuses_identical_base_objects():
    rng=np.random.default_rng(9);X=rng.normal(size=(36,4));y=np.arange(36)%3;g=np.arange(36)//3
    m=GroupProbabilityEnsemble().fit(X,y,groups=g);ids=list(map(id,m.bases_))
    expected=m.classes_[np.mean([b.predict_proba(X) for b in m.bases_],axis=0).argmax(1)]
    assert np.array_equal(expected,m.predict_average(X)) and list(map(id,m.bases_))==ids

@pytest.mark.parametrize('kind',['short_groups','missing_class','bad_weight','bad_combination'])
def test_ensemble_invalid_training_fails_closed(kind):
    X=np.arange(72).reshape(36,2);y=np.arange(36)%3;g=np.arange(36)//3;sw=np.ones(36);m=GroupProbabilityEnsemble()
    if kind=='short_groups':g=np.zeros(36)
    elif kind=='missing_class':g=y
    elif kind=='bad_weight':sw[0]=0
    else:m=GroupProbabilityEnsemble(combination='threshold')
    with pytest.raises(ValueError):m.fit(X,y,groups=g,sample_weight=sw)


def test_export_graph_declares_learned_combination():
    r=generic_registry();g=r.graph(Program(('temporal_bins',),'group_stack'),SPEC)
    assert g['ensemble_training']['meta']=='group_oof_logistic' and g['fixed_fault_threshold'] is None
    assert not g['ensemble_training']['group_ids_are_predictors']


def test_ensemble_depth_is_effective_search_dimension():
    p=Program(('a',),'group_stack',depth=3)
    assert p.key!=replace(p,depth=8).key


def test_planner_can_propose_group_composition():
    p=FeedbackPlanner(('a',),seeds=[Program(('a',),'group_stack')])
    assert p.propose().learner=='group_stack'
