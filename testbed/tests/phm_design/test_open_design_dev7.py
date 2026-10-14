import json
from dataclasses import replace
import numpy as np
import pytest
from phm_design.open_design import (ObservationSpec,Program,RiskPreference,FeedbackPlanner,generic_registry,pareto_indices,FeatureRegistry)
from phm_design.open_design_fit import TrainingExample,design,predict,validate_training,fit_estimator


def observations(k=0):
    t=np.linspace(0,30,48);return np.c_[t,np.sin(t/4)*(1+k),np.cos(t/7)+.02*k*t]


def spec(name='unseen_domain'):return ObservationSpec(name,('time_s','sensor_a','sensor_b'))


@pytest.mark.parametrize('atom',generic_registry().names)
def test_generic_atoms_do_not_require_healthy_prefix(atom):
    x=generic_registry().transform(Program((atom,),'linear'),observations(),spec())
    assert x.ndim==1 and np.isfinite(x).all()


@pytest.mark.parametrize('name',['ADCS','EPS','Thermal','Data','pump','unseen_domain'])
def test_contract_has_no_domain_name_switch(name):
    assert spec(name).array(observations()).shape==(48,3)


@pytest.mark.parametrize('field',['truth_label','fault_id','world_id','onset','injection','applied_torque_nm'])
def test_non_deployment_channels_rejected(field):
    with pytest.raises(ValueError):ObservationSpec('x',('time_s',field))


@pytest.mark.parametrize('bad', ['nan','reverse','duplicate','shape','short'])
def test_observation_validation(bad):
    x=observations()
    if bad=='nan':x[0,1]=np.nan
    elif bad=='reverse':x=x[::-1]
    elif bad=='duplicate':x[1,0]=x[0,0]
    elif bad=='shape':x=x[:,:2]
    else:x=x[:5]
    with pytest.raises(ValueError):spec().array(x)


def test_extra_truth_dictionary_never_reaches_feature():
    rows=[dict(zip(spec().fields,r)) for r in observations()];rows[0]['truth']=1
    with pytest.raises(ValueError):spec().rows(rows)


def test_prefix_contract_is_per_atom_not_global():
    r=generic_registry();r.register('anchored',lambda a:np.mean(a,axis=0),requires_healthy_prefix=True)
    with pytest.raises(ValueError,match='PREFIX'):r.transform(Program(('anchored',),'linear'),observations(),spec())
    assert r.transform(Program(('temporal_bins',),'linear'),observations(),spec()).size


def test_registry_duplicate_fails():
    r=generic_registry()
    with pytest.raises(ValueError):r.register('temporal_bins',lambda a:a)


def test_pareto_preserves_health_fault_tradeoff():
    assert pareto_indices([[0,.8],[.5,.1],[.9,.9]])==[0,1]


@pytest.mark.parametrize('risk',[(.6,.4),(1.,0.),(0.,1.)])
def test_preferences_are_explicit(risk):
    r=RiskPreference(*risk);assert 0<=r.loss([.1,.5,1.])<=1


@pytest.mark.parametrize('risk',[(-1.,2.),(.2,.2),(float('nan'),.3)])
def test_invalid_preferences(risk):
    with pytest.raises(ValueError):RiskPreference(*risk)


def test_same_seed_search_replays_given_same_feedback():
    a=FeedbackPlanner(('a','b'));b=FeedbackPlanner(('a','b'))
    for i in range(24):
        x=a.propose();y=b.propose();assert x==y
        e=[(i%3)/3,.1,.2];a.observe(x,e);b.observe(y,e)
    assert any(x['reason'].startswith('feedback') for x in a.proposals)


def test_feedback_can_change_future_programs():
    a=FeedbackPlanner(('a','b'));b=FeedbackPlanner(('a','b'));different=False
    for i in range(30):
        x=a.propose();y=b.propose();different|=x.key!=y.key
        a.observe(x,[.1 if i%2 else .9]*3);b.observe(y,[.9 if i%2 else .1]*3)
    assert different


def test_expert_proposals_ignore_outcome_values():
    a=FeedbackPlanner(('a','b'),mode='expert');b=FeedbackPlanner(('a','b'),mode='expert')
    for i in range(24):
        x=a.propose();y=b.propose();assert x==y;a.observe(x,[.1]*3);b.observe(y,[.9]*3)


def test_effective_duplicate_hyperparameters_deduplicated():
    a=Program(('temporal_bins',),'forest',strength=.1)
    assert a.key==replace(a,strength=100.).key
    assert Program(('temporal_bins',),'knn',seed=12).key==Program(('temporal_bins',),'knn',seed=98).key


@pytest.mark.parametrize('role',['calibration','sealed_blind','development_scoring','test'])
def test_protected_roles_refused(role):
    es=[TrainingExample(observations(i),i%3,str(i),role) for i in range(6)]
    with pytest.raises(ValueError,match='PROTECTED'):validate_training(es,spec(),(0,1,2))


def test_compound_graph_no_fixed_diagnostic_threshold():
    r=generic_registry();p=Program(('temporal_bins','random_kernels'),'linear');g=r.graph(p,spec())
    assert g['fixed_fault_threshold'] is None and not g['requires_initial_healthy_prefix']
    assert len(g['nodes'])==6


@pytest.mark.parametrize('learner',FeedbackPlanner.learners)
def test_all_learners_fit(learner):
    rng=np.random.default_rng(4);X=rng.normal(size=(24,5));y=np.arange(24)%3
    model,ws=fit_estimator(Program(('temporal_bins',),learner),X,y,0)
    assert model.predict(X).shape==(24,)


def test_callable_design_group_integrity_and_model_roundtrip(tmp_path):
    import joblib
    es=[TrainingExample(observations(c),c,str(w)) for w in range(4) for c in range(3)]
    m,r=design(es,spec(),(0,1,2),0,max_trials=3)
    for t in r['attempts']:
        assert t['status']=='COMPLETE'
        assert all(f['disjoint'] for f in t['folds'])
    dest=tmp_path/'m';joblib.dump(m,dest);mm=joblib.load(dest)
    assert predict(mm,observations(),generic_registry())==predict(m,observations(),generic_registry())
    assert r['selection_only'] and not r['release_qualified']
