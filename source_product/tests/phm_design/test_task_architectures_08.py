"""0.8 task-aware architecture selection tests: synthetic fixtures only."""
import numpy as np
import pytest
import joblib
from phm_design.task_architectures import TaskOntology, design, TaskAcceptancePolicy
from phm_design.platform_features import FIELDS


def fixture_train(world_count=8):
    rng=np.random.default_rng(1007)
    rows=[];labels=[];world=[];roles=[];health=[]
    for k in range(world_count):
        for y in range(5):
            n=181; t=np.linspace(0,3600,n);phase=.02*k
            light=(np.sin(t/800+phase)>.0).astype(float)
            heat_a=np.full(n,4.+.05*k);heat_b=np.full(n,4.1+.03*k)
            a=302.+.001*t+light*.75+rng.normal(0,.01,n)
            b=305.+.001*t+light*.72+rng.normal(0,.01,n)
            if y in (1,3):a[65:]+=(.5 if y==1 else .7)
            if y in (2,4):b[65:]+=(.5 if y==2 else .7)
            obs=np.column_stack([t,light,heat_a,heat_b,a,b])
            rows.append(obs);labels.append(y);world.append('w'+str(k));roles.append('design_train');health.append(True)
    return rows,labels,world,roles,health


def quick_fit():
    rows,y,g,r,h=fixture_train()
    return design(domain='Thermal_localization',observations=rows,fields=FIELDS['Thermal_localization'],
                  labels=y,world_ids=g,roles=r,healthy_prefix_confirmed=h,
                  candidate_specs=[('temporal','linear','direct'),('temporal','linear','factorized'),('temporal','linear','fusion')],
                  acceptance_policy=TaskAcceptancePolicy(0,1,0,0,8))


def test_ontology_is_strict():
    onto=TaskOntology.from_domain('Thermal_localization')
    assert len(onto.labels)==5
    assert onto.labels[0]=='NOMINAL'
    with pytest.raises(ValueError):TaskOntology.from_domain('Thermal')


def test_probability_product_is_normalized():
    onto=TaskOntology.from_domain('EPS_localization')
    p=onto.project(np.array([[.3,.7]]),np.array([[.4,.6]]),np.array([[.2,.8]]))
    assert p.shape==(1,5)
    assert np.isclose(p.sum(),1.)
    assert p[0,0]==pytest.approx(.3)


def test_new_designer_selects_only_on_grouped_training():
    out=quick_fit()
    assert out.audit['independent_training_worlds']==8
    assert out.audit['group_fold_intersection']==0
    assert out.audit['stage']=='DEVELOPMENT_ONLY'
    assert len(out.audit['candidate_results'])==3
    assert all(x['status']=='COMPLETE' for x in out.audit['candidate_results'])


def test_conflict_is_retained_and_ontology_legal():
    rows,*_=fixture_train()
    out=quick_fit();d=out.diagnose(rows[0],fields=FIELDS['Thermal_localization'],healthy_prefix_confirmed=True)
    assert set(d['proba'])==set(out.ontology.labels)
    assert sum(d['proba'].values())==pytest.approx(1.)
    assert d['prediction'] in (None,*out.ontology.labels)
    assert type(d['head_conflict']) is bool
    assert not d['qualified']


def test_extra_inference_field_blocked():
    rows,*_=fixture_train();out=quick_fit()
    with pytest.raises(ValueError,match='EXACT_DEPLOYMENT_FIELDS_REQUIRED'):
        out.diagnose(rows[0],fields=FIELDS['Thermal_localization']+('truth_fault',),healthy_prefix_confirmed=True)


def test_frozen_prefix_assumption_enforced():
    rows,*_=fixture_train();out=quick_fit()
    with pytest.raises(ValueError,match='HEALTHY_PREFIX_NOT_CONFIRMED'):
        out.diagnose(rows[0],fields=FIELDS['Thermal_localization'],healthy_prefix_confirmed=False)


def test_evaluation_role_fails_closed():
    rows,y,g,r,h=fixture_train();r[-1]='sealed_blind'
    with pytest.raises(ValueError,match='PROTECTED_OR_EVALUATION_ROLE'):
        design(domain='Thermal_localization',observations=rows,fields=FIELDS['Thermal_localization'],
               labels=y,world_ids=g,roles=r,healthy_prefix_confirmed=h)


def test_world_count_guard():
    rows,y,g,r,h=fixture_train(8)
    g=[f'w{int(item[1:])%3}' for item in g]
    with pytest.raises(ValueError,match='INSUFFICIENT_INDEPENDENT_WORLDS'):
        design(domain='Thermal_localization',observations=rows,fields=FIELDS['Thermal_localization'],
               labels=y,world_ids=g,roles=r,healthy_prefix_confirmed=h)


def test_training_cannot_accept_hidden_truth_as_column():
    rows,y,g,r,h=fixture_train()
    with pytest.raises(ValueError,match='EXACT_DEPLOYMENT_FIELDS_REQUIRED'):
        design(domain='Thermal_localization',observations=rows,fields=FIELDS['Thermal_localization']+('injected_fault',),
               labels=y,world_ids=g,roles=r,healthy_prefix_confirmed=h)


def test_group_oof_disallows_leakage_by_construct():
    from phm_design.task_architectures import _split
    _,_,g,_,_=fixture_train()
    ga=np.asarray(g)
    for tr,va in _split(ga):
        assert not (set(ga[tr]) & set(ga[va]))


def test_joblib_roundtrip(tmp_path):
    rows,*_=fixture_train();out=quick_fit()
    path=tmp_path/'model.joblib';joblib.dump(out,path)
    restore=joblib.load(path)
    a=out.diagnose(rows[1],fields=FIELDS['Thermal_localization'],healthy_prefix_confirmed=True)
    b=restore.diagnose(rows[1],fields=FIELDS['Thermal_localization'],healthy_prefix_confirmed=True)
    assert a==b


def test_threshold_derived_from_oof_and_budgeted():
    out=quick_fit()
    assert out.threshold>=0.
    assert out.max_unknown_fraction==pytest.approx(.15)
    assert out.audit['threshold_trained_without_evaluation']
    assert len(out.audit['threshold_options'])>0
