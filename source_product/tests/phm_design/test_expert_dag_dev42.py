import json
import numpy as np
import pytest
from phm_design.expert_dag import ExpertSignals,make_candidate,execute,search_multi_environment

CLASSES=['N','A','B']; EXPERTS=['reference','fixed','generated']

def rec(truth, ref, fixed, gen, group='g'):
    return {'truth':truth,'group_id':group,'experts':{'reference':ref,'fixed':fixed,'generated':gen}}

def env_rows(flip=False):
    # reference is simplest and mostly right; generated fixes the hard A case.
    rows=[
      rec('N',[.9,.05,.05],[.8,.1,.1],[.85,.1,.05],'n'),
      rec('A',[.1,.45,.45],[.1,.4,.5],[.05,.9,.05],'a1'),
      rec('A',[.05,.8,.15],[.05,.75,.2],[.05,.85,.1],'a2'),
      rec('B',[.05,.1,.85],[.05,.85,.1],[.05,.9,.05],'b'),
    ]
    if flip:
        rows[1]=rec('A',[.1,.44,.46],[.1,.42,.48],[.03,.94,.03],'a1b')
    return rows

def config(**kw):
    d={'schema':'phm.expert-dag-search.v1','data_role':'development','retrospective_replay':False,
       'classes':CLASSES,'experts':EXPERTS,'structures':['single_expert','confidence_fallback','max_confidence'],
       'confidence_thresholds':[.5,.6,.7],
       'constraints':{'min_macro_f1':.9,'min_balanced_accuracy':.9,'min_all_class_recall':.5}}
    d.update(kw); return d

def test_runtime_never_needs_truth_and_switches_expert():
    c=make_candidate(CLASSES,EXPERTS,{'kind':'confidence_fallback','primary':'reference','fallback':'generated','threshold':.6})
    s=ExpertSignals({'reference':np.array([[.1,.45,.45],[.9,.05,.05]]),'fixed':np.array([[.1,.5,.4],[.8,.1,.1]]),'generated':np.array([[.05,.9,.05],[.85,.1,.05]])})
    out=execute(c,s)
    assert out['class_index'].tolist()==[1,0]
    assert out['expert_used']==['generated','reference']
    assert c['truth_inputs_used_at_runtime'] is False

def test_multi_environment_search_can_choose_small_fallback_dag():
    r=search_multi_environment({'e1':env_rows(False),'e2':env_rows(True)},config())
    assert r['status']=='COMPLETED' and r['qualification_eligible']
    assert r['test_or_holdout_used_for_selection'] is False
    assert r['chosen_candidate']['grammar']['kind']=='confidence_fallback'
    assert r['chosen_candidate']['grammar']['primary']=='reference'
    assert r['chosen_candidate']['grammar']['fallback']=='generated'
    assert r['chosen_worst_all_class_recall']==1.0

def test_occam_prefers_single_expert_when_tied():
    rows=[]
    for i,t in enumerate(CLASSES):
        v=[.05,.05,.05];v[i]=.9
        rows.append(rec(t,v,v,v,str(i)))
    r=search_multi_environment({'e1':rows,'e2':rows},config(constraints={'min_macro_f1':.9,'min_balanced_accuracy':.9,'min_all_class_recall':.9}))
    assert r['chosen_candidate']['grammar']=={'kind':'single_expert','expert':'fixed'} or r['chosen_candidate']['grammar']['kind']=='single_expert'

def test_fail_closed_if_no_candidate_meets_constraints():
    bad=[rec('N',[.9,.05,.05],[.9,.05,.05],[.9,.05,.05]),rec('A',[.9,.05,.05],[.9,.05,.05],[.9,.05,.05]),rec('B',[.9,.05,.05],[.9,.05,.05],[.9,.05,.05])]
    r=search_multi_environment({'e1':bad,'e2':bad},config(constraints={'min_macro_f1':.99,'min_balanced_accuracy':.99,'min_all_class_recall':.99}))
    assert r['status']=='FAIL_CLOSED_NO_ELIGIBLE_CANDIDATE'
    assert r['chosen_candidate'] is None

def test_search_rejects_non_development_role():
    with pytest.raises(ValueError,match='FRESH_DEVELOPMENT'):
        search_multi_environment({'e1':env_rows(),'e2':env_rows()},config(data_role='test'))

def test_candidate_rejects_runtime_truth_flag_change():
    c=make_candidate(CLASSES,EXPERTS,{'kind':'single_expert','expert':'reference'})
    c['truth_inputs_used_at_runtime']=True
    with pytest.raises(ValueError,match='RUNTIME_TRUTH'):
        execute(c,ExpertSignals({e:np.ones((1,3)) for e in EXPERTS}))
