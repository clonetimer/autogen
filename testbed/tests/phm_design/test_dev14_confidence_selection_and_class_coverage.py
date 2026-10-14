import numpy as np

from phm_design.multi_environment_selection import (
    _confidence_aware_choice, balanced_environment_rank,
)
from phm_design.risk_calibration import choose_threshold


def metric(recalls, macro_f1, n=20):
    classes=[f'C{i}' for i in range(len(recalls))]
    cm=[]
    for i,r in enumerate(recalls):
        k=round(r*n)
        row=[0]*len(classes);row[i]=k;row[(i+1)%len(classes)]=n-k
        cm.append(row)
    correct=sum(cm[i][i] for i in range(len(classes)))
    cases=n*len(classes)
    return {
        'macro_f1':float(macro_f1),
        'recall_by_class':dict(zip(classes,map(float,recalls))),
        'confusion_matrix':cm,
        'cases':cases,
        'accepted_cases':cases,
        'accepted_errors':cases-correct,
        'accepted_correct':correct,
    }


def test_confidence_aware_maximin_does_not_let_one_of_twenty_dominate():
    old={
      'development':metric([1,.625,.6875,.75,.6875],.750,n=16),
      'selection_a':metric([1,.70,.80,.55,.75],.758),
      'selection_b':metric([1,.65,.65,.65,.75],.740),
      'selection_c':metric([1,.85,.65,.80,.80],.819),
    }
    grid={
      'development':metric([1,.625,.625,.75,.625],.722,n=16),
      'selection_a':metric([1,.70,.70,.65,.80],.767),
      'selection_b':metric([1,.75,.60,.70,.80],.769),
      'selection_c':metric([1,.95,.70,.65,.80],.818),
    }
    # Legacy point-estimate maximin prefers grid: .60 > .55 worst recall.
    assert balanced_environment_rank(grid) < balanced_environment_rank(old)
    chosen,contenders,_,intervals=_confidence_aware_choice(
        {'old':old,'grid':grid},complexity={'old':0,'grid':0},tie_preference=['old','grid'])
    assert contenders=={'old','grid'}
    assert intervals['old']['upper'] >= intervals['grid']['lower']
    assert chosen=='old'  # secondary worst-environment Macro-F1 is stronger.


def test_confidence_aware_maximin_keeps_large_recall_advantage_decisive():
    low={'development':metric([1,.4,.4],.90,n=100),'selection_a':metric([1,.4,.4],.90,n=100),'selection_b':metric([1,.4,.4],.90,n=100)}
    high={'development':metric([1,.9,.9],.70,n=100),'selection_a':metric([1,.9,.9],.70,n=100),'selection_b':metric([1,.9,.9],.70,n=100)}
    chosen,contenders,_,intervals=_confidence_aware_choice(
        {'low':low,'high':high},complexity={'low':0,'high':0},tie_preference=['low','high'])
    assert intervals['low']['upper'] < intervals['high']['lower']
    assert contenders=={'high'}
    assert chosen=='high'


def test_risk_v2_rejects_threshold_with_zero_class_coverage():
    classes=['C0','C1','C2','C3','C4']
    y=[];scores=[];samples=[]
    for ci in range(5):
        for j in range(10):
            y.append(ci);samples.append({'group_id':f'g{ci}_{j}'})
            # All predictions are correct. At .9, class C4 is deliberately low confidence.
            conf=.95 if ci<4 else .80
            rest=(1-conf)/4
            row=[rest]*5;row[ci]=conf;scores.append(row)
    policy={'schema':'phm.selective-risk-policy.v2','thresholds':[.6,.9],
            'alpha':.2,'max_group_error_upper_bound':.8,'min_case_coverage':.25,
            'min_class_case_coverage':.1}
    result=choose_threshold(np.asarray(y),np.asarray(scores),samples,classes,policy)
    by_t={c['threshold']:c for c in result['candidates']}
    assert by_t[.9]['case_coverage']==.8
    assert by_t[.9]['minimum_class_case_coverage']==0.0
    assert by_t[.9]['eligible'] is False
    assert result['status']=='QUALIFIED_THRESHOLD'
    assert result['threshold']==.6


def test_conservative_confidence_choice_prefers_simpler_artifact_when_primary_interval_ties():
    from phm_design.multi_environment_selection import _confidence_aware_conservative_choice
    fixed={
      'development':metric([1,.7,.7],.720,n=20),
      'selection_a':metric([1,.5,.8],.7166,n=20),
      'selection_b':metric([1,.7,.7],.780,n=20),
    }
    raw={
      'development':metric([1,.7,.7],.730,n=20),
      'selection_a':metric([1,.5,.8],.7171,n=20),
      'selection_b':metric([1,.7,.7],.790,n=20),
    }
    chosen,contenders,ranks,intervals=_confidence_aware_conservative_choice(
        {'fixed':fixed,'raw':raw},complexity={'fixed':1,'raw':3},tie_preference=['fixed','raw'])
    assert contenders=={'fixed','raw'}
    assert intervals['fixed']['lower']==intervals['raw']['lower']
    assert ranks['fixed'][0] < ranks['raw'][0]
    assert chosen=='fixed'


def test_conservative_confidence_choice_allows_clear_primary_advantage_to_override_complexity():
    from phm_design.multi_environment_selection import _confidence_aware_conservative_choice
    simple={
      'development':metric([1,.4,.4],.90,n=100),
      'selection_a':metric([1,.4,.4],.90,n=100),
      'selection_b':metric([1,.4,.4],.90,n=100),
    }
    complex_={
      'development':metric([1,.9,.9],.70,n=100),
      'selection_a':metric([1,.9,.9],.70,n=100),
      'selection_b':metric([1,.9,.9],.70,n=100),
    }
    chosen,contenders,_,intervals=_confidence_aware_conservative_choice(
        {'simple':simple,'complex':complex_},complexity={'simple':0,'complex':4},tie_preference=['simple','complex'])
    assert intervals['simple']['upper'] < intervals['complex']['lower']
    assert contenders=={'complex'}
    assert chosen=='complex'
