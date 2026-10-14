from phm_design.multi_environment_selection import _confidence_aware_balanced_choice


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


def test_balanced_selector_uses_recall_aligned_secondary_before_macro_f1_noise():
    # Same primary worst recall and overlapping confidence intervals. Candidate A
    # has a slightly higher Macro-F1 point estimate, but B has better balanced
    # recall in the weakest environment and should win the secondary objective.
    a={
      'development':metric([1,.60,.80],.800,n=20),
      'selection_a':metric([1,.50,.70],.751,n=20),
      'selection_b':metric([1,.70,.70],.810,n=20),
    }
    b={
      'development':metric([1,.65,.80],.790,n=20),
      'selection_a':metric([1,.50,.85],.750,n=20),
      'selection_b':metric([1,.75,.75],.800,n=20),
    }
    chosen,contenders,ranks,intervals=_confidence_aware_balanced_choice(
        {'a':a,'b':b},complexity={'a':0,'b':3},tie_preference=['a','b'])
    assert contenders=={'a','b'}
    assert intervals['a']['lower']==intervals['b']['lower']
    assert ranks['b'][0] < ranks['a'][0]  # better worst balanced accuracy
    assert chosen=='b'


def test_balanced_selector_keeps_complexity_as_late_tie_breaker():
    simple={
      'development':metric([1,.6,.8],.75,n=20),
      'selection_a':metric([1,.6,.8],.75,n=20),
      'selection_b':metric([1,.6,.8],.75,n=20),
    }
    complex_={k:dict(v) for k,v in simple.items()}
    chosen,contenders,ranks,_=_confidence_aware_balanced_choice(
        {'simple':simple,'complex':complex_},complexity={'simple':0,'complex':4},tie_preference=['simple','complex'])
    assert contenders=={'simple','complex'}
    assert ranks['simple'][:-3]==ranks['complex'][:-3]
    assert chosen=='simple'


def test_balanced_selector_preserves_clear_primary_recall_separation():
    low={
      'development':metric([1,.4,.4],.95,n=100),
      'selection_a':metric([1,.4,.4],.95,n=100),
      'selection_b':metric([1,.4,.4],.95,n=100),
    }
    high={
      'development':metric([1,.9,.9],.60,n=100),
      'selection_a':metric([1,.9,.9],.60,n=100),
      'selection_b':metric([1,.9,.9],.60,n=100),
    }
    chosen,contenders,_,intervals=_confidence_aware_balanced_choice(
        {'low':low,'high':high},complexity={'low':0,'high':4},tie_preference=['low','high'])
    assert intervals['low']['upper'] < intervals['high']['lower']
    assert contenders=={'high'}
    assert chosen=='high'
