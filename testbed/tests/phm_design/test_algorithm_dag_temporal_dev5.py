import copy
import numpy as np
import pytest
from phm_design.algorithm_dag_temporal import (
  make_temporal_candidate, execute_temporal_candidate)

@pytest.mark.parametrize('method,param,drift',[('rolling_mean',5,0.),('ewma',.2,0.),('two_sided_cusum',1,.1)])
def test_streaming_dag_no_future_and_unknown(method,param,drift):
    dag=make_temporal_candidate(method=method,parameter=param,threshold=.4,drift=drift)
    signal=np.r_[np.zeros(25),np.ones(15)*2]
    mask=np.ones(len(signal),bool)
    base=execute_temporal_candidate(dag,signal,mask)
    altered=signal.copy();altered[33:]+=8
    b=execute_temporal_candidate(dag,altered,mask)
    assert np.array_equal(base['scores'][:33],b['scores'][:33])
    assert not base['alert'][:25].any()
    assert base['alert'][-1]
    mask[31]=False
    c=execute_temporal_candidate(dag,signal,mask)
    assert c['status'][31]=='UNKNOWN_UNOBSERVABLE'
    assert c['alert'][31]==False


def test_fingerprint_cannot_mutate_model_policy_after_freeze():
    dag=make_temporal_candidate(method='two_sided_cusum',parameter=1.,drift=.01,threshold=.5)
    changed=copy.deepcopy(dag);changed['nodes']['event']['params']['threshold']=.001
    with pytest.raises(ValueError,match='FINGERPRINT_DRIFT'):
        execute_temporal_candidate(changed,[0.,1.,2.],[True]*3)


def test_invalid_and_nonfinite_rejection():
    with pytest.raises(ValueError):make_temporal_candidate(method='ewma',parameter=0,threshold=1)
    with pytest.raises(ValueError):make_temporal_candidate(method='ewma',parameter=.2,threshold=0)
    d=make_temporal_candidate(method='ewma',parameter=.2,threshold=.3)
    with pytest.raises(ValueError):execute_temporal_candidate(d,[0.,float('nan')],[True,True])


def test_unknown_is_not_healthy():
    d=make_temporal_candidate(method='ewma',parameter=.2,threshold=.3)
    out=execute_temporal_candidate(d,[0.,0.,0.],[False]*3)
    assert all(x=='UNKNOWN_UNOBSERVABLE' for x in out['status'])
