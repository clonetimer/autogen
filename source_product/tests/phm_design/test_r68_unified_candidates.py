"""R68 CPU architecture bridge; synthetic trajectories only, never blind worlds."""
import joblib
import numpy as np
import pytest

from phm_design.task_candidate_registry import (TaskCandidateRegistry, TorchTaskEstimator,
                                               deployment_sequence)
from phm_design.task_architectures import (design, TaskAcceptancePolicy, TaskDesignRejected)
from phm_design.platform_features import FIELDS
from test_task_architectures_08 import fixture_train


def test_registry_allowlist_and_feature_compatibility():
    r = TaskCandidateRegistry()
    assert r.describe('temporal','linear').backend == 'sklearn'
    assert r.describe('physical','extra').backend == 'sklearn'
    assert r.describe('sequence','torch_gru').backend == 'torch_cpu'
    for pair in [('physical','torch_gru'),('sequence','linear'),('sequence','arbitrary')]:
        with pytest.raises(ValueError):r.describe(*pair)


def test_deployment_sequence_excludes_time_and_never_looks_at_labels():
    t = np.array([1.,4.,6.,10.])
    obs = np.column_stack([t, np.array([2.,3.,6.,8.]), np.array([4.,6.,8.,10.])])
    x = deployment_sequence(obs, 8)
    assert x.shape == (8,2)
    assert x[0].tolist()==[2.,4.]
    assert x[-1].tolist()==[8.,10.]
    with pytest.raises(ValueError,match='INVALID_SEQUENCE_TIME_AXIS'):
        deployment_sequence(obs[[0,2,1,3]],8)


def test_torch_estimators_fit_predict_and_serialize(tmp_path):
    rng=np.random.default_rng(66)
    x=rng.normal(size=(12,24,3)).astype('float32')
    y=np.array([0,1]*6)
    for kind in ('torch_cnn1d','torch_gru','torch_transformer'):
        model=TorchTaskEstimator(kind,67,epochs=2).fit(x,y)
        p=model.predict_proba(x[:3])
        assert p.shape==(3,2) and np.all(np.isfinite(p))
        assert np.allclose(p.sum(axis=1),1.,atol=1e-6)
        assert model.training_['cuda_used'] is False
        path=tmp_path/f'{kind}.joblib';joblib.dump(model,path)
        assert np.allclose(joblib.load(path).predict_proba(x[:3]),p)
        with pytest.raises(ValueError,match='TASK_SEQUENCE_CHANNEL_MISMATCH'):
            model.predict_proba(x[:3,:,:2])


def test_normalizer_fitted_only_to_train_partition():
    rng=np.random.default_rng(10)
    x=rng.normal(size=(8,12,3)).astype('float32')
    m=TorchTaskEstimator('torch_cnn1d',33,epochs=1).fit(x,np.array([0,1]*4))
    old_mean=m.mean_.copy()
    # Predicting a held-out shifted sequence must not refit preprocessing.
    m.predict_proba(np.full((2,12,3),300.,dtype='float32'))
    assert np.array_equal(old_mean,m.mean_)


def test_mixed_backend_group_oof_stays_development_only(tmp_path):
    rows,labels,world,roles,health=fixture_train()
    result=design(domain='Thermal_localization',observations=rows,
        fields=FIELDS['Thermal_localization'],labels=labels,world_ids=world,roles=roles,
        healthy_prefix_confirmed=health,
        candidate_specs=[('temporal','linear','direct'),('sequence','torch_cnn1d','direct')],
        acceptance_policy=TaskAcceptancePolicy(0,1,0,0,8))
    records=result.audit['candidate_results']
    assert [r['backend'] for r in records]==['sklearn','torch_cpu']
    assert all(r['status']=='COMPLETE' for r in records)
    assert result.audit['group_fold_intersection']==0
    assert result.audit['candidate_registry']=='r68-v1'
    assert result.audit['not_deployment_qualified'] is True
    pred=result.diagnose(rows[0],fields=FIELDS['Thermal_localization'],healthy_prefix_confirmed=True)
    assert pred['qualified'] is False
    assert pred['training_role_fields_used_at_inference'] is False
    output=tmp_path/'artifact.joblib';joblib.dump(result,output)
    assert joblib.load(output).diagnose(rows[0],fields=FIELDS['Thermal_localization'],
                                      healthy_prefix_confirmed=True)==pred


def test_fail_closed_backend_mismatch_and_sealed_role():
    rows,y,g,r,h=fixture_train()
    with pytest.raises(ValueError,match='TASK_FEATURE_BACKEND_MISMATCH'):
        design(domain='Thermal_localization',observations=rows,
          fields=FIELDS['Thermal_localization'],labels=y,world_ids=g,roles=r,
          healthy_prefix_confirmed=h,candidate_specs=[('physical','torch_gru','direct')])
    r[0]='sealed_blind'
    with pytest.raises(ValueError,match='PROTECTED_OR_EVALUATION_ROLE'):
        design(domain='Thermal_localization',observations=rows,
          fields=FIELDS['Thermal_localization'],labels=y,world_ids=g,roles=r,
          healthy_prefix_confirmed=h,candidate_specs=[('sequence','torch_cnn1d','direct')])
