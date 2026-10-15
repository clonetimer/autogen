"""R78 synthetic TRAIN-only tests of neural risk extrapolation.

Never reads R71 or any sealed evaluation worlds. A real PyTorch optimizer,
backpropagation, source-risk-variance and deployment proba are exercised.
"""
import joblib
import numpy as np
import pytest

from phm_design.task_candidate_registry import TaskCandidateRegistry
from phm_design.neural_domain_generalization import (
    RiskExtrapolationEstimator, RiskExtrapolationPolicy,
    _risk_extrapolation_loss,
)
from phm_design.task_graph import _fit_head, design_task_graph, TaskGraphPolicy
from phm_design.platform_features import FIELDS
from phm_design.leakage_control import WorldLineage
from test_task_architectures_08 import fixture_train


def synthetic_training():
    rng = np.random.default_rng(7811)
    t = np.linspace(0, 1, 24)
    x, y, env = [], [], []
    for e, offset in [('env-a', .12), ('env-b', -.3), ('env-c', .5)]:
        for label in (0, 1):
            for _ in range(5):
                stable = label * .15 * np.sin(4 * np.pi * t) + rng.normal(0, .04, len(t))
                nuisance = offset + rng.normal(0, .07, len(t))
                x.append(np.stack((stable, nuisance), axis=-1))
                y.append(label)
                env.append(e)
    return np.asarray(x, dtype=np.float32), np.asarray(y), np.asarray(env)


def test_vrex_is_differentiable_and_penalizes_training_domain_disagreement():
    import torch
    losses = torch.tensor([1., 3.], requires_grad=True)
    objective = _risk_extrapolation_loss(losses, 2.0)
    assert objective.item() == pytest.approx(4.0)  # mean=2, variance=1
    objective.backward()
    assert losses.grad.tolist() == pytest.approx([-1.5, 2.5])
    with pytest.raises(ValueError, match='REX_REQUIRES_FINITE'):
        _risk_extrapolation_loss(torch.tensor([float('nan'), 1.]), 1.)


def test_rex_policy_rejects_invalid_weights_and_warmup():
    with pytest.raises(ValueError, match='INVALID_REX_RISK_VARIANCE_WEIGHT'):
        RiskExtrapolationPolicy(risk_variance_weight=-1)
    with pytest.raises(ValueError, match='INVALID_REX_WARMUP_EPOCHS'):
        RiskExtrapolationPolicy(epochs=2, risk_variance_warmup_epochs=2)
    with pytest.raises(ValueError, match='INVALID_REX_TRAINING_POLICY'):
        RiskExtrapolationEstimator('rex_tcn', 1, policy=object())


def test_rex_candidates_share_registry_and_trainable_encoder_contract():
    registry = TaskCandidateRegistry()
    for name in ('rex_cnn1d', 'rex_gru', 'rex_transformer', 'rex_tcn'):
        descriptor = registry.describe('sequence', name)
        assert descriptor.backend == 'torch_cpu_rex'
        model = registry.build('sequence', name, 18)
        assert isinstance(model, RiskExtrapolationEstimator)
        assert model.dg_name == name


def test_rex_tcn_trains_neural_parameters_and_roundtrips(tmp_path):
    x, y, env = synthetic_training()
    policy = RiskExtrapolationPolicy(epochs=3, steps_per_epoch=2,
                                    risk_variance_warmup_epochs=1)
    model = RiskExtrapolationEstimator('rex_tcn', 83, policy=policy)
    fitted = model.fit(x, y, domain_groups=env)
    assert fitted is model
    assert model.training_['objective'] == 'risk_variance'
    assert model.training_['optimizer_steps'] == 6
    assert model.training_['normalization'] == 'training_partition_only'
    assert model.training_['cuda_used'] is False
    assert model.training_['independent_qualification'] is False
    assert set(model.domain_weights_) == set(env)
    assert sum(model.domain_weights_.values()) == pytest.approx(1., abs=1e-6)
    p = model.predict_proba(x[:4])
    assert p.shape == (4, 2)
    assert np.isfinite(p).all()
    assert np.allclose(p.sum(axis=1), 1.)
    fn = tmp_path / 'rex-model.joblib'
    joblib.dump(model, fn)
    assert np.array_equal(joblib.load(fn).predict_proba(x[:4]), p)


def test_rex_training_deterministic_when_retrained_on_identical_source_worlds():
    x, y, env = synthetic_training()
    policy = RiskExtrapolationPolicy(epochs=2, steps_per_epoch=2, risk_variance_warmup_epochs=0)
    a = RiskExtrapolationEstimator('rex_gru', 12, policy=policy).fit(x,y,domain_groups=env)
    b = RiskExtrapolationEstimator('rex_gru', 12, policy=policy).fit(x,y,domain_groups=env)
    assert np.array_equal(a.predict_proba(x[:4]), b.predict_proba(x[:4]))


def test_rex_refuses_missing_or_leaked_training_environments():
    x, y, env = synthetic_training()
    model = RiskExtrapolationEstimator('rex_gru', 3,
        policy=RiskExtrapolationPolicy(epochs=1, steps_per_epoch=1,
                                      risk_variance_warmup_epochs=0))
    with pytest.raises(ValueError, match='DG_REQUIRES_TRAINING_ENVIRONMENT_METADATA'):
        model.fit(x,y)
    with pytest.raises(ValueError, match='DG_INSUFFICIENT_OR_EXCESSIVE_TRAINING_ENVIRONMENTS'):
        model.fit(x,y,domain_groups=['one']*len(y))


def test_rex_integrates_into_heldout_environment_head_search():
    rows, y, worlds, roles, healthy = fixture_train()
    env = np.asarray([f'env{int(w[1:]) // 2}' for w in worlds])
    lineage = [WorldLineage(w, w, 'TRAIN', {'synthetic_base': w}, (), e)
               for w,e in zip(worlds,env)]
    options = {h: [('sequence','rex_tcn')] if h == 'detection'
               else [('temporal','linear')]
               for h in ('detection','localization','mechanism')}
    with pytest.raises(ValueError, match='DG_REQUIRES_VERIFIED_ENVIRONMENT_EXCLUSIVE_SEARCH'):
        design_task_graph(domain='Thermal_localization',observations=rows,
                          fields=FIELDS['Thermal_localization'],labels=y,world_ids=worlds,
                          roles=roles,healthy_prefix_confirmed=healthy,candidate_specs=options)
    rng = np.random.default_rng(12)
    train_seq = {'sequence':rng.normal(size=(len(y),24,3)).astype(np.float32)}
    model = _fit_head('detection',('sequence','rex_tcn'),train_seq,
                      {'detection':np.asarray(y)!=0},np.asarray(y),
                      np.arange(len(y)),7,env)
    assert model.training_['objective']=='risk_variance'
    assert model.training_['domain_source']=='TRAIN_LINEAGE_ONLY'
    assert model.predict_proba(train_seq['sequence'][:2]).shape==(2,2)


def test_auto_neural_profile_compares_training_objectives_with_heterogeneous_heads():
    from phm_design.task_graph import _auto_neural_candidate_specs
    profile = _auto_neural_candidate_specs('compact_domain_generalization')
    assert set(profile) == {'detection','localization','mechanism'}
    assert all(len(x) == 3 for x in profile.values())
    registry = TaskCandidateRegistry()
    for head, options in profile.items():
        assert {name.split('_', 1)[0] for _, name in options} == {'torch','dg','rex'}
        assert all(registry.describe(*option).input_kind == 'sequence' for option in options)
    assert profile['detection'][0][1] == 'torch_tcn'
    assert profile['localization'][0][1] == 'torch_cnn1d'
    assert profile['mechanism'][0][1] == 'torch_gru'
    with pytest.raises(ValueError, match='UNREGISTERED_NEURAL_DESIGN_PROFILE'):
        _auto_neural_candidate_specs('unbounded')


def test_auto_neural_profile_fails_without_verified_ood_protocol():
    rows, y, worlds, roles, healthy = fixture_train()
    kwargs = dict(domain='Thermal_localization',
        observations=rows, fields=FIELDS['Thermal_localization'], labels=y,
        world_ids=worlds, roles=roles, healthy_prefix_confirmed=healthy,
        neural_design_profile='compact_domain_generalization',
        acceptance_policy=TaskGraphPolicy(0,1,0,0,0,0,0,1,8))
    with pytest.raises(ValueError, match='NEURAL_AUTO_PROFILE_REQUIRES_VERIFIED_OOD_PROTOCOL'):
        design_task_graph(**kwargs)
    with pytest.raises(ValueError, match='NEURAL_AUTO_PROFILE_AND_MANUAL_CANDIDATES_CONFLICT'):
        design_task_graph(**kwargs, candidate_specs={
          h:[('temporal','linear')] for h in ('detection','localization','mechanism')})
