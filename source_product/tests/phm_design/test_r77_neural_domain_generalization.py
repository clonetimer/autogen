"""R77 synthetic-only tests of *trained* neural domain-generalization candidates.

No R71 revealed data, Basilisk final worlds, or qualification data is read.
"""
import joblib
import numpy as np
import pytest

from phm_design.task_candidate_registry import TaskCandidateRegistry
from phm_design.neural_domain_generalization import DGSequenceEstimator, DGTrainingPolicy
from phm_design.task_graph import _fit_head, design_task_graph, TaskGraphPolicy
from phm_design.task_architectures import TaskDesignRejected
from phm_design.leakage_control import WorldLineage
from phm_design.platform_features import FIELDS
from test_task_architectures_08 import fixture_train


def _synthetic_source_groups():
    rng = np.random.default_rng(7719)
    x, y, env = [], [], []
    t = np.linspace(0, 1, 24)
    for domain in ('lab-a', 'lab-b', 'lab-c'):
        for label in (0, 1):
            for replicate in range(6):
                # Sensor 0 has a stable weak physical signature; sensor 1's
                # large nominal offset depends only on source environment.
                signal = 0.18 * label * np.sin(4 * np.pi * t)
                first = signal + rng.normal(0, 0.03, len(t))
                second = (0 if domain == 'lab-a' else 0.4 if domain == 'lab-b' else -0.3) + rng.normal(0, 0.03, len(t))
                x.append(np.stack((first, second), axis=-1))
                y.append(label)
                env.append(domain)
    return np.asarray(x, dtype=np.float32), np.asarray(y), np.asarray(env)


def test_neural_architecture_candidates_exist_as_trainable_models():
    registry = TaskCandidateRegistry()
    for name in ('torch_tcn', 'dg_tcn', 'dg_gru', 'dg_transformer', 'dg_cnn1d'):
        description = registry.describe('sequence', name)
        assert description.input_kind == 'sequence'
        if name.startswith('dg_'):
            assert description.backend == 'torch_cpu_dg'
            assert isinstance(registry.build('sequence', name, 17), DGSequenceEstimator)


def test_group_dro_actually_trains_neural_network_and_saves_cpu_weights(tmp_path):
    x, y, groups = _synthetic_source_groups()
    model = DGSequenceEstimator('dg_tcn', 83, policy=DGTrainingPolicy(epochs=6))
    assert model.fit(x, y, domain_groups=groups) is model
    scores = model.predict_proba(x[:5])
    assert scores.shape == (5, 2)
    assert np.isfinite(scores).all()
    assert np.allclose(scores.sum(1), 1)
    assert model.training_['objective'] == 'group_dro'
    assert model.training_['normalization'] == 'training_partition_only'
    assert model.training_['domain_source'] == 'TRAIN_LINEAGE_ONLY'
    assert model.training_['cuda_used'] is False
    assert set(model.domain_weights_) == {'lab-a', 'lab-b', 'lab-c'}
    assert sum(model.domain_weights_.values()) == pytest.approx(1., abs=1e-6)
    path = tmp_path / 'neural-dg.joblib'
    joblib.dump(model, path)
    restored = joblib.load(path)
    assert np.array_equal(restored.predict_proba(x[:5]), scores)


def test_dg_training_fails_closed_without_group_labels_or_domain_diversity():
    x, y, groups = _synthetic_source_groups()
    model = DGSequenceEstimator('dg_gru', 83, policy=DGTrainingPolicy(epochs=1))
    with pytest.raises(ValueError,match='DG_REQUIRES_TRAINING_ENVIRONMENT_METADATA'):
        model.fit(x, y)
    with pytest.raises(ValueError,match='DG_INSUFFICIENT_OR_EXCESSIVE_TRAINING_ENVIRONMENTS'):
        model.fit(x, y, domain_groups=['same'] * len(x))
    with pytest.raises(ValueError,match='DG_TRAINING_ENVIRONMENT_LENGTH_MISMATCH'):
        model.fit(x, y, domain_groups=groups[:-1])
    with pytest.raises(ValueError,match='INVALID_DG_TRAINING_RATE'):
        DGTrainingPolicy(group_step_size=20)


def test_dg_weights_stable_on_identical_cpu_seed():
    x, y, groups = _synthetic_source_groups()
    policy = DGTrainingPolicy(epochs=3)
    a = DGSequenceEstimator('dg_gru', 222, policy=policy).fit(x, y, domain_groups=groups)
    b = DGSequenceEstimator('dg_gru', 222, policy=policy).fit(x, y, domain_groups=groups)
    assert np.array_equal(a.predict_proba(x[:6]), b.predict_proba(x[:6]))


def test_graph_dg_requires_environment_exclusive_provenance():
    rows, y, world, role, healthy = fixture_train()
    options = {h: [('sequence', 'dg_tcn')] if h == 'detection' else [('temporal', 'linear')]
               for h in ('detection','localization','mechanism')}
    with pytest.raises(ValueError,match='DG_REQUIRES_VERIFIED_ENVIRONMENT_EXCLUSIVE_SEARCH'):
        design_task_graph(domain='Thermal_localization', observations=rows,
                          fields=FIELDS['Thermal_localization'], labels=y,
                          world_ids=world, roles=role, healthy_prefix_confirmed=healthy,
                          candidate_specs=options, acceptance_policy=TaskGraphPolicy(0,1,0,0,0,0,0,1,8))


def test_graph_can_fit_neural_detection_with_train_environment_only():
    rows, y, world, role, healthy = fixture_train()
    # This direct training-head test isolates training contract from any
    # threshold-dependent decision to deliver a final graph.
    rng = np.random.default_rng(8)
    matrix = {'sequence': rng.normal(size=(len(y), 24, 3)).astype(np.float32)}
    targets = {'detection': np.asarray(y) != 0}
    environments = np.array([f'env{int(w[1:]) // 2}' for w in world])
    model = _fit_head('detection', ('sequence','dg_tcn'), matrix, targets,
                      np.asarray(y), np.arange(len(y)), 7, environments)
    assert model.training_['train_environments'] == 4
    assert model.training_['independent_qualification'] is False
    assert model.predict_proba(matrix['sequence'][:3]).shape == (3, 2)
    with pytest.raises(ValueError,match='DG_REQUIRES_VERIFIED_TRAINING_ENVIRONMENTS'):
        _fit_head('detection', ('sequence','dg_tcn'), matrix, targets,
                  np.asarray(y), np.arange(len(y)), 7)


def test_nested_search_compares_erm_and_dg_neural_candidates_on_train_worlds():
    rows, y, world, role, healthy = fixture_train()
    env = [f'env{int(w[1:]) // 2}' for w in world]
    lineage = [WorldLineage(w, w, 'TRAIN', {'synthetic_base':w}, (), e)
               for w, e in zip(world, env)]
    options = {
        'detection': [('sequence','torch_tcn'), ('sequence','dg_tcn')],
        'localization': [('temporal','linear')],
        'mechanism': [('temporal','linear')],
    }
    kwargs = dict(domain='Thermal_localization', observations=rows,
                  fields=FIELDS['Thermal_localization'], labels=y, world_ids=world,
                  roles=role, healthy_prefix_confirmed=healthy,
                  candidate_specs=options, training_lineage=lineage,
                  environment_families=env,
                  selection_mode='worst_environment',
                  outer_split_mode='leave_environment_family_out',
                  acceptance_policy=TaskGraphPolicy(0,1,0,0,0,0,0,1,8))
    try:
        graph = design_task_graph(**kwargs)
        audit = graph.audit
    except TaskDesignRejected as exc:
        audit = exc.audit
    assert audit['stage'] == 'DEVELOPMENT_ONLY'
    assert not audit['sealed_or_calibration_worlds_used'] if 'sealed_or_calibration_worlds_used' in audit else True
    trials = [trial for outer in audit['outer_fold_history']
              for trial in outer['head_selections']['detection']['trials']]
    assert trials
    dg_trials = [t for t in trials if t['learner'] == 'dg_tcn']
    erm_trials = [t for t in trials if t['learner'] == 'torch_tcn']
    assert dg_trials and erm_trials
    assert all(t['inner_validation_mode']=='leave_one_environment_family_out' for t in trials)
    for candidate in dg_trials:
        for inner in candidate.get('inner_fold_history', []):
            assert not set(inner['train_environment_families']) & set(inner['heldout_environment_families'])
