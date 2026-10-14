"""R69 nested OOF task-DAG integration: synthetic design_train data only."""
import joblib
import numpy as np
import pytest

from phm_design.platform_features import FIELDS
from phm_design.task_architectures import TaskDesignRejected
from phm_design.task_graph import (
    TaskGraphArtifact, TaskGraphGate, TaskGraphPolicy,
    _gate, _scores, design_task_graph,
)
from test_task_architectures_08 import fixture_train

DOMAIN = 'Thermal_localization'
FIELDS_ = FIELDS[DOMAIN]
HETEROGENEOUS = {
    'detection': [('temporal', 'linear'), ('physical', 'extra')],
    'localization': [('physical', 'extra')],
    'mechanism': [('temporal', 'linear')],
}


def loose_policy():
    return TaskGraphPolicy(0, 1, 0, 0, 0, 0, 0, 1, 8)


def create_graph(**kwargs):
    rows, y, worlds, roles, healthy = fixture_train()
    params = dict(domain=DOMAIN, observations=rows, fields=FIELDS_, labels=y,
                  world_ids=worlds, roles=roles, healthy_prefix_confirmed=healthy,
                  candidate_specs=HETEROGENEOUS, acceptance_policy=loose_policy())
    params.update(kwargs)
    return design_task_graph(**params)


@pytest.fixture(scope='module')
def artifact():
    return create_graph()


def test_gates_are_validated_before_training():
    with pytest.raises(ValueError, match='INVALID_TASK_GRAPH_GATE'):
        TaskGraphGate(detector_min_confidence=.4)
    with pytest.raises(ValueError, match='INVALID_GRAPH_POLICY'):
        TaskGraphPolicy(min_independent_worlds=3)
    with pytest.raises(ValueError, match='INVALID_GRAPH_POLICY'):
        TaskGraphPolicy(detection_fault_recall_min=True)
    for bad in ({'detection': [('temporal', 'linear')]},
                {'detection': [('sequence', 'linear')], 'localization': [('temporal', 'linear')], 'mechanism': [('temporal', 'linear')]},
                {'detection': [('temporal', 'linear')], 'localization': [('temporal', 'linear')], 'mechanism': [('temporal', 'linear'), ('temporal', 'linear')]}):
        with pytest.raises(ValueError):
            create_graph(candidate_specs=bad)


def test_headwise_nested_group_oof_heterogeneous_model_choices(artifact):
    audit = artifact.audit
    assert audit['schema'] == 'phm.0.8.task-graph.v1'
    assert audit['task_graph']['nodes'] == ['observations', 'detection', 'localization', 'mechanism', 'decision_gate']
    assert audit['stage'] == 'DEVELOPMENT_ONLY'
    assert audit['group_fold_intersection'] == 0
    assert audit['not_deployment_qualified'] and not audit['sealed_or_calibration_worlds_used']
    assert len(audit['outer_fold_history']) == 4
    assert sum(fold['heldout_rows'] for fold in audit['outer_fold_history']) == 40
    assert all(set(fold['head_selections']) == {'detection', 'localization', 'mechanism'}
               for fold in audit['outer_fold_history'])
    assert audit['final_head_specs']['localization'] == ['physical', 'extra']
    assert audit['final_head_specs']['mechanism'] == ['temporal', 'linear']
    assert audit['model_delivered'] is True
    metrics = audit['nested_outer_oof_metrics']
    assert metrics['detection_fault_recall'] == pytest.approx(1)
    assert metrics['localization_conditional_accuracy'] == pytest.approx(1)
    assert metrics['mechanism_conditional_accuracy'] < 1
    assert metrics['end_to_end_accuracy'] < metrics['detection_fault_recall']
    assert metrics['localization_conditional_support'] == 32
    assert metrics['independent_worlds'] == 8
    assert sum(metrics['end_to_end_decisions'].values()) + metrics['unknown_count'] == 40


def test_gate_paths_and_conditional_denominators():
    from phm_design.task_architectures import TaskOntology
    onto = TaskOntology.from_domain(DOMAIN)
    d, c, m = np.array([.1, .9]), np.array([.9, .1]), np.array([.2, .8])
    assert _gate(onto, d, c, m, TaskGraphGate())[1] == 'FAULT'
    assert _gate(onto, np.array([.5, .5]), c, m, TaskGraphGate(.7))[1] == 'DETECTION_UNCERTAIN'
    assert _gate(onto, np.array([.99, .01]), c, m, TaskGraphGate())[1] == 'NOMINAL'
    assert _gate(onto, d, np.array([.6, .4]), m, TaskGraphGate(.5, .8))[1] == 'LOCALIZATION_UNCERTAIN'
    assert _gate(onto, d, c, np.array([.6, .4]), TaskGraphGate(.5, .5, .8))[1] == 'MECHANISM_UNCERTAIN'
    labels = np.array([0, 1, 2, 3, 4] * 2)
    groups = np.array(['w0'] * 5 + ['w1'] * 5)
    no_detection = np.tile([1., 0.], (10, 1))
    pc = pm = np.tile([.6, .4], (10, 1))
    metrics = _scores(labels, groups, onto, no_detection, pc, pm, TaskGraphGate())
    assert metrics['localization_conditional_accuracy'] is None
    assert metrics['mechanism_conditional_accuracy'] is None
    assert not loose_policy().evaluate(metrics)['eligible']


def test_strict_policy_rejects_without_fitting_deployment_model():
    with pytest.raises(TaskDesignRejected) as result:
        create_graph(acceptance_policy=TaskGraphPolicy())
    a = result.value.audit
    assert a['decision'] == 'REJECT_GRAPH_ACCEPTANCE'
    assert a['model_delivered'] is False
    assert a['acceptance']['failed_gates']
    assert any(x['metric'] == 'mechanism_conditional_accuracy' for x in a['acceptance']['failed_gates'])
    assert a['nested_outer_oof_metrics']['detection_fault_recall'] == 1


def test_inference_schema_prefix_and_joblib_roundtrip(artifact, tmp_path):
    rows, *_ = fixture_train()
    before = artifact.diagnose(rows[0], fields=FIELDS_, healthy_prefix_confirmed=True)
    assert before['prediction'] in (None, *artifact.ontology.labels)
    assert before['qualified'] is False
    assert before['training_role_fields_used_at_inference'] is False
    for head in ('detection', 'localization', 'mechanism'):
        assert head in before['head_specs']
    with pytest.raises(ValueError, match='EXACT_DEPLOYMENT_FIELDS_REQUIRED'):
        artifact.diagnose(rows[0], fields=FIELDS_ + ('world_id',), healthy_prefix_confirmed=True)
    with pytest.raises(ValueError, match='HEALTHY_PREFIX_NOT_CONFIRMED'):
        artifact.diagnose(rows[0], fields=FIELDS_, healthy_prefix_confirmed=False)
    file = tmp_path / 'task_graph.joblib'
    joblib.dump(artifact, file)
    assert joblib.load(file).diagnose(rows[0], fields=FIELDS_, healthy_prefix_confirmed=True) == before


def test_roles_worlds_training_schema_fail_closed():
    rows, y, worlds, roles, healthy = fixture_train()
    roles[0] = 'sealed_blind'
    with pytest.raises(ValueError, match='PROTECTED_OR_EVALUATION_ROLE'):
        create_graph(roles=roles)
    with pytest.raises(ValueError, match='EXACT_DEPLOYMENT_FIELDS_REQUIRED'):
        create_graph(fields=FIELDS_ + ('fault_truth',))
    with pytest.raises(ValueError, match='INSUFFICIENT_INDEPENDENT_WORLDS'):
        create_graph(world_ids=['duplicate'] * len(worlds))
    with pytest.raises(ValueError, match='HEALTHY_PREFIX_NOT_CONFIRMED'):
        create_graph(healthy_prefix_confirmed=[False] + healthy[1:])
    with pytest.raises(ValueError, match='FULL_ONTOLOGY_CLASS_SUPPORT_REQUIRED'):
        create_graph(labels=[0] * len(y))


def test_correct_per_head_truth_conditioning_does_not_use_fault_label_as_predictor(artifact):
    a = artifact.audit
    assert not a['predictor_uses_labels_world_ids_or_truth']
    assert a['observation_fields'] == list(FIELDS_)
    assert all('train_worlds' in f and 'heldout_worlds' in f for f in a['outer_fold_history'])
    assert all(f['train_worlds'] + f['heldout_worlds'] == 8 for f in a['outer_fold_history'])


def test_opt_in_torch_detection_with_different_sklearn_downstream_heads():
    specs = {'detection': [('sequence', 'torch_cnn1d')],
             'localization': [('physical', 'extra')],
             'mechanism': [('temporal', 'linear')]}
    graph = create_graph(candidate_specs=specs)
    assert graph.audit['final_head_specs']['detection'] == ['sequence', 'torch_cnn1d']
    assert graph.models['detection'].training_['cuda_used'] is False
    assert graph.models['detection'].training_['normalization'] == 'training_partition_only'
    rows, *_ = fixture_train()
    pred = graph.diagnose(rows[-1], fields=FIELDS_, healthy_prefix_confirmed=True)
    assert pred['qualified'] is False
    assert pred['head_specs']['detection']['learner'] == 'torch_cnn1d'


def test_inner_training_failure_keeps_candidate_audit(monkeypatch):
    from phm_design.task_candidate_registry import TaskCandidateRegistry
    def failed_build(self, feature, learner, seed):
        raise RuntimeError('FORCED_TEST_BACKEND_UNAVAILABLE')
    monkeypatch.setattr(TaskCandidateRegistry, 'build', failed_build)
    with pytest.raises(TaskDesignRejected, match='GRAPH_HEAD_FIT_FAILED') as exc:
        create_graph(candidate_specs={h: [('temporal', 'linear')] for h in ('detection','localization','mechanism')})
    audit = exc.value.audit
    assert audit['decision'] == 'REJECT_OUTER_HEAD_FAILURE'
    assert audit['model_delivered'] is False
    assert len(audit['failed_head_trials']) == 1
    assert 'FORCED_TEST_BACKEND_UNAVAILABLE' in audit['failed_head_trials'][0]['reason']
