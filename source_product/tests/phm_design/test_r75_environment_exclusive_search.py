"""R75 tests: only synthetic TRAIN fixtures, never FINAL/OOD data."""
import numpy as np
import pytest

from phm_design.leakage_control import WorldLineage
from phm_design.platform_features import FIELDS
from phm_design.task_graph import (
    GraphHeadSearchRejected, TaskGraphPolicy, TaskGraphGate,
    _choose_head, _scores, design_task_graph,
)
from phm_design.task_architectures import TaskOntology
from test_task_architectures_08 import fixture_train


def _inputs():
    rows, labels, worlds, roles, healthy = fixture_train()
    families = {f'w{i}': f'env-{i // 2}' for i in range(8)}
    lineage = [WorldLineage(w, w, 'TRAIN', {'synthetic_base': w}, (), families[w])
               for w in worlds]
    specs = {h: [('temporal', 'linear')]
             for h in ('detection', 'localization', 'mechanism')}
    return rows, labels, worlds, roles, healthy, lineage, specs


def test_cross_environment_design_head_inner_folds_leave_out_complete_family():
    rows, labels, worlds, roles, healthy, lineage, specs = _inputs()
    graph = design_task_graph(
        domain='Thermal_localization', observations=rows,
        fields=FIELDS['Thermal_localization'], labels=labels, world_ids=worlds,
        roles=roles, healthy_prefix_confirmed=healthy,
        training_lineage=lineage,
        environment_families=[x.environment_family for x in lineage],
        outer_split_mode='leave_environment_family_out',
        selection_mode='worst_environment', candidate_specs=specs,
        acceptance_policy=TaskGraphPolicy(0, 1, 0, 0, 0, 0, 0, 1, 8))
    audit = graph.audit
    assert audit['model_delivered'] is True
    assert len(audit['outer_fold_history']) == 4
    for outer in audit['outer_fold_history']:
        assert not set(outer['train_environments']) & set(outer['heldout_environments'])
        for head in ('detection', 'localization', 'mechanism'):
            complete = [trial for trial in outer['head_selections'][head]['trials']
                        if trial['status'] == 'COMPLETE']
            assert complete, head
            for trial in complete:
                assert trial['inner_validation_mode'] == 'leave_one_environment_family_out'
                folds = trial['inner_fold_history']
                assert len(folds) == 3
                assert set.union(*(set(f['heldout_environment_families']) for f in folds)) == set(outer['train_environments'])
                for fold in folds:
                    assert fold['environment_disjoint'] and fold['world_disjoint']
                    assert len(fold['heldout_environment_families']) == 1
                    assert not set(fold['train_environment_families']) & set(fold['heldout_environment_families'])


def test_failed_head_does_not_use_uninitialized_predictions(monkeypatch):
    from phm_design.task_candidate_registry import TaskCandidateRegistry
    def fail(self, feature, learner, seed):
        raise RuntimeError('SYNTHETIC_BACKEND_FAILURE')
    monkeypatch.setattr(TaskCandidateRegistry, 'build', fail)
    labels = np.tile([0, 1], 8)
    worlds = np.repeat([f'w{i}' for i in range(8)], 2)
    env = np.repeat([f'env{i // 2}' for i in range(8)], 2)
    with pytest.raises(GraphHeadSearchRejected) as e:
        _choose_head('detection', [('temporal', 'linear')],
                     {'temporal': np.arange(16).reshape(16, 1).astype(float)},
                     {'detection': labels}, labels, worlds, np.arange(16), 11,
                     environment_families=env, selection_mode='worst_environment')
    assert len(e.value.trials) == 1
    assert e.value.trials[0]['status'] == 'FAILED'
    assert 'SYNTHETIC_BACKEND_FAILURE' in e.value.trials[0]['reason']
    assert 'UnboundLocalError' not in e.value.trials[0]['reason']


def test_upstream_fault_miss_counts_as_joint_localization_and_mechanism_failure():
    ontology = TaskOntology.from_domain('Thermal_localization')
    y = np.array([0, 1, 2, 3, 4] * 2)
    world = np.repeat(['a', 'b'], 5)
    # All fault labels can be guessed perfectly by the two downstream heads,
    # but detection misses every fault. Joint stage scores must be ZERO.
    f, c, m = ontology.heads(y)
    pf = np.tile([.9, .1], (len(y), 1))
    pc = np.where(c[:, None] == np.arange(2), .9, .1)
    pm = np.where(m[:, None] == np.arange(2), .9, .1)
    scores = _scores(y, world, ontology, pf, pc, pm, TaskGraphGate())
    assert scores['localization_oracle_detection_accuracy'] == 1
    assert scores['localization_all_faults_joint_accuracy'] == 0
    assert scores['mechanism_all_faults_joint_accuracy'] == 0
    assert scores['all_faults_denominator'] == 8
    assert scores['localization_conditional_accuracy'] is None


def test_high_confidence_rejection_is_not_counted_as_successful_stage():
    onto = TaskOntology.from_domain('Thermal_localization')
    y = np.array([0, 1, 2, 3, 4] * 2)
    w = np.repeat(['a', 'b'], 5)
    f, c, m = onto.heads(y)
    pf = np.tile([.49, .51], (len(y), 1))
    pc = np.where(c[:, None] == np.arange(2), .55, .45)
    pm = np.where(m[:, None] == np.arange(2), .9, .1)
    out = _scores(y, w, onto, pf, pc, pm, TaskGraphGate(.8, .8, .5))
    assert out['detection_pre_gate_fault_recall'] == 1
    assert out['detection_fault_recall'] == 0
    assert out['localization_all_faults_joint_accuracy'] == 0
    assert out['mechanism_all_faults_joint_accuracy'] == 0
    assert out['end_to_end_unknown_fraction'] == 1
