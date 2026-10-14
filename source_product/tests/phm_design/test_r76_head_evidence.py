"""R76 negative controls use synthetic TRAIN-only arrays; never sealed worlds."""
import numpy as np
import pytest

from phm_design.task_head_evidence import HeadEvidencePolicy
from phm_design.task_graph import _choose_head, GraphHeadSearchRejected
from phm_design.task_candidate_registry import TaskCandidateRegistry


def _binary_three_env():
    y = np.tile([0, 0, 0, 0, 1, 1, 1, 1], 3)
    env = np.repeat(['seen-a', 'seen-b', 'unseen-in-training-c'], 8)
    return y, env


def test_worst_environment_failure_rejects_high_average_detection():
    y, env = _binary_three_env()
    pred = y.copy()
    pred[env == 'unseen-in-training-c'] = 0
    policy = HeadEvidencePolicy(min_per_class_per_environment=2)
    result = policy.evaluate(head='detection', truth=y, prediction=pred, environments=env)
    assert not result['eligible']
    assert any(x['reason'] == 'ENVIRONMENT_FAULT_RECALL'
               and x['environment'] == 'unseen-in-training-c' for x in result['failed_gates'])
    assert result['environment_support_and_metrics']['seen-a']['positive_class_recall'] == 1


def test_fault_strength_stratum_miss_rejected_even_if_aggregate_is_good():
    y, env = _binary_three_env()
    strata = np.array(['NOMINAL'] * 24, dtype=object)
    pred = y.copy()
    for i in range(3):
        start = 8 * i
        strata[start + 4:start + 6] = 'weak'
        strata[start + 6:start + 8] = 'strong'
    # Environment 'seen-b' loses all weak faults while strong faults remain detected.
    pred[12:14] = 0
    policy = HeadEvidencePolicy(detector_fault_recall_min=.50,
                                expected_fault_strata=('weak','strong'),
                                require_fault_strength_strata=True,
                                min_per_fault_stratum=2, fault_stratum_recall_min=.80)
    result = policy.evaluate(head='detection', truth=y, prediction=pred,
                             environments=env, fault_strata=strata)
    assert not result['eligible']
    assert any(x['reason'] == 'FAULT_STRATUM_RECALL' and x['stratum'] == 'weak'
               for x in result['failed_gates'])


def test_missing_fault_stratum_cannot_be_relabelled_away():
    y, env = _binary_three_env()
    labels = np.array(['NOMINAL' if x == 0 else 'strong' for x in y], dtype=object)
    policy = HeadEvidencePolicy(expected_fault_strata=('weak','strong'),
                                require_fault_strength_strata=True)
    result = policy.evaluate(head='detection', truth=y, prediction=y,
                             environments=env, fault_strata=labels)
    assert not result['eligible']
    assert any(x['reason']=='INSUFFICIENT_FAULT_STRATUM_SUPPORT' and x['stratum']=='weak'
               for x in result['failed_gates'])


def test_downstream_head_requires_both_mechanism_classes_in_every_environment():
    y, env = _binary_three_env()
    pred = y.copy()
    pred[(env == 'seen-b') & (y == 1)] = 0
    r = HeadEvidencePolicy().evaluate(head='mechanism', truth=y,
                                      prediction=pred, environments=env)
    assert not r['eligible']
    assert any(x['reason'] == 'ENVIRONMENT_WORST_CLASS_RECALL' for x in r['failed_gates'])


def test_invalid_policy_or_unavailable_strata_is_fail_closed():
    with pytest.raises(ValueError,match='EXPLICIT_FAULT_STRENGTH_STRATA_REQUIRED'):
        HeadEvidencePolicy(require_fault_strength_strata=True)
    with pytest.raises(ValueError,match='INVALID_HEAD_EVIDENCE_POLICY'):
        HeadEvidencePolicy(min_per_class_per_environment=0)
    y, env = _binary_three_env()
    p = HeadEvidencePolicy(expected_fault_strata=('weak','strong'),
                           require_fault_strength_strata=True)
    assert any(v['reason']=='FAULT_STRENGTH_STRATA_REQUIRED'
               for v in p.evaluate(head='detection',truth=y,prediction=y,
                                   environments=env)['failed_gates'])


def test_nested_candidate_rejected_without_altering_predictor_or_granting_qualification(monkeypatch):
    class ConstantNominal:
        classes_ = np.array([0,1])
        def fit(self, x, y):
            return self
        def predict_proba(self, x):
            return np.tile([1., 0.], (len(x),1))
    def build(self, feature, learner, seed):
        return ConstantNominal()
    monkeypatch.setattr(TaskCandidateRegistry,'build',build)
    y, env = _binary_three_env()
    groups = np.repeat([f'w{i}' for i in range(6)],4)
    with pytest.raises(GraphHeadSearchRejected) as captured:
        _choose_head('detection',[('temporal','linear')],{'temporal':np.ones((24,3))},
                     {'detection':y},y,groups,np.arange(24),12,
                     environment_families=env,selection_mode='worst_environment',
                     head_evidence_policy=HeadEvidencePolicy())
    trials = captured.value.trials
    assert len(trials)==1
    assert trials[0]['status']=='REJECTED_BY_HEAD_EVIDENCE'
    assert trials[0]['head_evidence']['not_independent_qualification']
    assert any(x['reason']=='ENVIRONMENT_FAULT_RECALL'
               for x in trials[0]['head_evidence']['failed_gates'])


def test_end_to_end_stratum_screen_counts_unknown_and_missed_faults():
    y, env = _binary_three_env()
    # Only positive labels are actual faults. Declared strata must be in all environments.
    strength = np.array(['NOMINAL' if value == 0 else ('weak' if i % 8 < 6 else 'strong')
                         for i,value in enumerate(y)], dtype=object)
    pred = y.copy()
    pred[12:14] = -1  # All weak faults in the middle environment are UNKNOWN.
    policy = HeadEvidencePolicy(expected_fault_strata=('weak','strong'),
                                require_fault_strength_strata=True,
                                min_per_fault_stratum=2)
    result = policy.evaluate_end_to_end_strata(
        truth=y, decisions=pred, environments=env, fault_strata=strength)
    assert not result['eligible']
    assert result['unknown_counts_as_failure']
    assert any(x['reason'] == 'END_TO_END_FAULT_STRATUM_RECALL' and x['stratum']=='weak'
               for x in result['failed_gates'])
    # The same perfect downstream predictions would not redeem a missed upstream fault.


def test_full_pipeline_missing_weak_support_is_never_accepted():
    y, env = _binary_three_env()
    strength = np.array(['NOMINAL' if x == 0 else 'strong' for x in y],dtype=object)
    p = HeadEvidencePolicy(expected_fault_strata=('weak','strong'),
                           require_fault_strength_strata=True)
    r = p.evaluate_end_to_end_strata(truth=y,decisions=y,
                                    environments=env,fault_strata=strength)
    assert not r['eligible']
    assert any(x['reason']=='INSUFFICIENT_END_TO_END_STRATUM_SUPPORT' and
               x['stratum']=='weak' for x in r['failed_gates'])


def test_head_evidence_requires_genuine_held_out_environment_in_outer_screen():
    from phm_design.task_graph import design_task_graph
    from phm_design.platform_features import FIELDS
    with pytest.raises(ValueError,match='HEAD_EVIDENCE_REQUIRES_OUTER_ENVIRONMENT_HOLDOUT'):
        design_task_graph(
            domain='Thermal_localization',
            observations=[None]*20, fields=FIELDS['Thermal_localization'],
            labels=[0]*20, world_ids=['fake']*20, roles=['design_train']*20,
            healthy_prefix_confirmed=[True]*20,
            training_lineage=[], selection_mode='worst_environment',
            environment_families=['env']*20,
            outer_split_mode='world_group',
            head_evidence_policy=HeadEvidencePolicy())
