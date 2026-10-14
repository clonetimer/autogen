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
