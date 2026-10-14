"""Synthetic-only robust-search design checks (not new qualification)."""
import numpy as np
import pytest
from phm_design.robust_design_policy import RobustDesignPolicy, RobustDesignRejected
from phm_design.leakage_control import WorldLineage
from phm_design.open_design import ObservationSpec
from phm_design.open_design_fit import TrainingExample, design


def synthetic_case():
    rng = np.random.default_rng(7200)
    spec = ObservationSpec('synthetic-strong-and-weak',('time_s','sensor_a','sensor_b'),min_rows=16)
    examples, lineage = [], []
    for env in range(4):
        for world in range(4):
            wid = f'environment-{env}-world-{world}'
            for label in (0,1,2):
                t = np.linspace(0,1,32)
                x = np.stack([t,np.sin(2*np.pi*t)+label*.02+rng.normal(0,.03,32),
                              np.cos(2*np.pi*t)-label*.02+rng.normal(0,.03,32)],axis=1)
                examples.append(TrainingExample(x,label,wid,'design_train'))
                lineage.append(WorldLineage(wid,wid,'TRAIN',{'synthetic_base':wid},(),f'env-{env}'))
    return spec, examples, lineage


def test_robust_policy_rejects_inadequate_environment_support():
    policy = RobustDesignPolicy(min_environment_families=3)
    assert not policy.evaluate({'family_metrics':{}}, [0,1,2], 0)['eligible']
    with pytest.raises(ValueError,match='INVALID_ROBUST_DESIGN_POLICY'):
        RobustDesignPolicy(worst_fault_recall_min=1.1)


def test_robust_open_design_fail_closed_and_preserves_trials():
    spec, examples, lineage = synthetic_case()
    with pytest.raises(RobustDesignRejected) as e:
        design(examples,spec,(0,1,2),0,max_trials=3,training_lineage=lineage,
               robust_policy=RobustDesignPolicy())
    audit=e.value.audit
    assert audit['decision']=='REJECT_ALL_CANDIDATES'
    assert audit['model_delivered'] is False
    assert audit['independent_qualification'] is False
    assert audit['training_lineage_receipt']['worlds']==16
    assert len(audit['attempts'])==3
    assert all(x['status'] in ('REJECTED_BY_POLICY','FAILED') for x in audit['attempts'])


def test_robust_search_disallows_revealed_feedback_and_missing_provenance():
    spec, examples, lineage=synthetic_case()
    with pytest.raises(ValueError,match='ROBUST_SEARCH_REQUIRES_VERIFIED_LINEAGE'):
        design(examples,spec,(0,1,2),0,max_trials=1,
               robust_policy=RobustDesignPolicy())
    with pytest.raises(ValueError,match='FINAL_FEEDBACK_TAINTED_DESIGN'):
        design(examples,spec,(0,1,2),0,max_trials=1,training_lineage=lineage,
               robust_policy=RobustDesignPolicy(),feedback_used_for_design=True)


def test_legacy_average_search_does_not_ignore_supplied_final_lineage():
    """Providing provenance must never disable the role gate by omitting policy."""
    spec, examples, lineage = synthetic_case()
    bad = list(lineage)
    r = bad[0]
    bad[0] = WorldLineage(r.world_id, r.generation_family_id, 'FINAL_OOD',
                          r.base_physics, r.observation_hashes, r.environment_family)
    with pytest.raises(ValueError, match='DESIGN_WORKER_PROTECTED_ROLE'):
        design(examples, spec, (0, 1, 2), 0, max_trials=1,
               training_lineage=bad)
    with pytest.raises(ValueError, match='TRAINING_LINEAGE_LENGTH_MISMATCH'):
        design(examples, spec, (0, 1, 2), 0, max_trials=1,
               training_lineage=lineage[:-1])


def test_identity_rejects_explicit_split_or_fault_truth_keys():
    for base in ({'mass': 1, 'split': 'TRAIN'},
                 {'mass': 1, 'physics': {'fault_label': 2}},
                 {'mass': 1, 'metadata': [{'data_role': 'FINAL_OOD'}]}):
        with pytest.raises(ValueError, match='SPLIT_OR_TRUTH_KEY_IN_BASE_PHYSICS'):
            WorldLineage('world', 'family', 'TRAIN', base, (), 'env')
