"""Synthetic-only R74 negative controls; no FINAL worlds or labels opened."""
import pytest
from phm_design.leakage_control import WorldLineage, inspect_partition, authorize_design


def row(world, family, env, base):
    return WorldLineage(world, family, 'TRAIN', base, (), env)


def test_one_generation_family_cannot_be_fake_multiple_unseen_environments():
    rows = [row('w0', 'same-generator', 'seen', {'orbit': 1}),
            row('w1', 'same-generator', 'unseen', {'orbit': 2})]
    with pytest.raises(ValueError, match='CROSS_ENVIRONMENT_FAMILY_OVERLAP'):
        authorize_design(rows)


def test_identical_base_physics_cannot_be_relabelled_as_ood_environment():
    rows = [row('w0', 'f0', 'low', {'orbit': 1, 'noise': {'sigma': .03}}),
            row('w1', 'f1', 'high', {'noise': {'sigma': .03}, 'orbit': 1})]
    with pytest.raises(ValueError, match='CROSS_ENVIRONMENT_PHYSICS_OVERLAP'):
        authorize_design(rows)


def test_distinct_physics_distinct_families_are_accepted_as_metadata_only():
    rows = [row('w0', 'f0', 'low', {'orbit': 1}),
            row('w1', 'f1', 'high', {'orbit': 2}),
            row('w2', 'f2', 'middle', {'orbit': 3})]
    receipt = authorize_design(rows)
    assert receipt['worlds'] == 3
    assert 'environment_family_coherence' in receipt['checked']
    assert receipt['independent_qualification'] is False


def test_same_family_same_environment_can_contain_paired_trajectories():
    rows = [row('w0', 'f0', 'low', {'orbit': 1}),
            row('w0', 'f0', 'low', {'orbit': 1})]
    assert inspect_partition(rows)['worlds'] == 1
