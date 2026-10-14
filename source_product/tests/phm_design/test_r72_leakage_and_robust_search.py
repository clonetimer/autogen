"""R72 guard tests; these do not use sealed or revealed PHM research worlds."""
import hashlib
import numpy as np
import pytest
from phm_design.leakage_control import (WorldLineage, inspect_partition,
    authorize_design, validate_manifest_separation, register_feedback)
from phm_design.task_graph import _choose_head


def record(world, family, role, physics, observation=b'values', environment='env0'):
    return WorldLineage(world, family, role, physics,
                        (hashlib.sha256(observation).hexdigest(),), environment)


def test_split_prefix_and_seed_change_do_not_establish_independence():
    train = record('TRAIN-A', 'physical-A', 'TRAIN', {'inertia': 2.0}, b'aa')
    with pytest.raises(ValueError, match='CROSS_ROLE_FAMILY_OVERLAP'):
        validate_manifest_separation([train], [record('FINAL-X', 'physical-A', 'FINAL_OOD', {'inertia': 2.1}, b'xx')])
    with pytest.raises(ValueError, match='CROSS_ROLE_PHYSICS_OVERLAP'):
        validate_manifest_separation([train], [record('FINAL-X', 'physical-B', 'FINAL_OOD', {'inertia': 2.0}, b'yy')])
    with pytest.raises(ValueError, match='CROSS_ROLE_OBSERVATIONS_OVERLAP'):
        validate_manifest_separation([train], [record('FINAL-X', 'physical-B', 'FINAL_OOD', {'inertia': 9}, b'aa')])


def test_guard_fails_closed_for_protected_or_tainted_roles():
    sample = record('w', 'f', 'TRAIN', {'capacity': 1})
    assert authorize_design([sample])['independent_qualification'] is False
    with pytest.raises(ValueError, match='DESIGN_WORKER_PROTECTED_ROLE'):
        authorize_design([record('w', 'f', 'FINAL_ID', {'capacity': 1})])
    with pytest.raises(ValueError, match='FINAL_FEEDBACK_TAINTED_DESIGN'):
        authorize_design([sample], feedback_used_for_design=True)
    assert register_feedback(source_role='FINAL_OOD', used_to_change_design=True)['independent_test_status'] == 'TAINTED_REQUIRES_NEW_FINAL'


def test_manifest_completeness_and_unsupported_claims():
    r = record('w', 'f', 'TRAIN', {'mass_kg': 1})
    assert 'near_duplicate' in inspect_partition([r])['not_proven']
    assert inspect_partition([r, r])['worlds'] == 1  # paired healthy/fault trajectories
    with pytest.raises(ValueError, match='INCONSISTENT_WORLD_IDENTITY'):
        inspect_partition([r, record('w', 'another', 'TRAIN', {'mass_kg': 2}, b'other')])
    with pytest.raises(ValueError, match='PHYSICS_CONFIGURATION_REQUIRED'):
        record('w', 'f', 'TRAIN', {})


def test_worst_environment_selector_requires_environment_groups():
    y = np.tile([0, 1, 2, 3, 4], 8)
    groups = np.repeat([f'w{i}' for i in range(8)], 5)
    matrix = {'temporal': np.arange(len(y)).reshape(-1, 1).astype(float)}
    labels = {'detection': (y > 0).astype(int)}
    with pytest.raises(ValueError, match='ENVIRONMENT_FAMILIES_REQUIRED'):
        _choose_head('detection', [('temporal', 'linear')], matrix, labels, y,
                     groups, np.arange(len(y)), 1, selection_mode='worst_environment')


def test_task_graph_family_grouping_and_guard_integration():
    from phm_design.platform_features import FIELDS
    from phm_design.task_graph import design_task_graph, TaskGraphPolicy
    from test_task_architectures_08 import fixture_train
    rows, y, worlds, roles, healthy = fixture_train()
    domains = ['env-low', 'env-high'] * 4
    env_map = {f'w{i}': domains[i] for i in range(8)}
    # The world name is a model-design group, not an observation feature.
    # Same physical world has five different class trajectories.
    lineage = [WorldLineage(w, w, 'TRAIN', {'sample_base_world': w}, (),
                            env_map[w]) for w in worlds]
    specs = {h: [('temporal', 'linear')] for h in ('detection','localization','mechanism')}
    graph = design_task_graph(domain='Thermal_localization', observations=rows,
                             fields=FIELDS['Thermal_localization'], labels=y,
                             world_ids=worlds, roles=roles,
                             healthy_prefix_confirmed=healthy,
                             training_lineage=lineage,
                             environment_families=[x.environment_family for x in lineage],
                             selection_mode='worst_environment',
                             candidate_specs=specs,
                             acceptance_policy=TaskGraphPolicy(0,1,0,0,0,0,0,1,8))
    assert graph.audit['selection_mode'] == 'worst_environment'
    assert graph.audit['provenance_receipt']['worlds'] == 8
    assert graph.audit['independent_family_overlap_checked']
    assert graph.audit['evaluation_feedback_independence_unverified'] is True
    assert graph.audit['stage'] == 'DEVELOPMENT_ONLY'
    # Role boundary cannot be overridden by the relaxed policy.
    unsafe = list(lineage)
    unsafe[0] = WorldLineage(unsafe[0].world_id, unsafe[0].generation_family_id,
                             'FINAL_OOD', unsafe[0].base_physics, (), unsafe[0].environment_family)
    with pytest.raises(ValueError, match='DESIGN_WORKER_PROTECTED_ROLE'):
        design_task_graph(domain='Thermal_localization', observations=rows,
                          fields=FIELDS['Thermal_localization'], labels=y,
                          world_ids=worlds, roles=roles, healthy_prefix_confirmed=healthy,
                          training_lineage=unsafe, candidate_specs=specs,
                          acceptance_policy=TaskGraphPolicy(0,1,0,0,0,0,0,1,8))


def test_environment_family_outer_holdout_prevents_same_environment_in_training():
    from phm_design.platform_features import FIELDS
    from phm_design.task_graph import design_task_graph, TaskGraphPolicy
    from test_task_architectures_08 import fixture_train
    rows, y, worlds, roles, healthy = fixture_train()
    # 4 environment families, 2 different physical-world configurations each.
    environments = {f'w{i}': f'environment-{i // 2}' for i in range(8)}
    lineage = [WorldLineage(w, w, 'TRAIN', {'base_world': w}, (), environments[w])
               for w in worlds]
    specs = {h: [('temporal', 'linear')] for h in ('detection','localization','mechanism')}
    graph = design_task_graph(domain='Thermal_localization', observations=rows,
                              fields=FIELDS['Thermal_localization'], labels=y,
                              world_ids=worlds, roles=roles, healthy_prefix_confirmed=healthy,
                              training_lineage=lineage,
                              environment_families=[x.environment_family for x in lineage],
                              selection_mode='worst_environment',
                              outer_split_mode='leave_environment_family_out',
                              candidate_specs=specs,
                              acceptance_policy=TaskGraphPolicy(0,1,0,0,0,0,0,1,8))
    folds = graph.audit['outer_fold_history']
    assert len(folds) == 4
    assert all(not set(f['train_environments']) & set(f['heldout_environments']) for f in folds)
    assert graph.audit['outer_split_mode'] == 'leave_environment_family_out'
