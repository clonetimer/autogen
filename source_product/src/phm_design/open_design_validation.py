"""Validation-only environment metadata, never predictor inputs.

World splitting remains the backwards-compatible default. Family extrapolation
holds out an entire declared environment family. Missing family annotations are
an error, not an excuse to use labels or silently guess a family.
"""
from collections.abc import Mapping
import numpy as np
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

def validation_splits(groups, family_by_group=None):
    g = np.asarray(groups)
    if g.ndim != 1 or len(g) == 0 or any(not isinstance(v, str) or not v for v in g):
        raise ValueError('INVALID_VALIDATION_GROUPS')
    if len(set(g)) < 3:
        raise ValueError('AT_LEAST_THREE_PHYSICAL_GROUPS_REQUIRED')
    if family_by_group is None:
        split = GroupKFold(3).split(np.zeros((len(g), 1)), groups=g)
        return [(tr, te, {'validation_mode': 'physical_world'}) for tr, te in split]
    if not isinstance(family_by_group, Mapping) or set(family_by_group) != set(g):
        raise ValueError('FAMILY_MAP_MUST_COVER_EXACT_PHYSICAL_GROUPS')
    if any(not isinstance(v, str) or not v for v in family_by_group.values()):
        raise ValueError('INVALID_FAMILY_NAME')
    families = np.asarray([family_by_group[v] for v in g])
    if len(set(families)) < 2:
        raise ValueError('AT_LEAST_TWO_ENVIRONMENT_FAMILIES_REQUIRED')
    result = []
    for tr, te in LeaveOneGroupOut().split(np.zeros((len(g), 1)), groups=families):
        if len(set(g[tr])) < 3:
            raise ValueError('FAMILY_FOLD_INSUFFICIENT_TRAIN_WORLDS')
        if set(g[tr]) & set(g[te]) or set(families[tr]) & set(families[te]):
            raise ValueError('VALIDATION_LEAKAGE')
        result.append((tr, te, {'validation_mode': 'leave_environment_family_out',
            'train_families': sorted(set(families[tr])),
            'validation_families': sorted(set(families[te])),
            'family_disjoint': True}))
    return result
