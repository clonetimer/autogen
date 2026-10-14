"""Offline information audit, not a threshold diagnostic or promotion gate.

Exact observation aliases with conflicting labels disprove zero-error mapping
for that declared deterministic input contract on this finite record multiset.
Absence of aliases does NOT prove feasibility. Training failures do NOT prove
infeasibility. No claim about arbitrary sensors, methods, or domains is made.
"""
from collections import Counter, defaultdict
import hashlib
import numpy as np
from .open_design_fit import validate_training

def audit_observation_aliases(examples, spec, classes):
    validate_training(examples, spec, classes)
    buckets = defaultdict(list)
    originals = [e for e in examples if e.profile == 'identity']
    if not originals:
        raise ValueError('IDENTITY_OBSERVATIONS_REQUIRED')
    for e in originals:
        a = np.array(spec.array(e.observations), dtype='<f8', copy=True)
        a[a == 0] = 0.0  # canonicalize signed zero, not finite measurement noise
        h = hashlib.sha256(repr(a.shape).encode() + a.tobytes()).hexdigest()
        buckets[h].append(e)
    witnesses = []; unavoidable = 0
    for h, rows in sorted(buckets.items()):
        counts = Counter(int(e.label) for e in rows)
        if len(counts) > 1:
            errors = len(rows) - max(counts.values()); unavoidable += errors
            witnesses.append({'observation_sha256': h, 'label_counts': dict(counts),
                'worlds': sorted({e.group for e in rows}), 'records': len(rows),
                'minimum_errors_on_this_multiset': errors})
    return {'schema': 'phm.observation-alias-audit.v1',
        'status': 'EXACT_INPUT_LABEL_CONFLICT' if witnesses else 'NO_EXACT_CONFLICT_FOUND_NOT_PROOF_OF_FEASIBILITY',
        'input_contract': {'name': spec.name, 'fields': list(spec.fields)},
        'identity_records': len(originals), 'physical_worlds': len({e.group for e in originals}),
        'conflicting_observation_groups': len(witnesses), 'witnesses': witnesses,
        'minimum_empirical_errors': unavoidable,
        'minimum_empirical_error_fraction': unavoidable / len(originals),
        'bound_scope': 'this unweighted identity-record multiset under the exact declared inputs; no confidence claim',
        'universal_infeasibility': False, 'release_qualified': False}
