"""R72 research preflight: split provenance, role controls and feedback taint.

This is an enforceable INPUT contract, not an OS data enclave, not a proof
of novel physics, and not an independent qualification certificate.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
from typing import Mapping, Sequence

DEV_ROLES = frozenset({'TRAIN', 'DEV_VALID', 'DEV_STRESS'})
FINAL_ROLES = frozenset({'FINAL_ID', 'FINAL_OOD'})
ROLES = DEV_ROLES | FINAL_ROLES


# Identity inputs must describe the *base* physical world, not experiment labels,
# data partition names or fault-injection truth. Reject these recursively instead
# of allowing an accidental split key to manufacture different physics digests.
_DISALLOWED_BASE_KEYS = frozenset({
    'split', 'split_name', 'dataset_split', 'role', 'data_role', 'dataset_role',
    'label', 'target', 'class_label', 'fault_label', 'fault_class',
    'fault_onset', 'fault_onset_s', 'fault_injection', 'fault_injection_label',
    'filename', 'file_path', 'output_path', 'directory',
})


def _check_base_physics(value):
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError('INVALID_BASE_PHYSICS_KEY')
            if key.strip().lower() in _DISALLOWED_BASE_KEYS:
                raise ValueError('SPLIT_OR_TRUTH_KEY_IN_BASE_PHYSICS:' + key)
            _check_base_physics(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _check_base_physics(child)


def _digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class WorldLineage:
    world_id: str
    generation_family_id: str
    role: str
    base_physics: Mapping
    # Observation hashes allow exact duplicate checks without reading target labels.
    observation_hashes: tuple[str, ...] = ()
    environment_family: str = ''

    def __post_init__(self):
        if not all(isinstance(x, str) and x.strip() for x in
                   (self.world_id, self.generation_family_id, self.role, self.environment_family)):
            raise ValueError('LINEAGE_IDENTITY_REQUIRED')
        if self.role not in ROLES:
            raise ValueError('UNRECOGNIZED_DATA_ROLE')
        if not isinstance(self.base_physics, Mapping) or not self.base_physics:
            raise ValueError('PHYSICS_CONFIGURATION_REQUIRED')
        _check_base_physics(self.base_physics)
        if not all(isinstance(h, str) and len(h) == 64 and
                   all(c in '0123456789abcdef' for c in h) for h in self.observation_hashes):
            raise ValueError('INVALID_OBSERVATION_DIGEST')

    @property
    def physics_digest(self) -> str:
        return _digest(self.base_physics)


def inspect_partition(entries: Sequence[WorldLineage]) -> dict:
    """Detect family/physics/observation overlaps *across* data roles.

    The base_physics mapping must be canonical upstream; changing names, roles,
    seeds or split prefixes does not establish target-domain independence.
    """
    if not entries:
        raise ValueError('EMPTY_STUDY_MANIFEST')
    seen_world, owners = {}, {'family': {}, 'physics': {}, 'observations': {}}
    environment_owners = {'family': {}, 'physics': {}}
    for rec in entries:
        if not isinstance(rec, WorldLineage):
            raise ValueError('INVALID_STUDY_RECORD')
        identity = (rec.generation_family_id, rec.role, rec.physics_digest, rec.environment_family)
        if rec.world_id in seen_world and seen_world[rec.world_id] != identity:
            raise ValueError('INCONSISTENT_WORLD_IDENTITY')
        seen_world[rec.world_id] = identity
        # A repeated physical configuration or generation family cannot be
        # renamed into a new 'unseen environment'. This holds even when both
        # records are TRAIN, where cross-role checks alone would miss it.
        for kind, key in (('family', rec.generation_family_id),
                          ('physics', rec.physics_digest)):
            prior = environment_owners[kind].get(key)
            if prior is not None and prior != rec.environment_family:
                raise ValueError('CROSS_ENVIRONMENT_' + kind.upper() + '_OVERLAP')
            environment_owners[kind][key] = rec.environment_family
        for kind, keys in (('family', (rec.generation_family_id,)),
                           ('physics', (rec.physics_digest,)),
                           ('observations', rec.observation_hashes)):
            for key in keys:
                previous = owners[kind].get(key)
                if previous is not None and previous != rec.role:
                    raise ValueError('CROSS_ROLE_' + kind.upper() + '_OVERLAP')
                owners[kind][key] = rec.role
    return {'schema': 'phm.leakage.partition.v1', 'worlds': len(seen_world),
            'roles': {role: sum(x.role == role for x in entries) for role in sorted(ROLES)},
            'manifest_sha256': _digest([{'world_id': x.world_id,
                                         'family': x.generation_family_id,
                                         'role': x.role, 'physics_sha256': x.physics_digest,
                                         'observation_hashes': x.observation_hashes,
                                         'environment_family': x.environment_family} for x in entries]),
            'checked': ['world', 'generation_family', 'base_physics', 'exact_observation',
                        'environment_family_coherence'],
            'not_proven': ['near_duplicate', 'physical_domain_independence',
                           'human_or_agent_test_feedback_isolation']}


def authorize_design(entries: Sequence[WorldLineage], *, feedback_used_for_design=False) -> dict:
    """Admission for a development worker. Never receives protected observations.

    Records are metadata-only. All input records MUST be TRAIN; prior access to
    final metrics taints the claimed independent design protocol even if model
    fitting itself never saw a final label.
    """
    if feedback_used_for_design:
        raise ValueError('FINAL_FEEDBACK_TAINTED_DESIGN')
    if any(rec.role != 'TRAIN' for rec in entries):
        raise ValueError('DESIGN_WORKER_PROTECTED_ROLE')
    receipt = inspect_partition(entries)
    receipt['stage'] = 'DEVELOPMENT_ONLY'
    receipt['final_data_visible_to_design'] = False
    receipt['independent_qualification'] = False
    return receipt


def validate_manifest_separation(design_entries, heldout_entries):
    """Called by a separate steward, never inside model-search with target data."""
    if any(x.role not in DEV_ROLES for x in design_entries):
        raise ValueError('NON_DEVELOPMENT_IN_DESIGN_MANIFEST')
    if any(x.role not in FINAL_ROLES for x in heldout_entries):
        raise ValueError('INVALID_FINAL_ROLE')
    return inspect_partition(tuple(design_entries) + tuple(heldout_entries))


def register_feedback(*, source_role: str, used_to_change_design: bool):
    """A full-test summary or error case can contaminate a later design."""
    if source_role not in ROLES:
        raise ValueError('UNRECOGNIZED_DATA_ROLE')
    return {'source_role': source_role,
            'independent_test_status': ('TAINTED_REQUIRES_NEW_FINAL'
                                        if source_role in FINAL_ROLES and used_to_change_design
                                        else 'NOT_TAINTED_BY_THIS_EVENT')}
