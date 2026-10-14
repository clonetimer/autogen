"""R69 experimental multi-stage diagnosis graph with heterogeneous head search.

Nodes: deployment observations -> detection -> localization -> mechanism ->
legal ontology / explicit UNKNOWN.  Head selection uses inner GroupKFold on
*design_train* worlds; pipeline screening uses separate outer held-out worlds.
No sealed-world or deployment qualification is claimed.  Diagnostic labels,
world IDs and fault truth never enter deployment features.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence
import time

import numpy as np
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

from .open_design import ObservationSpec
from .platform_features import FIELDS
from .task_architectures import TaskOntology, TaskDesignRejected, _proba, _transform
from .task_candidate_registry import TaskCandidateRegistry
from .leakage_control import WorldLineage, authorize_design

_HEADS = ('detection', 'localization', 'mechanism')
_DEFAULT_SPECS = (('temporal', 'linear'), ('physical', 'extra'))

# Declarative topology; models are independently selectable, topology is frozen.
_TASK_DAG = {
    'nodes': ['observations', 'detection', 'localization', 'mechanism', 'decision_gate'],
    'edges': [
        {'from': 'observations', 'to': 'detection', 'condition': 'always'},
        {'from': 'detection', 'to': 'localization', 'condition': 'detected_fault'},
        {'from': 'localization', 'to': 'mechanism', 'condition': 'localized_fault'},
        {'from': 'detection', 'to': 'decision_gate', 'condition': 'nominal_or_unknown'},
        {'from': 'localization', 'to': 'decision_gate', 'condition': 'localization_unknown'},
        {'from': 'mechanism', 'to': 'decision_gate', 'condition': 'fault_or_unknown'},
    ],
}


class GraphHeadSearchRejected(ValueError):
    def __init__(self, head, trials):
        super().__init__('GRAPH_NO_TRAINABLE_HEAD:' + head)
        self.trials = trials


@dataclass(frozen=True)
class TaskGraphGate:
    """Fixed *predeclared* confidence cutoffs, never optimized on outer OOF."""
    detector_min_confidence: float = 0.50
    localization_min_confidence: float = 0.50
    mechanism_min_confidence: float = 0.50

    def __post_init__(self):
        for key in ('detector_min_confidence', 'localization_min_confidence', 'mechanism_min_confidence'):
            v = getattr(self, key)
            if isinstance(v, bool) or not isinstance(v, (float, int)) or not np.isfinite(v) or not 0.5 <= v <= 1:
                raise ValueError('INVALID_TASK_GRAPH_GATE:' + key)


@dataclass(frozen=True)
class TaskGraphPolicy:
    """Development-only screen; distinct detection, conditional and full-DAG gates."""
    detection_fault_recall_min: float = 0.80
    detection_healthy_false_alarm_max: float = 0.10
    localization_conditional_accuracy_min: float = 0.80
    mechanism_conditional_accuracy_min: float = 0.80
    end_to_end_accuracy_min: float = 0.90
    end_to_end_worst_fault_recall_min: float = 0.80
    end_to_end_worst_world_accuracy_min: float = 0.80
    end_to_end_unknown_max: float = 0.10
    min_independent_worlds: int = 8

    def __post_init__(self):
        for key, val in vars(self).items():
            if key == 'min_independent_worlds':
                if type(val) is not int or val < 8:
                    raise ValueError('INVALID_GRAPH_POLICY:' + key)
            elif isinstance(val, bool) or not isinstance(val, (float, int)) or not np.isfinite(val) or not 0 <= val <= 1:
                raise ValueError('INVALID_GRAPH_POLICY:' + key)

    def evaluate(self, scores: Mapping) -> dict:
        rules = (
            ('detection_fault_recall', 'min', self.detection_fault_recall_min),
            ('detection_healthy_false_alarm_rate', 'max', self.detection_healthy_false_alarm_max),
            ('localization_conditional_accuracy', 'min', self.localization_conditional_accuracy_min),
            ('mechanism_conditional_accuracy', 'min', self.mechanism_conditional_accuracy_min),
            ('end_to_end_accuracy', 'min', self.end_to_end_accuracy_min),
            ('end_to_end_worst_fault_recall', 'min', self.end_to_end_worst_fault_recall_min),
            ('end_to_end_worst_world_accuracy', 'min', self.end_to_end_worst_world_accuracy_min),
            ('end_to_end_unknown_fraction', 'max', self.end_to_end_unknown_max),
            ('independent_worlds', 'min', self.min_independent_worlds),
        )
        failures = []
        for name, op, limit in rules:
            observed = scores.get(name)
            if (not isinstance(observed, (int, float)) or isinstance(observed, bool) or
                not np.isfinite(observed) or (observed < limit if op == 'min' else observed > limit)):
                failures.append({'metric': name, 'observed': observed,
                                 'operator': '>=' if op == 'min' else '<=', 'required': limit})
        return {'eligible': not failures, 'failed_gates': failures,
                'evidence': 'NESTED_GROUP_OOF_DESIGN_TRAIN_ONLY', 'deployment_qualified': False}


def _specs(candidate_specs) -> dict[str, tuple[tuple[str, str], ...]]:
    if candidate_specs is None:
        candidate_specs = {h: _DEFAULT_SPECS for h in _HEADS}
    if not isinstance(candidate_specs, Mapping) or set(candidate_specs) != set(_HEADS):
        raise ValueError('EXACT_GRAPH_HEADS_REQUIRED')
    registry = TaskCandidateRegistry()
    outcome = {}
    for head in _HEADS:
        options = candidate_specs[head]
        if not isinstance(options, (list, tuple)) or not options:
            raise ValueError('EMPTY_GRAPH_HEAD_CANDIDATES:' + head)
        selected = []
        for option in options:
            if not isinstance(option, (list, tuple)) or len(option) != 2:
                raise ValueError('INVALID_GRAPH_HEAD_CANDIDATE')
            f, l = option
            registry.describe(f, l)
            selected.append((f, l))
        if len(set(selected)) != len(selected):
            raise ValueError('DUPLICATE_GRAPH_HEAD_CANDIDATE:' + head)
        outcome[head] = tuple(selected)
    return outcome


def _head_targets(ontology: TaskOntology, y):
    fault, component, mechanism = ontology.heads(y)
    return {'detection': fault, 'localization': component, 'mechanism': mechanism}


def _train_indices(y, head: str):
    return np.arange(len(y)) if head == 'detection' else np.flatnonzero(y != 0)


def _fit_head(head, spec, matrix, labels, y, subset, seed):
    idx = _train_indices(y[subset], head)
    training_rows = subset[idx]
    train_y = labels[head][training_rows]
    if len(training_rows) == 0 or set(train_y.tolist()) != {0, 1}:
        raise ValueError('GRAPH_HEAD_MISSING_CLASS:' + head)
    f, l = spec
    return TaskCandidateRegistry().build(f, l, seed).fit(matrix[f][training_rows], train_y)


def _get_head_proba(model, matrix, spec, query_rows):
    return _proba(model, matrix[spec[0]][query_rows], (0, 1))


def _choose_head(head, choices, matrix, labels, y, groups, eligible_rows, seed,
                 environment_families=None, selection_mode="mean"):
    """Choose one head *within* the supplied training worlds only."""
    actual = _train_indices(y[eligible_rows], head)
    idx = eligible_rows[actual]
    subset_groups = groups[idx]
    if len(set(subset_groups)) < 3:
        raise ValueError('INSUFFICIENT_INNER_HEAD_WORLDS:' + head)
    if selection_mode not in ("mean", "worst_environment"):
        raise ValueError('INVALID_SELECTION_MODE')
    env = None
    if selection_mode == 'worst_environment':
        if environment_families is None:
            raise ValueError('ENVIRONMENT_FAMILIES_REQUIRED')
        # R75: a training fold must NEVER include the environment family whose
        # loss is being used for robust candidate ranking. A world-only
        # GroupKFold previously mixed source and held-out environments.
        env = np.asarray(environment_families)[idx]
        if len(set(env)) < 2:
            raise ValueError('MULTIPLE_ENVIRONMENT_FAMILIES_REQUIRED')
        splits = list(LeaveOneGroupOut().split(idx, groups=env))
        inner_validation_mode = 'leave_one_environment_family_out'
    else:
        splits = list(GroupKFold(n_splits=3).split(idx, groups=subset_groups))
        inner_validation_mode = 'grouped_world_3fold'
    candidates = []
    champion = None
    for spec in choices:
        start = time.process_time()
        try:
            oof = np.full((len(idx), 2), np.nan, dtype=float)
            inner_fold_history = []
            for tr, va in splits:
                if set(groups[idx[tr]]) & set(groups[idx[va]]):
                    raise AssertionError('GRAPH_INNER_WORLD_LEAKAGE')
                if env is not None and set(env[tr]) & set(env[va]):
                    raise AssertionError('GRAPH_INNER_ENVIRONMENT_LEAKAGE')
                inner_fold_history.append({
                    'train_world_count': len(set(groups[idx[tr]])),
                    'heldout_world_count': len(set(groups[idx[va]])),
                    'train_environment_families': sorted(set(env[tr])) if env is not None else None,
                    'heldout_environment_families': sorted(set(env[va])) if env is not None else None,
                    'world_disjoint': True,
                    'environment_disjoint': env is not None,
                })
                m = _fit_head(head, spec, matrix, labels, y, idx[tr], seed)
                oof[va] = _get_head_proba(m, matrix, spec, idx[va])
            if not np.isfinite(oof).all():
                raise ValueError('INCOMPLETE_INNER_OOF')
            true = labels[head][idx]
            pred = oof.argmax(axis=1)
            if head == 'detection':
                recall = float(np.mean(pred[true == 1] == 1))
                far = float(np.mean(pred[true == 0] == 1))
                loss = 0.5 * (1 - recall + far)
                metric = {'fault_recall': recall, 'healthy_false_alarm_rate': far}
            else:
                loss = float(np.mean(pred != true))
                metric = {'fault_conditional_accuracy': 1 - loss}
            worst_environment_loss = None
            if selection_mode == 'worst_environment':
                # A group must not be credited for absent classes. Domain groups
                # without both detection labels are infeasible for detection.
                environment_losses = {}
                for family in sorted(set(env)):
                    rows = env == family
                    if head == 'detection':
                        healthy = rows & (true == 0)
                        faults = rows & (true == 1)
                        if not healthy.any() or not faults.any():
                            raise ValueError('ENVIRONMENT_DETECTION_CLASS_MISSING')
                        family_loss = (1 - np.mean(pred[faults] == 1) +
                                       np.mean(pred[healthy] == 1)) / 2
                    else:
                        if not rows.any() or set(true[rows]) != {0, 1}:
                            raise ValueError('ENVIRONMENT_HEAD_CLASS_MISSING')
                        family_loss = np.mean(pred[rows] != true[rows])
                    environment_losses[str(family)] = float(family_loss)
                worst_environment_loss = max(environment_losses.values())
                metric['environment_losses'] = environment_losses
            record = {'feature': spec[0], 'learner': spec[1],
                      'backend': TaskCandidateRegistry().describe(*spec).backend,
                      'status': 'COMPLETE', 'inner_group_oof_loss': loss,
                      'inner_validation_mode': inner_validation_mode,
                      'inner_fold_history': inner_fold_history,
                      'inner_metrics': metric}
            rank = ((worst_environment_loss, loss, *spec) if selection_mode == "worst_environment"
                    else (loss, *spec))
            if champion is None or rank < champion[0]:
                champion = (rank, spec)
        except Exception as err:
            # Preserve the original candidate failure; never recompute an
            # environment score from an uninitialized or partial OOF array.
            record = {'feature': spec[0], 'learner': spec[1], 'status': 'FAILED',
                      'inner_validation_mode': inner_validation_mode,
                      'reason': type(err).__name__ + ': ' + str(err)}
        record['cpu_s'] = round(time.process_time() - start, 6)
        candidates.append(record)
    if champion is None:
        raise GraphHeadSearchRejected(head, candidates)
    return champion[1], candidates


def _gate(ontology, pf, pc, pm, gates: TaskGraphGate):
    """One trajectory, one ordered graph decision; -1 means UNKNOWN."""
    dc = int(np.argmax(pf))
    cc = int(np.argmax(pc))
    mc = int(np.argmax(pm))
    if float(np.max(pf)) < gates.detector_min_confidence:
        return -1, 'DETECTION_UNCERTAIN', dc, cc, mc
    if dc == 0:
        return 0, 'NOMINAL', dc, cc, mc
    if float(np.max(pc)) < gates.localization_min_confidence:
        return -1, 'LOCALIZATION_UNCERTAIN', dc, cc, mc
    if float(np.max(pm)) < gates.mechanism_min_confidence:
        return -1, 'MECHANISM_UNCERTAIN', dc, cc, mc
    comps = tuple(sorted(set(ontology.components[1:])))
    mechs = tuple(sorted(set(ontology.mechanisms[1:])))
    matches = [k for k in range(1, len(ontology.labels))
               if ontology.components[k] == comps[cc] and ontology.mechanisms[k] == mechs[mc]]
    if len(matches) != 1:
        raise ValueError('GRAPH_ONTOLOGY_PROJECTION_FAILED')
    return matches[0], 'FAULT', dc, cc, mc


def _scores(y, groups, ontology, pf, pc, pm, gates):
    fault, component, mechanism = ontology.heads(y)
    d = pf.argmax(axis=1)
    c = pc.argmax(axis=1)
    m = pm.argmax(axis=1)
    # Confidence rejects also count as stage nonacceptance. Without this,
    # aggressive gates can look like perfect localization while returning
    # UNKNOWN on precisely the weak, ambiguous fault trajectories.
    passed_detection = (d == 1) & (pf.max(axis=1) >= gates.detector_min_confidence)
    passed_localization = passed_detection & (pc.max(axis=1) >= gates.localization_min_confidence)
    passed_mechanism = passed_localization & (pm.max(axis=1) >= gates.mechanism_min_confidence)
    detected = (fault == 1) & passed_detection
    correctly_located = detected & passed_localization & (c == component)
    final = np.array([_gate(ontology, pf[i], pc[i], pm[i], gates)[0]
                      for i in range(len(y))], dtype=int)
    correct = final == y
    per_world = {str(g): float(np.mean(correct[groups == g])) for g in sorted(set(groups))}
    per_fault = {str(ontology.labels[k]): float(np.mean(correct[y == k]))
                 for k in range(1, len(ontology.labels))}
    return {
        'detection_fault_recall': float(np.mean(passed_detection[fault == 1])),
        'detection_healthy_false_alarm_rate': float(np.mean(passed_detection[fault == 0])),
        'detection_pre_gate_fault_recall': float(np.mean(d[fault == 1] == 1)),
        'detection_healthy_nonacceptance_rate': float(np.mean(final[fault == 0] != 0)),
        'localization_conditional_accuracy': (float(np.mean(
            passed_localization[detected] & (c[detected] == component[detected])))
            if detected.any() else None),
        'localization_conditional_support': int(detected.sum()),
        'mechanism_conditional_accuracy': (float(np.mean(passed_mechanism[correctly_located] &
                                            (m[correctly_located] == mechanism[correctly_located])))
                                            if correctly_located.any() else None),
        'mechanism_conditional_support': int(correctly_located.sum()),
        'localization_oracle_detection_accuracy': float(np.mean(c[fault == 1] == component[fault == 1])),
        'mechanism_oracle_detection_accuracy': float(np.mean(m[fault == 1] == mechanism[fault == 1])),
        # On ALL actual fault trajectories: failed upstream detection remains
        # a failure, rather than silently dropping the hardest weak faults.
        'localization_all_faults_joint_accuracy': float(np.mean(
            passed_localization[fault == 1] & (c[fault == 1] == component[fault == 1]))),
        'mechanism_all_faults_joint_accuracy': float(np.mean(
            passed_mechanism[fault == 1] & (c[fault == 1] == component[fault == 1]) &
            (m[fault == 1] == mechanism[fault == 1]))),
        'all_faults_denominator': int((fault == 1).sum()),
        'end_to_end_accuracy': float(np.mean(correct)),
        'end_to_end_healthy_false_alarm_rate': float(np.mean(final[fault == 0] > 0)),
        'end_to_end_worst_fault_recall': min(per_fault.values()),
        'end_to_end_worst_world_accuracy': min(per_world.values()),
        'end_to_end_unknown_fraction': float(np.mean(final == -1)),
        'independent_worlds': len(per_world),
        'per_world_accuracy': per_world,
        'per_fault_recall': per_fault,
        'total_trajectories': len(y),
        'unknown_count': int((final == -1).sum()),
        'end_to_end_decisions': {name: int((final == j).sum()) for j, name in enumerate(ontology.labels)},
    }


@dataclass
class TaskGraphArtifact:
    ontology: TaskOntology
    spec_fields: tuple[str, ...]
    models: dict
    head_specs: dict
    gate: TaskGraphGate
    audit: dict

    def diagnose(self, observations, *, fields, healthy_prefix_confirmed=False):
        if tuple(fields) != self.spec_fields:
            raise ValueError('EXACT_DEPLOYMENT_FIELDS_REQUIRED')
        if type(healthy_prefix_confirmed) is not bool or not healthy_prefix_confirmed:
            raise ValueError('HEALTHY_PREFIX_NOT_CONFIRMED')
        a = ObservationSpec('task-graph:' + self.ontology.domain, self.spec_fields).array(observations)
        p = {}
        for head in _HEADS:
            feature, learner = self.head_specs[head]
            x = _transform([a], self.ontology.domain, feature)
            p[head] = _proba(self.models[head], x, (0, 1))[0]
        label, route, dc, cc, mc = _gate(self.ontology, p['detection'], p['localization'], p['mechanism'], self.gate)
        comps = tuple(sorted(set(self.ontology.components[1:])))
        mechs = tuple(sorted(set(self.ontology.mechanisms[1:])))
        return {
            'prediction': None if label == -1 else self.ontology.labels[label],
            'status': 'UNKNOWN' if label == -1 else 'RESEARCH_PREDICTION',
            'gate_route': route,
            'detection': 'FAULT' if dc else 'NOMINAL',
            'component': comps[cc] if dc and route in ('FAULT', 'MECHANISM_UNCERTAIN') else None,
            'mechanism': mechs[mc] if route == 'FAULT' else None,
            'head_confidences': {h: float(np.max(p[h])) for h in _HEADS},
            'head_specs': {h: {'feature': f, 'learner': l} for h, (f, l) in self.head_specs.items()},
            'qualified': False,
            'training_role_fields_used_at_inference': False,
        }


def design_task_graph(*, domain: str, observations: Sequence, fields: Sequence[str],
                      labels: Sequence[int], world_ids: Sequence[str], roles: Sequence[str],
                      healthy_prefix_confirmed: Sequence[bool],
                      candidate_specs: Mapping | None = None,
                      environment_families: Sequence[str] | None = None,
                      selection_mode: str = 'mean',
                      outer_split_mode: str = 'world_group',
                      training_lineage: Sequence[WorldLineage] | None = None,
                      feedback_used_for_design: bool = False,
                      gate: TaskGraphGate | None = None,
                      acceptance_policy: TaskGraphPolicy | None = None,
                      seed: int = 6901) -> TaskGraphArtifact:
    """Nested grouped-OOF design-train screen; never opens sealed/evaluation worlds.

    Head architectures can differ. One *predeclared* graph and fixed gate are
    scored; this is NOT an unbiased model-development generalization estimate,
    a certification, or an arbitrary topology / cost-aware DAG search.
    """
    ontology = TaskOntology.from_domain(domain)
    schema = tuple(FIELDS[domain])
    if tuple(fields) != schema:
        raise ValueError('EXACT_DEPLOYMENT_FIELDS_REQUIRED')
    n = len(observations)
    if not (n == len(labels) == len(world_ids) == len(roles) == len(healthy_prefix_confirmed)) or n < 20:
        raise ValueError('INVALID_TRAINING_LENGTH')
    if any(role != 'design_train' for role in roles):
        raise ValueError('PROTECTED_OR_EVALUATION_ROLE')
    if any(type(x) is not bool or not x for x in healthy_prefix_confirmed):
        raise ValueError('HEALTHY_PREFIX_NOT_CONFIRMED')
    if any(not isinstance(w, str) or not w for w in world_ids):
        raise ValueError('WORLD_ID_REQUIRED')
    if type(seed) is not int or not 0 <= seed < 2**32 - 5:
        raise ValueError('INVALID_GRAPH_SEED')
    if selection_mode not in ('mean', 'worst_environment'):
        raise ValueError('INVALID_SELECTION_MODE')
    if outer_split_mode not in ('world_group', 'leave_environment_family_out'):
        raise ValueError('INVALID_OUTER_SPLIT_MODE')
    if outer_split_mode == 'leave_environment_family_out' and (
            training_lineage is None or environment_families is None):
        raise ValueError('PROVENANCE_AND_ENVIRONMENTS_REQUIRED_FOR_OOD')
    if selection_mode == 'worst_environment':
        if training_lineage is None:
            raise ValueError('PROVENANCE_REQUIRED_FOR_ROBUST_SEARCH')
        if environment_families is None or len(environment_families) != n or any(
                not isinstance(v, str) or not v.strip() for v in environment_families):
            raise ValueError('ENVIRONMENT_FAMILIES_REQUIRED')
    if feedback_used_for_design:
        raise ValueError('FINAL_FEEDBACK_TAINTED_DESIGN')
    receipt = None
    if training_lineage is not None:
        if len(training_lineage) != n:
            raise ValueError('LINEAGE_LENGTH_MISMATCH')
        receipt = authorize_design(training_lineage,
                                   feedback_used_for_design=feedback_used_for_design)
        if [x.world_id for x in training_lineage] != list(world_ids):
            raise ValueError('LINEAGE_WORLD_ID_MISMATCH')
        if environment_families is not None and [x.environment_family for x in training_lineage] != list(environment_families):
            raise ValueError('LINEAGE_ENVIRONMENT_MISMATCH')
    gates = gate if gate is not None else TaskGraphGate()
    policy = acceptance_policy if acceptance_policy is not None else TaskGraphPolicy()
    if not isinstance(gates, TaskGraphGate) or not isinstance(policy, TaskGraphPolicy):
        raise ValueError('INVALID_GRAPH_POLICY_TYPE')
    specs = _specs(candidate_specs)
    y = np.asarray(labels, dtype=int)
    if not np.array_equal(y, np.asarray(labels)) or set(y.tolist()) != set(range(len(ontology.labels))):
        raise ValueError('FULL_ONTOLOGY_CLASS_SUPPORT_REQUIRED')
    # Physical-family grouping supersedes arbitrary split/world names when the
    # caller supplies an independently reviewed lineage manifest.
    g = np.asarray([x.generation_family_id for x in training_lineage]
                   if training_lineage is not None else world_ids)
    if len(set(g)) < policy.min_independent_worlds:
        raise ValueError('INSUFFICIENT_INDEPENDENT_WORLDS')
    arrays = [ObservationSpec('task-graph:' + domain, schema).array(obs) for obs in observations]
    features = tuple(dict.fromkeys(f for options in specs.values() for f, _ in options))
    matrix = {f: _transform(arrays, domain, f) for f in features}
    targets = _head_targets(ontology, y)
    outer_groups = (np.asarray(environment_families)
                    if outer_split_mode == 'leave_environment_family_out' else g)
    if len(set(outer_groups)) < 4:
        raise ValueError('FOUR_INDEPENDENT_OUTER_GROUPS_REQUIRED')
    splits = list(GroupKFold(n_splits=4).split(np.zeros(n), groups=outer_groups))
    outer_oof = {h: np.full((n, 2), np.nan, dtype=float) for h in _HEADS}
    folds_audit = []
    for outer_id, (tr, va) in enumerate(splits):
        if set(g[tr]) & set(g[va]):
            raise AssertionError('GRAPH_OUTER_WORLD_LEAKAGE')
        if outer_split_mode == 'leave_environment_family_out' and (
                set(outer_groups[tr]) & set(outer_groups[va])):
            raise AssertionError('GRAPH_OUTER_ENVIRONMENT_LEAKAGE')
        details = {'outer_fold': outer_id,
                   'train_environments': sorted(set(outer_groups[tr])) if outer_split_mode == 'leave_environment_family_out' else None,
                   'heldout_environments': sorted(set(outer_groups[va])) if outer_split_mode == 'leave_environment_family_out' else None,
                   'train_worlds': len(set(g[tr])),
                   'heldout_worlds': len(set(g[va])), 'heldout_rows': len(va), 'head_selections': {}}
        for head_num, head in enumerate(_HEADS):
            try:
                choice, trials = _choose_head(head, specs[head], matrix, targets, y, g, tr,
                                              seed + head_num, environment_families,
                                              selection_mode)
                model = _fit_head(head, choice, matrix, targets, y, tr, seed + head_num)
                outer_oof[head][va] = _get_head_proba(model, matrix, choice, va)
                details['head_selections'][head] = {'candidate': list(choice), 'trials': trials}
            except Exception as e:
                audit = {'schema': 'phm.0.8.task-graph.v1', 'stage': 'DEVELOPMENT_ONLY',
                         'decision': 'REJECT_OUTER_HEAD_FAILURE', 'model_delivered': False,
                         'outer_fold_history': folds_audit + [details], 'failed_head': head,
                         'error': type(e).__name__ + ': ' + str(e),
                         'failed_head_trials': getattr(e, 'trials', []),
                         'not_deployment_qualified': True}
                raise TaskDesignRejected('GRAPH_HEAD_FIT_FAILED', audit) from e
        folds_audit.append(details)
    if any(not np.isfinite(v).all() for v in outer_oof.values()):
        raise AssertionError('INCOMPLETE_GRAPH_OUTER_OOF')
    metrics = _scores(y, g, ontology, *(outer_oof[h] for h in _HEADS), gates)
    acceptance = policy.evaluate(metrics)
    audit = {'schema': 'phm.0.8.task-graph.v1', 'stage': 'DEVELOPMENT_ONLY',
             'method': 'nested_outer_group_oof_with_environment_exclusive_inner_selection',
             'selection_mode': selection_mode,
             'outer_split_mode': outer_split_mode,
             'provenance_receipt': receipt,
             'independent_family_overlap_checked': receipt is not None,
             'evaluation_feedback_independence_unverified': True,
             'task_graph': _TASK_DAG,
             'ontology': list(ontology.labels), 'independent_training_worlds': len(set(g)),
             'unique_training_rows': n, 'group_fold_intersection': 0,
             'observation_fields': list(schema),
             'candidate_specs': {h: [list(s) for s in specs[h]] for h in _HEADS},
             'gate': vars(gates).copy(), 'acceptance_policy': vars(policy).copy(),
             'outer_fold_history': folds_audit, 'nested_outer_oof_metrics': metrics,
             'acceptance': acceptance, 'not_deployment_qualified': True,
             'predictor_uses_labels_world_ids_or_truth': False,
             'sealed_or_calibration_worlds_used': False,
             'limitations': ['five_class_two_component_two_mechanism_ontology',
                             'inner_design_validation_adaptively_reused_not_sealed',
                             'offline_complete_trajectories_only', 'cpu_opt_in_torch',
                             'training_world_screen_not_independent_release_qualification']}
    if not acceptance['eligible']:
        audit.update({'decision': 'REJECT_GRAPH_ACCEPTANCE', 'model_delivered': False})
        raise TaskDesignRejected('GRAPH_FAILS_STAGE_OR_END_TO_END_POLICY', audit)
    # Recompute final head choices on all design_train worlds. These final
    # selections are never assigned outer held-out predictions after the fact.
    all_rows = np.arange(n)
    final_selection = {}
    final_trials = {}
    fitted = {}
    try:
        for i, head in enumerate(_HEADS):
            choice, trials = _choose_head(head, specs[head], matrix, targets, y, g, all_rows,
                                          seed + i, environment_families, selection_mode)
            final_selection[head] = choice
            final_trials[head] = trials
            fitted[head] = _fit_head(head, choice, matrix, targets, y, all_rows, seed + i)
    except Exception as e:
        audit.update({'decision': 'REJECT_FINAL_HEAD_FIT', 'model_delivered': False,
                      'final_fit_error': type(e).__name__ + ': ' + str(e),
                      'failed_head_trials': getattr(e, 'trials', []),
                      'final_selection_trials': final_trials})
        raise TaskDesignRejected('GRAPH_FINAL_HEAD_FIT_FAILED', audit) from e
    audit.update({'decision': 'RESEARCH_GRAPH_SCREEN_ELIGIBLE', 'model_delivered': True,
                  'final_head_specs': {h: list(s) for h, s in final_selection.items()},
                  'final_selection_trials': final_trials})
    return TaskGraphArtifact(ontology, schema, fitted, final_selection, gates, audit)
