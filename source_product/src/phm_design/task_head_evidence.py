"""R76 predeclared TRAIN-only evidence screens for multi-stage head search.

Reject candidate heads that hide missed faults inside average loss. Optional
fault-strength strata are *training evaluation metadata*, not features and
never an inference input. This is not independent domain qualification.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import math
import numpy as np


@dataclass(frozen=True)
class HeadEvidencePolicy:
    min_environment_families: int = 3
    min_per_class_per_environment: int = 2
    detector_fault_recall_min: float = 0.80
    detector_healthy_false_alarm_max: float = 0.10
    downstream_accuracy_min: float = 0.80
    downstream_worst_class_recall_min: float = 0.80
    require_fault_strength_strata: bool = False
    expected_fault_strata: tuple[str, ...] = ()
    min_per_fault_stratum: int = 2
    fault_stratum_recall_min: float = 0.80

    def __post_init__(self):
        for name in ('min_environment_families', 'min_per_class_per_environment',
                     'min_per_fault_stratum'):
            value = getattr(self, name)
            if type(value) is not int or value < (3 if name == 'min_environment_families' else 1):
                raise ValueError('INVALID_HEAD_EVIDENCE_POLICY:' + name)
        for name in ('detector_fault_recall_min', 'detector_healthy_false_alarm_max',
                     'downstream_accuracy_min', 'downstream_worst_class_recall_min',
                     'fault_stratum_recall_min'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or (
                not math.isfinite(value)) or not 0 <= value <= 1:
                raise ValueError('INVALID_HEAD_EVIDENCE_POLICY:' + name)
        if type(self.require_fault_strength_strata) is not bool:
            raise ValueError('INVALID_HEAD_EVIDENCE_POLICY:require_fault_strength_strata')
        if not isinstance(self.expected_fault_strata, tuple) or any(
                type(v) is not str or not v.strip() for v in self.expected_fault_strata
        ) or len(self.expected_fault_strata) != len(set(self.expected_fault_strata)):
            raise ValueError('INVALID_HEAD_EVIDENCE_POLICY:expected_fault_strata')
        if self.require_fault_strength_strata and not self.expected_fault_strata:
            raise ValueError('EXPLICIT_FAULT_STRENGTH_STRATA_REQUIRED')

    def evaluate(self, *, head: str, truth: Sequence[int], prediction: Sequence[int],
                 environments: Sequence[str], fault_strata: Sequence[str] | None = None) -> dict:
        if head not in ('detection', 'localization', 'mechanism'):
            raise ValueError('INVALID_HEAD')
        y = np.asarray(truth)
        pred = np.asarray(prediction)
        env = np.asarray(environments)
        if y.ndim != 1 or pred.ndim != 1 or env.ndim != 1 or (
                len(y) == 0 or len(y) != len(pred) or len(y) != len(env)):
            raise ValueError('HEAD_EVIDENCE_LENGTH_MISMATCH')
        if any(type(value) is not str or not value.strip() for value in env.tolist()):
            raise ValueError('INVALID_ENVIRONMENT_NAME')
        if not np.isin(y, (0, 1)).all() or not np.isin(pred, (0, 1)).all():
            raise ValueError('HEAD_EVIDENCE_BINARY_LABELS_REQUIRED')

        failed = []
        families = {}
        uniques = sorted(set(env.tolist()))
        if len(uniques) < self.min_environment_families:
            failed.append({'reason': 'INSUFFICIENT_ENVIRONMENT_FAMILIES',
                           'observed': len(uniques), 'required': self.min_environment_families})
        for family in uniques:
            mask = env == family
            healthy = mask & (y == 0)
            fault = mask & (y == 1)
            healthy_n, fault_n = int(healthy.sum()), int(fault.sum())
            evidence = {'healthy_support': healthy_n, 'fault_support': fault_n}
            if min(healthy_n, fault_n) < self.min_per_class_per_environment:
                failed.append({'reason': 'INSUFFICIENT_CLASS_SUPPORT', 'environment': family,
                               'observed': min(healthy_n, fault_n),
                               'required': self.min_per_class_per_environment})
            else:
                recall = float(np.mean(pred[fault] == 1))
                false_alarm = float(np.mean(pred[healthy] == 1))
                evidence.update({'positive_class_recall': recall,
                                 'negative_class_false_alarm_rate': false_alarm})
                if head == 'detection':
                    if recall < self.detector_fault_recall_min:
                        failed.append({'reason': 'ENVIRONMENT_FAULT_RECALL', 'environment': family,
                                       'observed': recall, 'required': self.detector_fault_recall_min})
                    if false_alarm > self.detector_healthy_false_alarm_max:
                        failed.append({'reason': 'ENVIRONMENT_HEALTHY_FALSE_ALARM', 'environment': family,
                                       'observed': false_alarm,
                                       'required': self.detector_healthy_false_alarm_max})
                else:
                    accuracy = float(np.mean(pred[mask] == y[mask]))
                    specificity = 1.0 - false_alarm
                    evidence.update({'accuracy': accuracy, 'worst_class_recall': min(recall, specificity)})
                    if accuracy < self.downstream_accuracy_min:
                        failed.append({'reason': 'ENVIRONMENT_CONDITIONAL_ACCURACY', 'environment': family,
                                       'observed': accuracy, 'required': self.downstream_accuracy_min})
                    if min(recall, specificity) < self.downstream_worst_class_recall_min:
                        failed.append({'reason': 'ENVIRONMENT_WORST_CLASS_RECALL', 'environment': family,
                                       'observed': min(recall, specificity),
                                       'required': self.downstream_worst_class_recall_min})
            families[family] = evidence

        by_stratum = {}
        if head == 'detection' and (self.require_fault_strength_strata or fault_strata is not None):
            if fault_strata is None:
                failed.append({'reason': 'FAULT_STRENGTH_STRATA_REQUIRED'})
            else:
                strata = np.asarray(fault_strata)
                if strata.ndim != 1 or len(strata) != len(y):
                    raise ValueError('FAULT_STRENGTH_STRATA_LENGTH_MISMATCH')
                if any(type(v) is not str or not v.strip() for v in strata.tolist()):
                    raise ValueError('INVALID_FAULT_STRENGTH_STRATUM')
                # Evaluate every explicitly declared fault stratum, not only
                # whichever subset gives the candidate its highest recall.
                found = set(strata[y == 1].tolist())
                if self.expected_fault_strata and not found.issubset(set(self.expected_fault_strata)):
                    failed.append({'reason': 'UNDECLARED_FAULT_STRENGTH_STRATA',
                                   'observed': sorted(found - set(self.expected_fault_strata))})
                for family in uniques:
                    expected = (self.expected_fault_strata or
                                tuple(sorted(set(strata[(env == family) & (y == 1)].tolist()))))
                    for level in expected:
                        mask = (env == family) & (y == 1) & (strata == level)
                        count = int(mask.sum())
                        key = family + ':' + level
                        result = {'support': count}
                        if count < self.min_per_fault_stratum:
                            failed.append({'reason': 'INSUFFICIENT_FAULT_STRATUM_SUPPORT',
                                           'environment': family, 'stratum': level,
                                           'observed': count, 'required': self.min_per_fault_stratum})
                        else:
                            recall = float(np.mean(pred[mask] == 1))
                            result['recall'] = recall
                            if recall < self.fault_stratum_recall_min:
                                failed.append({'reason': 'FAULT_STRATUM_RECALL',
                                               'environment': family, 'stratum': level,
                                               'observed': recall,
                                               'required': self.fault_stratum_recall_min})
                        by_stratum[key] = result
                if self.require_fault_strength_strata and not by_stratum:
                    failed.append({'reason': 'NO_FAULT_STRENGTH_STRATA'})
        return {'eligible': not failed, 'failed_gates': failed,
                'environment_support_and_metrics': families,
                'fault_strata': by_stratum,
                'evidence': 'TRAIN_ONLY_INNER_ENVIRONMENT_OOF',
                'not_independent_qualification': True}

    def evaluate_end_to_end_strata(self, *, truth: Sequence[int], decisions: Sequence[int],
                                  environments: Sequence[str], fault_strata: Sequence[str]) -> dict:
        """All-fault weak-stratum gate AFTER detector, locator and mechanism gates.

        UNKNOWN (-1) and false NOMINAL are failures. This uses only the
        design TRAIN outer-OOF predictions, never a revealed FINAL cohort.
        """
        y, p, env, strata = (np.asarray(truth), np.asarray(decisions),
                            np.asarray(environments), np.asarray(fault_strata))
        if (any(x.ndim != 1 for x in (y, p, env, strata)) or
                not len(y) or len({len(x) for x in (y, p, env, strata)}) != 1):
            raise ValueError('END_TO_END_STRATUM_LENGTH_MISMATCH')
        if not np.issubdtype(y.dtype, np.integer) or (y < 0).any():
            raise ValueError('INVALID_END_TO_END_TRUTH')
        if not np.issubdtype(p.dtype, np.integer) or (p < -1).any():
            raise ValueError('INVALID_END_TO_END_DECISION')
        if any(not isinstance(v, str) or not v.strip() for v in env.tolist()+strata.tolist()):
            raise ValueError('INVALID_END_TO_END_STRATUM_METADATA')
        levels = (self.expected_fault_strata or
                  tuple(sorted(set(strata[y != 0].tolist()))))
        failures, report = [], {}
        if not levels:
            failures.append({'reason': 'NO_FAULT_STRENGTH_STRATA'})
        observed = set(strata[y != 0].tolist())
        if self.expected_fault_strata and not observed.issubset(set(levels)):
            failures.append({'reason': 'UNDECLARED_FAULT_STRENGTH_STRATA',
                             'observed': sorted(observed - set(levels))})
        for family in sorted(set(env.tolist())):
            for level in levels:
                mask = (env == family) & (y != 0) & (strata == level)
                count = int(mask.sum())
                key = family + ':' + level
                row = {'fault_support': count}
                if count < self.min_per_fault_stratum:
                    failures.append({'reason': 'INSUFFICIENT_END_TO_END_STRATUM_SUPPORT',
                                     'environment': family, 'stratum': level,
                                     'observed': count, 'required': self.min_per_fault_stratum})
                else:
                    recall = float(np.mean(y[mask] == p[mask]))
                    row['joint_fault_recall'] = recall
                    if recall < self.fault_stratum_recall_min:
                        failures.append({'reason': 'END_TO_END_FAULT_STRATUM_RECALL',
                                         'environment': family, 'stratum': level,
                                         'observed': recall, 'required': self.fault_stratum_recall_min})
                report[key] = row
        return {'eligible': not failures, 'failed_gates': failures, 'per_environment_stratum': report,
                'evidence': 'TRAIN_ONLY_OUTER_ENVIRONMENT_OOF',
                'unknown_counts_as_failure': True, 'not_independent_qualification': True}
