"""Variable-ontology factorized diagnostic candidate for industrial AutoDesign.

This is an *alternative fitted architecture* tested against direct
classification in the same TRAIN-source OOF folds, not a post-hoc rule that
can use held-out predictions to tune class mapping. Structure is supplied
only in the predeclared TaskSpec; no simulator truth or environment metadata
enters deployment prediction.
"""
from __future__ import annotations

import numpy as np

from ..task_candidate_registry import TaskCandidateRegistry


class FactorizedDiagnosisModel:
    """Learn detection plus conditional component/mechanism heads.

    Supports any legally injective component × mechanism fault ontology with
    a designated healthy label. Does not assume five labels, two components
    or two mechanisms. Single-valued conditional heads are deterministic.
    """

    def __init__(self, *, feature, learner, objective, seed,
                 labels, healthy_label, class_attributes):
        self.feature = feature
        self.learner = learner
        self.objective = objective
        self.seed = int(seed)
        self.labels = tuple(labels)
        self.healthy_label = healthy_label
        self.healthy_index = self.labels.index(healthy_label)
        self.fault_labels = tuple(label for label in self.labels if label != healthy_label)
        self.attributes = {label: dict(class_attributes[label])
                           for label in self.fault_labels}
        if len(self.fault_labels) < 1 or any(
                'component' not in self.attributes[k] or
                'mechanism' not in self.attributes[k] for k in self.fault_labels):
            raise ValueError('FACTORIZED_REQUIRES_COMPONENT_AND_MECHANISM')
        pairs = [(self.attributes[label]['component'],
                  self.attributes[label]['mechanism']) for label in self.fault_labels]
        if len(set(pairs)) != len(pairs):
            raise ValueError('FACTORIZED_NONINJECTIVE_ONTOLOGY')
        self.components = tuple(sorted({x[0] for x in pairs}))
        self.mechanisms = tuple(sorted({x[1] for x in pairs}))
        self.classes_ = np.arange(len(self.labels), dtype=int)
        self.component_index = np.asarray(
            [self.components.index(self.attributes[x]['component'])
             for x in self.fault_labels], dtype=int)
        self.mechanism_index = np.asarray(
            [self.mechanisms.index(self.attributes[x]['mechanism'])
             for x in self.fault_labels], dtype=int)
        self.models_ = None
        self.fit_count_ = 0

    @property
    def planned_model_fits(self):
        return 1 + int(len(self.components) > 1) + int(len(self.mechanisms) > 1)

    def fit(self, x, y, *, domain_groups=None):
        y = np.asarray(y, dtype=int)
        if (y.ndim != 1 or len(y) != len(x)
                or set(y.tolist()) != set(self.classes_.tolist())):
            raise ValueError('FACTORIZED_MISSING_TRAIN_CLASS')
        fault = y != self.healthy_index
        if not np.any(fault) or not np.any(~fault):
            raise ValueError('FACTORIZED_REQUIRES_FAULT_AND_HEALTHY')
        domains = None
        if self.objective in ('GroupDRO', 'V-REx'):
            if domain_groups is None or len(domain_groups) != len(y):
                raise ValueError('FACTORIZED_DG_REQUIRES_TRAIN_ENVIRONMENTS')
            domains = np.asarray(domain_groups)
        models = {}
        self.fit_count_ = 0

        def fit_one(head, values, targets, selection, suffix, size):
            if size == 1:
                return None
            if set(targets.tolist()) != set(range(size)):
                raise ValueError('FACTORIZED_HEAD_CLASS_SUPPORT:' + head)
            estimator = TaskCandidateRegistry().build(
                self.feature, self.learner, self.seed + suffix)
            subset_domain = None if domains is None else domains[selection]
            if subset_domain is not None:
                estimator.fit(values, targets, domain_groups=subset_domain)
            else:
                estimator.fit(values, targets)
            actual = set(np.asarray(estimator.classes_).tolist())
            if actual != set(range(size)):
                raise ValueError('FACTORIZED_HEAD_ESTIMATOR_CLASSES:' + head)
            self.fit_count_ += 1
            return estimator

        selected = np.arange(len(y))
        models['detection'] = fit_one('detection', x,
                                     fault.astype(int), selected, 101, 2)
        faulty_indices = np.flatnonzero(fault)
        component_lookup = {name: i for i, name in enumerate(self.components)}
        mechanism_lookup = {name: i for i, name in enumerate(self.mechanisms)}
        fault_names = [self.labels[int(k)] for k in y[fault]]
        comp_y = np.asarray([component_lookup[self.attributes[name]['component']]
                             for name in fault_names], dtype=int)
        mech_y = np.asarray([mechanism_lookup[self.attributes[name]['mechanism']]
                             for name in fault_names], dtype=int)
        models['component'] = fit_one('component', x[fault], comp_y,
                                      faulty_indices, 202, len(self.components))
        models['mechanism'] = fit_one('mechanism', x[fault], mech_y,
                                      faulty_indices, 303, len(self.mechanisms))
        self.models_ = models
        self.training_ = {
            'objective': self.objective,
            'topology': 'factorized_detection_component_mechanism',
            'head_training': 'TRAIN_ONLY',
            'trained_head_models': self.fit_count_,
            'factorization_assumption': 'conditional_independence_of_component_and_mechanism',
            'environment_identity_is_not_deployment_feature': True,
            'independent_qualification': False,
        }
        return self

    @staticmethod
    def _head_probabilities(model, x, classes):
        if classes == 1:
            return np.ones((len(x), 1), dtype=float)
        raw = np.asarray(model.predict_proba(x), dtype=float)
        keys = np.asarray(model.classes_, dtype=int)
        if (raw.shape != (len(x), classes)
                or set(keys.tolist()) != set(range(classes))
                or not np.isfinite(raw).all() or (raw < 0).any()):
            raise ValueError('INVALID_FACTORIZED_HEAD_OUTPUT')
        ordered = raw[:, np.argsort(keys)]
        scale = ordered.sum(axis=1, keepdims=True)
        if (scale <= 0).any():
            raise ValueError('ZERO_FACTORIZED_HEAD_PROBABILITY')
        return ordered / scale

    def predict_proba(self, x):
        if self.models_ is None:
            raise ValueError('FACTORIZED_MODEL_NOT_FITTED')
        detected = self._head_probabilities(self.models_['detection'], x, 2)
        components = self._head_probabilities(
            self.models_['component'], x, len(self.components))
        mechanisms = self._head_probabilities(
            self.models_['mechanism'], x, len(self.mechanisms))
        joint = (components[:, self.component_index] *
                 mechanisms[:, self.mechanism_index])
        totals = joint.sum(axis=1, keepdims=True)
        if not np.isfinite(joint).all() or (totals <= 0).any():
            raise ValueError('NONIDENTIFIABLE_FACTORIZED_CLASS_PROBABILITIES')
        # Only legal fault pairs exist. Normalize their conditional mass,
        # reserving detection healthy-vs-fault mass exactly.
        joint = joint / totals
        result = np.zeros((len(x), len(self.labels)), dtype=float)
        result[:, self.healthy_index] = detected[:, 0]
        for position, label in enumerate(self.fault_labels):
            result[:, self.labels.index(label)] = detected[:, 1] * joint[:, position]
        if not np.allclose(result.sum(axis=1), 1., atol=1e-4):
            raise ValueError('FACTORIZED_PROBABILITY_MASS_LOST')
        return result
