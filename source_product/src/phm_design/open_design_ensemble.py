"""Group-cross-fitted learned composition of heterogeneous classifiers.

The meta learner never sees in-sample base predictions. Both inner and outer
validation must preserve physical groups. This is research selection, not an
independent probability calibration or a claim of diagnostic qualification.
"""
from __future__ import annotations
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.utils.validation import check_is_fitted, check_array, check_X_y


class GroupProbabilityEnsemble(ClassifierMixin, BaseEstimator):
    """Three heterogeneous bases and an optional learned group-OOF combiner.

    No physical failure threshold, label-encoded domain, or group ID is a
    predictor. Group IDs are used solely to construct training partitions.
    ``predict_average`` provides a controlled terminal-combiner ablation using
    exactly the same fitted bases. Its compute is shared, not a second budget-
    matched search run.
    """
    def __init__(self, strength=1.0, depth=5, seed=49019, combination='learned'):
        self.strength = strength
        self.depth = depth
        self.seed = seed
        self.combination = combination

    def _base(self, name):
        if name == 'linear':
            m = LogisticRegression(C=self.strength, max_iter=300, random_state=self.seed)
        elif name == 'extra':
            m = ExtraTreesClassifier(n_estimators=24, max_depth=self.depth,
                                     random_state=self.seed, n_jobs=1)
        elif name == 'gnb':
            m = GaussianNB(var_smoothing=1e-5)
        else:
            raise ValueError('ENSEMBLE_BASE_NOT_SUPPORTED')
        # Importantly, this scaler is fit within each inner training fold.
        return make_pipeline(StandardScaler(), m)

    @staticmethod
    def _fit_base(model, X, y, weights):
        name = list(model.named_steps)[-1]
        model.fit(X, y, **{name + '__sample_weight': weights})
        return model

    def _aligned(self, model, X):
        if not np.array_equal(model.classes_, self.classes_):
            raise ValueError('ENSEMBLE_CLASS_ALIGNMENT')
        p = model.predict_proba(X)
        if not np.isfinite(p).all():
            raise ValueError('ENSEMBLE_NONFINITE_PROBABILITY')
        return p

    @staticmethod
    def _meta_features(p):
        # Numerical clipping only, not a decision/rejection threshold.
        return np.log(np.clip(p, np.finfo(float).eps, 1.0))

    def fit(self, X, y, *, groups=None, sample_weight=None):
        X, y = check_X_y(X, y, dtype=float)
        self.n_features_in_ = X.shape[1]
        g = np.asarray(groups) if groups is not None else np.array([])
        if g.ndim != 1 or len(g) != len(y) or len(set(g)) < 3:
            raise ValueError('ENSEMBLE_REQUIRES_THREE_PHYSICAL_GROUPS')
        if self.combination not in ('learned', 'average'):
            raise ValueError('ENSEMBLE_COMBINATION')
        self.classes_ = np.unique(y)
        if len(self.classes_) < 2:
            raise ValueError('ENSEMBLE_REQUIRES_MULTIPLE_CLASSES')
        weights = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, float)
        if weights.shape != y.shape or not np.isfinite(weights).all() or np.any(weights <= 0):
            raise ValueError('ENSEMBLE_SAMPLE_WEIGHT')
        self.base_names_ = ('linear', 'extra', 'gnb')
        k = len(self.classes_)
        oof = np.full((len(y), k * len(self.base_names_)), np.nan)
        self.group_audit_ = []
        for tr, va in GroupKFold(3).split(X, y, g):
            if not np.array_equal(np.unique(y[tr]), self.classes_):
                raise ValueError('ENSEMBLE_INNER_FOLD_MISSING_CLASS')
            if set(g[tr]) & set(g[va]):
                raise ValueError('ENSEMBLE_GROUP_LEAKAGE')
            for j, name in enumerate(self.base_names_):
                model = self._fit_base(self._base(name), X[tr], y[tr], weights[tr])
                oof[va, j*k:(j+1)*k] = self._aligned(model, X[va])
            self.group_audit_.append({
                'base_train_groups': sorted(map(str, set(g[tr]))),
                'meta_oof_groups': sorted(map(str, set(g[va]))),
                'disjoint': True, 'scalers_fitted_inside_base_train': True,
                'base_train_count': int(len(tr)), 'meta_oof_count': int(len(va))})
        if not np.isfinite(oof).all():
            raise ValueError('ENSEMBLE_INCOMPLETE_OOF')
        self.meta_ = make_pipeline(StandardScaler(), LogisticRegression(
            C=self.strength, max_iter=300, random_state=self.seed))
        self.meta_.fit(self._meta_features(oof), y,
                       logisticregression__sample_weight=weights)
        self.bases_ = [self._fit_base(self._base(name), X, y, weights)
                       for name in self.base_names_]
        self.meta_training_source_ = 'strict_group_out_of_fold_predictions'
        return self

    def _base_predictions(self, X):
        check_is_fitted(self, 'bases_')
        X = check_array(X, dtype=float)
        if X.shape[1] != self.n_features_in_:
            raise ValueError('ENSEMBLE_FEATURE_DIMENSION')
        return np.concatenate([self._aligned(m, X) for m in self.bases_], axis=1)

    def predict_proba(self, X):
        p = self._base_predictions(X)
        if self.combination == 'average':
            return p.reshape(len(p), len(self.bases_), len(self.classes_)).mean(axis=1)
        return self.meta_.predict_proba(self._meta_features(p))

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    def predict_average(self, X):
        p = self._base_predictions(X).reshape(-1, len(self.bases_), len(self.classes_))
        return self.classes_[p.mean(axis=1).argmax(axis=1)]
