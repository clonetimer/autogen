"""R86 bounded Takagi-Sugeno neuro-fuzzy diagnostic research prototype.

Gaussian antecedent membership, normalized rule firing and linear TSK
consequents, trained using ridge regression. Deterministic supervised
TRAIN-residual rule growth and optional tail-pruning. NOT a full gradient
ANFIS, not independently calibrated and not a flight-ready detector.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math

import numpy as np

from .r87_premise_learning import (
    gaussian_firing, fuzzy_design, refine_premises)


@dataclass(frozen=True)
class FuzzyRulePolicy:
    max_rules: int = 4
    ridge: float = 0.05
    complexity_penalty: float = 0.012
    premise_steps: int = 0
    premise_learning_rate: float = 0.1

    def __post_init__(self):
        if (type(self.max_rules) is not int or not 2 <= self.max_rules <= 8
            or type(self.ridge) not in (float, int) or
            not math.isfinite(self.ridge) or not 1e-6 <= self.ridge <= 10
            or type(self.complexity_penalty) not in (float, int) or
            not math.isfinite(self.complexity_penalty) or
            not 0 <= self.complexity_penalty <= 1 or
            type(self.premise_steps) is not int or
            not 0 <= self.premise_steps <= 16 or
            type(self.premise_learning_rate) not in (float, int) or
            not math.isfinite(self.premise_learning_rate) or
            not .0001 <= self.premise_learning_rate <= .5):
            raise ValueError("INVALID_R86_FUZZY_POLICY")


def sequence_summary(x):
    """Only observed temporal channels, no fault truth or cohort metadata."""
    x = np.asarray(x, dtype=np.float64)
    if (x.ndim != 3 or not 1 <= len(x) <= 100_000
        or not 8 <= x.shape[1] <= 512 or not 1 <= x.shape[2] <= 16
        or not np.isfinite(x).all()):
        raise ValueError("INVALID_R86_OBSERVED_SEQUENCE")
    return np.concatenate((np.mean(x, axis=1), np.std(x, axis=1),
                           x[:, -1] - x[:, 0]), axis=1)


def _activation(x, centers, log_widths=None):
    widths = np.zeros_like(centers) if log_widths is None else log_widths
    return gaussian_firing(x, centers, widths)


def _design(x, centers, log_widths=None):
    widths = np.zeros_like(centers) if log_widths is None else log_widths
    return fuzzy_design(x, centers, widths)


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / exp.sum(axis=1, keepdims=True)


class FuzzyRuleClassifier:
    """Opt-in research model with fixed or adaptive TRAIN-only fuzzy rules."""

    def __init__(self, policy: FuzzyRulePolicy = FuzzyRulePolicy(),
                 *, mode: str = "adaptive"):
        if not isinstance(policy, FuzzyRulePolicy):
            raise ValueError("INVALID_R86_FUZZY_POLICY_TYPE")
        if mode not in ("fixed", "adaptive"):
            raise ValueError("INVALID_R86_RULE_MODE")
        self.policy = policy
        self.mode = mode
        self.classes_ = None
        self._centers = None
        self._log_widths = None
        self._coef = None
        self._mean = None
        self._scale = None
        self.audit = None

    def _fit_consequents(self, x, yi, centers, nclasses, log_widths=None):
        z = _design(x, centers, log_widths)
        y = np.eye(nclasses)[yi]
        regularizer = np.eye(z.shape[1]) * self.policy.ridge
        coefs = np.linalg.solve(z.T @ z + regularizer, z.T @ y)
        probs = _softmax(z @ coefs)
        loss = float(-np.log(np.maximum(probs[np.arange(len(yi)), yi],
                                       1e-12)).mean())
        return coefs, probs, loss

    def fit(self, x, y):
        raw = sequence_summary(x)
        target = np.asarray(y)
        if (target.ndim != 1 or len(target) != len(raw)
                or len(raw) < max(6, self.policy.max_rules + 2)):
            raise ValueError("INVALID_R86_TRAIN_LABELS")
        if target.dtype.kind not in "iu" or np.any(target < 0):
            raise ValueError("R86_NONINTEGER_OR_NEGATIVE_CLASSES")
        classes, yi = np.unique(target, return_inverse=True)
        if len(classes) < 2 or len(classes) > 32:
            raise ValueError("R86_INSUFFICIENT_CLASS_SUPPORT")
        if (np.min(np.bincount(yi)) < 2):
            raise ValueError("R86_INSUFFICIENT_CLASS_EXAMPLES")
        mean = raw.mean(axis=0)
        std = np.maximum(raw.std(axis=0), 1e-4)
        norm = np.clip((raw - mean) / std, -20., 20.)
        centers = [np.median(norm, axis=0)]
        rounds = []
        selected = None
        for index in range(self.policy.max_rules):
            c = np.stack(centers)
            coef, probs, loss = self._fit_consequents(norm, yi, c, len(classes))
            log_widths = np.zeros_like(c)
            accepted_steps = 0
            refit_count = 1
            if self.policy.premise_steps:
                refined = refine_premises(
                    norm, yi, c, ridge=self.policy.ridge,
                    steps=self.policy.premise_steps,
                    learning_rate=self.policy.premise_learning_rate)
                c, log_widths, coef, probs, loss = (
                    refined["centers"], refined["log_widths"],
                    refined["coef"], refined["probabilities"],
                    refined["training_loss"])
                accepted_steps = refined["premise_steps_accepted"]
                refit_count = refined["consequent_refits"]
            score = loss + self.policy.complexity_penalty * len(centers)
            rounds.append({"rule_count": len(centers),
                           "train_cross_entropy": loss,
                           "train_penalized_objective": score,
                           "premise_updates_accepted": accepted_steps,
                           "ridge_consequent_refits": refit_count,
                           "source_feedback_only": True})
            if (selected is None or
                    (self.mode == "adaptive" and
                     score < selected["score"] - 1e-12) or
                    self.mode == "fixed"):
                selected = {"centers": c.copy(), "coef": coef.copy(),
                            "log_widths": log_widths.copy(),
                            "score": score, "rule_count": len(centers)}
            if index + 1 == self.policy.max_rules:
                continue
            dist = np.min(np.mean(
                (norm[:, None, :] - c[None, :, :]) ** 2, axis=2), axis=1)
            if self.mode == "adaptive":
                residual = 1.0 - probs[np.arange(len(yi)), yi]
                priority = residual * np.sqrt(np.maximum(dist, 0))
            else:
                priority = dist
            # Avoid exact repeats; ties resolve by first source TRAIN index.
            priority = np.where(dist > 1e-8, priority, -np.inf)
            next_index = int(np.argmax(priority))
            if not math.isfinite(float(priority[next_index])):
                # All identical source observations: reuse no false "new rule".
                # The run is not capable of fitting the declared rule budget.
                raise ValueError("R86_INSUFFICIENT_RULE_DIVERSITY")
            centers.append(norm[next_index].copy())
        self.classes_ = classes
        self._centers = selected["centers"]
        self._log_widths = selected["log_widths"]
        self._coef = selected["coef"]
        self._mean = mean
        self._scale = std
        nfeatures = len(mean)
        premise_width_parameters = (
            len(self._centers) * nfeatures if self.policy.premise_steps else 0)
        parameter_count = (len(self._centers) * nfeatures +
                           premise_width_parameters +
                           len(self._coef) * len(classes))
        self.audit = {
            "schema": "phm.r86.neuro-fuzzy-train.v1",
            "model_family": "GAUSSIAN_MEMBERSHIP_NORMALIZED_TSK_RIDGE",
            "rule_antecedent_learning": (
                "TRAIN_RESIDUAL_GUIDED_CENTER_GROWTH" if self.mode == "adaptive"
                else "UNSUPERVISED_FARTHEST_CENTER_GROWTH"),
            "consequent_learning": "TRAIN_ONLY_RIDGE",
            "premise_gradient_optimization_implemented": self.policy.premise_steps > 0,
            "premise_steps_budget_per_candidate": self.policy.premise_steps,
            "premise_learning_rate": self.policy.premise_learning_rate,
            "total_consequent_ridge_refits": sum(
                row["ridge_consequent_refits"] for row in rounds),
            "total_premise_updates_accepted": sum(
                row["premise_updates_accepted"] for row in rounds),
            "mode": self.mode, "max_rule_budget": self.policy.max_rules,
            "candidate_fit_count": len(rounds),
            "selected_rule_count": selected["rule_count"],
            "tail_rules_pruned": self.policy.max_rules-selected["rule_count"],
            "parameters_estimated": parameter_count,
            "learnable_width_parameter_count": premise_width_parameters,
            "selected_center_parameter_count": len(self._centers) * nfeatures,
            "rule_search_feedback_role": "TRAIN_FIT_ONLY",
            "fold_local_preprocessing": True,
            "source_train_learning_only": True,
            "independent_validation_performed": False,
            "independent_qualification": False,
            "training_trace": rounds,
        }
        return self

    def predict_proba(self, x):
        if self._centers is None:
            raise ValueError("R86_MODEL_NOT_FITTED")
        raw = sequence_summary(x)
        if raw.shape[1] != len(self._mean):
            raise ValueError("R86_OBSERVATION_CHANNEL_MISMATCH")
        normalized = np.clip((raw - self._mean) / self._scale, -20., 20.)
        return _softmax(_design(
            normalized, self._centers, self._log_widths) @ self._coef)

    def predict(self, x):
        return self.classes_[self.predict_proba(x).argmax(axis=1)]

    def explain_rules(self, x):
        if self._centers is None:
            raise ValueError("R86_MODEL_NOT_FITTED")
        raw = sequence_summary(x)
        if raw.shape[1] != len(self._mean):
            raise ValueError("R86_OBSERVATION_CHANNEL_MISMATCH")
        norm = np.clip((raw - self._mean) / self._scale, -20., 20.)
        fire = _activation(norm, self._centers, self._log_widths)
        return [{"dominant_rule_index": int(np.argmax(a)),
                 "rule_firing_strengths": [float(t) for t in a],
                 "independent_qualification": False} for a in fire]

    def source_receipt_sha256(self):
        if self.audit is None:
            raise ValueError("R86_MODEL_NOT_FITTED")
        payload = {"policy": asdict(self.policy), "audit": self.audit}
        return hashlib.sha256(json.dumps(payload, sort_keys=True,
                           allow_nan=False, separators=(",", ":")
                           ).encode("utf-8")).hexdigest()
