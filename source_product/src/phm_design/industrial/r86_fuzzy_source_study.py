"""R86 TRAIN-only cross-environment neuro-fuzzy source comparison.

Fixed Gaussian TSK rules versus supervised residual-grown Gaussian TSK rules,
plus non-fuzzy linear reference. No independent TEST/FINAL score or calibrated
alarm-delay claim. All representations use deployment-observed sequence data.
"""
from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import Mapping, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .design_engine import DesignAcceptance, _observations_digest, _sha
from .r86_neuro_fuzzy import FuzzyRuleClassifier, FuzzyRulePolicy, sequence_summary
from .taskspec import TaskSpec, TaskSpecRejected, compile_task_spec, validate_observations
from ..task_candidate_registry import deployment_sequence


def _metric(y, prediction, environment, healthy_idx, nclasses):
    recall = []
    for family in sorted(set(environment.tolist())):
        mask = environment == family
        for label in range(nclasses):
            support = mask & (y == label)
            if not support.any():
                raise TaskSpecRejected("R86_EMPTY_ENVIRONMENT_CLASS_CELL")
            recall.append(float(np.mean(prediction[support] == label)))
    healthy = y == healthy_idx
    return {
        "accuracy": float(np.mean(prediction == y)),
        "worst_environment_class_recall": min(recall),
        "healthy_false_alarm": float(np.mean(prediction[healthy] != healthy_idx)),
    }


def run_fuzzy_source_study(task_spec: Mapping | TaskSpec,
                           records: Sequence[Mapping], *,
                           policy: FuzzyRulePolicy = FuzzyRulePolicy(),
                           acceptance: DesignAcceptance = DesignAcceptance(),
                           include_learned_premise: bool = False):
    """TRAIN-only LOEO protocol. No test data or model artifact is emitted."""
    if not isinstance(policy, FuzzyRulePolicy) or not isinstance(
            acceptance, DesignAcceptance):
        raise ValueError("R86_INVALID_COMPARISON_POLICY")
    if type(include_learned_premise) is not bool:
        raise ValueError("R87_INVALID_PREMISE_ABLATION_SWITCH")
    if (include_learned_premise and policy.premise_steps == 0 or
            not include_learned_premise and policy.premise_steps != 0):
        raise ValueError("R87_EXPLICIT_PREMISE_TUNING_ABLATION_REQUIRED")
    if not isinstance(records, (list, tuple)) or not records:
        raise TaskSpecRejected("R86_EMPTY_TRAIN_SOURCE")
    raw = (task_spec.to_dict() if isinstance(task_spec, TaskSpec)
           else TaskSpec.from_dict(task_spec).to_dict())
    if raw["task"]["decision_mode"] != "offline_trajectory":
        raise TaskSpecRejected("R86_OFFLINE_DIAGNOSIS_ONLY")
    matrices, metadata = [], []
    fields = [raw["observations"]["time"]["field"]] + [
        c["name"] for c in raw["observations"]["channels"]]
    for row in records:
        if not isinstance(row, Mapping) or set(row) != {
                "measurements", "metadata"} or not isinstance(
                    row["metadata"], Mapping):
            raise TaskSpecRejected("R86_INVALID_TRAIN_RECORD")
        validate_observations(raw, row["measurements"])
        metadata.append(dict(row["metadata"]))
        matrices.append(np.asarray([
            [point[field] for field in fields] for point in row["measurements"]],
            dtype=np.float64))
    plan = compile_task_spec(raw, train_manifest=metadata)
    labels = list(plan["ontology"]["labels"])
    nclasses = len(labels)
    lab_to_idx = {name: i for i, name in enumerate(labels)}
    y = np.asarray([lab_to_idx[m["label"]] for m in metadata], dtype=int)
    groups = np.asarray([
        "|".join(m[name] for name in plan["train_data_contract"]["group_keys"])
        for m in metadata])
    env = np.asarray([
        "|".join(m[name] for name in plan["train_data_contract"]["environment_axes"])
        for m in metadata])
    if len(set(env.tolist())) < 3:
        raise TaskSpecRejected("R86_INSUFFICIENT_SOURCE_ENVIRONMENTS")
    seen = {}
    for m, group in zip(matrices, groups):
        digest = hashlib.sha256(np.ascontiguousarray(m, dtype="<f8").tobytes()).hexdigest()
        old_group = seen.setdefault(digest, group)
        if old_group != group:
            raise TaskSpecRejected("R86_DUPLICATE_TRAJECTORY_ACROSS_PHYSICAL_GROUPS")
    folds = list(LeaveOneGroupOut().split(y, groups=env))
    if any(set(groups[train]) & set(groups[test]) for train, test in folds):
        raise TaskSpecRejected("R86_GROUP_CROSSES_ENVIRONMENT")
    sequence = np.stack([deployment_sequence(m) for m in matrices])
    if len(sequence) < policy.max_rules + 2:
        raise TaskSpecRejected("R86_INSUFFICIENT_PHYSICAL_EXAMPLES")
    fixed_premise_policy = replace(policy, premise_steps=0)
    fuzzy_arms = [
        ("fixed_fuzzy", "fixed", fixed_premise_policy),
        ("adaptive_fuzzy", "adaptive", fixed_premise_policy),
    ]
    if include_learned_premise:
        fuzzy_arms.extend([
            ("fixed_fuzzy_learned_premise", "fixed", policy),
            ("adaptive_fuzzy_learned_premise", "adaptive", policy),
        ])
    predictions = {arm: np.full(len(y), -1, dtype=int)
                   for arm in ("linear_reference",) +
                   tuple(name for name, _, _ in fuzzy_arms)}
    fits = {arm: 0 for arm in predictions}
    ridge_refits = {arm: 0 for arm in predictions}
    rule_traces = {name: [] for name, _, _ in fuzzy_arms}
    for train, holdout in folds:
        if len(set(y[train])) != nclasses:
            raise TaskSpecRejected("R86_TRAIN_FOLD_MISSING_CLASS")
        if any(int(np.sum(y[train] == cls)) < 2 for cls in range(nclasses)):
            raise TaskSpecRejected("R86_TRAIN_FOLD_INSUFFICIENT_CLASS_GROUPS")
        # All three arms see the exact same partition and observation summary.
        features = sequence_summary(sequence)
        baseline = make_pipeline(
            StandardScaler(), LogisticRegression(C=1, max_iter=600))
        baseline.fit(features[train], y[train])
        predictions["linear_reference"][holdout] = baseline.predict(features[holdout])
        fits["linear_reference"] += 1
        for arm, mode, arm_policy in fuzzy_arms:
            model = FuzzyRuleClassifier(arm_policy, mode=mode).fit(
                sequence[train], y[train])
            predictions[arm][holdout] = model.predict(sequence[holdout])
            fits[arm] += model.audit["candidate_fit_count"]
            ridge_refits[arm] += model.audit["total_consequent_ridge_refits"]
            rule_traces[arm].append({
                "fold_heldout_environment": sorted(set(env[holdout].tolist())),
                "selected_rule_count": model.audit["selected_rule_count"],
                "tail_rules_pruned": model.audit["tail_rules_pruned"],
                "candidate_fit_count": model.audit["candidate_fit_count"],
                "ridge_consequent_refit_count": model.audit[
                    "total_consequent_ridge_refits"],
                "premise_gradient_steps_accepted": model.audit[
                    "total_premise_updates_accepted"],
                "premise_tuned": bool(arm_policy.premise_steps),
                "parameter_count": model.audit["parameters_estimated"],
                "heldout_group_overlap": False,
                "source_train_only_rule_growth": True,
            })
    scores = {arm: {
        "source_train_environment_oof": _metric(
            y, pred, env, labels.index(plan["ontology"]["healthy_label"]), nclasses),
        "accepted_by_train_oof_screen": None,
        "candidate_fits": fits[arm],
        "consequent_ridge_refits": ridge_refits[arm],
        "estimated_model_parameters_by_fold": (
            [item["parameter_count"] for item in rule_traces[arm]]
            if arm in rule_traces else None),
        "fold_rule_trace": rule_traces.get(arm, []),
    } for arm, pred in predictions.items()}
    for arm in scores:
        scores[arm]["accepted_by_train_oof_screen"] = acceptance.evaluate(
            scores[arm]["source_train_environment_oof"])["eligible"]
    return {
        "schema": "phm.r86.fuzzy-source-comparison.v1",
        "task_spec_sha256": TaskSpec.from_dict(raw).sha256,
        "train_manifest_sha256": _sha(metadata),
        "train_observations_sha256": _observations_digest(matrices),
        "equipment_family": raw["equipment"]["family"],
        "source_data_role": "TRAIN_ONLY",
        "environment_count": len(folds),
        "physical_group_count": len(set(groups.tolist())),
        "same_folds_all_arms": True,
        "four_cell_rule_premise_ablation": include_learned_premise,
        "fuzzy_candidate_fits_equal_across_all_cells": all(
            fits[name] == policy.max_rules * len(folds)
            for name, _, _ in fuzzy_arms),
        "ridge_refit_count_equal_across_premise_conditions": (
            len(set(ridge_refits[name] for name, _, _ in fuzzy_arms)) == 1),
        "premise_gradient_budget_per_rule_candidate": (
            policy.premise_steps if include_learned_premise else 0),
        "fixed_adaptive_equal_candidate_fit_count": (
            fits["fixed_fuzzy"] == fits["adaptive_fuzzy"] ==
            policy.max_rules * len(folds)),
        "three_arms_complete": all((pred >= 0).all() for pred in predictions.values()),
        "policy": {"max_rules": policy.max_rules, "ridge": policy.ridge,
                   "complexity_penalty": policy.complexity_penalty,
                   "premise_steps": policy.premise_steps,
                   "premise_learning_rate": policy.premise_learning_rate},
        "arms": scores,
        "compute_flops_or_runtime_equalized": False,
        "linear_reference_equal_fit_count": False,
        "independent_validation_performed": False,
        "blind_test_performed": False,
        "source_oof_is_unbiased_generalization": False,
        "fuzzy_mechanism_generalization_benefit_proven": False,
        "eligible_for_stable": False,
    }
