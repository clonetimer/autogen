"""R95 — the actual algorithm DESIGN loop, not another detector implementation.

Given R93 equipment/dataset/trial constraints, synthesize a bounded set of
algorithm+sensor-subset+hyperparameter designs, lock SOURCE TRAIN folds, and
compare *real provider executions* on held-out SOURCE groups. All labels,
including source internal OOF targets, stay in this research evaluator, not
in a design proposal or candidate.predict().

No external VALIDATION/CALIBRATION/FINAL ingestion is permitted. Scores here
cannot establish independent cross-industrial generalization or deployment.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import itertools
import json
import math

from .contracts import ContractRejected, compile_source_plan
from .torch_like_bridge import TorchLikeTabularCandidate, TABULAR_PROVIDER

SCHEMA = "phm.autogen.diagnostic-design-study.v1"
SOURCE_RESULT = "phm.autogen.source-oof-comparison.v1"
PARAMS = {
    "linear": ({"C": 1.0}, {"C": 0.1}),
    "tree": ({"max_depth": 5}, {"max_depth": 3}),
    "random_forest": ({"n_estimators": 40}, {"n_estimators": 20}),
    "isolation_forest": ({"n_estimators": 40}, {"n_estimators": 20}),
}
MAX_SOURCE_FITS = 64


class DesignRejected(ContractRejected):
    """A principled refusal, never a fabricated success score."""


def _fail(code):
    raise DesignRejected(code)


def _sha(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False).encode("utf-8")).hexdigest()


def _blueprint_channels(equipment, trial):
    """Deterministic declared-sensor search; no post-hoc feature selection."""
    channels = trial["input_channels"]
    items = [list(channels)]
    items.extend([[c] for c in channels])
    if len(channels) >= 3:
        items.extend([list(p) for p in itertools.combinations(channels, 2)])
    unique = []
    for item in items:
        if item not in unique:
            unique.append(item)
    return unique


def _proposals(trial, equipment):
    families = [f for f in trial["candidate_families"]
                if (trial["task_type"], f) in TABULAR_PROVIDER]
    not_yet_executable = [f for f in trial["candidate_families"]
                          if f not in families]
    if not families:
        return [], not_yet_executable
    sensor_sets = _blueprint_channels(equipment, trial)
    all_options = []
    # First compare all registered families with the exact same complete
    # input sensors and default budget, BEFORE any subset/hyperparameter
    # explorations; no one algorithm gets a privileged data representation.
    for family in families:
        all_options.append((family, sensor_sets[0], PARAMS[family][0]))
    # Additional source-only trials are fair alternatives, not hidden loops.
    for family in families:
        for sensors in sensor_sets[1:]:
            all_options.append((family, sensors, PARAMS[family][0]))
    for family in families:
        for sensors in sensor_sets:
            all_options.append((family, sensors, PARAMS[family][1]))
    proposals = []
    seen = set()
    for family, sensors, config in all_options:
        key = (family, tuple(sensors), tuple(sorted(config.items())))
        if key in seen:
            continue
        seen.add(key)
        proposal = {
            "design_id": "design_%02d" % (len(proposals) + 1),
            "algorithm_family": family,
            "input_channels": list(sensors),
            "hyperparameters": dict(config),
            "provider": "torch_like",
            "search_mechanism": "predeclared_family_sensor_subset_hyperparameter",
            "scientific_role": "source_only_exploratory",
        }
        proposals.append(proposal)
        if len(proposals) == trial["budget"]["max_trials"]:
            break
    return proposals, not_yet_executable


def _group_folds(dataset, trial):
    rows = dataset["records"]
    target = trial["validation_target"]
    if target == "future_same_device":
        # A single machine with healthy-only TRAIN cannot evaluate fault
        # recall/precision. Future-forward diagnostic scoring requires a
        # separate carefully predeclared temporal/event protocol.
        return [], "R95_TEMPORAL_OR_ANOMALY_REQUIRES_EVENT_SUPPORT"
    axis = ("physical_id" if target == "unseen_physical_device"
            else "environment_id")
    groups = defaultdict(list)
    for r in rows:
        groups[r[axis]].append(r)
    if target == "unseen_physical_device":
        classes = trial["labels"]
        independent_per_class = {c: {r["physical_id"] for r in rows
                                      if r["label"] == c} for c in classes}
        if any(len(v) < 2 for v in independent_per_class.values()):
            return [], "R95_INSUFFICIENT_INDEPENDENT_SOURCE_DEVICES_PER_CLASS"
    else:
        if len(groups) < 3:
            return [], "R95_AT_LEAST_THREE_SOURCE_ENVIRONMENTS_REQUIRED"
    if len(groups) < 3:
        return [], "R95_TOO_FEW_INDEPENDENT_SOURCE_GROUPS"
    folds = []
    for group_id in sorted(groups):
        held = groups[group_id]
        others = [r for r in rows if r[axis] != group_id]
        if set(r["label"] for r in others) != set(trial["labels"]):
            return [], "R95_SOURCE_FOLD_HAS_NO_CLASS_COVERAGE"
        if target == "unseen_environment" and set(
                r["label"] for r in held) != set(trial["labels"]):
            return [], "R95_SOURCE_ENVIRONMENT_MISSING_CLASS"
        # Group hashes in study receipts rather than exporting physical IDs.
        folds.append({
            "group_sha256": hashlib.sha256(
                (axis + ":" + group_id).encode("utf-8")).hexdigest(),
            "axis": axis,
            "fit_ids": [r["record_id"] for r in others],
            "score_ids": [r["record_id"] for r in held],
        })
    return folds, None


def propose_diagnostic_study(equipment, dataset_source_only, trial):
    """Pure research design generator: zero training and zero holdout access."""
    source = compile_source_plan(equipment, dataset_source_only, trial)
    candidates, unsupported = _proposals(trial, equipment)
    folds, reason = _group_folds(dataset_source_only, trial)
    if trial["task_type"] != "classification":
        reason = "R95_NO_SOURCE_FAULT_LABELS_FOR_ANOMALY_RANKING"
        folds = []
    if not candidates:
        reason = "R95_NO_IMPLEMENTED_PROVIDER_FOR_TASK"
    fit_cost = len(candidates) * len(folds)
    if fit_cost > MAX_SOURCE_FITS:
        reason = "R95_SOURCE_FIT_BUDGET_EXCEEDS_HARD_RESEARCH_CEILING"
        folds = []    # no partial exploration under over-budget evidence
    plan = {
        "schema": SCHEMA,
        "source_plan_sha256": _sha(source),
        "source_manifest_sha256": _sha(dataset_source_only),
        "trial_sha256": _sha(trial),
        "equipment_sha256": _sha(equipment),
        "seed": trial["budget"]["seed"],
        "task_type": trial["task_type"],
        "evaluation_target": trial["validation_target"],
        "candidate_count": len(candidates),
        "max_declared_candidate_trials": trial["budget"]["max_trials"],
        "source_oof_fit_calls_planned": len(candidates) * len(folds),
        "source_oof_fit_calls_ceiling": MAX_SOURCE_FITS,
        "candidate_blueprints": candidates,
        "unimplemented_candidate_families": unsupported,
        "folds": folds,
        "screening_refusal": reason,
        "training_already_run": False,
        "independent_validation_or_final_accessed": False,
        "source_data_origin_self_reported": dataset_source_only["origin"],
        "evidence_boundary": "TRAIN_INTERNAL_EXPLORATION_ONLY",
    }
    return {**plan, "study_sha256": _sha(plan)}


def _validate_plan(plan, equipment, train, trial):
    if not isinstance(plan, dict) or "study_sha256" not in plan:
        _fail("R95_INVALID_FROZEN_STUDY")
    # Recompute complete deterministic plan: catches mutated records, seed,
    # roles, provider mappings, hyperparameters, feature sets and folds.
    canonical = propose_diagnostic_study(equipment, train, trial)
    if plan != canonical or plan["study_sha256"] != canonical["study_sha256"]:
        _fail("R95_STUDY_OR_DATA_CHANGED_AFTER_FREEZE")
    return canonical


def _source_table(train, source_features, channels):
    if not isinstance(source_features, (tuple, list)):
        _fail("R95_INVALID_FEATURE_TABLE")
    expected = {r["record_id"] for r in train["records"]}
    if len(source_features) != len(expected):
        _fail("R95_SOURCE_FEATURE_COUNT_MISMATCH")
    lookup = {}
    all_channels = set(channels)
    for sample in source_features:
        if not isinstance(sample, dict) or set(sample) != {
                "record_id", "observations"}:
            _fail("R95_INVALID_SOURCE_FEATURE_RECORD")
        rid = sample["record_id"]
        if rid not in expected or rid in lookup:
            _fail("R95_UNKNOWN_OR_DUPLICATE_SOURCE_RECORD")
        observation = sample["observations"]
        if not isinstance(observation, dict) or set(observation) != all_channels:
            _fail("R95_SOURCE_FEATURE_SCHEMA_OR_TARGET_LEAK")
        if any(type(v) not in (int, float) or
               not math.isfinite(v) or abs(v) > 1e12
               for v in observation.values()):
            _fail("R95_BAD_NUMERIC_SOURCE_FEATURE")
        lookup[rid] = observation
    if set(lookup) != expected:
        _fail("R95_MISSING_SOURCE_FEATURE")
    return lookup


def execute_source_design_study(plan, equipment, source_train, trial,
                                source_features, *, provider=None):
    """Execute pre-registered SOURCE-ONLY grouped OOF, no protected roles.

    Isolated algorithm instances are created for each arm/fold. Only
    supervised SOURCE TRAIN labels are consulted by this evaluation
    steward; candidate.predict gets sensor-only records.
    """
    _validate_plan(plan, equipment, source_train, trial)
    # Verify features once BEFORE fitting or making a scoreboard.
    features = _source_table(source_train, source_features, trial["input_channels"])
    blocked = plan["screening_refusal"]
    if blocked:
        return {
            "schema": SOURCE_RESULT, "study_sha256": plan["study_sha256"],
            "status": "ABSTAIN_SOURCE_EVIDENCE_INSUFFICIENT",
            "reason": blocked, "evaluated_candidates": [],
            "provisional_source_shortlist": [],
            "source_fit_calls_actual": 0,
            "algorithm_generalization_proven": False,
            "calibration_validation_final_accessed": False,
            "production_qualified": False,
        }
    indexed = {r["record_id"]: r for r in source_train["records"]}
    results = []
    fits = 0
    for design in plan["candidate_blueprints"]:
        family = design["algorithm_family"]
        sensors = design["input_channels"]
        hyper = design["hyperparameters"]
        outcomes, costs = [], []
        for fold in plan["folds"]:
            fit_ids, score_ids = fold["fit_ids"], fold["score_ids"]
            source_subset = deepcopy(source_train)
            source_subset["records"] = [indexed[rid] for rid in fit_ids]
            child_trial = deepcopy(trial)
            child_trial["input_channels"] = list(sensors)
            child_trial["candidate_families"] = [family]
            training_features = [{
                "record_id": rid, "observations": {
                    channel: features[rid][channel] for channel in sensors}}
                for rid in fit_ids]
            inference = [{
                channel: features[rid][channel] for channel in sensors}
                for rid in score_ids]
            candidate = TorchLikeTabularCandidate(
                family, provider=provider, design_options=hyper)
            candidate.fit(equipment, source_subset, child_trial, training_features)
            predictions = candidate.predict(inference)
            if len(predictions) != len(score_ids):
                _fail("R95_INVALID_SOURCE_PREDICTION_COUNT")
            for rid, pred in zip(score_ids, predictions):
                if (not isinstance(pred, dict) or
                        pred.get("predicted_label") not in trial["labels"]):
                    _fail("R95_UNKNOWN_PREDICTED_CLASS")
                outcomes.append({
                    "group_sha256": fold["group_sha256"],
                    "source_true_class": indexed[rid]["label"],
                    "source_predicted_class": pred["predicted_label"],
                })
            costs.append(candidate.resource_receipt())
            fits += 1
        true = [x["source_true_class"] for x in outcomes]
        correct = [int(x["source_predicted_class"] == x["source_true_class"])
                   for x in outcomes]
        group = defaultdict(list)
        for item, accuracy in zip(outcomes, correct):
            group[item["group_sha256"]].append(accuracy)
        group_scores = [sum(v)/len(v) for v in group.values()]
        label_recall = {}
        for label in trial["labels"]:
            relevant = [correct[i] for i, value in enumerate(true) if value == label]
            label_recall[label] = (sum(relevant)/len(relevant)
                                   if relevant else None)
        healthy = trial["healthy_label"]
        healthy_samples = [x for x in outcomes if x["source_true_class"] == healthy]
        far = (sum(x["source_predicted_class"] != healthy
                   for x in healthy_samples) / len(healthy_samples)
               if healthy_samples else None)
        results.append({
            "design_id": design["design_id"],
            "algorithm_family": family,
            "input_channels": sensors,
            "hyperparameters": hyper,
            "source_oof_record_accuracy": sum(correct)/len(correct),
            "source_oof_group_macro_accuracy": sum(group_scores)/len(group_scores),
            "source_oof_worst_group_accuracy": min(group_scores),
            "source_oof_per_class_recall": label_recall,
            "source_oof_healthy_false_alarm_fraction": far,
            "source_oof_records_scored": len(outcomes),
            "source_oof_heldout_groups": len(group),
            "observed_train_fit_calls": len(costs),
            "observed_fit_wall_seconds_sum": sum(
                x["fit_wall_seconds_observed"] for x in costs),
            "matched_cpu_gpu_compute_proven": False,
        })
    ranking = sorted(results, key=lambda r: (
        -r["source_oof_worst_group_accuracy"],
        -r["source_oof_group_macro_accuracy"],
        r["source_oof_healthy_false_alarm_fraction"],
        r["observed_fit_wall_seconds_sum"],
        r["design_id"]))
    return {
        "schema": SOURCE_RESULT, "study_sha256": plan["study_sha256"],
        "status": "SOURCE_TRAIN_OOF_DESCRIPTIVE_SHORTLIST_ONLY",
        "reason": None,
        "evaluated_candidates": results,
        "provisional_source_shortlist": [x["design_id"] for x in ranking],
        "source_fit_calls_actual": fits,
        "source_fit_calls_planned": plan["source_oof_fit_calls_planned"],
        "metrics_source": "TRAIN_INTERNAL_GROUP_HOLDOUT_NOT_EXTERNAL_VALIDATION",
        "selected_design_not_independently_validated": True,
        "compute_budget_parity_proven": False,
        "algorithm_generalization_proven": False,
        "calibration_validation_final_accessed": False,
        "production_qualified": False,
    }
