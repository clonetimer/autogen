"""R94 / P1.2 — unified bounded TRAIN-only candidate fit and frozen inference.

This is a deliberately SMALL functional executor, not general AutoML or a
claim of independent diagnostic quality. Only three implemented families:
constant baseline, sklearn binary logistic linear, robust statistical
normal-only anomaly. All use the same strict R93 observation inputs and
artifact contract; unimplemented R93 candidate families fail closed.

Validation and FINAL are not arguments to this module. Source TRAIN labels
are used to fit classifiers but never exposed to deployment inference.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import math
import statistics
from time import perf_counter
from typing import Protocol, runtime_checkable

from .contracts import (
    ContractRejected, compile_source_plan, validate_trial,
)

SCHEMA = "phm.autogen.candidate-artifact.v1"
RECEIPT = "phm.autogen.candidate-resource.v1"
FAMILIES = frozenset({"constant", "linear", "statistical"})
MODEL_FORMAT = "sha256-bound-numeric-json-no-pickle"
MAX_ABS_FEATURE = 1.0e12
SCORE_MEANING = {
    "constant": "none",
    "linear": "probability_of_second_class_not_calibrated",
    "statistical": "max_robust_deviation_from_healthy_reference",
}


class CandidateRefused(ContractRejected):
    """Refusal does not establish that a diagnostic task is unsolvable."""


def _reject(reason):
    raise CandidateRefused(reason)


def _canonical(item):
    return json.dumps(item, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def _digest(item):
    return hashlib.sha256(_canonical(item).encode("utf-8")).hexdigest()


def _finite(v):
    return type(v) in (int, float) and math.isfinite(v) and abs(v) <= MAX_ABS_FEATURE


def _prediction_values(inputs, channels):
    if not isinstance(inputs, (list, tuple)) or not inputs:
        _reject("R94_EMPTY_INFERENCE_INPUT")
    expected = set(channels)
    vectors = []
    for row in inputs:
        if not isinstance(row, Mapping) or set(row) != expected:
            _reject("R94_PROTECTED_OR_UNDECLARED_INFERENCE_FIELD")
        values = [row[name] for name in channels]
        if not all(_finite(v) for v in values):
            _reject("R94_INVALID_NUMERICAL_INFERENCE_VALUE")
        vectors.append([float(v) for v in values])
    return vectors


def _read_train_features(train_manifest, observations, trial):
    """Exactly one precomputed feature row per TRAIN metadata record."""
    if not isinstance(observations, (list, tuple)):
        _reject("R94_INVALID_OBSERVATION_TABLE")
    expected = {row["record_id"]: row for row in train_manifest["records"]}
    if len(observations) != len(expected):
        _reject("R94_SOURCE_OBSERVATION_COUNT_MISMATCH")
    observed = {}
    for item in observations:
        if not isinstance(item, Mapping) or set(item) != {
                "record_id", "observations"}:
            _reject("R94_INVALID_SOURCE_OBSERVATION_SCHEMA")
        record_id = item["record_id"]
        if (type(record_id) is not str or
                record_id not in expected or record_id in observed):
            _reject("R94_UNKNOWN_OR_REPEATED_SOURCE_RECORD")
        observed[record_id] = item["observations"]
    channels = trial["input_channels"]
    ids = [record["record_id"] for record in train_manifest["records"]]
    features = _prediction_values([observed[k] for k in ids], channels)
    labels = [expected[k]["label"] for k in ids]
    return features, labels


def _train_constant(labels, trial):
    if trial["task_type"] == "anomaly_detection":
        selected = trial["healthy_label"]
    else:
        counts = Counter(labels)
        # Tie broken by predeclared task ontology, not model-output order.
        selected = max(trial["labels"],
                       key=lambda label: (counts[label], -trial["labels"].index(label)))
    return {"selected_class": selected, "class_counts": {
        label: labels.count(label) for label in trial["labels"]}}


def _train_linear(x, labels, trial):
    if trial["task_type"] != "classification":
        _reject("R94_LINEAR_ONLY_BINARY_CLASSIFICATION")
    if len(trial["labels"]) != 2 or set(labels) != set(trial["labels"]):
        _reject("R94_LINEAR_NEEDS_TWO_OBSERVED_TRAIN_CLASSES")
    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
    except ImportError as exc:
        raise CandidateRefused("R94_LINEAR_SCIKIT_LEARN_MISSING") from exc
    arr = np.asarray(x, dtype=np.float64)
    center = arr.mean(axis=0)
    scale = arr.std(axis=0)
    scale = np.where(scale < 1e-10, 1., scale)
    normalized = (arr - center) / scale
    algorithm = LogisticRegression(
        C=1., solver="lbfgs", max_iter=300,
        random_state=trial["budget"]["seed"],
    )
    algorithm.fit(normalized, labels)
    if algorithm.coef_.shape != (1, len(trial["input_channels"])):
        _reject("R94_UNEXPECTED_LINEAR_MODEL_SHAPE")
    return {
        "mean": center.tolist(),
        "scale": scale.tolist(),
        "coef": algorithm.coef_[0].tolist(),
        "intercept": float(algorithm.intercept_[0]),
        "classes": algorithm.classes_.tolist(),
        "converged": bool(int(algorithm.n_iter_[0]) < 300),
        "solver": "sklearn_LogisticRegression_lbfgs_C1_max_iter300",
    }


def _train_statistical(x, labels, trial):
    if trial["task_type"] != "anomaly_detection":
        _reject("R94_STATISTICAL_ONLY_NORMAL_TRAINING")
    if len(x) < 2 or any(label not in (None, trial["healthy_label"])
                         for label in labels):
        _reject("R94_STATISTICAL_NORMAL_SOURCE_REQUIRED")
    width = len(trial["input_channels"])
    centers, scales = [], []
    for j in range(width):
        column = [row[j] for row in x]
        median = statistics.median(column)
        mad = statistics.median([abs(v - median) for v in column])
        # MAD degeneracy fallback is source-only, no protected labels.
        sigma = max(1.4826 * mad, 1e-6)
        centers.append(float(median))
        scales.append(float(sigma))
    return {"median": centers, "robust_scale": scales,
            "fixed_deviation_threshold": 3.5,
            "threshold_selected_by_validation": False}


def _check_frozen_state(payload):
    """Shape/value checks before any portable JSON inference."""
    if not isinstance(payload, Mapping) or set(payload) != {
        "schema", "format", "family", "task_type", "input_channels",
        "labels", "healthy_label", "state", "source_plan_sha256",
        "training_provenance", "scope",
    }:
        _reject("R94_BAD_MODEL_PAYLOAD_SCHEMA")
    if payload["schema"] != SCHEMA or payload["format"] != MODEL_FORMAT:
        _reject("R94_UNSUPPORTED_FROZEN_MODEL")
    family = payload["family"]
    if family not in FAMILIES:
        _reject("R94_UNSUPPORTED_FAMILY")
    channels = payload["input_channels"]
    labels = payload["labels"]
    if (not isinstance(channels, list) or not channels or
            len(channels) != len(set(channels)) or
            not all(type(v) is str for v in channels) or
            not isinstance(labels, list) or len(labels) != 2 or
            not all(type(v) is str for v in labels) or
            len(labels) != len(set(labels)) or
            payload["healthy_label"] not in labels):
        _reject("R94_BAD_ARTIFACT_ONTOLOGY")
    if payload["task_type"] not in ("classification", "anomaly_detection"):
        _reject("R94_UNSUPPORTED_FROZEN_TASK")
    if type(payload["source_plan_sha256"]) is not str or len(payload["source_plan_sha256"]) != 64:
        _reject("R94_BAD_SOURCE_PLAN_DIGEST")
    if (payload["scope"] != {
            "research_only": True, "independent_model_validated": False,
            "industrial_production_approved": False}):
        _reject("R94_ARTIFACT_SCOPE_ESCALATION")
    state = payload["state"]
    if not isinstance(state, Mapping):
        _reject("R94_INVALID_FROZEN_STATE")
    width = len(channels)
    if family == "constant":
        if (set(state) != {"selected_class", "class_counts"} or
                state["selected_class"] not in labels or
                set(state["class_counts"]) != set(labels) or
                any(type(v) is not int or v < 0 for v in state["class_counts"].values())):
            _reject("R94_INVALID_CONSTANT_STATE")
    elif family == "linear":
        if (payload["task_type"] != "classification" or
                set(state) != {"mean", "scale", "coef", "intercept",
                               "classes", "converged", "solver"} or
                state["classes"] != sorted(labels) or
                state["solver"] != "sklearn_LogisticRegression_lbfgs_C1_max_iter300"):
            _reject("R94_INVALID_LINEAR_STATE")
        for name in ("mean", "scale", "coef"):
            if not isinstance(state[name], list) or len(state[name]) != width or not all(
                    _finite(v) for v in state[name]):
                _reject("R94_BAD_LINEAR_VECTOR")
        if (any(v <= 0 for v in state["scale"]) or
                not _finite(state["intercept"]) or
                type(state["converged"]) is not bool):
            _reject("R94_BAD_LINEAR_PARAMETERS")
    elif family == "statistical":
        if (payload["task_type"] != "anomaly_detection" or
                set(state) != {"median", "robust_scale", "fixed_deviation_threshold",
                               "threshold_selected_by_validation"} or
                state["fixed_deviation_threshold"] != 3.5 or
                state["threshold_selected_by_validation"] is not False):
            _reject("R94_INVALID_STATISTICAL_STATE")
        for name in ("median", "robust_scale"):
            if (not isinstance(state[name], list) or
                    len(state[name]) != width or not all(_finite(v) for v in state[name])):
                _reject("R94_BAD_STATISTICAL_VECTOR")
        if any(v <= 0 for v in state["robust_scale"]):
            _reject("R94_BAD_STATISTICAL_SCALE")
    return payload


@runtime_checkable
class CandidateInterface(Protocol):
    """Identical public methods across all P1.2 implemented candidates."""

    def fit(self, equipment: Mapping, train_manifest: Mapping,
            trial: Mapping, source_observations: Sequence[Mapping]):
        ...

    def predict(self, observations: Sequence[Mapping]) -> list[dict]:
        ...

    def export_model(self) -> dict:
        ...

    def resource_receipt(self) -> dict:
        ...


class CandidateAdapter:
    """Minimal, auditable candidate family adapter; no implicit split selection."""

    def __init__(self, family: str):
        if family not in FAMILIES:
            _reject("R94_UNIMPLEMENTED_EXECUTOR_FAMILY")
        self.family = family
        self._payload = None
        self._resource = None

    def fit(self, equipment, train_manifest, trial, source_observations):
        if self._payload is not None:
            _reject("R94_FITTED_ADAPTER_CANNOT_REFIT")
        source_plan = compile_source_plan(equipment, train_manifest, trial)
        if self.family not in trial["candidate_families"]:
            _reject("R94_FAMILY_NOT_PREDECLARED_IN_TASK")
        if len(trial["labels"]) != 2:
            _reject("R94_P12_ONLY_BINARY_TASKS")
        x, labels = _read_train_features(
            train_manifest, source_observations, trial)
        if self.family == "linear" and trial["task_type"] != "classification":
            _reject("R94_LINEAR_ONLY_BINARY_CLASSIFICATION")
        if self.family == "statistical" and trial["task_type"] != "anomaly_detection":
            _reject("R94_STATISTICAL_ONLY_NORMAL_TRAINING")
        t0 = perf_counter()
        if self.family == "constant":
            state = _train_constant(labels, trial)
        elif self.family == "linear":
            state = _train_linear(x, labels, trial)
        else:
            state = _train_statistical(x, labels, trial)
        wall = perf_counter() - t0
        if wall > trial["budget"]["max_fit_seconds"]:
            _reject("R94_SOURCE_FIT_EXCEEDED_DECLARED_WALL_SECONDS")
        payload = {
            "schema": SCHEMA,
            "format": MODEL_FORMAT,
            "family": self.family,
            "task_type": trial["task_type"],
            "input_channels": list(trial["input_channels"]),
            "labels": list(trial["labels"]),
            "healthy_label": trial["healthy_label"],
            "state": state,
            "source_plan_sha256": _digest(source_plan),
            "training_provenance": {
                "equipment_sha256": source_plan["equipment_sha256"],
                "dataset_train_sha256": source_plan["dataset_train_sha256"],
                "trial_sha256": source_plan["trial_sha256"],
                "source_group_count": source_plan["source_group_count"],
                "source_record_count": len(x),
                "source_data_origin_self_reported": train_manifest["origin"],
                "protected_split_read": False,
            },
            "scope": {
                "research_only": True,
                "independent_model_validated": False,
                "industrial_production_approved": False,
            },
        }
        _check_frozen_state(payload)
        artifact_bytes = len(_canonical(payload).encode("utf-8"))
        if artifact_bytes > trial["budget"]["max_memory_mb"] * 1024 * 1024:
            _reject("R94_SERIALIZED_MODEL_EXCEEDS_DECLARED_MEMORY_BUDGET")
        self._payload = payload
        self._resource = {
            "schema": RECEIPT, "candidate": self.family,
            "fit_calls": 1, "source_records": len(x),
            "source_physical_groups": source_plan["source_group_count"],
            "fit_elapsed_seconds_observed": wall,
            "frozen_model_bytes_observed": artifact_bytes,
            "peak_process_memory_bytes_measured": None,
            "training_flops_measured": None,
            "hard_preemptive_timeout_enforced": False,
            "max_fit_seconds_posthoc_checked": True,
            "max_serialized_model_bytes_checked": True,
            "candidate_trials_charged": 1,
            "computer_hardware_matched": False,
            "validation_or_final_accessed": False,
            "score_on_independent_real_equipment_obtained": False,
        }
        return self

    def _model(self):
        if self._payload is None:
            _reject("R94_MODEL_NOT_FITTED_OR_LOADED")
        return self._payload

    def predict(self, observations):
        p = _check_frozen_state(self._model())
        x = _prediction_values(observations, p["input_channels"])
        state, family = p["state"], p["family"]
        out = []
        for row in x:
            if family == "constant":
                label, value = state["selected_class"], None
            elif family == "linear":
                normalized = [(v - m) / s for v, m, s in
                              zip(row, state["mean"], state["scale"])]
                z = sum(a*b for a, b in zip(normalized, state["coef"])) + state["intercept"]
                probability = (1.0 / (1.0 + math.exp(-z))
                               if z >= 0 else math.exp(z) / (1.0 + math.exp(z)))
                label = state["classes"][1] if probability >= .5 else state["classes"][0]
                value = float(probability)
            else:
                value = float(max(abs(v-m) / scale for v, m, scale in
                                  zip(row, state["median"], state["robust_scale"])))
                label = (next(x for x in p["labels"] if x != p["healthy_label"])
                         if value > state["fixed_deviation_threshold"]
                         else p["healthy_label"])
            out.append({"predicted_label": label, "decision_score": value,
                        "score_semantics": SCORE_MEANING[family]})
        return out

    def export_model(self):
        payload = deepcopy(self._model())
        resource = self.resource_receipt()
        signed = {"payload": payload, "resource_receipt": resource}
        return {**signed, "sha256": _digest(signed)}

    def resource_receipt(self):
        if self._resource is None:
            _reject("R94_NO_TRAINING_RESOURCE_RECEIPT")
        return deepcopy(self._resource)


def load_model(artifact):
    """Portable inference using only SHA-bound numerical JSON; never pickle."""
    if not isinstance(artifact, Mapping) or set(artifact) != {
            "payload", "resource_receipt", "sha256"}:
        _reject("R94_INVALID_EXPORTED_ARTIFACT")
    bound = {"payload": artifact["payload"],
             "resource_receipt": artifact["resource_receipt"]}
    if type(artifact["sha256"]) is not str or artifact["sha256"] != _digest(bound):
        _reject("R94_FROZEN_ARTIFACT_HASH_MISMATCH")
    payload = deepcopy(_check_frozen_state(artifact["payload"]))
    resource = artifact["resource_receipt"]
    if (not isinstance(resource, Mapping) or
            set(resource) != {
                "schema", "candidate", "fit_calls", "source_records",
                "source_physical_groups", "fit_elapsed_seconds_observed",
                "frozen_model_bytes_observed", "peak_process_memory_bytes_measured",
                "training_flops_measured", "hard_preemptive_timeout_enforced",
                "max_fit_seconds_posthoc_checked", "max_serialized_model_bytes_checked",
                "candidate_trials_charged", "computer_hardware_matched",
                "validation_or_final_accessed",
                "score_on_independent_real_equipment_obtained",
            } or resource["schema"] != RECEIPT or
            resource["candidate"] != payload["family"] or
            resource["fit_calls"] != 1 or resource["candidate_trials_charged"] != 1 or
            type(resource["source_records"]) is not int or resource["source_records"] < 1 or
            type(resource["source_physical_groups"]) is not int or
            resource["source_physical_groups"] < 1 or
            not _finite(resource["fit_elapsed_seconds_observed"]) or
            not _finite(resource["frozen_model_bytes_observed"]) or
            resource["frozen_model_bytes_observed"] <= 0 or
            resource["hard_preemptive_timeout_enforced"] is not False or
            resource["max_fit_seconds_posthoc_checked"] is not True or
            resource["max_serialized_model_bytes_checked"] is not True or
            resource["computer_hardware_matched"] is not False or
            resource["validation_or_final_accessed"] is not False or
            resource["score_on_independent_real_equipment_obtained"] is not False or
            resource["peak_process_memory_bytes_measured"] is not None or
            resource["training_flops_measured"] is not None):
        _reject("R94_UNTRUSTED_OR_ESCALATED_RESOURCE_RECEIPT")
    model = CandidateAdapter(payload["family"])
    model._payload = payload
    model._resource = deepcopy(resource)
    return model
