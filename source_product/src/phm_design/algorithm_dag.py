"""High-level diagnostic Algorithm-DAG synthesis.

This module searches *decision topology* above an already-trained score model and
structured mechanism signals.  It deliberately does not train on holdout data and
it does not infer physics from free-form text.  Ground-truth labels/locations are
accepted only by the search/evaluation functions; the runtime executor has no
truth inputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable
import itertools
import hashlib
import json
import math

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.tree import DecisionTreeClassifier

SCHEMA = "phm.algorithm-dag.v1"
SEARCH_SCHEMA = "phm.algorithm-dag-search.v1"
REPORT_SCHEMA = "phm.algorithm-dag-search-report.v1"


@dataclass(frozen=True)
class RuntimeSignals:
    """Inputs available at diagnosis time only.

    ``mechanism_values`` is the backward-compatible default mechanism statistic.
    ``mechanism_banks`` may expose additional causal statistics derived from the
    same physical-program outputs (for example RMS or tail mean).  Search may
    choose among declared banks, but runtime execution never receives truth.
    """

    neural_scores: np.ndarray  # cases x classes
    mechanism_values: np.ndarray  # cases x components, default statistic
    mechanism_banks: dict[str, np.ndarray] | None = None


def _digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def _validate_scores(scores: np.ndarray, classes: list[str]) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[1] != len(classes) or not np.isfinite(scores).all():
        raise ValueError("ALGORITHM_DAG_SCORE_SHAPE")
    if np.any(scores < 0):
        raise ValueError("ALGORITHM_DAG_NEGATIVE_SCORE")
    sums = scores.sum(1)
    if np.any(sums <= 0):
        raise ValueError("ALGORITHM_DAG_ZERO_SCORE_SUM")
    return scores / sums[:, None]


def _validate_mechanism(values: np.ndarray, n: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != n or values.shape[1] < 1 or not np.isfinite(values).all():
        raise ValueError("ALGORITHM_DAG_MECHANISM_SHAPE")
    return values


def _mechanism_source(signals: RuntimeSignals, source: str, n: int | None = None) -> np.ndarray:
    """Resolve a declared runtime mechanism statistic without truth access."""
    if source == "default":
        values = signals.mechanism_values
    else:
        banks = signals.mechanism_banks or {}
        if source not in banks:
            raise ValueError("ALGORITHM_DAG_MECHANISM_SOURCE_MISSING:" + str(source))
        values = banks[source]
    count = len(signals.neural_scores) if n is None else int(n)
    return _validate_mechanism(values, count)


def temporal_mechanism_banks(sequence: np.ndarray) -> dict[str, np.ndarray]:
    """Causal, truth-free summary banks for physical-program output sequences.

    All returned statistics preserve the mechanism signal's physical unit.  The
    function intentionally avoids label-dependent thresholds or onset alignment.
    The input is cases x time x components and the full observed window is used.
    """
    x = np.asarray(sequence, dtype=np.float64)
    if x.ndim != 3 or x.shape[0] < 1 or x.shape[1] < 4 or x.shape[2] < 1 or not np.isfinite(x).all():
        raise ValueError("ALGORITHM_DAG_MECHANISM_SEQUENCE_SHAPE")
    a = np.abs(x)
    half = max(1, x.shape[1] // 2)
    quarter = max(1, x.shape[1] // 4)
    head = a[:, :quarter]
    tail = a[:, -quarter:]
    rms = np.sqrt(np.mean(x * x, axis=1))
    max_abs = np.max(a, axis=1)
    roughness = np.mean(np.abs(np.diff(x, axis=1)), axis=1)
    diff_std = np.std(np.diff(a, axis=1), axis=1)
    late_mean = np.mean(tail, axis=1)
    # Dimensionless shape summaries complement the unit-preserving amplitude
    # banks. They are causal and truth-free, and therefore reusable across
    # physical domains rather than encoding ADCS fault names.
    eps = np.maximum(rms, 1e-12)
    time = np.linspace(-1.0, 1.0, x.shape[1], dtype=np.float64)
    centered = time - np.mean(time)
    slope = np.einsum("t,ntc->nc", centered, a) / max(float(np.dot(centered, centered)), 1e-12)
    return {
        "rms": rms,
        "p90_abs": np.quantile(a, 0.90, axis=1),
        "max_abs": max_abs,
        "std_abs": np.std(a, axis=1),
        "late_half_mean_abs": np.mean(a[:, -half:], axis=1),
        "late_quarter_mean_abs": late_mean,
        "roughness_abs_diff": roughness,
        "tail_head_delta_abs": np.abs(late_mean - np.mean(head, axis=1)),
        "crest_factor": max_abs / eps,
        "roughness_over_rms": roughness / eps,
        "diff_std_over_rms": diff_std / eps,
        "trend_over_rms": slope / eps,
        "tail_cv": np.std(tail, axis=1) / np.maximum(late_mean, 1e-12),
    }


def _node(op: str, inputs: list[str] | None = None, **params: Any) -> dict:
    out = {"op": op}
    if inputs:
        out["inputs"] = inputs
    if params:
        out["params"] = params
    return out


def make_candidate(*, classes: list[str], nominal_index: int, gate_source: str,
                   classifier: str, localizer: str, threshold: float | None,
                   classifier_thresholds: list[float] | None = None,
                   classifier_class_order: list[int] | None = None, fallback_margin: float | None = None,
                   classifier_mechanism_source: str = "default") -> dict:
    """Build a declarative decision DAG from a small, auditable grammar."""
    if not (0 <= nominal_index < len(classes)):
        raise ValueError("ALGORITHM_DAG_NOMINAL_INDEX")
    if gate_source not in {"none", "physical_max", "neural_fault_probability"}:
        raise ValueError("ALGORITHM_DAG_GATE_SOURCE")
    if classifier not in {"neural_closed_set", "neural_fault_only", "physical_scalar_bins", "physical_scalar_bins_neural_fallback"}:
        raise ValueError("ALGORITHM_DAG_CLASSIFIER")
    if localizer not in {"none", "physical_argmax"}:
        raise ValueError("ALGORITHM_DAG_LOCALIZER")
    if not isinstance(classifier_mechanism_source, str) or not classifier_mechanism_source:
        raise ValueError("ALGORITHM_DAG_CLASSIFIER_MECHANISM_SOURCE")
    if gate_source == "none" and threshold is not None:
        raise ValueError("ALGORITHM_DAG_UNUSED_THRESHOLD")
    if gate_source != "none" and (threshold is None or not math.isfinite(float(threshold))):
        raise ValueError("ALGORITHM_DAG_THRESHOLD")

    fault_indices = [i for i in range(len(classes)) if i != nominal_index]
    nodes: dict[str, dict] = {
        "scores": _node("input_scores"),
        "mechanism": _node("input_mechanism", source="default"),
    }
    if classifier in {"physical_scalar_bins", "physical_scalar_bins_neural_fallback"}:
        nodes["classifier_mechanism"] = _node("input_mechanism", source=classifier_mechanism_source)
    if classifier == "neural_closed_set":
        nodes["class_if_fault"] = _node("argmax_scores", ["scores"], class_indices=list(range(len(classes))))
    elif classifier == "neural_fault_only":
        nodes["class_if_fault"] = _node("argmax_scores", ["scores"], class_indices=fault_indices)
    else:
        thresholds = [] if classifier_thresholds is None else [float(x) for x in classifier_thresholds]
        order = [] if classifier_class_order is None else [int(x) for x in classifier_class_order]
        if (len(thresholds) != len(fault_indices)-1 or any(not math.isfinite(x) for x in thresholds)
                or any(a >= b for a,b in zip(thresholds, thresholds[1:]))):
            raise ValueError("ALGORITHM_DAG_CLASSIFIER_THRESHOLDS")
        if sorted(order) != sorted(fault_indices):
            raise ValueError("ALGORITHM_DAG_CLASSIFIER_CLASS_ORDER")
        nodes["physical_max_classifier"] = _node("row_max", ["classifier_mechanism"])
        if classifier == "physical_scalar_bins":
            if fallback_margin is not None:
                raise ValueError("ALGORITHM_DAG_UNUSED_FALLBACK_MARGIN")
            nodes["class_if_fault"] = _node("scalar_bins", ["physical_max_classifier"],
                                              thresholds=thresholds, class_indices=order)
        else:
            if fallback_margin is None or not math.isfinite(float(fallback_margin)) or float(fallback_margin) < 0:
                raise ValueError("ALGORITHM_DAG_FALLBACK_MARGIN")
            nodes["class_if_fault"] = _node("scalar_bins_neural_fallback", ["physical_max_classifier", "scores"],
                                              thresholds=thresholds, class_indices=order,
                                              fallback_margin=float(fallback_margin), fault_indices=fault_indices)

    if gate_source == "none":
        class_output = "class_if_fault"
        nodes["fault_gate"] = _node("class_is_not", [class_output], class_index=nominal_index)
    elif gate_source == "physical_max":
        nodes["physical_max"] = _node("row_max", ["mechanism"])
        nodes["fault_gate"] = _node("threshold_ge", ["physical_max"], threshold=float(threshold))
        nodes["nominal_class"] = _node("constant_class", class_index=nominal_index)
        nodes["class_output"] = _node("mux_class", ["fault_gate", "class_if_fault", "nominal_class"])
        class_output = "class_output"
    else:
        nodes["fault_probability"] = _node("one_minus_class_score", ["scores"], class_index=nominal_index)
        nodes["fault_gate"] = _node("threshold_ge", ["fault_probability"], threshold=float(threshold))
        nodes["nominal_class"] = _node("constant_class", class_index=nominal_index)
        nodes["class_output"] = _node("mux_class", ["fault_gate", "class_if_fault", "nominal_class"])
        class_output = "class_output"

    if localizer == "physical_argmax":
        nodes["physical_location"] = _node("argmax_vector", ["mechanism"])
        nodes["no_location"] = _node("constant_location", location=-1)
        nodes["location_output"] = _node("mux_location", ["fault_gate", "physical_location", "no_location"])
        location_output = "location_output"
    else:
        nodes["location_output"] = _node("constant_location", location=-1)
        location_output = "location_output"

    candidate = {
        "schema": SCHEMA,
        "classes": list(classes),
        "nominal_index": int(nominal_index),
        "grammar": {
            "gate_source": gate_source,
            "classifier": classifier,
            "localizer": localizer,
            "threshold": threshold,
            **({"classifier_thresholds": [float(x) for x in classifier_thresholds],
                "classifier_class_order": [int(x) for x in classifier_class_order],
                "classifier_mechanism_source": classifier_mechanism_source,
                **({"fallback_margin": float(fallback_margin)} if classifier == "physical_scalar_bins_neural_fallback" else {})}
               if classifier in {"physical_scalar_bins", "physical_scalar_bins_neural_fallback"} else {}),
        },
        "nodes": nodes,
        "outputs": {"class_index": class_output, "location_index": location_output},
        "truth_inputs_used_at_runtime": False,
    }
    candidate["fingerprint"] = _digest(candidate)
    return candidate


def execute(candidate: dict, signals: RuntimeSignals) -> dict:
    """Execute a frozen Algorithm-DAG.  No truth arguments exist by design."""
    if candidate.get("schema") != SCHEMA:
        raise ValueError("ALGORITHM_DAG_SCHEMA")
    classes = candidate["classes"]
    scores = _validate_scores(signals.neural_scores, classes)
    mechanism = _mechanism_source(signals, "default", len(scores))
    n = len(scores)
    values: dict[str, Any] = {"scores": scores}
    nodes = candidate["nodes"]

    unresolved = set(nodes) - {"scores"}
    while unresolved:
        progressed = False
        for name in sorted(list(unresolved)):
            spec = nodes[name]
            inputs = spec.get("inputs", [])
            if any(i not in values for i in inputs):
                continue
            p = spec.get("params", {})
            op = spec["op"]
            if op == "input_mechanism":
                values[name] = _mechanism_source(signals, str(p.get("source", "default")), n)
            elif op == "argmax_scores":
                idx = np.asarray(p["class_indices"], dtype=int)
                local = scores[:, idx].argmax(1)
                values[name] = idx[local]
            elif op == "scalar_bins":
                scalar = np.asarray(values[inputs[0]], dtype=np.float64)
                thresholds = np.asarray(p["thresholds"], dtype=np.float64)
                classes_for_bins = np.asarray(p["class_indices"], dtype=int)
                bins = np.searchsorted(thresholds, scalar, side="right")
                values[name] = classes_for_bins[bins]
            elif op == "scalar_bins_neural_fallback":
                scalar = np.asarray(values[inputs[0]], dtype=np.float64)
                thresholds = np.asarray(p["thresholds"], dtype=np.float64)
                classes_for_bins = np.asarray(p["class_indices"], dtype=int)
                bins = np.searchsorted(thresholds, scalar, side="right")
                physical_pred = classes_for_bins[bins]
                distance = np.min(np.abs(scalar[:, None]-thresholds[None, :]), axis=1) if len(thresholds) else np.full(len(scalar), np.inf)
                fault_idx = np.asarray(p["fault_indices"], dtype=int)
                neural_pred = fault_idx[scores[:, fault_idx].argmax(1)]
                values[name] = np.where(distance <= float(p["fallback_margin"]), neural_pred, physical_pred).astype(int)
            elif op == "rule_tree":
                features = np.column_stack([np.asarray(values[i], dtype=np.float64) for i in inputs])
                children_left = np.asarray(p["children_left"], dtype=int)
                children_right = np.asarray(p["children_right"], dtype=int)
                feature_index = np.asarray(p["feature_index"], dtype=int)
                thresholds = np.asarray(p["thresholds"], dtype=np.float64)
                leaf_class = np.asarray(p["leaf_class_index"], dtype=int)
                pred = np.empty(n, dtype=int)
                for row in range(n):
                    node = 0
                    steps = 0
                    while children_left[node] != children_right[node]:
                        fi = int(feature_index[node])
                        if fi < 0 or fi >= features.shape[1]:
                            raise ValueError("ALGORITHM_DAG_RULE_TREE_FEATURE")
                        node = int(children_left[node] if features[row, fi] <= thresholds[node] else children_right[node])
                        steps += 1
                        if steps > len(children_left):
                            raise ValueError("ALGORITHM_DAG_RULE_TREE_CYCLE")
                    if leaf_class[node] < 0:
                        raise ValueError("ALGORITHM_DAG_RULE_TREE_LEAF")
                    pred[row] = int(leaf_class[node])
                values[name] = pred
            elif op == "rule_tree_neural_fallback":
                if len(inputs) < 2:
                    raise ValueError("ALGORITHM_DAG_RULE_TREE_FALLBACK_INPUTS")
                features = np.column_stack([np.asarray(values[i], dtype=np.float64) for i in inputs[:-1]])
                score_values = _validate_scores(np.asarray(values[inputs[-1]], dtype=np.float64), classes)
                children_left = np.asarray(p["children_left"], dtype=int)
                children_right = np.asarray(p["children_right"], dtype=int)
                feature_index = np.asarray(p["feature_index"], dtype=int)
                thresholds = np.asarray(p["thresholds"], dtype=np.float64)
                leaf_class = np.asarray(p["leaf_class_index"], dtype=int)
                physical = np.empty(n, dtype=int)
                boundary = np.full(n, np.inf, dtype=np.float64)
                for row in range(n):
                    node = 0
                    steps = 0
                    while children_left[node] != children_right[node]:
                        fi = int(feature_index[node])
                        if fi < 0 or fi >= features.shape[1]:
                            raise ValueError("ALGORITHM_DAG_RULE_TREE_FEATURE")
                        threshold_value = float(thresholds[node])
                        relative_distance = abs(float(features[row, fi]) - threshold_value) / max(abs(threshold_value), 1e-12)
                        boundary[row] = min(boundary[row], relative_distance)
                        node = int(children_left[node] if features[row, fi] <= threshold_value else children_right[node])
                        steps += 1
                        if steps > len(children_left):
                            raise ValueError("ALGORITHM_DAG_RULE_TREE_CYCLE")
                    if leaf_class[node] < 0:
                        raise ValueError("ALGORITHM_DAG_RULE_TREE_LEAF")
                    physical[row] = int(leaf_class[node])
                fault_idx = np.asarray(p["fault_indices"], dtype=int)
                neural = fault_idx[score_values[:, fault_idx].argmax(1)]
                use_neural = boundary <= float(p["fallback_margin"])
                values[name] = np.where(use_neural, neural, physical).astype(int)
            elif op == "row_max":
                values[name] = np.asarray(values[inputs[0]], dtype=np.float64).max(1)
            elif op == "threshold_ge":
                values[name] = np.asarray(values[inputs[0]]) >= float(p["threshold"])
            elif op == "one_minus_class_score":
                values[name] = 1.0 - scores[:, int(p["class_index"])]
            elif op == "constant_class":
                values[name] = np.full(n, int(p["class_index"]), dtype=int)
            elif op == "constant_location":
                values[name] = np.full(n, int(p["location"]), dtype=int)
            elif op == "class_is_not":
                values[name] = np.asarray(values[inputs[0]], dtype=int) != int(p["class_index"])
            elif op == "mux_class":
                values[name] = np.where(values[inputs[0]], values[inputs[1]], values[inputs[2]]).astype(int)
            elif op == "argmax_vector":
                values[name] = np.asarray(values[inputs[0]], dtype=np.float64).argmax(1).astype(int)
            elif op == "mux_location":
                values[name] = np.where(values[inputs[0]], values[inputs[1]], values[inputs[2]]).astype(int)
            else:
                raise ValueError("ALGORITHM_DAG_UNKNOWN_OP:" + str(op))
            unresolved.remove(name)
            progressed = True
        if not progressed:
            raise ValueError("ALGORITHM_DAG_CYCLE_OR_MISSING_INPUT")

    return {
        "class_index": np.asarray(values[candidate["outputs"]["class_index"]], dtype=int),
        "location_index": np.asarray(values[candidate["outputs"]["location_index"]], dtype=int),
    }


def _metrics(truth: np.ndarray, pred: np.ndarray, truth_location: np.ndarray,
             pred_location: np.ndarray, nominal_index: int, classes: list[str]) -> dict:
    labels = np.arange(len(classes))
    cm = confusion_matrix(truth, pred, labels=labels)
    recalls = np.divide(np.diag(cm), cm.sum(1), out=np.zeros(len(classes), dtype=float), where=cm.sum(1) > 0)
    tf = truth != nominal_index
    pf = pred != nominal_index
    tp = int(np.sum(tf & pf)); fn = int(np.sum(tf & ~pf)); fp = int(np.sum(~tf & pf)); tn = int(np.sum(~tf & ~pf))
    fault_recall = tp / (tp + fn) if tp + fn else 1.0
    nominal_fpr = fp / (fp + tn) if fp + tn else 0.0
    fault_idx = np.flatnonzero(tf)
    loc_mask = fault_idx[truth_location[fault_idx] >= 0]
    localization = float(np.mean(pred_location[loc_mask] == truth_location[loc_mask])) if len(loc_mask) else None
    mechanism = float(np.mean(pred[fault_idx] == truth[fault_idx])) if len(fault_idx) else None
    joint = None
    if len(loc_mask):
        joint = float(np.mean((pred[loc_mask] == truth[loc_mask]) & (pred_location[loc_mask] == truth_location[loc_mask])))
    return {
        "cases": int(len(truth)),
        "macro_f1": float(f1_score(truth, pred, labels=labels, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, pred)),
        "fault_detection_recall": float(fault_recall),
        "nominal_false_alarm_rate": float(nominal_fpr),
        "mechanism_accuracy_on_faults": mechanism,
        "component_localization_accuracy_on_faults": localization,
        "joint_mechanism_location_accuracy_on_faults": joint,
        "confusion_matrix": cm.tolist(),
        "recall_by_class": dict(zip(classes, recalls.tolist())),
    }


def _candidate_complexity(candidate: dict) -> int:
    # Number of executable, non-input nodes is an intentionally transparent proxy.
    complexity = sum(1 for x in candidate["nodes"].values() if not x["op"].startswith("input_"))
    # A compiled rule tree is one DAG node but contains multiple auditable rule
    # nodes; count those too so model selection does not hide tree complexity.
    tree_nodes = int(candidate.get("grammar", {}).get("tree_node_count", 1))
    if tree_nodes > 1:
        complexity += tree_nodes - 1
    return complexity


def _eligible(m: dict, constraints: dict, *, robustness_margin_fraction: float | None = None) -> tuple[bool, list[str]]:
    reasons = []
    if m["macro_f1"] < float(constraints.get("min_macro_f1", 0.0)):
        reasons.append("MACRO_F1")
    if m["fault_detection_recall"] < float(constraints.get("min_fault_detection_recall", 0.0)):
        reasons.append("FAULT_RECALL")
    if m["nominal_false_alarm_rate"] > float(constraints.get("max_nominal_false_alarm_rate", 1.0)):
        reasons.append("NOMINAL_FPR")
    min_loc = constraints.get("min_component_localization_accuracy")
    if min_loc is not None and (m["component_localization_accuracy_on_faults"] is None or m["component_localization_accuracy_on_faults"] < float(min_loc)):
        reasons.append("LOCALIZATION")
    min_mech = constraints.get("min_mechanism_accuracy_on_faults")
    if min_mech is not None and (m["mechanism_accuracy_on_faults"] is None or m["mechanism_accuracy_on_faults"] < float(min_mech)):
        reasons.append("MECHANISM_ACCURACY")
    min_joint = constraints.get("min_joint_mechanism_location_accuracy")
    if min_joint is not None and (m["joint_mechanism_location_accuracy_on_faults"] is None or m["joint_mechanism_location_accuracy_on_faults"] < float(min_joint)):
        reasons.append("JOINT_ACCURACY")
    min_all = constraints.get("min_all_class_recall")
    if min_all is not None and any(float(v) < float(min_all) for v in m["recall_by_class"].values()):
        reasons.append("CLASS_RECALL")
    required = constraints.get("min_recall_by_class", {})
    if not isinstance(required, dict):
        raise ValueError("ALGORITHM_DAG_CLASS_RECALL_CONSTRAINT")
    for name, threshold in required.items():
        if name not in m["recall_by_class"]:
            raise ValueError("ALGORITHM_DAG_UNKNOWN_RECALL_CLASS:" + str(name))
        if float(m["recall_by_class"][name]) < float(threshold):
            reasons.append("CLASS_RECALL:" + str(name))
    min_margin = constraints.get("min_robustness_margin_fraction")
    if min_margin is not None:
        min_margin = float(min_margin)
        if not math.isfinite(min_margin):
            raise ValueError("ALGORITHM_DAG_ROBUSTNESS_MARGIN_CONSTRAINT")
        # Threshold-free candidates have no deterministic decision boundary to
        # score with this physical/neural margin definition.  The constraint is
        # therefore applied only when a margin is defined; such candidates must
        # satisfy the ordinary performance constraints on their own.
        if robustness_margin_fraction is not None and float(robustness_margin_fraction) < min_margin:
            reasons.append("ROBUSTNESS_MARGIN")
    return not reasons, reasons


def _objective(m: dict, complexity: int, weights: dict) -> float:
    score = 0.0
    score += float(weights.get("macro_f1", 1.0)) * m["macro_f1"]
    score += float(weights.get("balanced_accuracy", 0.0)) * m["balanced_accuracy"]
    score += float(weights.get("fault_detection_recall", 0.0)) * m["fault_detection_recall"]
    score -= float(weights.get("nominal_false_alarm_rate", 0.0)) * m["nominal_false_alarm_rate"]
    if m["component_localization_accuracy_on_faults"] is not None:
        score += float(weights.get("component_localization_accuracy", 0.0)) * m["component_localization_accuracy_on_faults"]
    if m["joint_mechanism_location_accuracy_on_faults"] is not None:
        score += float(weights.get("joint_accuracy", 0.0)) * m["joint_mechanism_location_accuracy_on_faults"]
    score -= float(weights.get("complexity_penalty", 0.0)) * complexity
    return float(score)


def enumerate_candidates(config: dict) -> list[dict]:
    if config.get("schema") != SEARCH_SCHEMA:
        raise ValueError("ALGORITHM_DAG_SEARCH_SCHEMA")
    classes = config["classes"]
    if not isinstance(classes, list) or len(classes) < 2 or len(set(classes)) != len(classes):
        raise ValueError("ALGORITHM_DAG_CLASSES")
    nominal = config["nominal_class"]
    if nominal not in classes:
        raise ValueError("ALGORITHM_DAG_NOMINAL_CLASS")
    ni = classes.index(nominal)
    requested = config.get("topologies", [
        "neural_closed_set",
        "physics_gate_neural_fault_physical_localizer",
        "neural_gate_neural_fault_physical_localizer",
    ])
    out = []
    for name in requested:
        if name == "neural_closed_set":
            out.append(make_candidate(classes=classes, nominal_index=ni, gate_source="none",
                                      classifier="neural_closed_set", localizer="physical_argmax", threshold=None))
        elif name == "physics_gate_neural_fault_physical_localizer":
            for t in config.get("physical_gate_thresholds", []):
                out.append(make_candidate(classes=classes, nominal_index=ni, gate_source="physical_max",
                                          classifier="neural_fault_only", localizer="physical_argmax", threshold=float(t)))
        elif name == "physics_gate_neural_closed_physical_localizer":
            for t in config.get("physical_gate_thresholds", []):
                out.append(make_candidate(classes=classes, nominal_index=ni, gate_source="physical_max",
                                          classifier="neural_closed_set", localizer="physical_argmax", threshold=float(t)))
        elif name == "neural_gate_neural_fault_physical_localizer":
            for t in config.get("neural_fault_probability_thresholds", []):
                out.append(make_candidate(classes=classes, nominal_index=ni, gate_source="neural_fault_probability",
                                          classifier="neural_fault_only", localizer="physical_argmax", threshold=float(t)))
        elif name == "physics_gate_physical_bins_physical_localizer":
            fault_indices = [i for i in range(len(classes)) if i != ni]
            sources = config.get("physical_classifier_sources", ["default"])
            if not isinstance(sources, list) or not sources or any(not isinstance(x, str) or not x for x in sources):
                raise ValueError("ALGORITHM_DAG_PHYSICAL_CLASSIFIER_SOURCES")
            by_source = config.get("physical_classifier_thresholds_by_source", {})
            if not isinstance(by_source, dict):
                raise ValueError("ALGORITHM_DAG_PHYSICAL_CLASSIFIER_GRID_BY_SOURCE")
            for source in sources:
                grid = sorted({float(x) for x in by_source.get(source, config.get("physical_classifier_thresholds", []))})
                if len(grid) < max(0, len(fault_indices)-1):
                    raise ValueError("ALGORITHM_DAG_PHYSICAL_CLASSIFIER_GRID:" + source)
                for gate_t in config.get("physical_gate_thresholds", []):
                    for thresholds in itertools.combinations(grid, len(fault_indices)-1):
                        for order in itertools.permutations(fault_indices):
                            out.append(make_candidate(classes=classes, nominal_index=ni, gate_source="physical_max",
                                                      classifier="physical_scalar_bins", localizer="physical_argmax",
                                                      threshold=float(gate_t), classifier_thresholds=list(thresholds),
                                                      classifier_class_order=list(order), classifier_mechanism_source=source))
        elif name == "physics_gate_physical_bins_neural_fallback_physical_localizer":
            fault_indices = [i for i in range(len(classes)) if i != ni]
            sources = config.get("physical_classifier_sources", ["default"])
            if not isinstance(sources, list) or not sources or any(not isinstance(x, str) or not x for x in sources):
                raise ValueError("ALGORITHM_DAG_PHYSICAL_CLASSIFIER_SOURCES")
            by_source = config.get("physical_classifier_thresholds_by_source", {})
            margins_by_source = config.get("physical_classifier_fallback_margins_by_source", {})
            if not isinstance(by_source, dict) or not isinstance(margins_by_source, dict):
                raise ValueError("ALGORITHM_DAG_PHYSICAL_FALLBACK_GRID_BY_SOURCE")
            for source in sources:
                grid = sorted({float(x) for x in by_source.get(source, config.get("physical_classifier_thresholds", []))})
                margins = sorted({float(x) for x in margins_by_source.get(source, config.get("physical_classifier_fallback_margins", []))})
                if len(grid) < max(0, len(fault_indices)-1) or not margins or any(x < 0 or not math.isfinite(x) for x in margins):
                    raise ValueError("ALGORITHM_DAG_PHYSICAL_FALLBACK_GRID:" + source)
                for gate_t in config.get("physical_gate_thresholds", []):
                    for thresholds in itertools.combinations(grid, len(fault_indices)-1):
                        for order in itertools.permutations(fault_indices):
                            for margin in margins:
                                out.append(make_candidate(classes=classes, nominal_index=ni, gate_source="physical_max",
                                                          classifier="physical_scalar_bins_neural_fallback", localizer="physical_argmax",
                                                          threshold=float(gate_t), classifier_thresholds=list(thresholds),
                                                          classifier_class_order=list(order), fallback_margin=margin,
                                                          classifier_mechanism_source=source))
        else:
            raise ValueError("ALGORITHM_DAG_TOPOLOGY:" + str(name))
    if not out:
        raise ValueError("ALGORITHM_DAG_EMPTY_SEARCH_SPACE")
    fingerprints = [c["fingerprint"] for c in out]
    if len(fingerprints) != len(set(fingerprints)):
        raise ValueError("ALGORITHM_DAG_DUPLICATE_CANDIDATE")
    return out


def records_to_arrays(records: Iterable[dict], classes: list[str]) -> tuple[RuntimeSignals, np.ndarray, np.ndarray]:
    records = list(records)
    if not records:
        raise ValueError("ALGORITHM_DAG_EMPTY_RECORDS")
    truth = []
    scores = []
    mech = []
    banks_by_name: dict[str, list] = {}
    bank_names: set[str] | None = None
    locations = []
    for row in records:
        if row.get("truth") not in classes:
            raise ValueError("ALGORITHM_DAG_TRUTH_CLASS")
        truth.append(classes.index(row["truth"]))
        scores.append(row["neural_scores"])
        # The generic name is preferred; ADCS-v7 evidence uses residual_mean_abs_nms.
        mv = row.get("mechanism_values", row.get("residual_mean_abs_nms"))
        if mv is None:
            raise ValueError("ALGORITHM_DAG_MECHANISM_MISSING")
        mech.append(mv)
        row_banks = row.get("mechanism_banks", {})
        if not isinstance(row_banks, dict):
            raise ValueError("ALGORITHM_DAG_MECHANISM_BANKS")
        names = set(row_banks)
        if bank_names is None:
            bank_names = names
            for name in sorted(names):
                if not isinstance(name, str) or not name or name == "default":
                    raise ValueError("ALGORITHM_DAG_MECHANISM_BANK_NAME")
                banks_by_name[name] = []
        elif names != bank_names:
            raise ValueError("ALGORITHM_DAG_MECHANISM_BANK_MISMATCH")
        for name in sorted(names):
            banks_by_name[name].append(row_banks[name])
        loc = row.get("truth_location", row.get("truth_wheel", -1))
        locations.append(-1 if loc is None else int(loc))
    validated_banks = {name: _validate_mechanism(np.asarray(values), len(records))
                       for name, values in banks_by_name.items()}
    sig = RuntimeSignals(_validate_scores(np.asarray(scores), classes),
                         _validate_mechanism(np.asarray(mech), len(records)), validated_banks)
    return sig, np.asarray(truth, dtype=int), np.asarray(locations, dtype=int)


def _robustness_margin_fraction(candidate: dict, signals: RuntimeSignals, truth: np.ndarray, nominal_index: int) -> float | None:
    """Development-only normalized decision margin used solely as a deterministic tie-break.

    Positive values mean every development record lies on the correct side of the
    relevant threshold boundary.  The normalization makes physical and neural
    gate margins dimensionless; it is not a statistical confidence bound.
    """
    g = candidate["grammar"]
    margins: list[float] = []
    if g["gate_source"] == "physical_max":
        statistic = signals.mechanism_values.max(1)
        threshold = float(g["threshold"])
        signed = np.where(truth != nominal_index, statistic-threshold, threshold-statistic)
        scale = max(float(statistic.max()-statistic.min()), 1e-12)
        margins.append(float(signed.min()/scale))
    elif g["gate_source"] == "neural_fault_probability":
        scores = _validate_scores(signals.neural_scores, candidate["classes"])
        statistic = 1.0-scores[:, nominal_index]
        threshold = float(g["threshold"])
        signed = np.where(truth != nominal_index, statistic-threshold, threshold-statistic)
        scale = max(float(statistic.max()-statistic.min()), 1e-12)
        margins.append(float(signed.min()/scale))
    if g["classifier"] in {"physical_scalar_bins", "physical_scalar_bins_neural_fallback"}:
        classifier_values = _mechanism_source(signals, g.get("classifier_mechanism_source", "default"))
        statistic = classifier_values.max(1)
        thresholds = [float(x) for x in g["classifier_thresholds"]]
        order = [int(x) for x in g["classifier_class_order"]]
        fault_mask = truth != nominal_index
        fault_values = statistic[fault_mask]
        fault_truth = truth[fault_mask]
        if len(fault_values):
            scale = max(float(fault_values.max()-fault_values.min()), 1e-12)
            class_to_bin = {c:i for i,c in enumerate(order)}
            signed=[]
            for value, cls in zip(fault_values, fault_truth):
                if int(cls) not in class_to_bin:
                    signed.append(-scale); continue
                idx=class_to_bin[int(cls)]
                low = thresholds[idx-1] if idx>0 else None
                high = thresholds[idx] if idx<len(thresholds) else None
                distances=[]
                if low is not None: distances.append(float(value-low))
                if high is not None: distances.append(float(high-value))
                signed.append(min(distances) if distances else scale)
            margins.append(float(min(signed)/scale))
    return min(margins) if margins else None


def search(records: Iterable[dict], config: dict) -> dict:
    """Search decision topology using development labels only.

    Non-development inputs are permitted only when ``retrospective_replay`` is
    true; such a report is explicitly ineligible as fresh qualification evidence.
    """
    role = config.get("data_role", "development")
    retrospective = bool(config.get("retrospective_replay", False))
    if role != "development" and not retrospective:
        raise ValueError("ALGORITHM_DAG_SEARCH_REQUIRES_DEVELOPMENT")
    classes = config["classes"]
    nominal_index = classes.index(config["nominal_class"])
    signals, truth, truth_location = records_to_arrays(records, classes)
    constraints = config.get("constraints", {})
    weights = config.get("objective_weights", {"macro_f1": 1.0, "complexity_penalty": 1e-4})
    results = []
    for candidate in enumerate_candidates(config):
        out = execute(candidate, signals)
        m = _metrics(truth, out["class_index"], truth_location, out["location_index"], nominal_index, classes)
        complexity = _candidate_complexity(candidate)
        robustness_margin = _robustness_margin_fraction(candidate, signals, truth, nominal_index)
        eligible, reasons = _eligible(m, constraints, robustness_margin_fraction=robustness_margin)
        objective = _objective(m, complexity, weights)
        results.append({
            "candidate_fingerprint": candidate["fingerprint"],
            "grammar": candidate["grammar"],
            "complexity_nodes": complexity,
            "eligible": eligible,
            "ineligibility_reasons": reasons,
            "objective": objective,
            "robustness_margin_fraction": robustness_margin,
            "metrics": m,
            "candidate": candidate,
        })
    eligible = [r for r in results if r["eligible"]]
    pool = eligible or results
    # Deterministic: objective, macro-F1, fault recall, then simpler, then fingerprint.
    chosen = sorted(pool, key=lambda r: (-r["objective"], -r["metrics"]["macro_f1"],
                                         -r["metrics"]["fault_detection_recall"],
                                         -(r["robustness_margin_fraction"] if r["robustness_margin_fraction"] is not None else -1e300),
                                         r["complexity_nodes"], r["candidate_fingerprint"]))[0]
    report = {
        "schema": REPORT_SCHEMA,
        "status": "COMPLETED" if eligible else "FAIL_CLOSED_NO_ELIGIBLE_CANDIDATE",
        "selection_status": "ELIGIBLE_CANDIDATE_SELECTED" if eligible else "NO_ELIGIBLE_CANDIDATE",
        "data_role": role,
        "retrospective_replay": retrospective,
        "qualification_eligible": role == "development" and not retrospective and bool(eligible),
        "chosen_is_eligible": bool(chosen["eligible"]),
        "test_or_holdout_used_for_selection": role in {"test", "holdout", "blind"},
        "classes": classes,
        "nominal_class": config["nominal_class"],
        "candidate_count": len(results),
        "eligible_candidate_count": len(eligible),
        "chosen_fingerprint": chosen["candidate_fingerprint"],
        "chosen_grammar": chosen["grammar"],
        "chosen_metrics": chosen["metrics"],
        "chosen_objective": chosen["objective"],
        "chosen_candidate": chosen["candidate"],
        "results": [{k: v for k, v in r.items() if k != "candidate"} for r in results],
        "truth_inputs_used_at_runtime": False,
        "truth_used_for_search_scoring_only": True,
        "claim_status": "RESEARCH_ONLY" if role == "development" and not retrospective else "RETROSPECTIVE_REPLAY_ONLY",
    }
    report["fingerprint"] = _digest(report)
    return report



def _compile_rule_tree(model: DecisionTreeClassifier) -> dict:
    tree = model.tree_
    leaf_class = []
    model_classes = np.asarray(model.classes_, dtype=int)
    for node in range(tree.node_count):
        if tree.children_left[node] == tree.children_right[node]:
            local = int(np.argmax(tree.value[node][0]))
            leaf_class.append(int(model_classes[local]))
        else:
            leaf_class.append(-1)
    return {
        "children_left": tree.children_left.astype(int).tolist(),
        "children_right": tree.children_right.astype(int).tolist(),
        "feature_index": tree.feature.astype(int).tolist(),
        "thresholds": tree.threshold.astype(float).tolist(),
        "leaf_class_index": leaf_class,
        "node_count": int(tree.node_count),
        "max_depth": int(tree.max_depth),
    }


def make_physical_rule_tree_candidate(*, classes: list[str], nominal_index: int,
                                      gate_threshold: float, feature_sources: list[str],
                                      compiled_tree: dict,
                                      localizer: str = "physical_argmax",
                                      neural_fallback_margin: float | None = None) -> dict:
    """Compile a shallow, truth-free physical decision tree into Algorithm-DAG."""
    if not feature_sources or len(set(feature_sources)) != len(feature_sources):
        raise ValueError("ALGORITHM_DAG_RULE_TREE_FEATURE_SOURCES")
    if localizer not in {"none", "physical_argmax"}:
        raise ValueError("ALGORITHM_DAG_LOCALIZER")
    nodes: dict[str, dict] = {
        "scores": _node("input_scores"),
        "mechanism": _node("input_mechanism", source="default"),
        "physical_max": _node("row_max", ["mechanism"]),
        "fault_gate": _node("threshold_ge", ["physical_max"], threshold=float(gate_threshold)),
        "nominal_class": _node("constant_class", class_index=int(nominal_index)),
    }
    scalar_inputs = []
    for i, source in enumerate(feature_sources):
        inp = f"tree_mechanism_{i}"
        scalar = f"tree_scalar_{i}"
        nodes[inp] = _node("input_mechanism", source=source)
        nodes[scalar] = _node("row_max", [inp])
        scalar_inputs.append(scalar)
    tree_params = {k: compiled_tree[k] for k in ["children_left", "children_right", "feature_index", "thresholds", "leaf_class_index"]}
    if neural_fallback_margin is None:
        nodes["class_if_fault"] = _node("rule_tree", scalar_inputs, **tree_params)
        classifier_name = "physical_rule_tree"
    else:
        margin = float(neural_fallback_margin)
        if not math.isfinite(margin) or margin < 0:
            raise ValueError("ALGORITHM_DAG_RULE_TREE_FALLBACK_MARGIN")
        fault_indices = [i for i in range(len(classes)) if i != nominal_index]
        nodes["class_if_fault"] = _node("rule_tree_neural_fallback", scalar_inputs + ["scores"],
            **tree_params, fallback_margin=margin, fault_indices=fault_indices)
        classifier_name = "physical_rule_tree_neural_fallback"
    nodes["class_output"] = _node("mux_class", ["fault_gate", "class_if_fault", "nominal_class"])
    if localizer == "physical_argmax":
        nodes["physical_location"] = _node("argmax_vector", ["mechanism"])
        nodes["no_location"] = _node("constant_location", location=-1)
        nodes["location_output"] = _node("mux_location", ["fault_gate", "physical_location", "no_location"])
    else:
        nodes["location_output"] = _node("constant_location", location=-1)
    candidate = {
        "schema": SCHEMA,
        "classes": list(classes),
        "nominal_index": int(nominal_index),
        "grammar": {
            "gate_source": "physical_max",
            "classifier": classifier_name,
            "localizer": localizer,
            "threshold": float(gate_threshold),
            "classifier_feature_sources": list(feature_sources),
            "tree_node_count": int(compiled_tree["node_count"]),
            "tree_max_depth": int(compiled_tree["max_depth"]),
            **({"neural_fallback_margin": float(neural_fallback_margin)} if neural_fallback_margin is not None else {}),
        },
        "nodes": nodes,
        "outputs": {"class_index": "class_output", "location_index": "location_output"},
        "truth_inputs_used_at_runtime": False,
    }
    candidate["fingerprint"] = _digest(candidate)
    return candidate


def search_physical_rule_tree_multi_environment(environments: dict[str, Iterable[dict]], config: dict) -> dict:
    """Fit/select a shallow physical rule-tree using development environments only.

    Hyperparameters and feature subsets are selected by multi-environment maximin
    performance. The final tree is refit on the union of development environments
    and compiled into a truth-free Algorithm-DAG runtime. Leave-one-environment-out
    metrics are computed using development environments only and, by default,
    are a hard robustness gate. They never use blind/test data.
    """
    if not isinstance(environments, dict) or len(environments) < 2:
        raise ValueError("ALGORITHM_DAG_MULTI_ENVIRONMENTS")
    if config.get("data_role", "development") != "development" or bool(config.get("retrospective_replay", False)):
        raise ValueError("ALGORITHM_DAG_MULTI_REQUIRES_FRESH_DEVELOPMENT")
    classes = config["classes"]
    nominal_index = classes.index(config["nominal_class"])
    constraints = config.get("constraints", {})
    weights = config.get("objective_weights", {"macro_f1": 1.0, "complexity_penalty": 1e-4})
    feature_sets = config.get("physical_tree_feature_sets", [])
    if not isinstance(feature_sets, list) or not feature_sets:
        raise ValueError("ALGORITHM_DAG_RULE_TREE_FEATURE_SETS")
    depths = [int(x) for x in config.get("physical_tree_depths", [2, 3])]
    leaves = [int(x) for x in config.get("physical_tree_min_samples_leaf", [1, 2])]
    gate_thresholds = [float(x) for x in config.get("physical_gate_thresholds", [])]
    fallback_values = config.get("physical_tree_neural_fallback_margins", [None])
    if not isinstance(fallback_values, list) or not fallback_values:
        raise ValueError("ALGORITHM_DAG_RULE_TREE_FALLBACK_GRID")
    fallback_margins = []
    for value in fallback_values:
        if value is None:
            fallback_margins.append(None)
        else:
            value = float(value)
            if not math.isfinite(value) or value < 0:
                raise ValueError("ALGORITHM_DAG_RULE_TREE_FALLBACK_MARGIN")
            fallback_margins.append(value)
    require_loo = bool(config.get("require_leave_one_environment_out", True))
    if not gate_thresholds:
        raise ValueError("ALGORITHM_DAG_RULE_TREE_GATE_GRID")
    parsed = {}
    for name, records in sorted(environments.items()):
        parsed[name] = records_to_arrays(records, classes)

    def matrix(signals: RuntimeSignals, sources: list[str]) -> np.ndarray:
        return np.column_stack([_mechanism_source(signals, src).max(1) for src in sources])

    results = []
    for sources_raw in feature_sets:
        sources = [str(x) for x in sources_raw]
        if not sources or len(set(sources)) != len(sources):
            raise ValueError("ALGORITHM_DAG_RULE_TREE_FEATURE_SOURCES")
        pooled_x=[]; pooled_y=[]
        for signals,truth,_ in parsed.values():
            fm = truth != nominal_index
            pooled_x.append(matrix(signals,sources)[fm]); pooled_y.append(truth[fm])
        X=np.concatenate(pooled_x,axis=0); y=np.concatenate(pooled_y,axis=0)
        for depth in depths:
            for min_leaf in leaves:
                if depth < 1 or min_leaf < 1:
                    raise ValueError("ALGORITHM_DAG_RULE_TREE_HYPERPARAMETER")
                model=DecisionTreeClassifier(max_depth=depth,min_samples_leaf=min_leaf,
                                             class_weight="balanced",random_state=0)
                model.fit(X,y)
                compiled=_compile_rule_tree(model)
                for gate in gate_thresholds:
                    for fallback_margin in fallback_margins:
                        candidate=make_physical_rule_tree_candidate(classes=classes,nominal_index=nominal_index,
                            gate_threshold=gate,feature_sources=sources,compiled_tree=compiled,
                            neural_fallback_margin=fallback_margin)
                        complexity=_candidate_complexity(candidate)
                        env_reports={}
                        for name,(signals,truth,truth_location) in parsed.items():
                            out=execute(candidate,signals)
                            m=_metrics(truth,out["class_index"],truth_location,out["location_index"],nominal_index,classes)
                            margin=_robustness_margin_fraction(candidate,signals,truth,nominal_index)
                            ok,reasons=_eligible(m,constraints,robustness_margin_fraction=margin)
                            env_reports[name]={"eligible":ok,"ineligibility_reasons":reasons,
                                "objective":_objective(m,complexity,weights),"robustness_margin_fraction":margin,"metrics":m}
                        # Leave-one-environment-out diagnostics for the same architecture/hyperparameters.
                        loo={}
                        loo_reports={}
                        for hold,(hold_signals,hold_truth,hold_loc) in parsed.items():
                            train_x=[];train_y=[]
                            for name,(sig,tr,_) in parsed.items():
                                if name==hold: continue
                                fm=tr!=nominal_index;train_x.append(matrix(sig,sources)[fm]);train_y.append(tr[fm])
                            lm=DecisionTreeClassifier(max_depth=depth,min_samples_leaf=min_leaf,class_weight="balanced",random_state=0)
                            lm.fit(np.concatenate(train_x),np.concatenate(train_y))
                            lc=make_physical_rule_tree_candidate(classes=classes,nominal_index=nominal_index,
                                gate_threshold=gate,feature_sources=sources,compiled_tree=_compile_rule_tree(lm),
                            neural_fallback_margin=fallback_margin)
                            lo=execute(lc,hold_signals)
                            lmtr=_metrics(hold_truth,lo["class_index"],hold_loc,lo["location_index"],nominal_index,classes)
                            lmargin=_robustness_margin_fraction(lc,hold_signals,hold_truth,nominal_index)
                            lok,lreasons=_eligible(lmtr,constraints,robustness_margin_fraction=lmargin)
                            loo[hold]=lmtr
                            loo_reports[hold]={"eligible":lok,"ineligibility_reasons":lreasons,
                                "robustness_margin_fraction":lmargin,"metrics":lmtr}
                        objs=[v["objective"] for v in env_reports.values()]
                        macros=[v["metrics"]["macro_f1"] for v in env_reports.values()]
                        recalls=[min(v["metrics"]["recall_by_class"].values()) for v in env_reports.values()]
                        loo_macros=[v["macro_f1"] for v in loo.values()]
                        loo_recalls=[min(v["recall_by_class"].values()) for v in loo.values()]
                        env_eligible=all(v["eligible"] for v in env_reports.values())
                        loo_eligible=all(v["eligible"] for v in loo_reports.values())
                        results.append({
                            "candidate_fingerprint":candidate["fingerprint"],"grammar":candidate["grammar"],
                            "neural_fallback_margin":fallback_margin,
                            "complexity_nodes":complexity,"eligible":bool(env_eligible and (loo_eligible or not require_loo)),
                            "environment_eligible":bool(env_eligible),"loo_eligible":bool(loo_eligible),
                            "environment_reports":env_reports,"leave_one_environment_out":loo,
                            "leave_one_environment_out_reports":loo_reports,
                            "worst_objective":float(min(objs)),"mean_objective":float(np.mean(objs)),
                            "worst_macro_f1":float(min(macros)),"worst_all_class_recall":float(min(recalls)),
                            "loo_worst_macro_f1":float(min(loo_macros)),"loo_worst_all_class_recall":float(min(loo_recalls)),
                            "candidate":candidate,
                        })
    eligible=[r for r in results if r["eligible"]]
    pool=eligible or results
    chosen=sorted(pool,key=lambda r:(-r["worst_objective"],-r["worst_macro_f1"],-r["worst_all_class_recall"],
                                      -r["loo_worst_macro_f1"],-r["loo_worst_all_class_recall"],
                                      r["complexity_nodes"],
                                      (0.0 if r.get("neural_fallback_margin") is None else 1.0+float(r["neural_fallback_margin"])),
                                      r["candidate_fingerprint"]))[0]
    report={
        "schema":"phm.algorithm-dag-physical-rule-tree-search-report.v1",
        "status":"COMPLETED" if eligible else "FAIL_CLOSED_NO_ELIGIBLE_CANDIDATE",
        "selection_status":"ELIGIBLE_CANDIDATE_SELECTED" if eligible else "NO_ELIGIBLE_CANDIDATE",
        "data_role":"development","retrospective_replay":False,"qualification_eligible":bool(eligible),
        "environment_names":sorted(parsed),"environment_count":len(parsed),
        "candidate_count":len(results),"eligible_candidate_count":len(eligible),
        "selection_rule":"maximin_development_with_leave_one_environment_out_hard_gate" if require_loo else "maximin_development_then_leave_one_environment_out_diagnostic",
        "require_leave_one_environment_out":require_loo,
        "chosen_is_eligible":bool(chosen["eligible"]),"chosen_fingerprint":chosen["candidate_fingerprint"],
        "chosen_grammar":chosen["grammar"],"chosen_worst_macro_f1":chosen["worst_macro_f1"],
        "chosen_worst_all_class_recall":chosen["worst_all_class_recall"],
        "chosen_loo_worst_macro_f1":chosen["loo_worst_macro_f1"],
        "chosen_loo_worst_all_class_recall":chosen["loo_worst_all_class_recall"],
        "chosen_environment_reports":chosen["environment_reports"],
        "chosen_leave_one_environment_out":chosen["leave_one_environment_out"],
        "chosen_leave_one_environment_out_reports":chosen["leave_one_environment_out_reports"],
        "chosen_candidate":chosen["candidate"],
        "results":[{k:v for k,v in r.items() if k!="candidate"} for r in results],
        "truth_inputs_used_at_runtime":False,"truth_used_for_search_scoring_only":True,
        "test_or_holdout_used_for_selection":False,"claim_status":"RESEARCH_ONLY",
    }
    report["fingerprint"]=_digest(report)
    return report


def search_multi_environment(environments: dict[str, Iterable[dict]], config: dict) -> dict:
    """Select one Algorithm-DAG across multiple *development* environments.

    Candidate generation is shared across environments.  A candidate is eligible
    only when every environment satisfies the declared constraints.  Selection is
    maximin-first to reduce single-environment threshold overfitting; this remains
    development evidence, not calibration or holdout qualification.
    """
    if not isinstance(environments, dict) or len(environments) < 2 or any(not isinstance(k, str) or not k for k in environments):
        raise ValueError("ALGORITHM_DAG_MULTI_ENVIRONMENTS")
    if config.get("data_role", "development") != "development" or bool(config.get("retrospective_replay", False)):
        raise ValueError("ALGORITHM_DAG_MULTI_REQUIRES_FRESH_DEVELOPMENT")
    classes = config["classes"]
    nominal_index = classes.index(config["nominal_class"])
    constraints = config.get("constraints", {})
    weights = config.get("objective_weights", {"macro_f1": 1.0, "complexity_penalty": 1e-4})
    parsed = {}
    for name, records in sorted(environments.items()):
        signals, truth, truth_location = records_to_arrays(records, classes)
        parsed[name] = (signals, truth, truth_location)
    results=[]
    for candidate in enumerate_candidates(config):
        complexity=_candidate_complexity(candidate)
        env_reports={}
        for name,(signals,truth,truth_location) in parsed.items():
            out=execute(candidate,signals)
            m=_metrics(truth,out["class_index"],truth_location,out["location_index"],nominal_index,classes)
            margin=_robustness_margin_fraction(candidate,signals,truth,nominal_index)
            eligible,reasons=_eligible(m,constraints,robustness_margin_fraction=margin)
            obj=_objective(m,complexity,weights)
            env_reports[name]={"eligible":eligible,"ineligibility_reasons":reasons,"objective":obj,
                               "robustness_margin_fraction":margin,"metrics":m}
        objs=[v["objective"] for v in env_reports.values()]
        margins=[v["robustness_margin_fraction"] for v in env_reports.values() if v["robustness_margin_fraction"] is not None]
        macros=[v["metrics"]["macro_f1"] for v in env_reports.values()]
        faults=[v["metrics"]["fault_detection_recall"] for v in env_reports.values()]
        fprs=[v["metrics"]["nominal_false_alarm_rate"] for v in env_reports.values()]
        locs=[v["metrics"]["component_localization_accuracy_on_faults"] for v in env_reports.values()
              if v["metrics"]["component_localization_accuracy_on_faults"] is not None]
        results.append({
            "candidate_fingerprint":candidate["fingerprint"],"grammar":candidate["grammar"],
            "complexity_nodes":complexity,"eligible":all(v["eligible"] for v in env_reports.values()),
            "environment_reports":env_reports,
            "worst_objective":float(min(objs)),"mean_objective":float(np.mean(objs)),
            "worst_macro_f1":float(min(macros)),"worst_fault_detection_recall":float(min(faults)),
            "worst_nominal_false_alarm_rate":float(max(fprs)),
            "worst_component_localization_accuracy":float(min(locs)) if locs else None,
            "worst_robustness_margin_fraction":float(min(margins)) if margins else None,
            "candidate":candidate,
        })
    eligible=[r for r in results if r["eligible"]]
    pool=eligible or results
    chosen=sorted(pool,key=lambda r:(-r["worst_objective"],-r["worst_macro_f1"],
                                      -r["worst_fault_detection_recall"],
                                      -(r["worst_robustness_margin_fraction"] if r["worst_robustness_margin_fraction"] is not None else -1e300),
                                      -r["mean_objective"],r["complexity_nodes"],r["candidate_fingerprint"]))[0]
    report={
        "schema":"phm.algorithm-dag-multi-environment-search-report.v1",
        "status":"COMPLETED" if eligible else "FAIL_CLOSED_NO_ELIGIBLE_CANDIDATE",
        "selection_status":"ELIGIBLE_CANDIDATE_SELECTED" if eligible else "NO_ELIGIBLE_CANDIDATE",
        "data_role":"development","retrospective_replay":False,"qualification_eligible":bool(eligible),
        "chosen_is_eligible":bool(chosen["eligible"]),
        "environment_names":sorted(parsed),"environment_count":len(parsed),
        "candidate_count":len(results),"eligible_candidate_count":len(eligible),
        "selection_rule":"maximin_objective_then_worst_metrics_then_development_margin",
        "chosen_fingerprint":chosen["candidate_fingerprint"],"chosen_grammar":chosen["grammar"],
        "chosen_worst_objective":chosen["worst_objective"],"chosen_mean_objective":chosen["mean_objective"],
        "chosen_worst_macro_f1":chosen["worst_macro_f1"],
        "chosen_worst_fault_detection_recall":chosen["worst_fault_detection_recall"],
        "chosen_worst_nominal_false_alarm_rate":chosen["worst_nominal_false_alarm_rate"],
        "chosen_worst_component_localization_accuracy":chosen["worst_component_localization_accuracy"],
        "chosen_worst_robustness_margin_fraction":chosen["worst_robustness_margin_fraction"],
        "chosen_environment_reports":chosen["environment_reports"],"chosen_candidate":chosen["candidate"],
        "results":[{k:v for k,v in r.items() if k!="candidate"} for r in results],
        "truth_inputs_used_at_runtime":False,"truth_used_for_search_scoring_only":True,
        "test_or_holdout_used_for_selection":False,"claim_status":"RESEARCH_ONLY",
    }
    report["fingerprint"]=_digest(report)
    return report
