"""R94 / P1.2: orchestrate the *existing* AAS detector contract without cloning it.

Autogen owns TRAIN-only TaskSpec/data contracts, supervised-vs-unlabelled
sample access, experiment identity, execution receipts, and comparison.
AAS owns detector fit/score/decision-mode/threshold/final labels, persistence
and process isolation. torch_like owns optional native algorithm backends.

This bridge does NOT import AAS at import time, load arbitrary plugin paths,
serialize sklearn/pickle, invent scores for label-only detectors, tune
thresholds from test data, or claim a cross-equipment result.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import math
from time import perf_counter

from .contracts import ContractRejected, compile_source_plan

SCHEMA = "phm.autogen.aas-detector-bridge.v1"
DECISION_MODES = frozenset({"score_threshold", "native", "label"})
_THRESHOLD_CONFLICTS = frozenset({
    "threshold_method", "threshold_mode", "threshold_plugin", "threshold",
    "threshold_window", "threshold_alpha", "threshold_sigma",
    "threshold_update_on_anomaly", "threshold_quantile_method",
})
_MAX_ABS_FEATURE = 1e12


class DetectorBridgeRejected(ContractRejected):
    """No implicit fallback when an AAS contract or evidence is unavailable."""


def _reject(reason):
    raise DetectorBridgeRejected(reason)


def _digest(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ensure_ascii=False).encode("utf-8")).hexdigest()


def _table(rows, channels, *, train_ids=None):
    """Convert R93 feature records to AAS' native array-style fit/detect input."""
    if not isinstance(rows, (tuple, list)) or not rows:
        _reject("R94_EMPTY_RECORD_BATCH")
    result = []
    found = set()
    for element in rows:
        if train_ids is not None:
            if not isinstance(element, Mapping) or set(element) != {
                    "record_id", "observations"}:
                _reject("R94_TRAIN_TABLE_HAS_UNDECLARED_PROPERTIES")
            record_id = element["record_id"]
            if (type(record_id) is not str or record_id not in train_ids
                    or record_id in found):
                _reject("R94_MISSING_DUPLICATE_OR_UNKNOWN_TRAIN_RECORD")
            found.add(record_id)
            row = element["observations"]
        else:
            row = element
        if not isinstance(row, Mapping) or set(row) != set(channels):
            _reject("R94_UNKNOWN_PROTECTED_OR_MISSING_SENSOR_FIELD")
        values = [row[c] for c in channels]
        if any(type(v) not in (int, float) or not math.isfinite(v)
               or abs(v) > _MAX_ABS_FEATURE for v in values):
            _reject("R94_NONFINITE_OR_INVALID_SENSOR_VALUE")
        result.append([float(v) for v in values])
    if train_ids is not None and found != train_ids:
        _reject("R94_TRAIN_RECORDS_DO_NOT_MATCH_SOURCE_MANIFEST")
    return result


def _validate_decision(result, n, mode):
    if not isinstance(result, Mapping):
        _reject("R94_AAS_DETECT_MUST_RETURN_MAPPING")
    # AAS canonical result is created inside the detector; never rethreshold.
    if result.get("decision_mode") != mode:
        _reject("R94_AAS_DECISION_MODE_MISMATCH")
    labels = result.get("pred_label")
    if not isinstance(labels, (list, tuple)) or len(labels) != n or any(
            type(v) is not int or v not in (0, 1) for v in labels):
        _reject("R94_INVALID_AAS_PRED_LABEL")
    scores = result.get("anomaly_score")
    if scores is not None and (
            not isinstance(scores, (list, tuple)) or len(scores) != n or
            any(type(v) not in (int, float) or not math.isfinite(v)
                for v in scores)):
        _reject("R94_INVALID_AAS_ANOMALY_SCORE")
    if mode == "score_threshold" and scores is None:
        _reject("R94_SCORE_THRESHOLD_MODE_REQUIRES_SCORE")
    thresholds = result.get("thresholds")
    if thresholds is None:
        thresholds = []
    if (not isinstance(thresholds, (list, tuple)) or
            len(thresholds) not in (0, n) or any(
                type(v) not in (int, float) or not math.isfinite(v)
                for v in thresholds)):
        _reject("R94_INVALID_AAS_THRESHOLD_TRACE")
    threshold = result.get("threshold")
    if threshold is not None and (
            type(threshold) not in (int, float) or not math.isfinite(threshold)):
        _reject("R94_INVALID_AAS_THRESHOLD")
    return {
        "schema": "phm.autogen.detector-result.v1",
        "decision_mode": mode,
        "pred_label": list(labels),
        "anomaly_score": None if scores is None else [float(v) for v in scores],
        "threshold": None if threshold is None else float(threshold),
        "thresholds": [float(v) for v in thresholds],
        "threshold_owner": "detector_not_autogen",
        "label_owner": "detector_not_autogen",
        "independent_model_validation": False,
    }


def _freeze_detector_decision(detector, batch):
    """AAS' supplied loader is authoritative when importable.

    An embedded AAS application should pass its original detect_with_detector
    bound function, so BaseDetector's non-overridable dispatch and snapshot/
    restore logic run exactly as in AAS. Standalone callers may rely on the
    AAS detector's detect() and export/restore-evaluation-state capabilities.
    """
    export = getattr(detector, "export_evaluation_state", None)
    restore = getattr(detector, "restore_evaluation_state", None)
    if not callable(export) or not callable(restore):
        _reject("R94_AAS_STATE_PRESERVATION_HOOKS_REQUIRED")
    snapshot = deepcopy(export())
    try:
        return detector.detect(batch)
    finally:
        restore(deepcopy(snapshot))


class AASDetectorBridge:
    """Single-use TRAIN-only adapter around one AAS detector instance.

    Explicitly supports only binary NORMAL/FAULT or NORMAL/ANOMALY for
    AAS' original 0/1 decision contract. No universal multiclass coercion.
    """

    def __init__(self, detector, *, descriptor):
        if not isinstance(descriptor, Mapping) or set(descriptor) != {
                "name", "class_path", "decision_mode", "config",
                "candidate_family"}:
            _reject("R94_INVALID_AAS_DESCRIPTOR")
        name, mode, config = (descriptor[k] for k in (
            "name", "decision_mode", "config"))
        if type(name) is not str or not name.strip():
            _reject("R94_AAS_DETECTOR_NAME_REQUIRED")
        if type(descriptor["class_path"]) is not str or not descriptor["class_path"].strip():
            _reject("R94_AAS_CLASS_PATH_REQUIRED")
        if mode not in DECISION_MODES:
            _reject("R94_AAS_DECISION_MODE_REQUIRED")
        if type(descriptor["candidate_family"]) is not str or not descriptor["candidate_family"]:
            _reject("R94_AAS_CANDIDATE_FAMILY_REQUIRED")
        if not isinstance(config, Mapping) or any(
                type(k) is not str for k in config):
            _reject("R94_INVALID_AAS_CONFIG")
        if config.get("decision_mode", mode) != mode:
            _reject("R94_CONFLICTING_DECLARED_DECISION_MODES")
        if mode != "score_threshold" and set(config) & _THRESHOLD_CONFLICTS:
            _reject("R94_NON_THRESHOLD_DETECTOR_CANNOT_SET_GENERIC_THRESHOLD")
        if getattr(detector, "decision_mode", None) != mode:
            _reject("R94_RUNTIME_DECISION_MODE_MISMATCH")
        for method in ("fit", "detect", "export_evaluation_state",
                       "restore_evaluation_state"):
            if not callable(getattr(detector, method, None)):
                _reject("R94_INCOMPLETE_AAS_DETECTOR_PROTOCOL")
        self.detector = detector
        self.descriptor = deepcopy(dict(descriptor))
        self._plan = None
        self._receipt = None
        self._channels = None
        self._labels = None

    def fit(self, equipment, train_manifest, trial, train_features):
        if self._plan is not None:
            _reject("R94_REFIT_AFTER_TRAIN_FREEZE_FORBIDDEN")
        source_plan = compile_source_plan(equipment, train_manifest, trial)
        labels = trial["labels"]
        if len(labels) != 2 or labels[0] != trial["healthy_label"]:
            _reject("R94_AAS_BINARY_ZERO_HEALTHY_ONE_FAULT_REQUIRED")
        if trial["task_type"] == "classification" and self.descriptor["decision_mode"] != "label":
            _reject("R94_SUPERVISED_CLASSIFICATION_REQUIRES_LABEL_MODE")
        if trial["task_type"] == "anomaly_detection" and self.descriptor["decision_mode"] == "label":
            _reject("R94_UNSUPERVISED_ANOMALY_DISALLOWS_LABEL_MODE")
        # AAS descriptor name != Autogen abstract family; bind explicitly.
        if self.descriptor["candidate_family"] not in trial["candidate_families"]:
            _reject("R94_AAS_ALGORITHM_NOT_AUTHORIZED_BY_TRIAL")
        indexed = {r["record_id"]: r for r in train_manifest["records"]}
        if any(r["role"] != "TRAIN" for r in indexed.values()):
            _reject("R94_PROTECTED_SPLIT_IN_TRAIN_DATA")
        # Exact manifest order, regardless of caller row ordering.
        if not isinstance(train_features, (list, tuple)) or not all(
                isinstance(item, Mapping) and "record_id" in item
                for item in train_features):
            _reject("R94_MALFORMED_TRAIN_FEATURE_BATCH")
        source_ids = [item["record_id"] for item in train_features]
        if len(source_ids) != len(set(source_ids)):
            _reject("R94_DUPLICATE_SOURCE_FEATURE_RECORD")
        if set(source_ids) != set(indexed):
            _reject("R94_TRAIN_RECORDS_DO_NOT_MATCH_SOURCE_MANIFEST")
        table = {item["record_id"]: item for item in train_features}
        batch = _table([table[rid] for rid in indexed],
                       trial["input_channels"], train_ids=set(indexed))
        train_y = ([labels.index(indexed[rid]["label"]) for rid in indexed]
                   if trial["task_type"] == "classification" else None)
        # AAS' detector-specific fit and decision calibration remain the
        # only authority, with CAL/VAL/FINAL always absent. The detector may
        # choose to use train_y for classification, but never sees VAL.
        import numpy as np
        matrix = np.asarray(batch, dtype=np.float64)
        start = perf_counter()
        # Real AAS BaseDetector accepts positional train_y, not 'y='.
        self.detector.fit(matrix, train_y)
        if (self.descriptor["decision_mode"] == "score_threshold" and
                callable(getattr(self.detector, "needs_threshold_calibration", None)) and
                self.detector.needs_threshold_calibration()):
            scores = self.detector.score(matrix)
            self.detector.fit_threshold(scores)
        elapsed = perf_counter() - start
        if elapsed > trial["budget"]["max_fit_seconds"]:
            _reject("R94_POSTHOC_FIT_WALL_TIME_LIMIT_EXCEEDED")
        self._plan = source_plan
        self._channels = tuple(trial["input_channels"])
        self._labels = tuple(labels)
        self._receipt = {
            "schema": "phm.autogen.aas-candidate-resource.v1",
            "detector_name": self.descriptor["name"],
            "backend_contract": "AAS.BaseDetector",
            "descriptor_sha256": _digest(self.descriptor),
            "source_plan_sha256": _digest(source_plan),
            "fit_calls": 1,
            "fit_wall_seconds_observed": elapsed,
            "source_record_count": len(matrix),
            "source_physical_group_count": source_plan["source_group_count"],
            "max_seconds_checked_after_fit": True,
            "hard_process_timeout_enforced_here": False,
            "memory_peak_measured_here": False,
            "model_bytes_measured_here": False,
            "backend_worker_isolation_verified_here": False,
            "calibration_validation_final_read": False,
            "independent_model_performance_proven": False,
            "production_approved": False,
        }
        return self

    def predict(self, observations):
        if self._plan is None:
            _reject("R94_CANDIDATE_NOT_FITTED")
        features = _table(observations, self._channels)
        import numpy as np
        matrix = np.asarray(features, dtype=np.float64)
        result = _freeze_detector_decision(self.detector, matrix)
        decision = _validate_decision(result, len(matrix),
                                      self.descriptor["decision_mode"])
        decision["predicted_class"] = [
            self._labels[n] for n in decision["pred_label"]]
        return decision

    def export_model(self):
        """Emit auditable handoff metadata; do not replace AAS' save/load.

        AAS controls artifact locations, pickle trust, model registration and
        resumption from Phase 39-41. No untrusted model executable is emitted.
        """
        if self._plan is None:
            _reject("R94_CANDIDATE_NOT_FITTED")
        return {
            "schema": SCHEMA, "status": "AAS_OWNED_PERSISTENCE",
            "descriptor": deepcopy(self.descriptor),
            "input_channels": list(self._channels),
            "classes_zero_one": list(self._labels),
            "source_plan_sha256": _digest(self._plan),
            "model_binary": None,
            "model_storage_and_loading_owned_by": "AAS",
            "serialized_pickled_model_created_by_autogen": False,
            "model_performance_qualified": False,
            "industrial_production_ready": False,
        }

    def resource_receipt(self):
        if self._receipt is None:
            _reject("R94_NO_FIT_RECEIPT")
        return deepcopy(self._receipt)
