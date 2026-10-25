"""R94 AAS-compatibility orchestration, not a duplicate detector library.

These are protocol doubles of the *actual uploaded AAS BaseDetector* API:
fit(train_X, train_y=None), detect(test_X) and evaluation-state
snapshot/restore. A separate local compatibility audit against the user ZIP
is documented; the CI does not claim AAS was bundled into the repository.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import json

import numpy as np
import pytest

from phm_design.industrial_contracts.aas_bridge import (
    AASDetectorBridge, DetectorBridgeRejected, bind_existing_aas_detector,
)
from phm_design.industrial_contracts.contracts import ContractRejected

EXAMPLES = Path(__file__).resolve().parents[2] / "examples/industrial_contracts"


def load(name):
    return json.loads((EXAMPLES / name).read_text(encoding="utf-8"))


def fixtures(prefix):
    return (load(prefix+"_equipment.json"),
            load(prefix+"_train_only.json"),
            load(prefix+"_trial.json"),
            load(prefix+"_train_features.json"),
            load(prefix+"_inference_features.json"))


class AASScoreThresholdDouble:
    decision_mode = "score_threshold"

    def __init__(self):
        self.fit_calls = 0
        self.calibration_calls = 0
        self.ref_median = None
        self.threshold = None
        self.counter = 0

    def fit(self, train_X, train_y=None, valid_X=None, valid_y=None):
        assert valid_X is None and valid_y is None
        assert train_y is None
        assert train_X.shape[1] == 3
        self.fit_calls += 1
        self.ref_median = np.median(train_X, axis=0)

    def score(self, test_X):
        return np.max(np.abs(test_X-self.ref_median), axis=1).tolist()

    def needs_threshold_calibration(self):
        return self.threshold is None

    def fit_threshold(self, scores):
        self.calibration_calls += 1
        self.threshold = max(scores)+1e-9

    def detect(self, test_X):
        assert self.threshold is not None
        self.counter += len(test_X)
        scores = self.score(test_X)
        return {"decision_mode": self.decision_mode,
                "pred_label": [int(v > self.threshold) for v in scores],
                "anomaly_score": scores, "threshold": self.threshold,
                "thresholds": [self.threshold]*len(scores)}

    def export_evaluation_state(self):
        return {"counter": self.counter, "threshold": self.threshold}

    def restore_evaluation_state(self, snapshot):
        self.counter = snapshot["counter"]
        self.threshold = snapshot["threshold"]


class AASNativeDouble(AASScoreThresholdDouble):
    decision_mode = "native"

    def detect(self, test_X):
        self.counter += len(test_X)
        scores = self.score(test_X)
        return {"decision_mode": "native",
                "pred_label": [int(v>1) for v in scores],
                "anomaly_score": scores, "threshold": None, "thresholds": []}


class AASLabelDouble(AASScoreThresholdDouble):
    decision_mode = "label"

    def fit(self, train_X, train_y=None, valid_X=None, valid_y=None):
        assert valid_X is None and valid_y is None
        assert train_y == [0, 1, 0, 1]
        self.fit_calls += 1
        self.midpoint = (train_X[:, 0].max()+train_X[:, 0].min())/2

    def detect(self, test_X):
        self.counter += len(test_X)
        return {"decision_mode": "label", "pred_label":
                [int(row[0] > self.midpoint) for row in test_X],
                "anomaly_score": None, "threshold": None, "thresholds": []}


def descriptor(name, family, mode):
    return {"name": name, "class_path": "detectors.fake.TestOnly",
            "candidate_family": family, "decision_mode": mode, "config": {}}


@pytest.mark.parametrize("mode,factory", [
    ("score_threshold", AASScoreThresholdDouble),
    ("native", AASNativeDouble),
])
def test_aas_anomaly_score_and_native_decisions_remain_detector_owned(mode, factory):
    eq, train, trial, rows, inference = fixtures("compressor")
    detector = factory()
    bridge = AASDetectorBridge(detector, descriptor=descriptor(
        "aas_"+mode, "statistical", mode)).fit(eq, train, trial, rows)
    assert detector.fit_calls == 1
    assert detector.counter == 0
    assert (detector.calibration_calls == 1 if mode == "score_threshold"
            else detector.calibration_calls == 0)
    first = bridge.predict(inference)
    assert first["decision_mode"] == mode
    assert first["pred_label"] == [0, 1]
    assert first["predicted_class"] == ["NORMAL", "ANOMALY"]
    assert first["threshold_owner"] == "detector_not_autogen"
    assert first["label_owner"] == "detector_not_autogen"
    assert len(first["anomaly_score"]) == len(inference)
    assert first["independent_model_validation"] is False
    assert detector.counter == 0  # snapshot + restore
    assert first == bridge.predict(inference)
    receipt = bridge.resource_receipt()
    assert receipt["fit_calls"] == 1 and receipt["source_record_count"] == 2
    assert receipt["calibration_validation_final_read"] is False
    assert receipt["hard_process_timeout_enforced_here"] is False
    assert receipt["backend_worker_isolation_verified_here"] is False
    handoff = bridge.export_model()
    assert handoff["model_storage_and_loading_owned_by"] == "AAS"
    assert handoff["model_binary"] is None
    assert handoff["industrial_production_ready"] is False
    with pytest.raises(DetectorBridgeRejected, match="R94_REFIT_AFTER_TRAIN_ATTEMPT_FORBIDDEN"):
        bridge.fit(eq, train, trial, rows)


def test_label_only_classification_does_not_invent_anomaly_score_or_threshold():
    eq, train, trial, rows, inference = fixtures("bearing")
    detector = AASLabelDouble()
    bridge = AASDetectorBridge(detector, descriptor=descriptor(
        "label_classifier", "linear", "label")).fit(eq, train, trial, rows)
    got = bridge.predict(inference)
    assert got["decision_mode"] == "label"
    assert got["anomaly_score"] is None
    assert got["threshold"] is None
    assert got["pred_label"] == [0, 1]
    assert got["predicted_class"] == ["HEALTHY", "FAULT"]
    assert detector.counter == 0
    assert detector.fit_calls == 1


@pytest.mark.parametrize("mode,family", [
    ("score_threshold", "linear"),
    ("native", "linear"),
    ("label", "statistical"),
])
def test_task_and_aas_decision_mode_must_agree(mode, family):
    target = "bearing" if family == "linear" else "compressor"
    eq, train, trial, rows, _ = fixtures(target)
    d = {"label": AASLabelDouble, "score_threshold": AASScoreThresholdDouble,
         "native": AASNativeDouble}[mode]()
    with pytest.raises(DetectorBridgeRejected, match="R94_(SUPERVISED_CLASSIFICATION_REQUIRES_LABEL_MODE|UNSUPERVISED_ANOMALY_DISALLOWS_LABEL_MODE)"):
        AASDetectorBridge(d, descriptor=descriptor(
            "algorithm", family, mode)).fit(eq, train, trial, rows)


def test_generic_threshold_parameters_rejected_in_native_mode():
    spec = descriptor("native", "statistical", "native")
    spec["config"] = {"threshold_method": "fixed", "threshold": 1}
    with pytest.raises(DetectorBridgeRejected, match="R94_NON_THRESHOLD_DETECTOR_CANNOT_SET_GENERIC_THRESHOLD"):
        AASDetectorBridge(AASNativeDouble(), descriptor=spec)


def test_train_or_validation_roles_cannot_cross_bridge_boundary():
    eq, _, trial, rows, _ = fixtures("compressor")
    complete = load("compressor_trusted_audit.json")
    bridge = AASDetectorBridge(AASScoreThresholdDouble(),
        descriptor=descriptor("three_sigma", "statistical", "score_threshold"))
    with pytest.raises(ContractRejected, match="R93_PROTECTED_ROLE_IN_DESIGN_WORKER"):
        bridge.fit(eq, complete, trial, rows)


def test_duplicate_record_rows_and_unknown_feature_properties_refused():
    eq, train, trial, rows, _ = fixtures("compressor")
    extra = deepcopy(rows)
    extra[1] = deepcopy(extra[0])
    with pytest.raises(DetectorBridgeRejected, match="R94_DUPLICATE_SOURCE_FEATURE_RECORD"):
        AASDetectorBridge(AASScoreThresholdDouble(), descriptor=descriptor(
            "three_sigma", "statistical", "score_threshold")).fit(eq, train, trial, extra)
    extra = deepcopy(rows)
    extra[0]["observations"]["ground_truth"] = 1
    with pytest.raises(DetectorBridgeRejected, match="R94_UNKNOWN_PROTECTED_OR_MISSING_SENSOR_FIELD"):
        AASDetectorBridge(AASScoreThresholdDouble(), descriptor=descriptor(
            "three_sigma", "statistical", "score_threshold")).fit(eq, train, trial, extra)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), True, 1e16])
def test_numeric_finiteness_and_boolean_coercion_refused(bad):
    eq, train, trial, rows, _ = fixtures("compressor")
    rows[0]["observations"]["pressure"] = bad
    with pytest.raises(DetectorBridgeRejected, match="R94_NONFINITE_OR_INVALID_SENSOR_VALUE"):
        AASDetectorBridge(AASScoreThresholdDouble(), descriptor=descriptor(
            "three_sigma", "statistical", "score_threshold")).fit(eq, train, trial, rows)


def test_cannot_infer_candidate_family_from_arbitrary_detector_name():
    eq, train, trial, rows, _ = fixtures("compressor")
    spec = descriptor("whatever_algorithm", "tree", "native")
    with pytest.raises(DetectorBridgeRejected, match="R94_AAS_ALGORITHM_NOT_AUTHORIZED_BY_TRIAL"):
        AASDetectorBridge(AASNativeDouble(), descriptor=spec).fit(eq, train, trial, rows)


def test_detector_cannot_be_invoked_without_source_fit():
    eq, _, _, _, inference = fixtures("compressor")
    bridge = AASDetectorBridge(AASNativeDouble(), descriptor=descriptor(
        "native", "statistical", "native"))
    for call in (lambda: bridge.predict(inference), bridge.export_model,
                 bridge.resource_receipt):
        with pytest.raises(DetectorBridgeRejected):
            call()


def test_training_and_detection_outputs_must_preserve_aas_final_label_semantics():
    eq, train, trial, rows, inference = fixtures("compressor")
    class BadAAS(AASScoreThresholdDouble):
        def detect(self, test_X):
            result = super().detect(test_X)
            result["pred_label"] = [0]  # length spoof
            return result
    bridge = AASDetectorBridge(BadAAS(), descriptor=descriptor(
        "three_sigma", "statistical", "score_threshold")).fit(eq, train, trial, rows)
    with pytest.raises(DetectorBridgeRejected, match="R94_INVALID_AAS_PRED_LABEL"):
        bridge.predict(inference)
    assert bridge.detector.counter == 0


def test_aas_threshold_must_be_train_calibrated_not_invented_from_review_rows():
    eq, train, trial, rows, _ = fixtures("compressor")
    class BrokenCalibration(AASScoreThresholdDouble):
        def fit_threshold(self, scores):
            self.calibration_calls += 1   # never initializes threshold
    with pytest.raises(DetectorBridgeRejected, match="R94_AAS_TRAIN_ONLY_THRESHOLD_NOT_CALIBRATED"):
        AASDetectorBridge(BrokenCalibration(), descriptor=descriptor(
            "three_sigma", "statistical", "score_threshold")).fit(eq, train, trial, rows)



def test_existing_aas_descriptor_binding_calls_original_state_preserving_loader():
    eq, train, trial, features, inference = fixtures("compressor")
    det = AASScoreThresholdDouble()
    calls = []

    def authoritative_service(detector, matrix, *, preserve_state):
        assert preserve_state is True
        calls.append("AAS.detectors.loader.detect_with_detector")
        before = detector.export_evaluation_state()
        try:
            return detector.detect(matrix)
        finally:
            detector.restore_evaluation_state(before)

    # Structure copied from supplied AAS configs/detectors/three_sigma.yaml;
    # class_path remains descriptive and is NOT imported by the adapter.
    original = {
        "name": "three_sigma", "label": "3-Sigma",
        "description": "statistical detector", "min_training_samples": 2,
        "class_path": "detectors.statistical_detectors.ThreeSigmaDetector",
        "type": "statistical", "dependencies": ["numpy"],
        "config": {}, "decision_mode": "score_threshold",
        "decision_modes": ["score_threshold"],
    }
    bridge = bind_existing_aas_detector(
        det, original, candidate_family="statistical",
        detection_service=authoritative_service).fit(eq, train, trial, features)
    first = bridge.predict(inference)
    assert first["pred_label"] == [0, 1]
    assert calls == ["AAS.detectors.loader.detect_with_detector"]
    assert bridge.resource_receipt()["backend_contract"] == "AAS.BaseDetector"
    assert bridge.export_model()["descriptor"]["class_path"] == original["class_path"]


def test_external_detection_service_cannot_change_mode_or_length():
    eq, train, trial, features, inference = fixtures("compressor")
    det = AASScoreThresholdDouble()
    def spoofed_service(detector, matrix, *, preserve_state):
        return {"decision_mode": "native", "pred_label": [0, 1],
                "anomaly_score": None}
    bridge = bind_existing_aas_detector(det, {
        "name": "zscore", "class_path": "detectors.statistical_detectors.ZScoreDetector",
        "config": {}, "decision_mode": "score_threshold",
    }, candidate_family="statistical",
    detection_service=spoofed_service).fit(eq, train, trial, features)
    with pytest.raises(DetectorBridgeRejected, match="R94_AAS_DECISION_MODE_MISMATCH"):
        bridge.predict(inference)
