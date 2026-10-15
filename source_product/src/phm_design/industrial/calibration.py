"""R81c source-TRAIN-only probability calibration and selective UNKNOWN decisions.

The calibrator is fitted only from out-of-fold probabilities supplied by a
caller that enforces a separate, disjoint training inner fold. It never accesses
independent FINAL labels and makes no distribution-free coverage guarantee.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


class CalibrationRejected(ValueError):
    """Invalid or unsupported TRAIN calibration evidence."""


@dataclass(frozen=True)
class TemperatureSelectiveCalibrator:
    temperature: float
    threshold: float
    calibration_samples: int
    calibration_coverage: float
    calibration_error_upper_bound: float
    method: str = "temperature_selective"

    def probabilities(self, values) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        _validate_probabilities(values)
        power = np.power(np.maximum(values, 1.e-12), 1. / self.temperature)
        return power / power.sum(axis=1, keepdims=True)

    def decide(self, values) -> np.ndarray:
        calibrated = self.probabilities(values)
        predicted = calibrated.argmax(axis=1).astype(int)
        predicted[calibrated.max(axis=1) < self.threshold] = -1
        return predicted

    def decisions(self, values) -> list[dict]:
        calibrated = self.probabilities(values)
        indices = calibrated.argmax(axis=1)
        confidence = calibrated.max(axis=1)
        return [
            {"class_index": (int(idx) if conf >= self.threshold else -1),
             "confidence": float(conf),
             "abstained": bool(conf < self.threshold)}
            for idx, conf in zip(indices, confidence)
        ]


def _validate_probabilities(values):
    if (values.ndim != 2 or values.shape[0] == 0 or values.shape[1] < 2
            or not np.isfinite(values).all() or (values < 0).any()
            or not np.allclose(values.sum(axis=1), 1., atol=1.e-4)):
        raise CalibrationRejected("INVALID_CALIBRATION_PROBABILITIES")


def _risk_upper_bound(errors: int, n: int) -> float:
    """One-sided Wilson 95% upper risk bound (descriptive TRAIN evidence)."""
    z = 1.645
    p = errors / n
    z2 = z * z
    return float((p + z2 / (2 * n) +
                  z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)))
                 / (1 + z2 / n))


def fit_temperature_selective(probabilities, labels, policy: dict):
    """Fit temperature and refusal cut on *inner TRAIN OOF* only.

    max_selective_error is applied to a conservative Wilson upper bound,
    rather than to a point estimate. Risk/coverage are TRAIN descriptive
    screens, not performance or calibration guarantees on new equipment.
    """
    values = np.asarray(probabilities, dtype=float)
    y = np.asarray(labels)
    _validate_probabilities(values)
    n, k = values.shape
    if (y.shape != (n,) or y.dtype.kind not in "iu" or
            (y < 0).any() or (y >= k).any() or
            len(set(y.tolist())) != k):
        raise CalibrationRejected("CALIBRATION_CLASS_SUPPORT_MISSING")
    if n < policy["min_calibration_samples"]:
        raise CalibrationRejected("CALIBRATION_SAMPLE_SUPPORT_INSUFFICIENT")
    best = None
    # Fixed predeclared calibration family; no outer heldout labels are read.
    for temperature in (0.5, 0.75, 1., 1.5, 2., 3., 4.):
        scores = np.power(np.maximum(values, 1.e-12), 1. / temperature)
        scores /= scores.sum(axis=1, keepdims=True)
        loss = -float(np.log(np.maximum(scores[np.arange(n), y], 1.e-12)).mean())
        if best is None or loss < best[0]:
            best = (loss, temperature, scores)
    _, temperature, calibrated = best
    confidence = calibrated.max(axis=1)
    predicted = calibrated.argmax(axis=1)
    eligible = []
    for threshold in np.unique(confidence):
        chosen = confidence >= threshold
        count = int(chosen.sum())
        coverage = count / n
        if coverage < policy["minimum_coverage"]:
            continue
        upper = _risk_upper_bound(int(np.count_nonzero(predicted[chosen] != y[chosen])),
                                  count)
        if upper <= policy["max_selective_error"]:
            eligible.append((coverage, -float(threshold), float(threshold), upper))
    if not eligible:
        raise CalibrationRejected("CALIBRATION_RISK_OR_COVERAGE_UNMET")
    coverage, _, threshold, upper = max(eligible)
    return TemperatureSelectiveCalibrator(
        temperature=float(temperature), threshold=threshold,
        calibration_samples=n, calibration_coverage=float(coverage),
        calibration_error_upper_bound=upper)
