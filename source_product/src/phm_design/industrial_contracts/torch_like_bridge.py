"""R94 P1.2 — thin torch_like provider bridge; NO duplicate algorithm engine.

autogen owns source-role validation, feature admission, trial budget bookkeeping
and evidence qualification. torch_like owns actual sklearn/PyTorch algorithm
instantiation, runtime operations and persistence. This bridge currently admits
only the explicitly mapped *tabular native sklearn* candidates. Other
candidate families are NOT silently emulated.

For trust: torch_like's sklearn load_model uses joblib/pickle. Never load a
directory from an untrusted source. No loader is exposed by this bridge.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter
import hashlib
import json
import math

from .contracts import ContractRejected, compile_source_plan


class ProviderRefused(ContractRejected):
    pass


def _reject(reason):
    raise ProviderRefused(reason)


# Map product-facing families to REAL provider implementations, never the
# other way around. Backend type and tasks are immutable policy decisions.
TABULAR_PROVIDER = {
    ("classification", "linear"): ("LogisticRegression", "classification"),
    ("classification", "tree"): ("DecisionTreeClassifier", "classification"),
    ("classification", "random_forest"): ("RandomForestClassifier", "classification"),
    ("anomaly_detection", "isolation_forest"): ("IsolationForest", "anomaly_detection"),
}
MAX_ABS = 1e12


def _vector_rows(rows, channels):
    if not isinstance(rows, (list, tuple)) or not rows:
        _reject("R94_EMPTY_OBSERVATIONS")
    expected = set(channels)
    output = []
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != expected:
            _reject("R94_UNDECLARED_OR_PROTECTED_CHANNEL")
        vector = [row[channel] for channel in channels]
        if any(type(value) not in (int, float) or not math.isfinite(value)
               or abs(value) > MAX_ABS for value in vector):
            _reject("R94_NONFINITE_OR_INVALID_SENSOR")
        output.append([float(value) for value in vector])
    return output


def _source_features(train, values, input_channels):
    """Never accept extra/duplicate/protected records through source features."""
    if not isinstance(values, (list, tuple)) or len(values) != len(train["records"]):
        _reject("R94_SOURCE_RECORD_COUNT_MISMATCH")
    expected = [r["record_id"] for r in train["records"]]
    lookup = {}
    for sample in values:
        if not isinstance(sample, Mapping) or set(sample) != {"record_id", "observations"}:
            _reject("R94_SOURCE_FEATURE_SCHEMA")
        rid = sample["record_id"]
        if not isinstance(rid, str) or rid not in expected or rid in lookup:
            _reject("R94_UNKNOWN_OR_REPEATED_RECORD")
        lookup[rid] = sample["observations"]
    return _vector_rows([lookup[i] for i in expected], input_channels)


def _torch_like_provider():
    """Lazy optional dependency. Neither Stable main nor R93 requires it."""
    try:
        import torch_like as tl
    except ImportError as exc:
        raise ProviderRefused("R94_OPTIONAL_TORCH_LIKE_NOT_INSTALLED") from exc
    return tl


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def _sha(value):
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


class TorchLikeTabularCandidate:
    """Lifecycle bridge over existing torch_like.ModelRuntime/native estimator.

    The provider's estimator runs fit/score/predict. autogen never reimplements
    the mathematical estimator or writes a parallel joblib serializer.
    """

    def __init__(self, family: str, *, provider=None):
        self.family = family
        self._provider = provider
        self._runtime = None
        self._model = None
        self._plan = None
        self._receipt = None
        self._trial = None

    def fit(self, equipment: Mapping, source_train: Mapping,
            trial: Mapping, features: Sequence[Mapping]):
        if self._runtime is not None:
            _reject("R94_FITTED_CANDIDATE_CANNOT_RETRAIN")
        plan = compile_source_plan(equipment, source_train, trial)
        if self.family not in trial["candidate_families"]:
            _reject("R94_CANDIDATE_NOT_PREDECLARED")
        key = (trial["task_type"], self.family)
        if key not in TABULAR_PROVIDER:
            _reject("R94_UNSUPPORTED_PROVIDER_TASK_AND_FAMILY")
        provider = self._provider or _torch_like_provider()
        algorithm, required_kind = TABULAR_PROVIDER[key]
        # First inspect the REAL provider's native contract, rather than
        # advertising every future registry entry as executable.
        descriptor = provider.get_estimator_info(algorithm)
        if descriptor["kind"] != required_kind:
            _reject("R94_PROVIDER_CAPABILITY_MISMATCH")
        instance = provider.create_estimator(
            algorithm, **self._constructor_options(algorithm, trial))
        runtime = provider.as_runtime(instance)
        if not runtime.supports("fit") or not runtime.supports("predict"):
            _reject("R94_PROVIDER_MISSING_REQUIRED_NATIVE_OPERATION")
        x = _source_features(source_train, features, trial["input_channels"])
        labels = [row["label"] for row in source_train["records"]]
        if trial["task_type"] == "classification":
            if len(trial["labels"]) != 2 or set(labels) != set(trial["labels"]):
                _reject("R94_CLASSIFICATION_NEEDS_TWO_SOURCE_CLASSES")
        else:
            if len(x) < 2 or any(label not in (None, trial["healthy_label"]) for label in labels):
                _reject("R94_ANOMALY_REQUIRES_ONLY_NORMAL_SOURCE")
            if not callable(getattr(instance, "decision_function", None)):
                _reject("R94_ANOMALY_PROVIDER_REQUIRES_DECISION_FUNCTION")
        started = perf_counter()
        if trial["task_type"] == "classification":
            runtime.fit(x, labels)
        else:
            runtime.fit(x)
        elapsed = perf_counter() - started
        if elapsed > trial["budget"]["max_fit_seconds"]:
            _reject("R94_POSTHOC_FIT_SECONDS_EXCEEDED")
        self._provider = provider
        self._model = instance
        self._runtime = runtime
        self._plan = plan
        self._trial = trial
        self._receipt = {
            "schema": "phm.autogen.external-provider-resource.v1",
            "provider": "torch_like", "native_algorithm": algorithm,
            "task_type": trial["task_type"],
            "source_plan_sha256": _sha(plan),
            "source_groups": plan["source_group_count"],
            "source_records": len(x),
            "train_fit_calls": 1,
            "fit_wall_seconds_observed": float(elapsed),
            "hard_deadline_enforced": False,
            "cpu_gpu_time_parity_proven": False,
            "flops_or_peak_memory_measured": False,
            "validation_or_final_used_for_fit": False,
            "independent_model_validated": False,
            "production_qualified": False,
        }
        return self

    @staticmethod
    def _constructor_options(algorithm, trial):
        seed = trial["budget"]["seed"]
        if algorithm == "LogisticRegression":
            return {"max_iter": 400, "random_state": seed}
        if algorithm == "DecisionTreeClassifier":
            return {"random_state": seed, "max_depth": 5}
        if algorithm == "RandomForestClassifier":
            return {"random_state": seed, "n_estimators": 40, "n_jobs": 1}
        if algorithm == "IsolationForest":
            return {"random_state": seed, "n_estimators": 40, "n_jobs": 1,
                    "contamination": "auto"}
        _reject("R94_UNKNOWN_PROVIDER_CONSTRUCTOR")

    def predict(self, records):
        if self._runtime is None:
            _reject("R94_FIT_REQUIRED_BEFORE_PREDICTION")
        x = _vector_rows(records, self._trial["input_channels"])
        native = self._runtime.predict(x)
        if self._trial["task_type"] == "classification":
            labels = [str(v) for v in native]
            if not set(labels) <= set(self._trial["labels"]):
                _reject("R94_PROVIDER_PREDICTED_UNDECLARED_CLASS")
            return [{"predicted_label": label,
                     "diagnostic_score": None,
                     "score_semantics": "native_class_no_probability_claim"}
                    for label in labels]
        # IsolationForest native predict returns +1/-1. Decision function
        # positive indicates inlier; negate so larger = more anomalous.
        values = self._model.decision_function(x)
        if len(native) != len(x) or len(values) != len(x):
            _reject("R94_NATIVE_PREDICTION_SHAPE_MISMATCH")
        abnormal = next(v for v in self._trial["labels"] if v != self._trial["healthy_label"])
        output = []
        for decision, score in zip(native, values):
            if int(decision) not in (-1, 1) or not math.isfinite(float(score)):
                _reject("R94_NONFINITE_OR_UNEXPECTED_ANOMALY_OUTPUT")
            output.append({
                "predicted_label": abnormal if int(decision) == -1 else self._trial["healthy_label"],
                "diagnostic_score": float(-score),
                "score_semantics": "minus_native_isolationforest_decision_function",
            })
        return output

    def export_model(self, path):
        """Delegate native persistence to torch_like; NOT safe for untrusted loading."""
        if self._runtime is None:
            _reject("R94_NO_FITTED_PROVIDER_MODEL")
        root = Path(path)
        if root.exists():
            _reject("R94_REFUSE_TO_OVERWRITE_PROVIDER_ARTIFACT")
        created = self._provider.save_model(self._model, root, overwrite=False)
        manifest = Path(created) / "model.json"
        binary = Path(created) / "estimator.joblib"
        if not manifest.is_file() or not binary.is_file():
            _reject("R94_INCOMPLETE_PROVIDER_NATIVE_ARTIFACT")
        return {
            "schema": "phm.autogen.external-model-artifact.v1",
            "provider": "torch_like",
            "artifact_root": str(created),
            "native_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "native_estimator_joblib_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
            "contains_pickle_unsafe_for_untrusted_loading": True,
            "independent_model_validated": False,
            "production_qualified": False,
        }

    def resource_receipt(self):
        if self._receipt is None:
            _reject("R94_FIT_REQUIRED_BEFORE_RESOURCE_RECEIPT")
        return dict(self._receipt)


def available_adapters():
    """No import required; this is a MAP of provider-owned capabilities."""
    return [{"task_type": task, "family": family,
             "provider": "torch_like", "native_name": name,
             "native_kind": kind, "implementation_in_autogen": False}
            for (task, family), (name, kind) in sorted(TABULAR_PROVIDER.items())]
