"""R94: autogen orchestrates; torch_like owns algorithms/runtime/save.

The initial no-dependency tests use a narrow capability spy, not pretend
measurements. A separately enabled CI job uses the REAL pinned torch_like
library and native sklearn estimators for execution/persistence.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import json
import math

import pytest

from phm_design.industrial_contracts.torch_like_bridge import (
    ProviderRefused, TorchLikeTabularCandidate, available_adapters)

EX = Path(__file__).resolve().parents[2] / "examples/industrial_contracts"


def read(name):
    return json.loads((EX / name).read_text(encoding="utf-8"))


def fixture(device):
    return (
        read(f"{device}_equipment.json"),
        read(f"{device}_train_only.json"),
        read(f"{device}_trial.json"),
        read(f"{device}_train_features.json"),
        read(f"{device}_inference_features.json"),
    )


class SpyRuntime:
    def __init__(self, estimator):
        self.estimator = estimator
    def supports(self, name):
        return name in ("fit", "predict")
    def fit(self, data, labels=None):
        self.estimator.fit(data, labels)
    def predict(self, data):
        return self.estimator.predict(data)


class SpyEstimator:
    def __init__(self, name):
        self.name = name
        self.fit_called = 0
        self.labels = None
    def fit(self, data, labels=None):
        self.fit_called += 1
        self.labels = labels
        self.columns = len(data[0])
        return self
    def predict(self, data):
        if self.name == "IsolationForest":
            return [1 if row[0] < 10 else -1 for row in data]
        return ["HEALTHY" if row[0] < 1.0 else "FAULT" for row in data]
    def decision_function(self, data):
        return [1.0 if row[0] < 10 else -2.0 for row in data]


class SpyTorchLike:
    def __init__(self):
        self.created = []
        self.last = None
        self.saved = []
    def get_estimator_info(self, name):
        return {"name": name, "kind":
                "anomaly_detection" if name == "IsolationForest"
                else "classification", "backend": "sklearn"}
    def create_estimator(self, name, **kwargs):
        self.created.append((name, kwargs))
        self.last = SpyEstimator(name)
        return self.last
    def as_runtime(self, estimator):
        return SpyRuntime(estimator)
    def save_model(self, model, path, *, overwrite=False):
        path = Path(path)
        assert not overwrite
        path.mkdir(exist_ok=False)
        (path/"model.json").write_text('{"library":"torch_like"}', encoding="utf-8")
        (path/"estimator.joblib").write_bytes(b"fake-mocked-joblib-not-usable")
        self.saved.append((model, path))
        return path


@pytest.mark.parametrize("candidate,device,algorithm", [
    ("linear", "bearing", "LogisticRegression"),
    ("tree", "bearing", "DecisionTreeClassifier"),
    ("random_forest", "bearing", "RandomForestClassifier"),
    ("isolation_forest", "compressor", "IsolationForest"),
])
def test_native_provider_owns_fit_predict_and_no_duplicate_math(
        candidate, device, algorithm):
    eq, train, trial, features, incoming = fixture(device)
    if candidate in {"tree", "random_forest"}:
        trial["candidate_families"].append(candidate)
    provider = SpyTorchLike()
    bridge = TorchLikeTabularCandidate(candidate, provider=provider)
    assert bridge.fit(eq, train, trial, features) is bridge
    assert provider.created[0][0] == algorithm
    assert provider.last.fit_called == 1
    if device == "bearing":
        assert provider.last.labels == ["HEALTHY", "FAULT", "HEALTHY", "FAULT"]
    else:
        assert provider.last.labels is None  # unsupervised native IsolationForest
    output = bridge.predict(incoming)
    assert len(output) == 2
    assert [x["predicted_label"] for x in output] == (
        ["HEALTHY", "FAULT"] if device == "bearing"
        else ["NORMAL", "ANOMALY"])
    assert bridge.resource_receipt()["train_fit_calls"] == 1
    assert bridge.resource_receipt()["validation_or_final_used_for_fit"] is False
    assert bridge.resource_receipt()["cpu_gpu_time_parity_proven"] is False
    assert bridge.resource_receipt()["hard_deadline_enforced"] is False
    assert bridge.resource_receipt()["production_qualified"] is False


def test_all_reported_implementations_are_provider_owned():
    registry = available_adapters()
    assert len(registry) == 4
    assert all(not a["implementation_in_autogen"] and a["provider"] == "torch_like"
               for a in registry)
    assert {a["native_name"] for a in registry} == {
        "LogisticRegression", "DecisionTreeClassifier",
        "RandomForestClassifier", "IsolationForest"}


def test_source_roles_and_protected_input_fail_before_provider_fit():
    eq, _, trial, values, _ = fixture("bearing")
    provider = SpyTorchLike()
    with pytest.raises(ValueError, match="R93_PROTECTED_ROLE_IN_DESIGN_WORKER"):
        TorchLikeTabularCandidate("linear", provider=provider).fit(
            eq, read("bearing_trusted_audit.json"), trial, values)
    assert provider.last is None
    eq, train, trial, values, _ = fixture("bearing")
    values[0]["observations"]["ground_truth"] = "FAULT"
    with pytest.raises(ProviderRefused, match="R94_UNDECLARED_OR_PROTECTED_CHANNEL"):
        TorchLikeTabularCandidate("linear", provider=provider).fit(
            eq, train, trial, values)


@pytest.mark.parametrize("pollution", [
    {"label":"FAULT"}, {"physical_id":"bearing_007"},
    {"role":"VALIDATION"}, {"vibration":True},
    {"vibration":math.nan}, {"vibration":math.inf},
])
def test_never_forward_metadata_or_nonfinite_measurements_to_native_predict(
        pollution):
    eq, train, trial, features, inference = fixture("bearing")
    m = TorchLikeTabularCandidate("linear", provider=SpyTorchLike())
    m.fit(eq, train, trial, features)
    bad = dict(inference[0])
    bad.update(pollution)
    with pytest.raises(ProviderRefused,
        match="R94_UNDECLARED_OR_PROTECTED_CHANNEL|R94_NONFINITE_OR_INVALID_SENSOR"):
        m.predict([bad])


def test_wrong_task_or_unavailable_implementation_must_fail_closed():
    eq, train, trial, features, _ = fixture("bearing")
    trial["candidate_families"].append("statistical")
    with pytest.raises(ProviderRefused, match="R94_UNSUPPORTED_PROVIDER_TASK_AND_FAMILY"):
        TorchLikeTabularCandidate("statistical", provider=SpyTorchLike()).fit(
            eq, train, trial, features)
    with pytest.raises(ProviderRefused, match="R94_CANDIDATE_NOT_PREDECLARED"):
        TorchLikeTabularCandidate("random_forest", provider=SpyTorchLike()).fit(
            eq, train, trial, features)


def test_provider_capability_mismatch_refused_before_training():
    class MaliciousProvider(SpyTorchLike):
        def get_estimator_info(self, name):
            return {"kind":"regression"}
    eq, train, trial, features, _ = fixture("bearing")
    with pytest.raises(ProviderRefused, match="R94_PROVIDER_CAPABILITY_MISMATCH"):
        TorchLikeTabularCandidate("linear", provider=MaliciousProvider()).fit(
            eq, train, trial, features)


def test_artifacts_delegate_to_library_and_newonly_outputs(tmp_path):
    eq, train, trial, features, _ = fixture("bearing")
    provider = SpyTorchLike()
    m = TorchLikeTabularCandidate("linear", provider=provider).fit(
        eq, train, trial, features)
    dest = tmp_path/"native"
    receipt = m.export_model(dest)
    assert provider.saved and provider.saved[0][0] is provider.last
    assert receipt["native_estimator_joblib_sha256"]
    assert receipt["contains_pickle_unsafe_for_untrusted_loading"] is True
    assert receipt["independent_model_validated"] is False
    with pytest.raises(ProviderRefused, match="R94_REFUSE_TO_OVERWRITE_PROVIDER_ARTIFACT"):
        m.export_model(dest)


def test_adapter_refuses_refit_in_place_and_fresh_predict():
    eq, train, trial, features, incoming = fixture("bearing")
    m = TorchLikeTabularCandidate("linear", provider=SpyTorchLike())
    with pytest.raises(ProviderRefused, match="R94_FIT_REQUIRED_BEFORE_PREDICTION"):
        m.predict(incoming)
    m.fit(eq, train, trial, features)
    with pytest.raises(ProviderRefused, match="R94_FITTED_CANDIDATE_CANNOT_RETRAIN"):
        m.fit(eq, train, trial, features)


@pytest.mark.parametrize("candidate,device", [
    ("linear", "bearing"),
    ("tree", "bearing"),
    ("random_forest", "bearing"),
    ("isolation_forest", "compressor"),
])
def test_real_torch_like_native_estimator_runtime_fit_and_replay(
        candidate, device, tmp_path):
    tl = pytest.importorskip("torch_like")
    eq, train, trial, features, incoming = fixture(device)
    if candidate in ("tree", "random_forest"):
        trial["candidate_families"].append(candidate)
    m = TorchLikeTabularCandidate(candidate, provider=tl)
    m.fit(eq, train, trial, features)
    predictions = m.predict(incoming)
    assert len(predictions) == 2
    assert predictions[0]["predicted_label"] in trial["labels"]
    assert predictions[1]["predicted_label"] in trial["labels"]
    path = tmp_path / candidate
    artifact = m.export_model(path)
    assert artifact["contains_pickle_unsafe_for_untrusted_loading"] is True
    # Model bytes originated here in this test only. Never load unknown joblib!
    restored = tl.load_model(path)
    from hashlib import sha256
    assert sha256((path/"estimator.joblib").read_bytes()).hexdigest() == (
        artifact["native_estimator_joblib_sha256"])
    native = restored.predict([[v[c] for c in trial["input_channels"]]
                               for v in incoming])
    assert len(native) == 2
    if device == "bearing":
        assert [str(x) for x in native] == [
            p["predicted_label"] for p in predictions]
    else:
        assert all(int(x) in (-1, 1) for x in native)
