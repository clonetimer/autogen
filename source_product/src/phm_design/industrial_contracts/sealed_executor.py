"""R101 trusted-steward *contract fixture* execution and evidence receipts.

This module proves that a locally supplied runner was invoked with bytes whose
hashes match the frozen R99 study. It deliberately does NOT authenticate the
runner, load native torch_like joblib, prove model provenance or execute models
on real hardware. A future audited provider adapter must replace fixture mode.
No model training, arbitrary imports, code execution or deserialization here.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import math
import os
import platform
import time
import tracemalloc

from .contracts import validate_dataset
from .independent_review import (
    freeze_comparison, evaluate_sealed_measured_comparison,
    _verify_local_source_files,
)

SCHEMA = "phm.autogen.r101-fixture-execution-seal.v1"
RECEIPT = "phm.autogen.r101-fixture-execution-receipt.v1"
FILES = ("model.json", "preprocessor.json", "graph.json")
MAX_ARTIFACT_FILE_BYTES = 64 * 1024
MAX_RECORD_BYTES = 32 * 1024 * 1024


class SealedExecutionRejected(ValueError):
    """Refuse unsafe/mismatched artifacts, invalid data, or untrusted execution."""


def _fail(code):
    raise SealedExecutionRejected(code)


def _json_hash(value):
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _byte_hash(content):
    return hashlib.sha256(content).hexdigest()


def _check_frozen(frozen, trial):
    if not isinstance(frozen, Mapping):
        _fail("R101_R99_COMPARISON_REQUIRED")
    try:
        actual = freeze_comparison(
            frozen.get("source_design_sha256"),
            frozen.get("challenger_artifact_id"),
            frozen.get("baseline_artifact_id"), trial)
    except (TypeError, ValueError, KeyError) as exc:
        raise SealedExecutionRejected("R101_INVALID_R99_COMPARISON") from exc
    if frozen != actual:
        _fail("R101_R99_COMPARISON_DIGEST_OR_SCOPE_CHANGED")
    return (frozen["challenger_artifact_id"], frozen["baseline_artifact_id"])


def _read_bundle(root):
    if type(root) not in (str, Path):
        _fail("R101_ARTIFACT_DIRECTORY_REQUIRED")
    path = Path(root)
    if path.is_symlink() or not path.is_dir():
        _fail("R101_ARTIFACT_ROOT_MISSING_OR_SYMLINK")
    if {p.name for p in path.iterdir()} != set(FILES):
        _fail("R101_ONLY_SAFE_FIXTURE_JSON_FILES_ALLOWED_NO_JOBLIB_PICKLE")
    digests = {}
    documents = {}
    for name in FILES:
        file = path / name
        if file.is_symlink() or not file.is_file():
            _fail("R101_UNSAFE_ARTIFACT_PATH")
        if not 0 < file.stat().st_size <= MAX_ARTIFACT_FILE_BYTES:
            _fail("R101_ARTIFACT_FILE_SIZE_OUTSIDE_LIMIT")
        raw = file.read_bytes()
        try:
            document = json.loads(raw.decode("utf-8"))
        except (UnicodeError, ValueError) as exc:
            raise SealedExecutionRejected("R101_ARTIFACT_NOT_STRICT_JSON") from exc
        if not isinstance(document, dict):
            _fail("R101_ARTIFACT_JSON_OBJECT_REQUIRED")
        digests[name] = _byte_hash(raw)
        documents[name] = document
    model = documents["model.json"]
    if (set(model) != {"schema", "fixture_id", "not_a_trained_model"}
            or model["schema"] != "phm.autogen.r101-fixture-model.v1"
            or type(model["fixture_id"]) is not str or not model["fixture_id"]
            or model["not_a_trained_model"] is not True):
        _fail("R101_ONLY_DECLARED_NONEXECUTABLE_FIXTURE_MODEL")
    pre = documents["preprocessor.json"]
    if (pre != {"schema": "phm.autogen.r101-fixture-preprocessor.v1",
                "source_only": True, "mode": "identity_fixture"}):
        _fail("R101_FIXTURE_PREPROCESSOR_NOT_REGISTERED")
    graph = documents["graph.json"]
    if (graph != {"schema": "phm.autogen.r101-fixture-graph.v1",
                  "stages": ["original_bytes_fixture"],
                  "uses_protected_truth": False}):
        _fail("R101_FIXTURE_GRAPH_NOT_REGISTERED")
    return {"files_sha256": digests, "bundle_sha256": _json_hash(digests),
            "documents": documents}


def seal_fixture_execution(frozen, trial, bundle_directories):
    """Freeze the exact safe TEST FIXTURE bytes. No model loading occurs.

    Does not authenticate the creation time, physical evidence or runner.
    """
    ids = _check_frozen(frozen, trial)
    if not isinstance(bundle_directories, Mapping) or set(bundle_directories) != set(ids):
        _fail("R101_EXACT_FROZEN_ARTIFACT_ID_SET_REQUIRED")
    bundles = {model: _read_bundle(bundle_directories[model]) for model in ids}
    if len({v["bundle_sha256"] for v in bundles.values()}) != 2:
        _fail("R101_CHALLENGER_AND_BASELINE_BUNDLES_IDENTICAL")
    body = {
        "schema": SCHEMA, "comparison_sha256": frozen["comparison_sha256"],
        "trial_sha256": _json_hash(trial),
        "artifact_bundle_sha256": {
            model: bundles[model]["bundle_sha256"] for model in ids},
        "file_sha256": {
            model: bundles[model]["files_sha256"] for model in ids},
        "execution_kind": "NONEXECUTABLE_CONTRACT_FIXTURE_ONLY",
        "no_native_model_load": True,
        "fixture_only_not_production": True,
        "frozen_before_validation_labels_externally_attested": False,
        "provider_executable_independently_authenticated": False,
        "independent_device_superiority_proven": False,
    }
    return {**body, "seal_sha256": _json_hash(body)}


def _verified_blind_bytes(row, original_files):
    path = Path(original_files[row["record_id"]])
    if path.is_symlink() or not path.is_file():
        _fail("R101_RAW_FILE_NOT_AVAILABLE")
    if not 128 <= path.stat().st_size <= MAX_RECORD_BYTES:
        _fail("R101_RAW_FILE_EXCEEDS_BOUNDED_FIXTURE_SIZE")
    raw = path.read_bytes()
    if _byte_hash(raw) != row["source_sha256"]:
        _fail("R101_ORIGINAL_RAW_BYTES_CHANGED_BEFORE_RUNNER")
    return raw


def execute_fixture_pair(seal, frozen, equipment, manifest, trial,
                         original_files, bundle_directories, *,
                         fixture_runner=None, explicit_fixture_mode=False):
    """Execute two *non-production fixture callbacks* on identical VAL raw bytes.

    The callback receives ONLY (artifact_id, record_id, original_bytes,
    deepcopy(nonexecutable_bundle_documents)); it never receives the full
    dataset, label, R99 evaluation receipt, or a model-training interface.
    An untrusted Python callback still has ambient filesystem capabilities;
    this process is NOT a security sandbox or third-party attestation.
    """
    if explicit_fixture_mode is not True or not callable(fixture_runner):
        _fail("R101_NO_TRUSTED_NATIVE_RUNNER_FIXTURE_MODE_ONLY")
    expected = seal_fixture_execution(frozen, trial, bundle_directories)
    if seal != expected:
        _fail("R101_SEAL_OR_REGISTERED_FILE_BYTES_CHANGED")
    validate_dataset(manifest, equipment, trial, train_only=False)
    if (manifest["origin"] != "measured"
            or manifest["split_protocol"]["mode"] != "physical_group"
            or trial["validation_target"] != "unseen_physical_device"
            or trial["task_type"] != "classification"):
        _fail("R101_R99_INDEPENDENT_MEASURED_CLASSIFICATION_SCOPE_REQUIRED")
    # Every manifest raw-file byte identity is checked on the steward side.
    _verify_local_source_files(manifest, original_files)
    rows = sorted((r for r in manifest["records"] if r["role"] == "VALIDATION"),
                  key=lambda row: row["record_id"])
    if not rows:
        _fail("R101_NO_VALIDATION_RECORDS")
    ids = _check_frozen(frozen, trial)
    bundles = {model: _read_bundle(bundle_directories[model]) for model in ids}
    # Do not pass metadata, roles or truth to the fixture runner.
    predictions = {}
    invocations = {}
    was_tracking = tracemalloc.is_tracing()
    if not was_tracking:
        tracemalloc.start()
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    try:
        for model in ids:
            prediction_stream = {}
            input_to_output = []
            for row in rows:
                raw = _verified_blind_bytes(row, original_files)
                try:
                    result = fixture_runner(
                        model, row["record_id"], raw,
                        deepcopy(bundles[model]["documents"]))
                except Exception as exc:
                    raise SealedExecutionRejected(
                        "R101_REGISTERED_FIXTURE_RUNNER_ERROR") from exc
                if (not isinstance(result, Mapping)
                        or set(result) != {"predicted_label"}
                        or type(result["predicted_label"]) is not str
                        or result["predicted_label"] not in trial["labels"]):
                    _fail("R101_RUNNER_RETURNED_INVALID_PREDICTION")
                prediction_stream[row["record_id"]] = result["predicted_label"]
                input_to_output.append({
                    "record_id": row["record_id"],
                    "original_file_sha256": row["source_sha256"],
                    "artifact_bundle_sha256": bundles[model]["bundle_sha256"],
                    "predicted_label": result["predicted_label"],
                })
            predictions[model] = prediction_stream
            invocations[model] = {
                "artifact_bundle_sha256": bundles[model]["bundle_sha256"],
                "number_of_executed_fixture_callbacks": len(input_to_output),
                "input_prediction_pair_sha256": _json_hash(input_to_output),
                "prediction_stream_sha256": _json_hash(prediction_stream),
            }
        wall = max(0.0, time.perf_counter() - wall_start)
        cpu = max(0.0, time.process_time() - cpu_start)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        if not was_tracking:
            tracemalloc.stop()
    # Catch source or artifact mutation during invocation. This is still not
    # an atomic filesystem snapshot, secure sandbox, or signed model execution.
    for row in rows:
        _verified_blind_bytes(row, original_files)
    for model in ids:
        if _read_bundle(bundle_directories[model])["bundle_sha256"] != (
                expected["artifact_bundle_sha256"][model]):
            _fail("R101_MODEL_BUNDLE_MUTATED_DURING_EXECUTION")
    # R99's independent metrics stay STRICTLY non-certified.
    reviewed = evaluate_sealed_measured_comparison(
        frozen, equipment, manifest, trial, original_files, predictions)
    body = {
        "schema": RECEIPT, "seal_sha256": expected["seal_sha256"],
        "comparison_sha256": frozen["comparison_sha256"],
        "manifest_sha256": _json_hash(manifest),
        "status": "SYNTHETIC_FIXTURE_INVOCATION_HASHED_NOT_NATIVE_ATTESTED",
        "execution": invocations,
        "r99_review_receipt_sha256": reviewed["receipt_sha256"],
        "r99_directional_submission_signal_only": reviewed[
            "directional_submission_evidence_meets_preregistered_threshold"],
        "observed_resources": {
            "both_models_combined_wall_seconds": wall,
            "both_models_combined_process_cpu_seconds": cpu,
            "traced_python_peak_bytes": peak,
            "process_cpu_count_self_reported": os.cpu_count(),
            "platform_self_reported": platform.platform(),
            "feature_pipeline_and_prediction_fixture_callback_included": True,
            "native_library_peak_memory_verified": False,
            "gpu_memory_or_flops_measured": False,
            "resource_limits_enforced": False,
            "hardware_and_software_independently_attested": False,
            "per_model_matched_compute_proven": False,
        },
        "runner_implementation_sha256_authenticated": False,
        "fitted_model_native_bytes_safely_loaded": False,
        "predictions_cryptographically_bound_to_real_model_execution": False,
        "validation_truth_passed_as_callback_argument": False,
        "post_selection_bias_corrected": False,
        "real_physical_device_identity_independently_certified": False,
        "independent_real_device_superiority_proven": False,
        "eligible_for_stable": False,
    }
    return {**body, "receipt_sha256": _json_hash(body)}
