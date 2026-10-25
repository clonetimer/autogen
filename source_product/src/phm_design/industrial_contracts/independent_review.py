"""R99 trusted-side paired independent-device comparison evidence gate.

This is deliberately NOT part of the SOURCE-TRAIN algorithm designer.
An independent data steward supplies a R93 full-role measured DatasetCard,
trusted local original file bytes, fixed predictions and a SHA-locked
pre-registered pair of model artifacts. Actual bytes are hashed here.
It still cannot independently authenticate physical specimens, instrument
calibration, or whether third-party predictions came from the frozen model.
Therefore it may report paired effect and statistical directional evidence,
but NEVER claim full externally proven algorithm superiority or Stable status.

IMPORTANT: don't send VAL/FINAL manifest or truth to autogen design workers.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import math

from .contracts import validate_dataset, ContractRejected

SCHEMA = "phm.autogen.sealed-independent-comparison.v1"
RESULT = "phm.autogen.sealed-measured-comparison-receipt.v1"
MAX_BYTES = 256*1024*1024
MIN_GROUPS_PER_CLASS = 3


class IndependentReviewRejected(ContractRejected):
    """Data source, provenance, frozen decision or pairing unacceptable."""


def _fail(reason):
    raise IndependentReviewRejected(reason)


def _digest(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ensure_ascii=False).encode("utf-8")).hexdigest()


def freeze_comparison(source_plan_sha256, challenger_id, baseline_id,
                      trial, *, preregistered_primary="physical_group_accuracy"):
    """Create BEFORE independent labels/recordings are inspected."""
    if (type(source_plan_sha256) is not str or len(source_plan_sha256) != 64
            or any(c not in "0123456789abcdef" for c in source_plan_sha256)
            or not all(type(x) is str and x for x in
                        (challenger_id, baseline_id))
            or challenger_id == baseline_id):
        _fail("R99_UNREGISTERED_OR_DUPLICATED_FROZEN_MODELS")
    if preregistered_primary != "physical_group_accuracy":
        _fail("R99_ONLY_PREREGISTERED_GROUP_ACCURACY_SUPPORTED")
    if trial.get("task_type") != "classification":
        _fail("R99_INDEPENDENT_BINARY_CLASSIFICATION_ONLY")
    doc = {
        "schema": SCHEMA,
        "source_design_sha256": source_plan_sha256,
        "challenger_artifact_id": challenger_id,
        "baseline_artifact_id": baseline_id,
        "trial_sha256": _digest(trial),
        "preregistered_primary": preregistered_primary,
        "paired_on": "original_physical_device",
        "healthy_false_alarm_must_not_increase": True,
        "min_validation_physical_devices_per_class": MIN_GROUPS_PER_CLASS,
        "read_independent_truth": False,
        "no_model_selection_after_evaluation": True,
    }
    return {**doc, "comparison_sha256": _digest(doc)}


def _verify_local_source_files(manifest, original_files):
    """Hash each ORIGINAL measurement file once, not features or metadata."""
    if not isinstance(original_files, Mapping):
        _fail("R99_ORIGINAL_FILE_MAPPING_REQUIRED")
    rows = manifest["records"]
    lookup = {r["record_id"]: r for r in rows}
    if set(original_files) != set(lookup):
        _fail("R99_ORIGINAL_FILE_COVERAGE_MUST_BE_COMPLETE")
    checked = {}
    for rid, row in lookup.items():
        file = original_files[rid]
        if type(file) is not str:
            _fail("R99_RAW_FILE_PATH_REQUIRED")
        path = Path(file)
        if path.is_symlink() or not path.is_file():
            _fail("R99_ORIGINAL_RAW_FILE_NOT_ACCESSIBLE")
        if path.stat().st_size < 128 or path.stat().st_size > MAX_BYTES:
            _fail("R99_MEASUREMENT_RAW_FILE_SIZE_OUTSIDE_LIMIT")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(2**20), b""):
                digest.update(chunk)
        if digest.hexdigest() != row["source_sha256"]:
            _fail("R99_ORIGINAL_RAW_BYTES_SHA_MISMATCH")
        checked[rid] = {
            "measured_binary_sha256": digest.hexdigest(),
            "source_sha256_matches_original_bytes": True,
        }
    return checked


def evaluate_sealed_measured_comparison(frozen, equipment, manifest, trial,
                                        original_files, predictions):
    """Audit *submitted* baseline/challenger predictions on external devices.

    Cannot certify that prediction arrays were produced by the frozen model.
    No training, hyperparameter search or threshold tuning is performed here.
    Use actual verifiable artifact execution in a separate trusted worker to
    achieve that missing integrity requirement.
    """
    if not isinstance(frozen, Mapping):
        _fail("R99_COMPARISON_NOT_FROZEN")
    expected = freeze_comparison(
        frozen.get("source_design_sha256"),
        frozen.get("challenger_artifact_id"),
        frozen.get("baseline_artifact_id"), trial)
    if frozen != expected:
        _fail("R99_COMPARISON_CHANGED_AFTER_SEALING")
    validate_dataset(manifest, equipment, trial, train_only=False)
    if manifest["origin"] != "measured":
        _fail("R99_EXPERIMENTAL_MEASURED_DATA_ONLY")
    if manifest["split_protocol"]["mode"] != "physical_group":
        _fail("R99_INDEPENDENT_PHYSICAL_GROUP_SPLIT_REQUIRED")
    if trial["validation_target"] != "unseen_physical_device":
        _fail("R99_INDEPENDENT_DEVICE_GENERALIZATION_REQUIRED")
    all_rows = manifest["records"]
    heldout = [r for r in all_rows if r["role"] == "VALIDATION"]
    if any(r["label"] not in trial["labels"] for r in heldout):
        _fail("R99_MISSING_TRUSTED_EXTERNAL_LABEL")
    for label in trial["labels"]:
        count = len({r["physical_id"] for r in heldout if r["label"] == label})
        if count < MIN_GROUPS_PER_CLASS:
            _fail("R99_TOO_FEW_REAL_HELDOUT_DEVICES_PER_CLASS")
    physical_classes = defaultdict(set)
    for row in heldout:
        physical_classes[row["physical_id"]].add(row["label"])
    if any(len(values) != 1 for values in physical_classes.values()):
        _fail("R99_AMBIGUOUS_PHYSICAL_DEVICE_CLASS_LABEL")
    # Do not assume file paths, SHA, device IDs or measurement time based
    # solely on caller strings. Hash real ORIGINAL files on evaluator side.
    checked = _verify_local_source_files(manifest, original_files)
    if (not isinstance(predictions, Mapping) or set(predictions) != {
            frozen["challenger_artifact_id"], frozen["baseline_artifact_id"]}):
        _fail("R99_EXACT_TWO_PREREGISTERED_PREDICTION_STREAMS_REQUIRED")
    valid_ids = {r["record_id"] for r in heldout}
    streams = {}
    for model, predictions_by_record in predictions.items():
        if (not isinstance(predictions_by_record, Mapping)
                or set(predictions_by_record) != valid_ids
                or any(type(v) is not str or v not in trial["labels"]
                       for v in predictions_by_record.values())):
            _fail("R99_UNPAIRED_EXTRA_OR_INVALID_PREDICTIONS")
        streams[model] = predictions_by_record
    group = defaultdict(list)
    for row in heldout:
        group[row["physical_id"]].append(row)
    differences, base_means, challenger_means = [], [], []
    healthy_counts = [0, 0]
    healthy_total = 0
    for entity in sorted(group):
        rows = group[entity]
        b = sum(streams[frozen["baseline_artifact_id"]][r["record_id"]]
                == r["label"] for r in rows)/len(rows)
        c = sum(streams[frozen["challenger_artifact_id"]][r["record_id"]]
                == r["label"] for r in rows)/len(rows)
        differences.append(c-b)
        base_means.append(b)
        challenger_means.append(c)
        for row in rows:
            if row["label"] == trial["healthy_label"]:
                healthy_total += 1
                healthy_counts[0] += (streams[frozen["baseline_artifact_id"]][
                    row["record_id"]] != trial["healthy_label"])
                healthy_counts[1] += (streams[frozen["challenger_artifact_id"]][
                    row["record_id"]] != trial["healthy_label"])
    positive = sum(d > 0 for d in differences)
    negative = sum(d < 0 for d in differences)
    informative = positive + negative
    # One-sided EXACT paired sign test over independent *physical groups*.
    # Ties are excluded, and repeated windows never inflate sample count.
    # This quantifies submitted data only; it is not proof of model
    # provenance or post-selection-free independent generalization.
    p = (sum(math.comb(informative, k) for k in range(
        positive, informative+1)) / 2**informative
         if informative else 1.0)
    mean_gap = sum(differences)/len(differences)
    far_base = healthy_counts[0]/healthy_total
    far_challenger = healthy_counts[1]/healthy_total
    descriptive_signal = (
        mean_gap > 0 and p <= .05 and far_challenger <= far_base)
    receipt = {
        "schema": RESULT,
        "comparison_sha256": frozen["comparison_sha256"],
        "manifest_sha256": _digest(manifest),
        "status": "PAIRED_MEASURED_SUBMISSION_AUDITED_NOT_MODEL_CERTIFIED",
        "raw_original_file_count_hashed": len(checked),
        "raw_file_sha_matches_declared_metadata": True,
        "declared_heldout_physical_groups": len(group),
        "declared_heldout_physical_groups_per_class": {
            label: len([entity for entity in group if
                        next(iter(physical_classes[entity])) == label])
            for label in trial["labels"]},
        "baseline_group_macro_accuracy": sum(base_means)/len(base_means),
        "challenger_group_macro_accuracy": sum(challenger_means)/len(challenger_means),
        "observed_challenger_minus_baseline_group_macro": mean_gap,
        "paired_positive_groups": positive,
        "paired_negative_groups": negative,
        "paired_tied_groups": len(differences)-informative,
        "one_sided_group_paired_sign_p": p,
        "baseline_healthy_false_alarm": far_base,
        "challenger_healthy_false_alarm": far_challenger,
        "directional_submission_evidence_meets_preregistered_threshold":
            descriptive_signal,
        "measurements_raw_bytes_authenticated_against_manifest": True,
        "physical_entity_authenticity_independently_certified": False,
        "predictions_bound_to_frozen_executable_artifacts": False,
        "model_was_retrained_on_holdout": False,
        "autogen_design_worker_accessed_holdout": False,
        "independent_real_device_superiority_proven": False,
        "eligible_for_stable": False,
        "required_for_proof": [
            "third_party_physically_authenticated_bearing_ID_and_label_manifest",
            "trusted_runner_executes_sealed_model_binaries_on_untouched_holdout",
            "independent_blind_paired_result_with_predeclared_familywise_error",
            "matched_end_to_end_feature_compute_and_inference_budget",
        ],
    }
    return {**receipt, "receipt_sha256": _digest(receipt)}
