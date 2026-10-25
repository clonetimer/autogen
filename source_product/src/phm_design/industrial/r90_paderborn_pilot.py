"""R90 prospectively fixed Paderborn single-heldout-condition pilot.

NEW exploratory protocol; never weakens the stricter R88 two-condition gate.
Only named vibration_1 is accepted from MATLAB, with unverified native units.
Do not interpret any synthetic fixture as original Paderborn equipment data.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path

import numpy as np

from .r86_neuro_fuzzy import FuzzyRulePolicy
from .r88_fuzzy_validation_steward import (
    ARMS, SCHEMA as FROZEN_SCHEMA, _metric, _predict,
    freeze_fuzzy_source_models, _rows)
from .taskspec import TaskSpec, TaskSpecRejected
from ..task_candidate_registry import deployment_sequence

SCHEMA = "phm.r90.single-unseen-condition.v1"
TRAIN = (
    ("N15_M07_F10", "K001", "HEALTHY"),
    ("N15_M07_F10", "KI03", "FAULT"),
    ("N09_M07_F10", "K002", "HEALTHY"),
    ("N09_M07_F10", "KA01", "FAULT"),
    ("N15_M01_F10", "K003", "HEALTHY"),
    ("N15_M01_F10", "KI05", "FAULT"),
)
VALIDATION = (
    ("N15_M07_F04", "K004", "HEALTHY"),
    ("N15_M07_F04", "K005", "HEALTHY"),
    ("N15_M07_F04", "K006", "HEALTHY"),
    ("N15_M07_F04", "KA03", "FAULT"),
    ("N15_M07_F04", "KA04", "FAULT"),
    ("N15_M07_F04", "KI04", "FAULT"),
)
RECORDINGS = (1, 2)
BINS = 256
NOMINAL_HZ = 64000.


def _fail(reason):
    raise TaskSpecRejected(reason)


def protocol_sha256():
    return hashlib.sha256(json.dumps(
        {"schema": SCHEMA, "train": TRAIN, "val": VALIDATION,
         "recordings": RECORDINGS, "bins": BINS, "nominal_hz": NOMINAL_HZ,
         "channel": "vibration_1"},
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _named_signals(obj, depth=0):
    if depth > 10:
        return
    if isinstance(obj, Mapping):
        fields = {str(k): v for k, v in obj.items()
                  if not str(k).startswith("__")}
    elif hasattr(obj, "_fieldnames"):
        fields = {k: getattr(obj, k) for k in obj._fieldnames}
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _named_signals(v, depth + 1)
        return
    elif isinstance(obj, np.ndarray) and obj.dtype.kind == "O":
        for v in obj.flat:
            yield from _named_signals(v, depth + 1)
        return
    else:
        return
    low = {k.lower(): v for k, v in fields.items()}
    if "name" in low and "data" in low:
        name = str(np.asarray(low["name"]).squeeze()).strip().lower()
        if name == "vibration_1":
            try:
                yield np.asarray(low["data"], dtype=float).ravel()
            except (ValueError, TypeError) as exc:
                raise TaskSpecRejected("R90_NAMED_SIGNAL_NOT_NUMERIC") from exc
    for v in fields.values():
        yield from _named_signals(v, depth + 1)


def read_vibration_mat(path):
    from scipy.io import loadmat
    try:
        d = loadmat(str(path), squeeze_me=True, struct_as_record=False)
    except (OSError, ValueError, NotImplementedError) as exc:
        raise TaskSpecRejected("R90_UNSUPPORTED_MAT_FORMAT") from exc
    found = list(_named_signals(d))
    if len(found) != 1:
        _fail("R90_ONE_EXPLICIT_NAMED_VIBRATION_CHANNEL_REQUIRED")
    x = found[0]
    if not 200000 <= len(x) <= 300000 or not np.isfinite(x).all():
        _fail("R90_INVALID_SIGNAL_SAMPLES")
    return x


def rms_observations(x):
    raw = np.asarray(x, dtype=float)
    if raw.ndim != 1 or len(raw) < 200000 or not np.isfinite(raw).all():
        _fail("R90_INVALID_RMS_INPUT")
    width = len(raw) // BINS
    frame = raw[:width * BINS].reshape(BINS, width)
    rms = np.sqrt(np.mean(frame ** 2, axis=1))
    if not np.isfinite(rms).all():
        _fail("R90_NONFINITE_RMS")
    return [{"nominal_elapsed_s": float((i+.5)*width/NOMINAL_HZ),
             "vibration_rms": float(v)} for i, v in enumerate(rms)]


def _check_task(raw):
    spec = raw.to_dict() if isinstance(raw, TaskSpec) else TaskSpec.from_dict(raw).to_dict()
    if (spec["equipment"]["adapter"] != "paderborn_verified_vibration_rms_r90"
        or spec["task"]["labels"] != ["HEALTHY", "FAULT"]
        or spec["task"]["decision_mode"] != "offline_trajectory"
        or spec["data"]["group_keys"] != ["bearing_instance_id"]
        or spec["data"]["environment_axes"] != ["operating_condition"]
        or spec["observations"]["time"] != {
            "field": "nominal_elapsed_s", "sampling": "fixed"}
        or spec["observations"]["channels"] != [{
            "name": "vibration_rms", "role": "sensor",
            "unit": "raw_source_unit_unverified",
            "available_at_decision": True}]):
        _fail("R90_PINNED_TASKSPEC_REQUIRED")
    return spec


def read_pinned_recordings(raw_root, *, role):
    if role not in ("TRAIN", "VALIDATION"):
        _fail("R90_PROTECTED_ROLE")
    root = Path(raw_root).resolve(strict=True)
    expected = TRAIN if role == "TRAIN" else VALIDATION
    output, receipt = [], []
    for cond, specimen, label in expected:
        for idx in RECORDINGS:
            relative = Path(specimen) / (f"{cond}_{specimen}_{idx}.mat")
            p = (root / relative).resolve(strict=True)
            if not p.is_relative_to(root) or not p.is_file():
                _fail("R90_OUTSIDE_ROOT_OR_NONFILE")
            sha = hashlib.sha256(p.read_bytes()).hexdigest()
            x = read_vibration_mat(p)
            output.append({"measurements": rms_observations(x),
                           "metadata": {"role": role, "label": label,
                           "bearing_instance_id": specimen,
                           "operating_condition": cond}})
            receipt.append({"relative_path": relative.as_posix(),
                            "raw_sha256": sha,
                            "nominal_hz_not_clock_verified": NOMINAL_HZ,
                            "physical_sensor_unit_verified": False})
    return output, receipt


def _check_roster(records, *, role):
    expected = TRAIN if role == "TRAIN" else VALIDATION
    count = Counter((role, label, specimen, cond) for cond, specimen, label
                    in expected for _ in RECORDINGS)
    seen = Counter()
    for x in records:
        m = x["metadata"]
        if set(m) != {"role", "label", "bearing_instance_id",
                      "operating_condition"}:
            _fail("R90_INVALID_METADATA")
        seen[(m["role"], m["label"], m["bearing_instance_id"],
              m["operating_condition"])] += 1
    if count != seen:
        _fail("R90_LOCKED_PHYSICAL_BEARING_ROSTER_MISMATCH")


def freeze_paderborn_source(spec, raw_root, frozen_dir, *, policy):
    task = _check_task(spec)
    data, hashes = read_pinned_recordings(raw_root, role="TRAIN")
    _check_roster(data, role="TRAIN")
    frozen = freeze_fuzzy_source_models(task, data, frozen_dir,
                                       policy=policy,
                                       evidence_origin="UNVERIFIED")
    return {"schema": SCHEMA, "roster_sha256": protocol_sha256(),
            "frozen": frozen, "source_raw_file_receipts": hashes,
            "validation_accessed": False, "independent_qualification": False}


def evaluate_paderborn_validation(frozen_dir, raw_root, *,
                                  expected_manifest_sha256):
    """Read-only single heldout condition; NOT the unchanged R88 2-env test."""
    if (type(expected_manifest_sha256) is not str or
            len(expected_manifest_sha256) != 64 or
            any(c not in "0123456789abcdef" for c in expected_manifest_sha256)):
        _fail("R90_EXTERNAL_MANIFEST_SHA_REQUIRED")
    base = Path(frozen_dir)
    path = base / "manifest.json"
    if (not path.is_file() or
            hashlib.sha256(path.read_bytes()).hexdigest() !=
            expected_manifest_sha256):
        _fail("R90_MANIFEST_HASH_MISMATCH")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (manifest.get("schema") != FROZEN_SCHEMA
        or manifest.get("source_train_only") is not True
        or manifest.get("test_or_final_accessed") is not False
        or manifest.get("independent_model_qualification") is not False):
        _fail("R90_FROZEN_PROTOCOL_UNTRUSTED")
    task = _check_task(manifest["source_task_spec"])
    if TaskSpec.from_dict(task).sha256 != manifest["source_task_spec_sha256"]:
        _fail("R90_TASKSPEC_HASH_CHANGED")
    expected_source = {
        hashlib.sha256(specimen.encode("utf-8")).hexdigest()
        for _, specimen, _ in TRAIN}
    if set(manifest["source_group_hashes"]) != expected_source:
        _fail("R90_SOURCE_GROUP_SET_CHANGED")
    if manifest["evidence_origin_self_reported"] not in (
            "UNVERIFIED", "REVIEWER_DOCUMENTED"):
        _fail("R90_SYNTHETIC_ORIGIN_FROZEN_MODEL_REJECTED")
    for arm in ARMS:
        fp = base / (arm + ".npz")
        if (not fp.is_file() or
                hashlib.sha256(fp.read_bytes()).hexdigest() !=
                manifest["frozen_model_hashes"].get(arm)):
            _fail("R90_WEIGHT_HASH_MISMATCH")
    val, raw_receipts = read_pinned_recordings(raw_root, role="VALIDATION")
    _check_roster(val, role="VALIDATION")
    rows = _rows(task, val, role="VALIDATION")
    seen_source = set(manifest["source_observation_hashes"])
    if any(x[4] in seen_source for x in rows):
        _fail("R90_TRAJECTORY_REPLAY")
    seq = np.stack([deployment_sequence(x[0]) for x in rows])
    true = np.asarray([task["task"]["labels"].index(x[3]) for x in rows])
    envs = np.asarray([x[2] for x in rows])
    groups = np.asarray([x[1] for x in rows])
    results, predictions = {}, {}
    for arm in ARMS:
        with np.load(base / (arm + ".npz"), allow_pickle=False) as blob:
            weights = {k: blob[k].copy() for k in blob.files}
        pred = _predict(weights, seq)
        if pred.shape != true.shape or not np.isin(pred, (0, 1)).all():
            _fail("R90_UNEXPECTED_FROZEN_PREDICTIONS")
        predictions[arm] = pred
        results[arm] = _metric(true, pred, envs, healthy_idx=0, nclasses=2)
    pairs = (("adaptive_fuzzy", "fixed_fuzzy"),
             ("fixed_fuzzy_learned_premise", "fixed_fuzzy"),
             ("adaptive_fuzzy_learned_premise", "fixed_fuzzy_learned_premise"),
             ("adaptive_fuzzy_learned_premise", "linear_reference"))
    bootstrap = {}
    rng = np.random.default_rng(90023)
    physical = sorted(set(groups))
    for a, b in pairs:
        diffs = np.asarray([
            np.mean(predictions[a][groups == g] == true[groups == g]) -
            np.mean(predictions[b][groups == g] == true[groups == g])
            for g in physical])
        indices = rng.integers(0, len(diffs), size=(499, len(diffs)))
        samples = diffs[indices].mean(axis=1)
        bootstrap[a + "_vs_" + b] = {
            "physical_group_macro_accuracy_difference": float(diffs.mean()),
            "percentile_95_interval": [float(np.quantile(samples, .025)),
                                       float(np.quantile(samples, .975))],
            "physical_groups": len(physical), "replicates": 499,
            "confirmatory_significance_claim": False}
    return {
        "schema": "phm.r90.single-condition-exploratory-result.v1",
        "source_task_spec_sha256": manifest["source_task_spec_sha256"],
        "locked_protocol_sha256": protocol_sha256(),
        "frozen_manifest_sha256": expected_manifest_sha256,
        "validation_role": "VALIDATION_ONLY",
        "single_heldout_condition": "N15_M07_F04",
        "healthy_independent_bearings_in_validation": 3,
        "fault_independent_bearings_in_validation": 3,
        "physical_groups_bootstrap_unit": True,
        "raw_mat_receipts": raw_receipts,
        "source_sensor_units_verified": False,
        "source_sampling_clock_verified": False,
        "frozen_all_five_models_verified": True,
        "validation_fit_or_selection_performed": False,
        "validation_replay_protection_one_time_enforced": False,
        "results": results, "paired_physical_group_intervals": bootstrap,
        "blind_test_or_final_used": False,
        "multi_condition_external_generalization_proven": False,
        "neuro_fuzzy_improvement_proven": False, "eligible_for_stable": False,
    }
