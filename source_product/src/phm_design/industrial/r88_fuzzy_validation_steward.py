"""R88 frozen neuro-fuzzy VALIDATION steward: no fitting on heldout records.

Research-only two-stage protocol. TRAIN-side refit writes hash-checked NPZ
arrays (no pickle); a separate evaluator reads only frozen weights and
explicit VALIDATION measurements. Neither path admits TEST/FINAL. Hash
checks demonstrate file integrity, NOT authenticity or actual independent
physical provenance. External users can still rerun/tune using validation;
this API cannot enforce a one-time legal or scientific blind.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from collections.abc import Mapping, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .design_engine import _observations_digest, _sha
from .r86_neuro_fuzzy import FuzzyRuleClassifier, FuzzyRulePolicy, sequence_summary
from .r86_fuzzy_source_study import run_fuzzy_source_study, _metric
from .r87_premise_learning import fuzzy_design
from .taskspec import TaskSpec, TaskSpecRejected, validate_observations
from ..task_candidate_registry import deployment_sequence


SCHEMA = "phm.r88.frozen-fuzzy-validation.v1"
ARMS = ("linear_reference", "fixed_fuzzy", "adaptive_fuzzy",
        "fixed_fuzzy_learned_premise", "adaptive_fuzzy_learned_premise")


def _reject(value):
    raise TaskSpecRejected(value)


def _digest(array):
    a = np.ascontiguousarray(array, dtype="<f8")
    h = hashlib.sha256()
    h.update(len(a.shape).to_bytes(1, "little"))
    for width in a.shape:
        h.update(int(width).to_bytes(8, "little"))
    h.update(a.tobytes())
    return h.hexdigest()


def _rows(raw, records, *, role):
    if not isinstance(records, (list, tuple)) or not records:
        _reject("R88_EMPTY_OR_INVALID_RECORDS")
    group_keys = raw["data"]["group_keys"]
    env_keys = raw["data"]["environment_axes"]
    fields = [raw["observations"]["time"]["field"]] + [
        c["name"] for c in raw["observations"]["channels"]]
    expected = {"role", "label", *group_keys, *env_keys}
    rows = []
    group_env = {}
    by_obs = {}
    for item in records:
        if not isinstance(item, Mapping) or set(item) != {"metadata", "measurements"}:
            _reject("R88_INVALID_RECORD_SCHEMA")
        m = item["metadata"]
        if (not isinstance(m, Mapping) or set(m) != expected or
                m.get("role") != role or
                m.get("label") not in raw["task"]["labels"] or
                any(type(m[k]) is not str or not m[k].strip()
                    for k in (*group_keys, *env_keys))):
            _reject("R88_PROTECTED_ROLE_OR_METADATA")
        validate_observations(raw, item["measurements"])
        x = np.asarray([[v[k] for k in fields]
                        for v in item["measurements"]], dtype=np.float64)
        group = "\x1f".join(m[k] for k in group_keys)
        env = "\x1f".join(m[k] for k in env_keys)
        if group in group_env and group_env[group] != env:
            _reject("R88_PHYSICAL_GROUP_CROSSES_ENVIRONMENTS")
        group_env[group] = env
        obs = _digest(x)
        if obs in by_obs and by_obs[obs] != group:
            _reject("R88_RELABELED_DUPLICATE_OBSERVATION")
        by_obs[obs] = group
        rows.append((x, group, env, m["label"], obs))
    return rows


def _fuzzy_params(model):
    return {
        "kind": np.asarray(["fuzzy"]),
        "centers": model._centers,
        "log_widths": model._log_widths,
        "coef": model._coef,
        "mean": model._mean,
        "scale": model._scale,
        "classes": model.classes_.astype(np.int64),
    }


def _linear_params(model, scaler):
    return {
        "kind": np.asarray(["linear"]),
        "coef": model.coef_.astype(np.float64),
        "intercept": model.intercept_.astype(np.float64),
        "mean": scaler.mean_.astype(np.float64),
        "scale": scaler.scale_.astype(np.float64),
        "classes": model.classes_.astype(np.int64),
    }


def _predict(weights, sequence):
    kind = str(weights["kind"][0])
    x = sequence_summary(sequence)
    mean, scale = weights["mean"], weights["scale"]
    if mean.ndim != 1 or scale.shape != mean.shape or np.any(scale <= 0):
        _reject("R88_BAD_FROZEN_NORMALIZER")
    if x.shape[1] != len(mean):
        _reject("R88_INFERENCE_CHANNEL_MISMATCH")
    if kind == "linear":
        z = (x - mean) / scale
        logits = z @ weights["coef"].T + weights["intercept"]
        if logits.shape[1] == 1:
            idx = np.where(logits[:, 0] > 0, 1, 0)
        else:
            idx = logits.argmax(axis=1)
    elif kind == "fuzzy":
        z = np.clip((x - mean) / scale, -20, 20)
        logits = fuzzy_design(
            z, weights["centers"], weights["log_widths"]) @ weights["coef"]
        idx = logits.argmax(axis=1)
    else:
        _reject("R88_UNREGISTERED_FROZEN_MODEL")
    return weights["classes"][idx]


def freeze_fuzzy_source_models(task_spec: Mapping | TaskSpec,
                               train_records: Sequence[Mapping],
                               output_dir: str | Path, *,
                               policy: FuzzyRulePolicy,
                               evidence_origin: str = "UNVERIFIED"):
    """Fit only from TRAIN and save immutable-by-hash, NOT authenticated weights.

    No VAL/TEST/FINAL records or labels accepted. Requires a full source OOF
    report before any model can be frozen. No promotion if train results fail.
    """
    if not isinstance(policy, FuzzyRulePolicy) or policy.premise_steps <= 0:
        raise ValueError("R88_EXPLICIT_LEARNABLE_PREMISE_POLICY_REQUIRED")
    if evidence_origin not in ("UNVERIFIED", "SYNTHETIC_TEST_FIXTURE",
                               "REVIEWER_DOCUMENTED"):
        raise ValueError("R88_INVALID_EVIDENCE_ORIGIN")
    raw = (task_spec.to_dict() if isinstance(task_spec, TaskSpec)
           else TaskSpec.from_dict(task_spec).to_dict())
    records = _rows(raw, train_records, role="TRAIN")
    source = run_fuzzy_source_study(
        raw, train_records, policy=policy, include_learned_premise=True)
    if not source["all_declared_arms_complete"] or not source["same_folds_all_arms"]:
        _reject("R88_INCOMPLETE_SOURCE_OOF_STUDY")
    out = Path(output_dir)
    if out.exists():
        _reject("R88_FROZEN_OUTPUT_MUST_BE_NEW")
    source_groups = sorted({hashlib.sha256(group.encode("utf-8")).hexdigest()
                            for _, group, _, _, _ in records})
    source_observations = sorted({hash_v for _, _, _, _, hash_v in records})
    matrices = [x for x, _, _, _, _ in records]
    seq = np.stack([deployment_sequence(x) for x in matrices])
    label_to_index = {name: i for i, name in enumerate(raw["task"]["labels"])}
    y = np.asarray([label_to_index[label] for _, _, _, label, _ in records])
    feature = sequence_summary(seq)

    # Models are fitted exclusively after all source-only guards have run.
    scaler = StandardScaler().fit(feature)
    linear = LogisticRegression(C=1, max_iter=600).fit(
        scaler.transform(feature), y)
    models = {"linear_reference": _linear_params(linear, scaler)}
    from dataclasses import replace
    controls = replace(policy, premise_steps=0)
    for name, mode, arm_policy in (
        ("fixed_fuzzy", "fixed", controls),
        ("adaptive_fuzzy", "adaptive", controls),
        ("fixed_fuzzy_learned_premise", "fixed", policy),
        ("adaptive_fuzzy_learned_premise", "adaptive", policy),
    ):
        model = FuzzyRuleClassifier(arm_policy, mode=mode).fit(seq, y)
        models[name] = _fuzzy_params(model)
    out.mkdir(parents=True, exist_ok=False)
    artifact_hashes = {}
    for name in ARMS:
        path = out / (name + ".npz")
        np.savez_compressed(path, **models[name])
        artifact_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "schema": SCHEMA,
        "source_task_spec": raw,
        "source_task_spec_sha256": TaskSpec.from_dict(raw).sha256,
        "source_train_manifest_sha256": source["train_manifest_sha256"],
        "source_train_observations_sha256": source["train_observations_sha256"],
        "source_oof_report_sha256": _sha(source),
        "source_group_hashes": source_groups,
        "source_observation_hashes": source_observations,
        "source_group_hash_contract": "UTF8_UNIT_SEPARATOR_SHA256_V1",
        "observation_hash_contract": "SHAPE_LENGTH_FLOAT64LE_SHA256_V1",
        "frozen_model_hashes": artifact_hashes,
        "evidence_origin_self_reported": evidence_origin,
        "provenance_independently_verified": False,
        "source_oof_eligible_all_arms": all(
            a["accepted_by_train_oof_screen"] for a in source["arms"].values()),
        "source_train_only": True,
        "independent_model_qualification": False,
        "validation_used_for_model_selection": False,
        "test_or_final_accessed": False,
        "test_or_final_evaluation_supported": False,
    }
    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False,
                             sort_keys=True, indent=2, allow_nan=False) + "\n",
                             encoding="utf-8")
    return {"artifact_root": str(out),
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "source_oof_report": source,
            "qualified": False}


def evaluate_frozen_fuzzy_validation(frozen_root: str | Path,
                                     validation_records: Sequence[Mapping],
                                     *, expected_manifest_sha256: str):
    """Heldout steward: only VALIDATION records, no fit/selection/threshold."""
    if (type(expected_manifest_sha256) is not str or
            len(expected_manifest_sha256) != 64):
        _reject("R88_EXPECTED_MANIFEST_HASH_REQUIRED")
    root = Path(frozen_root)
    manifest_path = root / "manifest.json"
    if (not manifest_path.is_file() or
            hashlib.sha256(manifest_path.read_bytes()).hexdigest() !=
            expected_manifest_sha256):
        _reject("R88_FROZEN_MANIFEST_HASH_MISMATCH")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("schema") != SCHEMA or
            not manifest.get("source_train_only") is True or
            manifest.get("test_or_final_accessed") is not False or
            manifest.get("independent_model_qualification") is not False):
        _reject("R88_FROZEN_PROTOCOL_MISMATCH")
    raw = TaskSpec.from_dict(manifest["source_task_spec"]).to_dict()
    if TaskSpec.from_dict(raw).sha256 != manifest["source_task_spec_sha256"]:
        _reject("R88_TASKSPEC_HASH_MISMATCH")
    source_groups = set(manifest["source_group_hashes"])
    source_observations = set(manifest["source_observation_hashes"])
    data = _rows(raw, validation_records, role="VALIDATION")
    validation_groups = {}
    for _, group, env, label, digest in data:
        if hashlib.sha256(group.encode("utf-8")).hexdigest() in source_groups:
            _reject("R88_VALIDATION_SOURCE_PHYSICAL_GROUP_OVERLAP")
        if digest in source_observations:
            _reject("R88_VALIDATION_SOURCE_TRAJECTORY_REPLAY")
        validation_groups.setdefault(env, {}).setdefault(label, set()).add(group)
    labels = raw["task"]["labels"]
    if len(validation_groups) < 2 or any(
            set(cell) != set(labels) or
            min(len(v) for v in cell.values()) < 2
            for cell in validation_groups.values()):
        _reject("R88_VALIDATION_NEEDS_TWO_ENVIRONMENTS_TWO_GROUPS_PER_CLASS")
    for arm in ARMS:
        p = root / (arm + ".npz")
        if (not p.is_file() or
                hashlib.sha256(p.read_bytes()).hexdigest() !=
                manifest["frozen_model_hashes"].get(arm)):
            _reject("R88_FROZEN_MODEL_HASH_MISMATCH")
    # All contents hash checked BEFORE reading any model, and no pickle load.
    seq = np.stack([deployment_sequence(x) for x, _, _, _, _ in data])
    reference = {i: label for i, label in enumerate(labels)}
    expected = np.asarray([labels.index(label) for _, _, _, label, _ in data])
    env = np.asarray([e for _, _, e, _, _ in data])
    groups = [g for _, g, _, _, _ in data]
    results = {}
    predictions = {}
    for arm in ARMS:
        with np.load(root / (arm + ".npz"), allow_pickle=False) as obj:
            frozen = {key: obj[key].copy() for key in obj.files}
        prediction = _predict(frozen, seq)
        if (prediction.shape != expected.shape or
                not np.isin(prediction, list(reference)).all()):
            _reject("R88_FROZEN_MODEL_PREDICTION_INVALID")
        predictions[arm] = prediction
        results[arm] = _metric(
            expected, prediction, env,
            labels.index(raw["task"]["healthy_label"]), len(labels))
    # Deterministic physical-group macro delta + group-paired bootstrap.
    # Keep this purely NumPy: frozen inference must not depend on importing
    # a neural search trainer, Torch, pickle, or any mutable study worker.
    def paired_interval(p_a, p_b):
        by_group = {}
        for i, group in enumerate(groups):
            by_group.setdefault(group, []).append(i)
        per_group = np.asarray([
            np.mean(p_a[by_group[g]] == expected[by_group[g]]) -
            np.mean(p_b[by_group[g]] == expected[by_group[g]])
            for g in sorted(by_group)], dtype=float)
        rng = np.random.default_rng(88017)
        draws = rng.integers(0, len(per_group),
                             size=(499, len(per_group)))
        means = per_group[draws].mean(axis=1)
        return {
            "point_delta_group_macro_accuracy": float(per_group.mean()),
            "cluster_bootstrap_95_percentile_interval": [
                float(np.quantile(means, .025)),
                float(np.quantile(means, .975))],
            "bootstrap_seed": 88017, "replicates": 499,
            "unit": "SELF_REPORTED_PHYSICAL_GROUP",
            "confirmatory_significance_claim": False,
        }
    effects = {}
    for a, b in (
        ("adaptive_fuzzy", "fixed_fuzzy"),
        ("fixed_fuzzy_learned_premise", "fixed_fuzzy"),
        ("adaptive_fuzzy_learned_premise", "adaptive_fuzzy"),
        ("adaptive_fuzzy_learned_premise", "fixed_fuzzy_learned_premise"),
        ("adaptive_fuzzy_learned_premise", "linear_reference"),
    ):
        effects[a + "_vs_" + b] = paired_interval(
            predictions[a], predictions[b])
    return {
        "schema": "phm.r88.readonly-source-separated-validation.v1",
        "manifest_sha256": expected_manifest_sha256,
        "validation_role": "VALIDATION_ONLY",
        "validation_observation_sha256": _observations_digest(
            [x for x, _, _, _, _ in data]),
        "validation_group_count": len(set(groups)),
        "validation_environment_count": len(validation_groups),
        "source_group_overlap_rejected": True,
        "source_observation_replay_rejected": True,
        "source_artifact_fits_performed": 0,
        "post_validation_refit_or_selection_performed": False,
        "frozen_model_hashes_verified": True,
        "origin_self_reported": manifest["evidence_origin_self_reported"],
        "physical_independence_attested_by_external_steward": False,
        "one_time_validation_enforced": False,
        "source_train_screen_all_arms_passed":
            manifest["source_oof_eligible_all_arms"],
        "arms": results,
        "paired_group_intervals": effects,
        "eligible_for_stable": False,
        "independent_generalization_proven": False,
        "blind_test_or_final_performed": False,
    }
