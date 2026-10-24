"""R83c sealed frozen-model VALIDATION evaluator, never a search worker.

This is a second-stage *reviewer-controlled* offline reader. It does not
generate, filter, tune or train source models. Its heldout metrics are only
independent of SOURCE TRAIN when the steward actually supplies separately
generated, disjoint physical groups and authentic evidence; hashes alone
cannot prove physical independence or provenance.

Never use this API with TEST/FINAL data. Those belong to an independent R84
steward, not the iterative AutoDesign research development process.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from collections.abc import Mapping, Sequence

import numpy as np

from .design_engine import _sha
from .taskspec import TaskSpec, TaskSpecRejected, validate_observations
from .self_organizing import SelfOrganizingArtifact
from ..io import sha_file, read_json, write_json

SCHEMA = "phm.r83.frozen-source-independent-validation.v1"
ARMS = ("fixed_portfolio", "static_topology", "self_organizing")
BOOTSTRAP_SEED = 83021
BOOTSTRAP_REPLICATES = 499


def _fail(message: str):
    raise TaskSpecRejected(message)


def _fingerprint(rows, fields):
    arr = np.asarray([[entry[k] for k in fields] for entry in rows],
                     dtype="<f8")
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def _digest_group(group: str):
    return hashlib.sha256(group.encode("utf-8")).hexdigest()


def _preflight(report: Mapping, raw: dict, validation_records: Sequence[Mapping]):
    if (report.get("schema") != "phm.r83.matched-source-train-study.v1" or
            report.get("source_task_spec_sha256") != TaskSpec.from_dict(raw).sha256 or
            report.get("source_data_role") != "TRAIN_ONLY" or
            report.get("independent_qualification") is not False or
            report.get("source_train_test_or_final_accessed") is not False or
            report.get("complete_three_arm_comparison") is not True or
            report.get("fairness_gate", {}).get("source_train_comparison_allowed") is not True or
            set(report.get("arms", {})) != set(ARMS) or
            not isinstance(report.get("source_train_manifest_sha256"), str) or
            not isinstance(report.get("source_train_observations_sha256"), str) or
            report.get("source_trajectory_digest_contract") !=
            "RAW_FLOAT64_LE_NO_METADATA_V1" or
            report.get("source_group_digest_contract") != "UTF8_UNIT_SEPARATOR_V1"):
        _fail("R83_VALIDATION_REQUIRES_FROZEN_COMPLETE_TRAIN_REPORT")
    source_groups = report.get("source_train_physical_group_sha256")
    source_traces = report.get("source_train_trajectory_sha256")
    if (not isinstance(source_groups, list) or not source_groups or
            not isinstance(source_traces, list) or not source_traces or
            any(not isinstance(x, str) or len(x) != 64 for x in
                source_groups + source_traces)):
        _fail("R83_VALIDATION_REQUIRES_SOURCE_DISJOINTNESS_HASHES")
    if (raw["task"]["decision_mode"] != "offline_trajectory" or
            raw["data"]["role"] != "TRAIN"):
        _fail("R83_VALIDATION_TASK_CONTRACT_INVALID")
    if not isinstance(validation_records, (list, tuple)) or not validation_records:
        _fail("R83_VALIDATION_EMPTY_OR_NONFINITE_RECORDS")
    label_set = set(raw["task"]["labels"])
    cols = [raw["observations"]["time"]["field"]] + [
        c["name"] for c in raw["observations"]["channels"]]
    group_cols, env_cols = raw["data"]["group_keys"], raw["data"]["environment_axes"]
    expected = set(("role", "label", *group_cols, *env_cols))
    group_to_env, fingerprints, out = {}, {}, []
    all_groups = set(source_groups)
    all_traces = set(source_traces)
    for item in validation_records:
        if not isinstance(item, Mapping) or set(item) != {"metadata", "measurements"}:
            _fail("R83_VALIDATION_RECORD_SCHEMA_INVALID")
        meta = item["metadata"]
        if (not isinstance(meta, Mapping) or set(meta) != expected or
                meta.get("role") != "VALIDATION" or meta.get("label") not in label_set or
                any(type(meta[c]) is not str or not meta[c].strip()
                    for c in group_cols + env_cols)):
            _fail("R83_VALIDATION_PROTECTED_ROLE_OR_LABEL_INVALID")
        validate_observations(raw, item["measurements"])
        group = "\x1f".join(meta[c] for c in group_cols)
        env = "\x1f".join(meta[c] for c in env_cols)
        if _digest_group(group) in all_groups:
            _fail("R83_VALIDATION_PHYSICAL_GROUP_OVERLAPS_TRAIN")
        if group in group_to_env and group_to_env[group] != env:
            _fail("R83_VALIDATION_GROUP_CROSSES_ENVIRONMENTS")
        group_to_env[group] = env
        digest = _fingerprint(item["measurements"], cols)
        if digest in all_traces:
            _fail("R83_VALIDATION_TRAJECTORY_REUSED_FROM_TRAIN")
        if digest in fingerprints and fingerprints[digest] != group:
            _fail("R83_VALIDATION_RELABELED_DUPLICATE_TRAJECTORY")
        fingerprints[digest] = group
        out.append((meta["label"], group, env, item["measurements"]))
    # Fixed a priori sample support check, not a way of selecting convenient
    # validation cases after measuring models.
    supports = {}
    for label, group, env, _ in out:
        supports.setdefault(env, {}).setdefault(label, set()).add(group)
    if len(supports) < 3:
        _fail("R83_VALIDATION_TOO_FEW_HELDOUT_REGIMES")
    for counts in supports.values():
        if set(counts) != label_set or min(len(s) for s in counts.values()) < 2:
            _fail("R83_VALIDATION_INSUFFICIENT_INDEPENDENT_CLASS_GROUPS")
    return out, {env: {c: len(groups) for c, groups in counts.items()}
                 for env, counts in sorted(supports.items())}, sorted(set(group_to_env))


def _metrics(labels, predictions, groups, envs, classes, healthy):
    y, p = np.asarray(labels), np.asarray(predictions)
    environment = {}
    for env in sorted(set(envs)):
        environment[env] = {}
        for cls in classes:
            mask = np.asarray([e == env and label == cls
                               for e, label in zip(envs, labels)])
            environment[env][cls] = {
                "support": int(mask.sum()),
                "recall": float(np.mean(p[mask] == cls)),
            }
    group_acc = {
        group: float(np.mean(p[np.asarray(groups) == group] ==
                             y[np.asarray(groups) == group]))
        for group in sorted(set(groups))}
    return {
        "trajectory_accuracy": float(np.mean(y == p)),
        "physical_group_macro_accuracy": float(np.mean(list(group_acc.values()))),
        "worst_environment_class_recall": float(min(
            x["recall"] for env in environment.values() for x in env.values())),
        "healthy_false_alarm": float(np.mean(p[y == healthy] != healthy)),
        "per_environment_class": environment,
        "physical_groups": len(group_acc),
        "trajectories": len(labels),
        "unqualified_research_only": True,
    }


def _paired_group_interval(labels, group_keys, predictions_a, predictions_b,
                           seed=BOOTSTRAP_SEED):
    """Cluster-paired bootstrap of group-mean accuracy difference only."""
    by_group = {}
    for index, group in enumerate(group_keys):
        by_group.setdefault(group, []).append(index)
    ordered = sorted(by_group)
    a = np.asarray([float(np.mean([
        predictions_a[i] == labels[i] for i in by_group[g]])) for g in ordered])
    b = np.asarray([float(np.mean([
        predictions_b[i] == labels[i] for i in by_group[g]])) for g in ordered])
    differences = a - b
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(ordered),
                           size=(BOOTSTRAP_REPLICATES, len(ordered)))
    means = differences[samples].mean(axis=1)
    return {
        "point_delta_group_macro_accuracy": float(differences.mean()),
        "cluster_bootstrap_95_percentile_interval": [
            float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
        "bootstrap_seed": seed,
        "replicates": BOOTSTRAP_REPLICATES,
        "unit": "REVIEWED_PHYSICAL_GROUP",
        "confirmatory_significance_claim": False,
    }


def evaluate_r83_frozen_validation(
    report: Mapping, task_spec: Mapping | TaskSpec,
    frozen_root: str | Path, validation_records: Sequence[Mapping], *,
    trusted_fixed_pickle: bool = False,
) -> dict:
    """One read-only pass on steward-supplied VALIDATION cases.

    Fixed portfolio artifact uses joblib; loading it is only permitted with
    explicit trusted_fixed_pickle=True, after manifest hashes were verified.
    This evaluator does not write weights, train, select, or expose scores to
    a design worker. Any TEST, FINAL, TRAIN or mixed-role record is refused.
    """
    raw = task_spec.to_dict() if isinstance(task_spec, TaskSpec) else TaskSpec.from_dict(task_spec).to_dict()
    rows, supports, groups = _preflight(report, raw, validation_records)
    if trusted_fixed_pickle is not True:
        raise ValueError("R83_VALIDATION_EXPLICIT_FIXED_PICKLE_TRUST_REQUIRED")
    root = Path(frozen_root)
    # Verify *all* required artifacts and all report hashes before unpickling
    # any fixed model, without allowing the report to choose arbitrary paths.
    for arm in ARMS:
        receipt = report["arms"][arm]
        freeze = receipt.get("frozen_research_artifact", {})
        if (receipt.get("model_delivered") is not True or
                freeze.get("relative_directory") != arm or
                freeze.get("qualified") is not False or
                not isinstance(freeze.get("manifest_sha256"), str)):
            _fail("R83_VALIDATION_MISSING_FROZEN_ARM")
        metadata = root / arm / (
            "audit.json" if arm == "fixed_portfolio" else "manifest.json")
        if not metadata.is_file() or sha_file(metadata) != freeze["manifest_sha256"]:
            _fail("R83_VALIDATION_FROZEN_MANIFEST_HASH_MISMATCH")
    from .artifact_io import load_trusted_artifact
    models = {
        "fixed_portfolio": load_trusted_artifact(root / "fixed_portfolio",
                                                trust_pickle=True),
        "static_topology": SelfOrganizingArtifact.load(root / "static_topology"),
        "self_organizing": SelfOrganizingArtifact.load(root / "self_organizing"),
    }
    # Source task declaration can differ only in the design section between
    # registered fixed portfolio and R83 topology refit. All observation and
    # task ontology contracts must remain identical.
    for model in models.values():
        recorded_manifest_hash = (
            model.audit["training_manifest_sha256"] if hasattr(model, "trial")
            else model.audit["train_manifest_sha256"])
        recorded_observation_hash = (
            model.audit["training_observations_sha256"] if hasattr(model, "trial")
            else model.audit["train_observations_sha256"])
        if (recorded_manifest_hash != report["source_train_manifest_sha256"] or
                recorded_observation_hash !=
                report["source_train_observations_sha256"]):
            _fail("R83_VALIDATION_FROZEN_SOURCE_DATA_MISMATCH")
        model_task = (model.audit["source_task_spec"] if hasattr(model, "trial")
                      else model.task_spec)
        for field in ("task", "observations", "equipment", "data"):
            if model_task[field] != raw[field]:
                _fail("R83_VALIDATION_FROZEN_INPUT_TASK_MISMATCH")
    labels = [x[0] for x in rows]
    group_names = [x[1] for x in rows]
    environments = [x[2] for x in rows]
    prediction = {}
    evaluation = {}
    for arm in ARMS:
        model = models[arm]
        if arm == "fixed_portfolio":
            results = [model.predict_decisions(row[3])[0]["label"] for row in rows]
        else:
            results = [model.predict(row[3])["label"] for row in rows]
        if any(y not in raw["task"]["labels"] for y in results):
            _fail("R83_VALIDATION_UNEXPECTED_MODEL_DECISION")
        prediction[arm] = results
        evaluation[arm] = _metrics(labels, results, group_names, environments,
                                   raw["task"]["labels"], raw["task"]["healthy_label"])
    paired = {
        "adaptive_minus_static": _paired_group_interval(
            labels, group_names, prediction["self_organizing"],
            prediction["static_topology"]),
        "adaptive_minus_fixed": _paired_group_interval(
            labels, group_names, prediction["self_organizing"],
            prediction["fixed_portfolio"]),
    }
    metadata_digest = _sha([
        {"label": item[0], "group_sha": _digest_group(item[1]),
         "environment": item[2]} for item in rows])
    return {
        "schema": SCHEMA,
        "source_train_report_sha256": _sha(report),
        "task_spec_sha256": TaskSpec.from_dict(raw).sha256,
        "validation_metadata_sha256": metadata_digest,
        "validation_trajectory_hashes_sha256": _sha(sorted(
            _fingerprint(item[3], [raw["observations"]["time"]["field"]] + [
                c["name"] for c in raw["observations"]["channels"]])
            for item in rows)),
        "validation_role": "VALIDATION_ONLY_REVIEWER_CONTROLLED",
        "train_group_overlap": False,
        "train_trajectory_overlap": False,
        "validation_physical_groups": len(groups),
        "physical_group_support_by_environment_class": supports,
        "minimum_groups_per_cell": min(
            min(cell.values()) for cell in supports.values()),
        "recommend_five_groups_per_cell_met": all(
            min(cell.values()) >= 5 for cell in supports.values()),
        "three_frozen_models_only": True,
        "model_fit_or_selection_performed": False,
        "all_arms_evaluated_on_same_records": True,
        "evaluation_metrics": evaluation,
        "paired_differences": paired,
        "statistical_significance_or_physical_independence_proven": False,
        "independent_final_test_performed": False,
        "independent_qualification": False,
    }
