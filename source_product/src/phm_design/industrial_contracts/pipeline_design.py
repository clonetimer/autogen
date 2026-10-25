"""R96 — design a *diagnostic pipeline*, not another estimator library.

Pipeline DAG (bounded): legal sensor subset -> fold-local preprocessing
-> existing torch_like estimator -> source OOF decision. R95 chooses estimator,
sensors and its knobs. R96 searches meaningful normalization variants and
fits the transformation independently INSIDE EACH training fold.

Only SOURCE TRAIN records can enter these functions. There is no external
VALIDATION/FINAL parameter, independent performance claim, or production
promotion. Optional providers supply the actual estimator implementations.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
import math
from statistics import median

from .design_engine import (
    DesignRejected, propose_diagnostic_study, _source_table)
from .torch_like_bridge import TorchLikeTabularCandidate

SCHEMA = "phm.autogen.pipeline-design.v1"
RECEIPT = "phm.autogen.pipeline-source-oof.v1"
TRANSFORMS = ("identity", "standard_scale", "robust_scale", "signed_log1p")
MAX_SOURCE_FITS = 64


def _fail(message):
    raise DesignRejected(message)


def _hash(obj):
    return hashlib.sha256(json.dumps(
        obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
        allow_nan=False).encode("utf-8")).hexdigest()


def _numeric(matrix, width):
    if not isinstance(matrix, (tuple, list)) or not matrix:
        _fail("R96_EMPTY_NUMERIC_MATRIX")
    result = []
    for row in matrix:
        if not isinstance(row, (tuple, list)) or len(row) != width:
            _fail("R96_INVALID_NUMERIC_ROW_WIDTH")
        if any(type(x) not in (int, float) or not math.isfinite(x) or
               abs(x) > 1e12 for x in row):
            _fail("R96_NONFINITE_OR_PROTECTED_NUMERIC_INPUT")
        result.append([float(x) for x in row])
    return result


def fit_preprocessor(transform, columns, train_matrix):
    """Learn zero parameters from heldout/source-validation rows.

    No class labels, group IDs or target timestamps are parameters. All
    location/scale values are based exclusively on the provided train matrix.
    """
    if transform not in TRANSFORMS:
        _fail("R96_UNKNOWN_PREPROCESSING_TRANSFORM")
    if (not isinstance(columns, (tuple, list)) or not columns or
            len(columns) != len(set(columns)) or
            not all(type(c) is str for c in columns)):
        _fail("R96_INVALID_PREPROCESSOR_CHANNELS")
    x = _numeric(train_matrix, len(columns))
    loc, scale = [], []
    for j in range(len(columns)):
        col = [row[j] for row in x]
        if transform == "standard_scale":
            mean = math.fsum(col) / len(col)
            variance = math.fsum((v-mean)**2 for v in col) / len(col)
            std = math.sqrt(variance)
            loc.append(mean)
            scale.append(max(std, 1e-8))
        elif transform == "robust_scale":
            middle = median(col)
            mad = median([abs(v-middle) for v in col])
            loc.append(float(middle))
            scale.append(max(1.4826*mad, 1e-8))
        else:
            loc.append(0.)
            scale.append(1.)
    state = {
        "schema": "phm.autogen.fold-fitted-preprocessor.v1",
        "name": transform,
        "input_channels": list(columns),
        "location": loc, "scale": scale,
        "training_row_count": len(x),
        "fitted_on": "SOURCE_FOLD_TRAIN_ONLY",
    }
    return {**state, "state_sha256": _hash(state)}


def apply_preprocessor(state, matrix):
    if not isinstance(state, Mapping) or "state_sha256" not in state:
        _fail("R96_INVALID_PREPROCESSOR_STATE")
    body = {k: v for k, v in state.items() if k != "state_sha256"}
    if (set(body) != {
            "schema", "name", "input_channels", "location", "scale",
            "training_row_count", "fitted_on"} or
            state["state_sha256"] != _hash(body) or
            state["name"] not in TRANSFORMS or
            state["fitted_on"] != "SOURCE_FOLD_TRAIN_ONLY"):
        _fail("R96_MUTATED_OR_UNAUTHORIZED_PREPROCESSOR_STATE")
    columns = state["input_channels"]
    width = len(columns)
    if (not width or
            len(state["location"]) != width or
            len(state["scale"]) != width or
            any(type(v) not in (int, float) or not math.isfinite(v)
                for v in state["location"] + state["scale"]) or
            any(v <= 0 for v in state["scale"])):
        _fail("R96_INVALID_PREPROCESSOR_PARAMETERS")
    values = _numeric(matrix, width)
    result = []
    for row in values:
        if state["name"] == "signed_log1p":
            transformed = [math.copysign(math.log1p(abs(v)), v)
                           for v in row]
        else:
            transformed = [(v-m)/s for v,m,s in zip(
                row, state["location"], state["scale"])]
        if any(not math.isfinite(v) or abs(v) > 1e12
               for v in transformed):
            _fail("R96_PREPROCESSING_NUMERICAL_OVERFLOW")
        result.append(transformed)
    return result


def propose_pipeline_study(equipment, source_train, trial):
    """Freeze model + sensor + preprocessing alternatives before scoring."""
    base = propose_diagnostic_study(equipment, source_train, trial)
    designs = base["candidate_blueprints"]
    # Prioritize full-sensor fair family comparisons, then ALL families
    # at the same predeclared standard/robust/log transforms. This is not
    # outcome-driven search: the proposal is a pure function of TaskSpec.
    base_full = [d for d in designs if
                 d["input_channels"] == trial["input_channels"]]
    # Keep one original primary configuration per family, not variants
    # of hyperparameters until every primary family has been offered.
    representatives = []
    recorded = set()
    for d in base_full:
        if d["algorithm_family"] not in recorded:
            recorded.add(d["algorithm_family"])
            representatives.append(d)
    options = []
    for transform in TRANSFORMS:
        for d in representatives:
            options.append((d, transform))
    for d in designs:
        if d not in representatives:
            options.append((d, "identity"))
    selected = []
    known = set()
    for design, transform in options:
        key = (design["algorithm_family"],
               tuple(design["input_channels"]),
               tuple(sorted(design["hyperparameters"].items())), transform)
        if key in known:
            continue
        known.add(key)
        selected.append({
            "pipeline_id": f"pipeline_{len(selected)+1:02d}",
            "algorithm_family": design["algorithm_family"],
            "input_channels": list(design["input_channels"]),
            "hyperparameters": deepcopy(design["hyperparameters"]),
            "preprocessing": transform,
            "stages": ["sensor_select", "preprocess", "provider_estimator"],
            "source_only": True,
        })
        if len(selected) == trial["budget"]["max_trials"]:
            break
    folds = deepcopy(base["folds"])
    refusal = base["screening_refusal"]
    if len(selected) * len(folds) > MAX_SOURCE_FITS:
        refusal = "R96_PREDECLARED_PIPELINE_FIT_BUDGET_EXCEEDED"
        folds = []
    doc = {
        "schema": SCHEMA,
        "base_r95_source_plan_sha256": base["study_sha256"],
        "trial_sha256": _hash(trial),
        "equipment_sha256": _hash(equipment),
        "source_train_sha256": _hash(source_train),
        "seed": trial["budget"]["seed"],
        "candidate_count": len(selected),
        "pipeline_designs": selected,
        "folds": folds,
        "planned_source_fit_calls": len(selected) * len(folds),
        "hard_fit_limit": MAX_SOURCE_FITS,
        "unsupported_provider_families": list(
            base["unimplemented_candidate_families"]),
        "screening_refusal": refusal,
        "preprocessing_selection_uses_labels": False,
        "validation_calibration_final_used": False,
        "scope": "SOURCE_TRAIN_INTERNAL_RESEARCH_ONLY",
    }
    return {**doc, "study_sha256": _hash(doc)}


def evaluate_pipeline_study(study, equipment, train, trial, features, *,
                            provider=None):
    """Train every transform on fold TRAIN; never fit on fold-heldout rows."""
    if not isinstance(study, Mapping) or study != propose_pipeline_study(
            equipment, train, trial):
        _fail("R96_PIPELINE_OR_SOURCE_PLAN_MUTATED")
    source = _source_table(train, features, trial["input_channels"])
    if study["screening_refusal"]:
        return {
            "schema": RECEIPT, "study_sha256": study["study_sha256"],
            "status": "ABSTAIN_INSUFFICIENT_SOURCE_EVIDENCE",
            "reason": study["screening_refusal"],
            "candidates": [], "shortlist": [],
            "actual_fit_calls": 0,
            "independent_model_validated": False,
            "production_qualified": False,
        }
    indexed = {r["record_id"]: r for r in train["records"]}
    outcomes = []
    fit_calls = 0
    for pipeline in study["pipeline_designs"]:
        sensors = pipeline["input_channels"]
        kind = pipeline["preprocessing"]
        per_fold = []
        visits = []
        for fold in study["folds"]:
            fit_ids = fold["fit_ids"]
            heldout_ids = fold["score_ids"]
            if set(fit_ids) & set(heldout_ids):
                _fail("R96_HELDOUT_RECORD_IN_PREPROCESSOR_FIT_SET")
            x_train = [[source[rid][c] for c in sensors] for rid in fit_ids]
            x_held = [[source[rid][c] for c in sensors] for rid in heldout_ids]
            scaler = fit_preprocessor(kind, sensors, x_train)
            scaled_train = apply_preprocessor(scaler, x_train)
            scaled_held = apply_preprocessor(scaler, x_held)
            training = deepcopy(train)
            training["records"] = [indexed[rid] for rid in fit_ids]
            child_trial = deepcopy(trial)
            child_trial["input_channels"] = list(sensors)
            child_trial["candidate_families"] = [pipeline["algorithm_family"]]
            # The provider sees only transformed numerical sensor columns.
            sensor_train = [
                {"record_id": rid,
                 "observations": dict(zip(sensors, vector))}
                for rid, vector in zip(fit_ids, scaled_train)]
            sensor_held = [dict(zip(sensors, vector)) for vector in scaled_held]
            model = TorchLikeTabularCandidate(
                pipeline["algorithm_family"], provider=provider,
                design_options=pipeline["hyperparameters"])
            model.fit(equipment, training, child_trial, sensor_train)
            predicted = model.predict(sensor_held)
            fit_calls += 1
            if len(predicted) != len(heldout_ids):
                _fail("R96_HELDOUT_PREDICTION_COUNT_MISMATCH")
            correctness = []
            for rid, prediction in zip(heldout_ids, predicted):
                label = prediction.get("predicted_label")
                if label not in trial["labels"]:
                    _fail("R96_PREDICTED_LABEL_NOT_IN_TASK")
                truth = indexed[rid]["label"]
                per_fold.append({"group_hash": fold["group_sha256"],
                                 "correct": int(label == truth),
                                 "true": truth, "predicted": label})
                correctness.append(int(label == truth))
            visits.append({
                "heldout_group_sha256": fold["group_sha256"],
                "preprocessor_fit_count": len(fit_ids),
                "preprocessor_state_sha256": scaler["state_sha256"],
                "fold_fit_ids_sha256": _hash(fit_ids),
                "fold_heldout_ids_sha256": _hash(heldout_ids),
                "transform_fitted_only_on_fold_train": True,
                "model_fit_call_count": model.resource_receipt()["train_fit_calls"],
                "observed_fit_seconds": model.resource_receipt()[
                    "fit_wall_seconds_observed"],
            })
        by_group = defaultdict(list)
        for r in per_fold:
            by_group[r["group_hash"]].append(r["correct"])
        accuracy = [sum(v)/len(v) for v in by_group.values()]
        healthy = trial["healthy_label"]
        known_healthy = [r for r in per_fold if r["true"] == healthy]
        false_alarm = (sum(r["predicted"] != healthy for r in known_healthy)/
                       len(known_healthy)) if known_healthy else None
        outcomes.append({
            "pipeline_id": pipeline["pipeline_id"],
            "algorithm_family": pipeline["algorithm_family"],
            "input_channels": sensors,
            "preprocessing": kind,
            "hyperparameters": pipeline["hyperparameters"],
            "source_group_macro_accuracy": sum(accuracy)/len(accuracy),
            "source_worst_group_accuracy": min(accuracy),
            "source_healthy_false_alarm_fraction": false_alarm,
            "total_provider_fit_calls": len(visits),
            "sum_observed_fit_seconds": sum(
                v["observed_fit_seconds"] for v in visits),
            "preprocessor_fit_audit": visits,
            "source_oof_only": True,
        })
    # Source-only relative order: this is NOT a deployment score or
    # post-search independent estimate, nor is runtime hardware-matched.
    ranked = sorted(outcomes, key=lambda item: (
        -item["source_worst_group_accuracy"],
        -item["source_group_macro_accuracy"],
        item["source_healthy_false_alarm_fraction"],
        item["pipeline_id"],
    ))
    return {
        "schema": RECEIPT,
        "study_sha256": study["study_sha256"],
        "status": "SOURCE_GROUP_OOF_PIPELINE_SHORTLIST_ONLY",
        "candidates": outcomes,
        "shortlist": [x["pipeline_id"] for x in ranked],
        "actual_fit_calls": fit_calls,
        "planned_fit_calls": study["planned_source_fit_calls"],
        "preprocessor_fold_local": True,
        "compute_matched": False,
        "source_data_independent_of_prior_design_iterations": False,
        "independent_model_validated": False,
        "production_qualified": False,
    }
