"""R81a: actual device-independent TRAIN-only AutoDesign trial executor.

Executes a bounded R80 plan against deployment-observable time series.
All model ranking uses *source TRAIN* leave-one-environment-out predictions;
there is no independent or release-qualified score in this module.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import LeaveOneGroupOut

from .taskspec import (
    TaskSpec, TaskSpecRejected, compile_task_spec, validate_observations,
)
from .calibration import CalibrationRejected, fit_temperature_selective
from .feature_program import decode_program, execute_program, program_digest
from ..open_design import temporal_bins, random_temporal_kernels
from ..task_candidate_registry import TaskCandidateRegistry, deployment_sequence


@dataclass(frozen=True)
class TrialSpec:
    backbone: str
    objective: str
    feature: str
    learner: str
    seed: int
    topology: str = "direct"


@dataclass(frozen=True)
class DesignAcceptance:
    """Predeclared TRAIN-OOF research screen; never a qualification gate."""
    accuracy_min: float = .90
    worst_environment_class_recall_min: float = .80
    healthy_false_alarm_max: float = .10

    def __post_init__(self):
        for name, value in asdict(self).items():
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not np.isfinite(value) or not 0. <= value <= 1.):
                raise ValueError("INVALID_DESIGN_ACCEPTANCE:" + name)

    def evaluate(self, metrics: dict) -> dict:
        requirements = (
            ("accuracy", self.accuracy_min, "min"),
            ("worst_environment_class_recall", self.worst_environment_class_recall_min, "min"),
            ("healthy_false_alarm", self.healthy_false_alarm_max, "max"),
        )
        failures = []
        # Explicit loop: no data-dependent change to the declared threshold.
        for name, limit, direction in requirements:
            actual = metrics[name]
            if (actual < limit if direction == "min" else actual > limit):
                failures.append({"metric": name, "observed": actual,
                                 "required": limit,
                                 "relation": ">=" if direction == "min" else "<="})
        return {"eligible": not failures, "failed_gates": failures,
                "evidence": "TRAIN_ENVIRONMENT_OOF_ONLY",
                "independent_qualification": False}


class IndustrialDesignRejected(RuntimeError):
    def __init__(self, audit: dict):
        self.audit = audit
        super().__init__("NO_ELIGIBLE_INDUSTRIAL_DESIGN")


def _sha(payload) -> str:
    return hashlib.sha256(
        json.dumps(payload, allow_nan=False, sort_keys=True,
                   ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _candidate(spec: Mapping, seed: int) -> TrialSpec:
    backbone, objective = spec["backbone"], spec["objective"]
    representation = spec.get("representation", "tabular_signal_summary")
    topology = spec.get("topology", "direct")
    if representation == "feature_program":
        feature_identifier = spec.get("feature_program")
        decode_program(feature_identifier)
    else:
        feature_identifier = None
    if topology not in ("direct", "factorized"):
        raise TaskSpecRejected("UNREGISTERED_R81_TOPOLOGY")
    mapping = {
        "cnn1d": "cnn1d", "tcn": "tcn", "gru": "gru",
        "transformer": "transformer",
    }
    if backbone == "linear":
        if objective != "ERM":
            raise TaskSpecRejected("INVALID_NONNEURAL_OBJECTIVE")
        feature = {"tabular_signal_summary": "temporal",
                   "temporal_bins": "temporal",
                   "random_kernels": "random_kernels",
                   "difference_kernels": "difference_kernels"}.get(representation)
        if feature_identifier is not None:
            feature = feature_identifier
        if feature is None:
            raise TaskSpecRejected("INVALID_TABULAR_REPRESENTATION")
        return TrialSpec(backbone, objective, feature, "linear", seed, topology)
    if backbone == "tree":
        if objective != "ERM":
            raise TaskSpecRejected("INVALID_NONNEURAL_OBJECTIVE")
        feature = {"tabular_signal_summary": "temporal",
                   "temporal_bins": "temporal",
                   "random_kernels": "random_kernels",
                   "difference_kernels": "difference_kernels"}.get(representation)
        if feature_identifier is not None:
            feature = feature_identifier
        if feature is None:
            raise TaskSpecRejected("INVALID_TABULAR_REPRESENTATION")
        return TrialSpec(backbone, objective, feature, "extra", seed, topology)
    if backbone not in mapping:
        raise TaskSpecRejected("UNREGISTERED_R81_BACKBONE")
    prefix = {"ERM": "torch_", "GroupDRO": "dg_", "V-REx": "rex_"}.get(objective)
    if prefix is None:
        raise TaskSpecRejected("UNREGISTERED_R81_OBJECTIVE")
    if representation != "learned_sequence":
        raise TaskSpecRejected("INVALID_NEURAL_REPRESENTATION")
    return TrialSpec(backbone, objective, "sequence",
                     prefix + mapping[backbone], seed, topology)


def _features(array: np.ndarray, mode: str) -> np.ndarray:
    if mode.startswith("program:"):
        return execute_program(array, mode)
    if mode == "temporal":
        return np.asarray(temporal_bins(array), dtype=np.float64)
    if mode == "random_kernels":
        return np.asarray(random_temporal_kernels(array), dtype=np.float64)
    if mode == "difference_kernels":
        return np.asarray(random_temporal_kernels(array, differences=True),
                          dtype=np.float64)
    if mode == "sequence":
        return np.asarray(deployment_sequence(array), dtype=np.float32)
    raise ValueError("INVALID_TRIAL_FEATURE")


def _fitted(candidate: TrialSpec, train_x, train_y, domains, ontology=None):
    if candidate.topology == "factorized":
        if ontology is None:
            raise TaskSpecRejected("FACTORIZED_ONTOLOGY_REQUIRED")
        from .factorized_model import FactorizedDiagnosisModel
        model = FactorizedDiagnosisModel(
            feature=("temporal" if candidate.feature in
                     ("random_kernels", "difference_kernels") or
                     candidate.feature.startswith("program:")
                     else candidate.feature),
            learner=candidate.learner, objective=candidate.objective,
            seed=candidate.seed, labels=ontology["labels"],
            healthy_label=ontology["healthy_label"],
            class_attributes=ontology["class_attributes"])
        # The model's heads receive the exact selected representation matrix;
        # TaskCandidateRegistry tabular backbones take "temporal" descriptors.
        return model.fit(train_x, train_y, domain_groups=domains)
    registry = TaskCandidateRegistry()
    registry_feature = ("temporal" if candidate.feature in
                        ("random_kernels", "difference_kernels") or
                        candidate.feature.startswith("program:")
                        else candidate.feature)
    # Registry input_kind remains tabular/sequence. Feature-program identity
    # is carried by TrialSpec and materialized before the classifier is built.
    model = registry.build(registry_feature, candidate.learner, candidate.seed)
    if candidate.objective in ("GroupDRO", "V-REx"):
        return model.fit(train_x, train_y, domain_groups=np.asarray(domains))
    return model.fit(train_x, train_y)


def _factorized_fit_cost(ontology: Mapping) -> int:
    attrs = ontology["class_attributes"]
    components = {row["component"] for row in attrs.values()}
    mechanisms = {row["mechanism"] for row in attrs.values()}
    return 1 + int(len(components) > 1) + int(len(mechanisms) > 1)


def _probabilities(model, features, expected_classes: int) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(features), dtype=np.float64)
    classes = np.asarray(model.classes_)
    if (probabilities.shape != (len(features), expected_classes)
            or len(classes) != expected_classes
            or set(classes.tolist()) != set(range(expected_classes))
            or not np.isfinite(probabilities).all()
            or np.any(probabilities < 0)
            or not np.allclose(probabilities.sum(axis=1), 1., atol=1.e-4)):
        raise ValueError("UNSUPPORTED_MODEL_CLASS_OR_PROBABILITY_CONTRACT")
    return probabilities[:, np.argsort(classes.astype(int))]


def _predict_indices(model, features, expected_classes: int) -> np.ndarray:
    return np.argmax(_probabilities(model, features, expected_classes), axis=1)


def _score(truth, predicted, environments, groups, labels, healthy_idx) -> dict:
    y, p, env, group = (
        np.asarray(truth), np.asarray(predicted),
        np.asarray(environments), np.asarray(groups),
    )
    families = sorted(set(env.tolist()))
    families_out = {}
    minimum_recalls = []
    for family in families:
        mask = env == family
        support = {labels[k]: int(np.count_nonzero(mask & (y == k)))
                   for k in range(len(labels))}
        if any(value == 0 for value in support.values()):
            raise ValueError("INCOMPLETE_ENVIRONMENT_CLASS_SUPPORT")
        recalls = {labels[k]: float(np.mean(p[mask & (y == k)] == k))
                   for k in range(len(labels))}
        family_groups = {str(v) for v in group[mask]}
        families_out[family] = {
            "worlds": len(family_groups),
            "samples": int(mask.sum()),
            "class_support": support, "class_recall": recalls,
            "accuracy": float(np.mean(p[mask] == y[mask])),
            "unknown_rate": float(np.mean(p[mask] == -1)),
        }
        minimum_recalls.extend(recalls.values())
    normal = y == healthy_idx
    if not normal.any():
        raise ValueError("HEALTHY_SUPPORT_REQUIRED")
    return {
        "accuracy": float(np.mean(y == p)),
        "macro_f1": float(f1_score(y, p, labels=list(range(len(labels))),
                                   average="macro", zero_division=0)),
        "worst_environment_class_recall": float(min(minimum_recalls)),
        "worst_environment_accuracy": float(
            min(v["accuracy"] for v in families_out.values())),
        "healthy_false_alarm": float(np.mean(p[normal] != healthy_idx)),
        "unknown_rate": float(np.mean(p == -1)),
        "coverage": float(np.mean(p != -1)),
        "accepted_accuracy": (float(np.mean(p[p != -1] == y[p != -1]))
                              if np.any(p != -1) else 0.),
        "source_environment_metrics": families_out,
        "samples": int(len(y)),
        "physical_groups": len(set(group.tolist())),
        "evidence": "TRAIN_SOURCE_LOEO_OOF_NOT_INDEPENDENT_VALIDATION",
    }


class IndustrialDesignArtifact:
    """Refitted model + source-only audit. Loading pickle requires trusted files."""
    def __init__(self, *, model, trial: TrialSpec, plan: dict,
                 acceptance: dict, audit: dict, calibrator=None):
        self.model = model
        self.trial = trial
        self.plan = plan
        self.acceptance = acceptance
        self.audit = audit
        self.calibrator = calibrator

    def predict(self, rows: Sequence[Mapping], *, decision_timestamp=None) -> list[str]:
        return [item["label"] for item in self.predict_decisions(
            rows, decision_timestamp=decision_timestamp)]

    def predict_decisions(self, rows: Sequence[Mapping], *,
                          decision_timestamp=None) -> list[dict]:
        """Return deployable confidence and explicit UNKNOWN/abstention."""
        task = self.audit["source_task_spec"]
        validate_observations(task, rows, decision_timestamp=decision_timestamp)
        field_order = ([self.plan["feature_contract"]["time_field"]]
                       + self.plan["feature_contract"]["ordered_channel_names"])
        sequence = np.asarray([[r[field] for field in field_order] for r in rows])
        feature = _features(sequence, self.trial.feature)
        probabilities = _probabilities(
            self.model, feature[None, ...], len(self.plan["ontology"]["labels"]))
        labels = self.plan["ontology"]["labels"]
        if self.calibrator is not None:
            decision = self.calibrator.decisions(probabilities)
        else:
            decision = [{"class_index": int(np.argmax(row)),
                         "confidence": float(max(row)), "abstained": False}
                        for row in probabilities]
        return [{"label": ("__UNKNOWN__" if item["abstained"]
                           else labels[item["class_index"]]),
                 "confidence": item["confidence"],
                 "abstained": item["abstained"]}
                for item in decision]

    def export(self, directory: str | Path) -> dict:
        """Write research-only artifact and content hash, no qualified metadata."""
        import joblib
        directory = Path(directory)
        if directory.exists() and any(directory.iterdir()):
            raise ValueError("ARTIFACT_DIRECTORY_MUST_BE_EMPTY")
        directory.mkdir(parents=True, exist_ok=True)
        model_path = directory / "model.joblib"
        joblib.dump(self.model, model_path)
        binary_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
        description = {
            "schema": "phm.industrial.design-artifact.v0",
            "task_spec_sha256": self.plan["task_spec_sha256"],
            "trial": asdict(self.trial),
            "feature_program_sha256": (
                program_digest(self.trial.feature)
                if self.trial.feature.startswith("program:") else None),
            "candidate_grid_sha256": _sha(self.plan["candidate_grid"]),
            "acceptance_policy_sha256": _sha(self.acceptance),
            "calibration_policy_sha256": (
                _sha(self.plan["calibration_policy"])
                if self.plan["calibration_policy"] is not None else None),
            "training_manifest_sha256": self.audit["training_manifest_sha256"],
            "training_observations_sha256": self.audit["training_observations_sha256"],
            "trial_selection_audit": self.audit,
            "model_sha256": binary_hash,
            "calibration_state": (asdict(self.calibrator)
                                  if self.calibrator is not None else None),
            "model_file": "model.joblib",
            "trained": True,
            "development_selection_only": True,
            "independent_qualification": False,
            "warning": "Untrusted joblib/pickle model files must never be loaded.",
        }
        (directory / "audit.json").write_text(
            json.dumps(description, indent=2, ensure_ascii=False,
                       sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return {"directory": str(directory),
                "model_sha256": binary_hash, "qualified": False}


def run_design(task_spec: Mapping | TaskSpec,
               records: Sequence[Mapping], *,
               acceptance: DesignAcceptance = DesignAcceptance()) -> IndustrialDesignArtifact:
    """Run a bounded, source-only cross-environment selection and TRAIN refit.

    records: [{measurements: [{time_field: float, channel: float, ...},...],
               metadata: {role: TRAIN, label: ..., group_keys, env_keys}}, ...].
    These are already segmented trajectories; the sampling protocol must
    partition *physical instances before windowing* upstream.
    """
    if not isinstance(acceptance, DesignAcceptance):
        raise ValueError("INVALID_DESIGN_ACCEPTANCE_TYPE")
    if not isinstance(records, (tuple, list)) or not records:
        raise TaskSpecRejected("EMPTY_DESIGN_RECORDS")
    raw = task_spec.to_dict() if isinstance(task_spec, TaskSpec) else TaskSpec.from_dict(task_spec).to_dict()
    source_digest = TaskSpec.from_dict(raw).sha256
    if raw["task"]["decision_mode"] != "offline_trajectory":
        raise TaskSpecRejected("CAUSAL_R81_EXECUTION_NOT_IMPLEMENTED")
    train_metadata = []
    matrices = []
    for index, row in enumerate(records):
        if not isinstance(row, Mapping) or set(row) != {"measurements", "metadata"}:
            raise TaskSpecRejected("INVALID_R81_RECORD", str(index))
        metadata = row["metadata"]
        if not isinstance(metadata, Mapping):
            raise TaskSpecRejected("INVALID_R81_METADATA")
        train_metadata.append(dict(metadata))
        measurements = row["measurements"]
        validate_observations(raw, measurements)
        fields = [raw["observations"]["time"]["field"]] + [
            channel["name"] for channel in raw["observations"]["channels"]]
        matrices.append(np.asarray([[q[field] for field in fields]
                                    for q in measurements], dtype=np.float64))
    plan = compile_task_spec(raw, train_manifest=train_metadata)
    budget = plan["budget"]
    calibration_policy = plan["calibration_policy"]
    labels = plan["ontology"]["labels"]
    healthy_idx = labels.index(plan["ontology"]["healthy_label"])
    label_to_idx = {k: i for i, k in enumerate(labels)}
    true = np.asarray([label_to_idx[row["label"]] for row in train_metadata], dtype=int)
    group_columns = plan["train_data_contract"]["group_keys"]
    env_columns = plan["train_data_contract"]["environment_axes"]
    # Provenance is never passed to a deployment input.
    groups = np.asarray(["|".join(row[k] for k in group_columns)
                         for row in train_metadata])
    environment = np.asarray(["|".join(row[k] for k in env_columns)
                              for row in train_metadata])
    if len(set(environment.tolist())) < 3:
        raise TaskSpecRejected("INSUFFICIENT_SOURCE_ENVIRONMENTS")
    seen_trajectories = {}
    for sample_index, (matrix, group) in enumerate(zip(matrices, groups)):
        # Exact duplicate observations under different physical unit IDs can
        # masquerade as independent examples even with perfect label-role
        # checks. Do not treat renaming as a new sample/world.
        fingerprint = hashlib.sha256(
            np.ascontiguousarray(matrix, dtype="<f8").tobytes()
        ).hexdigest()
        previously = seen_trajectories.get(fingerprint)
        if previously is not None and previously != group:
            raise TaskSpecRejected(
                "DUPLICATE_TRAJECTORY_ACROSS_PHYSICAL_GROUPS",
                str(sample_index))
        seen_trajectories[fingerprint] = group
    fold_plan = list(LeaveOneGroupOut().split(true, groups=environment))
    if any(set(groups[train]) & set(groups[test]) or
           set(environment[train]) & set(environment[test])
           for train, test in fold_plan):
        raise AssertionError("R81_PHYSICAL_ENVIRONMENT_SPLIT_LEAKAGE")
    required_features = {
        _candidate(spec, budget["seed"] + i).feature
        for i, spec in enumerate(plan["candidate_grid"])
    }
    features = {
        feature: np.stack([_features(m, feature) for m in matrices])
        for feature in sorted(required_features)
    }
    attempts = []
    selected = None
    fits = 0
    budget_exhausted = False
    begun = time.process_time()
    receipt = {
        "schema": "phm.industrial.design-trials.v0",
        "source_task_spec_sha256": source_digest,
        "source_task_spec": raw,
        "training_manifest_sha256": _sha(train_metadata),
        "training_observations_sha256": hashlib.sha256(
            b"".join(np.ascontiguousarray(a, dtype="<f8").tobytes()
                     for a in matrices)).hexdigest(),
        "source_data_role": "TRAIN_ONLY",
        "split": "LEAVE_ONE_ENTIRE_SOURCE_ENVIRONMENT_OUT",
        "source_environment_count": len(set(environment.tolist())),
        "source_physical_group_count": len(set(groups.tolist())),
        "preprocessing": "MODEL_FIT_INSIDE_TRAIN_FOLD",
        "search_budget": budget,
        "candidate_grid_sha256": _sha(plan["candidate_grid"]),
        "feature_programs_sha256": _sha(plan["registered_feature_programs"]),
        "calibration_policy": calibration_policy,
        "calibration_role": ("TRAIN_INNER_OOF_ONLY" if calibration_policy else None),
        "compute_ledger": {
            "fit_unit": "ESTIMATOR_HEAD_FIT_CHARGE",
            "failed_fit_charge": "CONSERVATIVE_FULL_HEAD_RESERVATION",
            "cpu_seconds": "PYTHON_PROCESS_CPU_BEST_EFFORT",
            "wall_seconds": "HOST_WALL_BEST_EFFORT",
            "inference_wall_seconds": "HOST_WALL_INCLUDES_PYTHON_OVERHEAD",
            "flops_parity_claimed": False,
            "gpu_memory_or_energy_measured": False,
        },
        "attempts": attempts,
        "target_feedback_used": False,
        "final_evaluation_performed": False,
        "independent_qualification": False,
    }
    def fit_budgeted(trial, selection):
        nonlocal fits
        head_cost = (1 if trial.topology == "direct" else
                     _factorized_fit_cost(plan["ontology"]))
        if fits + head_cost > budget["max_fits"]:
            raise RuntimeError("TRIAL_FIT_BUDGET_EXCEEDED")
        if time.process_time() - begun > budget["max_cpu_seconds"]:
            raise RuntimeError("TRIAL_CPU_BUDGET_EXCEEDED")
        if set(true[selection].tolist()) != set(range(len(labels))):
            raise RuntimeError("TRAIN_FOLD_CLASS_SUPPORT_MISSING")
        # Reserve the full known head-fit cost *before* attempting the fit.
        # If a nested head fails midway, conservative reservations remain
        # charged; otherwise failures could secretly exceed the fit budget.
        fits += head_cost
        if trial.topology == "factorized":
            model = _fitted(trial, features[trial.feature][selection],
                            true[selection], environment[selection],
                            plan["ontology"])
        else:
            model = _fitted(trial, features[trial.feature][selection],
                            true[selection], environment[selection])
        if getattr(model, "fit_count_", 1) != head_cost:
            raise RuntimeError("UNDECLARED_MODEL_FIT_COST")
        return model

    def nested_calibration(trial, training_indices):
        # The outer heldout environment is never supplied here.
        inner_env = environment[training_indices]
        if len(set(inner_env.tolist())) < 3:
            raise CalibrationRejected("INSUFFICIENT_INNER_CALIBRATION_ENVIRONMENTS")
        values = np.zeros((len(training_indices), len(labels)), dtype=float)
        for inner_train, inner_holdout in LeaveOneGroupOut().split(
                training_indices, groups=inner_env):
            estimator = fit_budgeted(trial, training_indices[inner_train])
            values[inner_holdout] = _probabilities(
                estimator, features[trial.feature][training_indices[inner_holdout]],
                len(labels))
        return fit_temperature_selective(
            values, true[training_indices], calibration_policy)

    for i, spec in enumerate(plan["candidate_grid"]):
        trial = _candidate(spec, budget["seed"] + i)
        trial_start_cpu = time.process_time()
        trial_start_wall = time.perf_counter()
        trial_fits_before = fits
        entry = {"trial_index": i, "trial": asdict(trial),
                 "trial_sha256": _sha(asdict(trial)),
                 "feature_program_sha256": (
                     program_digest(trial.feature)
                     if trial.feature.startswith("program:") else None),
                 "folds": [], "status": "STARTED",
                 "inference_wall_seconds": 0.}
        attempts.append(entry)
        predictions = np.full(len(true), -2, dtype=int)
        try:
            for train, validation in fold_plan:
                calibrated = (nested_calibration(trial, train)
                              if calibration_policy is not None else None)
                model = fit_budgeted(trial, train)
                infer_started = time.perf_counter()
                if calibrated is None:
                    predictions[validation] = _predict_indices(
                        model, features[trial.feature][validation], len(labels))
                else:
                    probabilities = _probabilities(
                        model, features[trial.feature][validation], len(labels))
                    predictions[validation] = calibrated.decide(probabilities)
                entry["inference_wall_seconds"] += time.perf_counter() - infer_started
                entry["folds"].append({
                    "train_source_environments": sorted(set(environment[train].tolist())),
                    "heldout_environment": sorted(set(environment[validation].tolist())),
                    "train_physical_groups": len(set(groups[train].tolist())),
                    "heldout_physical_groups": len(set(groups[validation].tolist())),
                    "disjoint": True,
                    "inner_calibration": (asdict(calibrated)
                                          if calibrated is not None else None),
                })
            if (predictions == -2).any():
                raise RuntimeError("INCOMPLETE_TRAIN_SOURCE_OOF")
            metrics = _score(true, predictions, environment, groups, labels, healthy_idx)
            decision = acceptance.evaluate(metrics)
            if (calibration_policy is not None and
                    metrics["coverage"] < calibration_policy["minimum_coverage"]):
                decision["eligible"] = False
                decision["failed_gates"].append({
                    "metric": "source_oof_coverage",
                    "observed": metrics["coverage"],
                    "required": calibration_policy["minimum_coverage"],
                    "relation": ">="})
            entry["metrics"], entry["acceptance"] = metrics, decision
            entry["status"] = "ELIGIBLE" if decision["eligible"] else "REJECTED_BY_POLICY"
            if decision["eligible"]:
                rank = (-(metrics["worst_environment_class_recall"]),
                        -(metrics["worst_environment_accuracy"]),
                        metrics["healthy_false_alarm"], -metrics["macro_f1"],
                        -metrics["coverage"], i)
                if selected is None or rank < selected[0]:
                    selected = (rank, trial)
        except Exception as exc:
            entry["status"] = "FAILED"
            entry["failure"] = repr(exc)
            if str(exc) in ("TRIAL_FIT_BUDGET_EXCEEDED", "TRIAL_CPU_BUDGET_EXCEEDED"):
                budget_exhausted = True
        entry["cpu_seconds"] = round(time.process_time() - trial_start_cpu, 5)
        entry["wall_seconds"] = round(time.perf_counter() - trial_start_wall, 5)
        entry["inference_wall_seconds"] = round(entry["inference_wall_seconds"], 6)
        entry["model_fits"] = fits - trial_fits_before
        if budget_exhausted:
            break
    receipt["fold_model_fits"] = fits
    receipt["total_cpu_seconds_at_selection"] = round(time.process_time() - begun, 5)
    receipt["acceptance_policy"] = asdict(acceptance)
    if selected is None:
        receipt["decision"] = ("BUDGET_EXHAUSTED_BEFORE_SELECTION"
                               if budget_exhausted else "NO_ELIGIBLE_DESIGN")
        receipt["model_delivered"] = False
        raise IndustrialDesignRejected(receipt)
    if budget_exhausted:
        # Refuse a truncated portfolio: an earlier eligible arm cannot mask
        # an unevaluated candidate after the predeclared fit budget expires.
        receipt["decision"] = "BUDGET_EXHAUSTED_BEFORE_FINAL_REFIT"
        receipt["model_delivered"] = False
        raise IndustrialDesignRejected(receipt)
    trial = selected[1]
    refit_fits_before = fits
    refit_started_wall = time.perf_counter()
    head_cost = (1 if trial.topology == "direct" else
                 _factorized_fit_cost(plan["ontology"]))
    final_refit_cost = head_cost * (
        (len(set(environment.tolist())) + 1) if calibration_policy else 1)
    if (fits + final_refit_cost > budget["max_fits"]
            or time.process_time() - begun > budget["max_cpu_seconds"]):
        receipt["decision"] = "BUDGET_EXHAUSTED_BEFORE_FINAL_REFIT"
        receipt["model_delivered"] = False
        raise IndustrialDesignRejected(receipt)
    try:
        calibrator = (nested_calibration(trial, np.arange(len(true)))
                      if calibration_policy is not None else None)
        trained = fit_budgeted(trial, np.arange(len(true)))
    except Exception as exc:
        receipt["decision"] = ("BUDGET_EXHAUSTED_BEFORE_FINAL_REFIT"
                               if str(exc).startswith("TRIAL_") else
                               "SELECTED_REFIT_FAILED")
        receipt["failure"] = repr(exc)
        receipt["model_delivered"] = False
        raise IndustrialDesignRejected(receipt) from exc
    receipt["decision"] = "TRAIN_ONLY_SELECTION_REFITTED"
    receipt["selected_trial"] = asdict(trial)
    receipt["selected_trial_sha256"] = _sha(asdict(trial))
    receipt["total_model_fits"] = fits
    receipt["final_refit_model_fits"] = fits - refit_fits_before
    receipt["final_refit_wall_seconds"] = round(
        time.perf_counter() - refit_started_wall, 5)
    receipt["model_delivered"] = True
    receipt["model_qualified"] = False
    receipt["calibration_state"] = (asdict(calibrator) if calibrator else None)
    return IndustrialDesignArtifact(model=trained, trial=trial, plan=plan,
                                    acceptance=asdict(acceptance), audit=receipt,
                                    calibrator=calibrator)
