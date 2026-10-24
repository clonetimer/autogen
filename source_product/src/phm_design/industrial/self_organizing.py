"""Opt-in R83 research prototype: bounded, feedback-driven temporal network growth.

This module does not change R80/R81 TaskSpec grammar or the Stable design entry.
Only SOURCE TRAIN trajectories enter model selection. Selection OOF is not an
independent test, and exported artifacts are never flight/industrial qualified.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Mapping, Sequence

import numpy as np
from sklearn.model_selection import LeaveOneGroupOut

from .design_engine import (
    DesignAcceptance, IndustrialDesignRejected, _observations_digest, _sha,
    _matched_train_loss,
)
from .taskspec import TaskSpec, TaskSpecRejected, compile_task_spec, validate_observations
from ..task_candidate_registry import TorchTaskEstimator, deployment_sequence


@dataclass(frozen=True)
class NetworkGenome:
    """Finite, declarative topology grammar; never arbitrary Python modules."""
    width: int = 8
    depth: int = 1
    kernel: int = 3
    pooling: str = "mean"
    residual: bool = False

    def __post_init__(self):
        if (type(self.width) is not int or self.width not in (8, 16, 32) or
                type(self.depth) is not int or self.depth not in (1, 2, 3) or
                type(self.kernel) is not int or self.kernel not in (3, 5) or
                self.pooling not in ("mean", "mean_max") or
                type(self.residual) is not bool):
            raise ValueError("UNREGISTERED_NETWORK_TOPOLOGY")

    @property
    def digest(self) -> str:
        return _sha(asdict(self))

    def neighbors(self) -> tuple["NetworkGenome", ...]:
        """One edit per child, unique deterministic order, including pruning."""
        pool = []
        for width in (8, 16, 32):
            if width != self.width:
                pool.append(NetworkGenome(width, self.depth, self.kernel,
                                          self.pooling, self.residual))
        for depth in (1, 2, 3):
            if depth != self.depth:
                pool.append(NetworkGenome(self.width, depth, self.kernel,
                                          self.pooling, self.residual))
        pool.append(NetworkGenome(self.width, self.depth, 5 if self.kernel == 3 else 3,
                                  self.pooling, self.residual))
        pool.append(NetworkGenome(self.width, self.depth, self.kernel,
                                  "mean_max" if self.pooling == "mean" else "mean",
                                  self.residual))
        pool.append(NetworkGenome(self.width, self.depth, self.kernel,
                                  self.pooling, not self.residual))
        return tuple(pool)


@dataclass(frozen=True)
class EvolutionPolicy:
    max_candidates: int = 3
    epochs: int = 3
    max_parameters: int = 30_000

    def __post_init__(self):
        if (type(self.max_candidates) is not int or
                not 2 <= self.max_candidates <= 12 or
                type(self.epochs) is not int or not 1 <= self.epochs <= 40 or
                type(self.max_parameters) is not int or
                not 100 <= self.max_parameters <= 200_000):
            raise ValueError("INVALID_EVOLUTION_POLICY")


import torch
from torch import nn


class _TemporalTopology(nn.Module):
    """Module-scope Torch architecture to support Windows spawn serialization."""
    def __init__(self, channels: int, classes: int, genome: NetworkGenome):
        super().__init__()
        self.genome = genome
        self.stem = nn.Conv1d(channels, genome.width, 1)
        self.blocks = nn.ModuleList()
        for stage in range(genome.depth):
            dilation = 2 ** stage
            self.blocks.append(nn.Sequential(
                nn.Conv1d(genome.width, genome.width, genome.kernel,
                          dilation=dilation, padding=dilation * (genome.kernel // 2)),
                nn.GroupNorm(1, genome.width), nn.GELU()))
        features = genome.width * (2 if genome.pooling == "mean_max" else 1)
        self.head = nn.Linear(features, classes)

    def forward(self, observations):
        hidden = self.stem(observations.transpose(1, 2))
        for block in self.blocks:
            changed = block(hidden)
            hidden = hidden + changed if self.genome.residual else changed
        pooled = hidden.mean(-1)
        if self.genome.pooling == "mean_max":
            pooled = torch.cat((pooled, hidden.amax(-1)), dim=1)
        return self.head(pooled)


class EvolvingTemporalEstimator(TorchTaskEstimator):
    """Reuse fold-local normalization, deterministic CPU training and inference."""
    def __init__(self, genome: NetworkGenome, seed: int, policy: EvolutionPolicy):
        self.genome = genome
        super().__init__("torch_tcn", seed, epochs=policy.epochs,
                         max_parameters=policy.max_parameters)

    def _network(self, steps: int, channels: int, classes: int):
        return _TemporalTopology(channels, classes, self.genome)


STATIC_R83_GENOMES = (
    NetworkGenome(),
    NetworkGenome(16, 2, 3, "mean_max", True),
    NetworkGenome(8, 2, 3, "mean", False),
)


def propose_genome(history: Sequence[Mapping], visited: set[str]) -> tuple[NetworkGenome, str | None]:
    """After two prespecified seeds, choose an untried child of best TRAIN OOF parent."""
    seeds = (NetworkGenome(), NetworkGenome(16, 2, 3, "mean_max", True))
    for seed in seeds:
        if seed.digest not in visited:
            return seed, None
    ranked = sorted(history, key=lambda item: (
        item["selection_loss"], item["genome_sha256"]))
    for parent in ranked:
        genome = NetworkGenome(**parent["genome"])
        for child in genome.neighbors():
            if child.digest not in visited:
                return child, parent["genome_sha256"]
    raise ValueError("EXHAUSTED_TOPOLOGY_NEIGHBORHOOD")


def _metrics(actual: np.ndarray, predicted: np.ndarray, env: np.ndarray,
             classes: int, healthy_idx: int) -> dict:
    recall = []
    for regime in sorted(set(env.tolist())):
        subset = env == regime
        for cls in range(classes):
            support = subset & (actual == cls)
            if not support.any():
                raise TaskSpecRejected("INSUFFICIENT_ENVIRONMENT_CLASS_SUPPORT")
            recall.append(float(np.mean(predicted[support] == cls)))
    healthy = actual == healthy_idx
    return {
        "accuracy": float(np.mean(actual == predicted)),
        "worst_environment_class_recall": min(recall),
        "healthy_false_alarm": float(np.mean(predicted[healthy] != healthy_idx)),
    }


def _risk(metrics: Mapping) -> float:
    """Single shared R83 TRAIN-selection loss for both comparison arms."""
    return _matched_train_loss(metrics)


@dataclass
class SelfOrganizingArtifact:
    genome: NetworkGenome
    model: EvolvingTemporalEstimator
    task_spec: dict
    audit: dict

    def predict(self, measurements: Sequence[Mapping]) -> dict:
        validate_observations(self.task_spec, measurements)
        fields = [self.task_spec["observations"]["time"]["field"]] + [
            x["name"] for x in self.task_spec["observations"]["channels"]]
        matrix = np.asarray([[row[field] for field in fields] for row in measurements],
                            dtype=np.float64)
        sequence = deployment_sequence(matrix)[None, ...]
        p = self.model.predict_proba(sequence)[0]
        idx = int(np.argmax(p))
        return {"label": self.task_spec["task"]["labels"][idx],
                "confidence_uncalibrated": float(p[idx]),
                "independent_qualification": False}

    def export(self, folder: str | Path) -> dict:
        """Non-pickle numerical state; integrity checks are NOT authenticity."""
        target = Path(folder)
        target.mkdir(parents=True, exist_ok=False)
        if self.model.model_ is None:
            raise ValueError("UNFITTED_SELF_ORGANIZING_MODEL")
        weights = {"state__" + name: value.detach().cpu().numpy()
                   for name, value in self.model.model_.state_dict().items()}
        weights["normalizer_mean"] = np.asarray(self.model.mean_, dtype=np.float32)
        weights["normalizer_scale"] = np.asarray(self.model.scale_, dtype=np.float32)
        np.savez_compressed(target / "weights.npz", **weights)
        binary_sha = hashlib.sha256((target / "weights.npz").read_bytes()).hexdigest()
        manifest = {
            "schema": "phm.self-organizing.artifact.v0",
            "genome": asdict(self.genome),
            "policy": self.audit["policy"],
            "seed": self.audit["seed"],
            "task_spec": self.task_spec,
            "audit": self.audit,
            "weights_sha256": binary_sha,
            "independent_qualification": False,
            "inference_confidence_calibrated": False,
        }
        (target / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        return {"directory": str(target), "weights_sha256": binary_sha,
                "independent_qualification": False}

    @classmethod
    def load(cls, folder: str | Path) -> "SelfOrganizingArtifact":
        """Load only checked numerical arrays, never arbitrary pickle code."""
        root = Path(folder)
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if (manifest.get("schema") != "phm.self-organizing.artifact.v0" or
                manifest.get("independent_qualification") is not False):
            raise ValueError("INVALID_SELF_ORGANIZING_MANIFEST")
        raw = TaskSpec.from_dict(manifest["task_spec"]).to_dict()
        genome = NetworkGenome(**manifest["genome"])
        policy = EvolutionPolicy(**manifest["policy"])
        blob = (root / "weights.npz").read_bytes()
        if len(blob) > 12_000_000 or hashlib.sha256(blob).hexdigest() != manifest["weights_sha256"]:
            raise ValueError("SELF_ORGANIZING_MODEL_INTEGRITY_FAILURE")
        estimator = EvolvingTemporalEstimator(genome, manifest["seed"], policy)
        labels = raw["task"]["labels"]
        channels = len(raw["observations"]["channels"])
        model = estimator._network(48, channels, len(labels))
        import torch
        with np.load(root / "weights.npz", allow_pickle=False) as bundle:
            expected = {"state__" + name for name in model.state_dict()}
            expected.update(("normalizer_mean", "normalizer_scale"))
            if set(bundle.files) != expected:
                raise ValueError("INVALID_SELF_ORGANIZING_STATE_KEYS")
            state = {}
            for name, tensor in model.state_dict().items():
                arr = np.array(bundle["state__" + name])
                if arr.shape != tuple(tensor.shape) or not np.isfinite(arr).all():
                    raise ValueError("INVALID_SELF_ORGANIZING_STATE_SHAPE")
                state[name] = torch.from_numpy(arr.copy())
            mean = np.asarray(bundle["normalizer_mean"], dtype=np.float32)
            scale = np.asarray(bundle["normalizer_scale"], dtype=np.float32)
        if (mean.shape != (1, 1, channels) or scale.shape != mean.shape or
                not np.isfinite(mean).all() or not np.isfinite(scale).all() or
                (scale <= 0).any()):
            raise ValueError("INVALID_SELF_ORGANIZING_NORMALIZATION")
        model.load_state_dict(state, strict=True)
        model.eval()
        estimator.model_, estimator.classes_ = model, np.arange(len(labels))
        estimator.mean_, estimator.scale_ = mean, scale
        return cls(genome, estimator, raw, manifest["audit"])


def run_self_organizing_design(
    task_spec: Mapping | TaskSpec, records: Sequence[Mapping], *,
    policy: EvolutionPolicy = EvolutionPolicy(),
    acceptance: DesignAcceptance = DesignAcceptance(),
    _guarded_execution: bool = False,
    seed_schedule: str = "legacy",
    proposal_mode: str = "feedback",
) -> SelfOrganizingArtifact:
    """Bounded, opt-in experiment. Not an alternate Stable/qualified entrypoint."""
    if not isinstance(policy, EvolutionPolicy) or not isinstance(acceptance, DesignAcceptance):
        raise ValueError("INVALID_SELF_ORGANIZING_ARGUMENTS")
    if seed_schedule not in ("legacy", "r83_matched_train"):
        raise ValueError("UNREGISTERED_SELF_ORGANIZING_SEED_SCHEDULE")
    if proposal_mode not in ("feedback", "predeclared_static"):
        raise ValueError("UNREGISTERED_SELF_ORGANIZING_PROPOSAL_MODE")
    if proposal_mode == "predeclared_static" and policy.max_candidates != 3:
        raise ValueError("STATIC_TOPOLOGY_REQUIRES_EXACTLY_THREE_CANDIDATES")
    if not isinstance(records, (list, tuple)) or not records:
        raise TaskSpecRejected("EMPTY_DESIGN_RECORDS")
    raw = task_spec.to_dict() if isinstance(task_spec, TaskSpec) else TaskSpec.from_dict(task_spec).to_dict()
    if raw["task"]["decision_mode"] != "offline_trajectory":
        raise TaskSpecRejected("CAUSAL_SELF_ORGANIZING_NOT_IMPLEMENTED")
    design = raw["design"]
    if (design["compare"] != ["tcn"] or design["objectives"] != ["ERM"] or
            design.get("topologies", ["direct"]) != ["direct"] or
            any(key in design for key in (
                "adaptive_feature_search", "feature_programs", "calibration"))):
        raise TaskSpecRejected("SELF_ORGANIZING_REQUIRES_TCN_DIRECT_ERM_TEMPLATE")
    matrices, manifest = [], []
    fields = [raw["observations"]["time"]["field"]] + [
        channel["name"] for channel in raw["observations"]["channels"]]
    for row in records:
        if not isinstance(row, Mapping) or set(row) != {"measurements", "metadata"}:
            raise TaskSpecRejected("INVALID_SELF_ORGANIZING_RECORD")
        validate_observations(raw, row["measurements"])
        manifest.append(dict(row["metadata"]))
        matrices.append(np.asarray([[sample[name] for name in fields]
                                    for sample in row["measurements"]], dtype=np.float64))
    plan = compile_task_spec(raw, train_manifest=manifest)
    budget = plan["budget"]
    if policy.max_candidates > budget["max_trials"]:
        raise TaskSpecRejected("SELF_ORGANIZING_TRIAL_BUDGET_EXCEEDED")
    if "max_wall_seconds" in budget and not _guarded_execution:
        raise TaskSpecRejected("SELF_ORGANIZING_REQUIRES_GUARDED_WALL_RUNNER")
    labels = plan["ontology"]["labels"]
    label_idx = {label: i for i, label in enumerate(labels)}
    y = np.asarray([label_idx[m["label"]] for m in manifest])
    group_cols = plan["train_data_contract"]["group_keys"]
    env_cols = plan["train_data_contract"]["environment_axes"]
    groups = np.asarray(["|".join(m[k] for k in group_cols) for m in manifest])
    env = np.asarray(["|".join(m[k] for k in env_cols) for m in manifest])
    if len(set(env.tolist())) < 3:
        raise TaskSpecRejected("INSUFFICIENT_SOURCE_ENVIRONMENTS")
    seen = {}
    for ix, (matrix, group) in enumerate(zip(matrices, groups)):
        key = hashlib.sha256(np.ascontiguousarray(matrix, dtype="<f8").tobytes()).hexdigest()
        if key in seen and seen[key] != group:
            raise TaskSpecRejected("DUPLICATE_TRAJECTORY_ACROSS_PHYSICAL_GROUPS", str(ix))
        seen[key] = group
    splits = list(LeaveOneGroupOut().split(y, groups=env))
    if any(set(groups[a]) & set(groups[b]) for a, b in splits):
        raise TaskSpecRejected("PHYSICAL_GROUP_CROSSES_ENVIRONMENTS")
    for regime in set(env):
        for cls in range(len(labels)):
            if not ((env == regime) & (y == cls)).any():
                raise TaskSpecRejected("INSUFFICIENT_ENVIRONMENT_CLASS_SUPPORT")
    begun = time.process_time()
    sequences = np.stack([deployment_sequence(matrix) for matrix in matrices])
    receipt = {
        "schema": "phm.self-organizing.train-search.v0",
        "source_task_spec_sha256": TaskSpec.from_dict(raw).sha256,
        "train_manifest_sha256": _sha(manifest),
        "train_observations_sha256": _observations_digest(matrices),
        "source_data_role": "TRAIN_ONLY",
        "split": "LEAVE_ONE_ENTIRE_SOURCE_ENVIRONMENT_OUT",
        "source_environment_count": len(set(env.tolist())),
        "source_physical_group_count": len(set(groups.tolist())),
        "policy": asdict(policy),
        "seed": budget["seed"],
        "seed_schedule": seed_schedule,
        "proposal_mode": proposal_mode,
        "budget": budget,
        "attempts": [],
        "target_feedback_used": False,
        "independent_qualification": False,
        "model_delivered": False,
        "feedback_is_independent_test": False,
        "preprocessing": "FOLD_TRAIN_ONLY",
        "wall_guard": ("PARENT_SPAWN_TERMINATION" if _guarded_execution and
                       "max_wall_seconds" in budget else "NOT_CONFIGURED"),
    }
    def refuse(reason):
        receipt["decision"] = reason
        raise IndustrialDesignRejected(receipt)

    visited = set()
    charged_fits = 0
    for trial in range(policy.max_candidates):
        if charged_fits + len(splits) + 1 > budget["max_fits"]:
            refuse("SEARCH_FIT_BUDGET_EXHAUSTED")
        if time.process_time() - begun > budget["max_cpu_seconds"]:
            refuse("SEARCH_CPU_BUDGET_EXHAUSTED")
        if proposal_mode == "predeclared_static":
            genome, parent = STATIC_R83_GENOMES[trial], None
        else:
            genome, parent = propose_genome(receipt["attempts"], visited)
        visited.add(genome.digest)
        prediction = np.full(len(y), -1, dtype=int)
        parameter_count = None
        for fold_idx, (train, holdout) in enumerate(splits):
            if len(set(y[train])) != len(labels):
                refuse("TRAIN_FOLD_CLASS_SUPPORT_MISSING")
            charged_fits += 1  # Reserve before invoking Torch fit.
            receipt["charged_fits"] = charged_fits
            candidate_seed = (budget["seed"] + trial
                              if seed_schedule == "r83_matched_train"
                              else budget["seed"] + trial * 100 + fold_idx)
            estimator = EvolvingTemporalEstimator(genome, candidate_seed, policy)
            try:
                estimator.fit(sequences[train], y[train])
                current_params = estimator.training_["parameter_count"]
                if parameter_count is None:
                    parameter_count = current_params
                prediction[holdout] = estimator.predict_proba(sequences[holdout]).argmax(1)
            except (ValueError, RuntimeError, FloatingPointError) as exc:
                receipt["failed_fit_type"] = type(exc).__name__
                refuse("SELF_ORGANIZING_FIT_FAILED")
            if time.process_time() - begun > budget["max_cpu_seconds"]:
                refuse("SEARCH_CPU_BUDGET_EXHAUSTED")
        if (prediction < 0).any():
            refuse("SOURCE_TRAIN_OOF_INCOMPLETE")
        metrics = _metrics(y, prediction, env, len(labels),
                           labels.index(plan["ontology"]["healthy_label"]))
        attempt = {
            "genome": asdict(genome),
            "genome_sha256": genome.digest,
            "parent_genome_sha256": parent,
            "proposal": ("PREDECLARED_STATIC" if proposal_mode == "predeclared_static"
                         else "PREDECLARED_SEED" if parent is None
                         else "TRAIN_OOF_FEEDBACK_MUTATION"),
            "metrics": metrics,
            "selection_loss": _risk(metrics),
            "parameter_count": parameter_count,
            "eligible": acceptance.evaluate(metrics)["eligible"],
            "feedback_role": "SOURCE_TRAIN_OOF_ONLY",
        }
        receipt["attempts"].append(attempt)
    receipt["charged_fits"] = charged_fits
    eligible = [entry for entry in receipt["attempts"] if entry["eligible"]]
    if not eligible:
        refuse("NO_ELIGIBLE_SELF_ORGANIZING_DESIGN")
    champion = min(eligible, key=lambda entry: (entry["selection_loss"], entry["genome_sha256"]))
    if charged_fits + 1 > budget["max_fits"]:
        refuse("FINAL_REFIT_BUDGET_EXHAUSTED")
    charged_fits += 1
    receipt["charged_fits"] = charged_fits
    genome = NetworkGenome(**champion["genome"])
    if seed_schedule == "r83_matched_train":
        champion_index = next(i for i, entry in enumerate(receipt["attempts"])
                              if entry["genome_sha256"] == genome.digest)
        refit_seed = budget["seed"] + champion_index
    else:
        refit_seed = budget["seed"] + 999
    estimator = EvolvingTemporalEstimator(genome, refit_seed, policy)
    try:
        estimator.fit(sequences, y)
    except (ValueError, RuntimeError, FloatingPointError) as exc:
        receipt["failed_fit_type"] = type(exc).__name__
        refuse("FINAL_REFIT_FAILED")
    if time.process_time() - begun > budget["max_cpu_seconds"]:
        refuse("FINAL_REFIT_CPU_BUDGET_EXHAUSTED")
    receipt["decision"] = "RESEARCH_TRAIN_OOF_ELIGIBLE_NOT_QUALIFIED"
    receipt["selected_genome_sha256"] = genome.digest
    receipt["charged_fits"] = charged_fits
    receipt["model_delivered"] = True
    return SelfOrganizingArtifact(genome, estimator, raw, receipt)
