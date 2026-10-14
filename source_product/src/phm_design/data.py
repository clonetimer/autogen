"""Strict, device-independent fixed-length classification data contract.

Each sample is an entire independent case, not an independently counted window.
The last W measurements are selected without onset/label information. Dataset
preparation is a trusted role. fit() never receives the holdout manifest.
"""
from __future__ import annotations
import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from .io import digest, read_json, safe_child, sha_file

FORBIDDEN = ("label.", "truth.", "status.", "estimate.")
FORBIDDEN_PARTS = ("actual_torque", "applied_torque", "effective_", "friction_torque", "fault_active")
SUPPORTED_TASKS = {"fixed_length_case_classification", "fixed_length_fault_mechanism_and_component_localization"}


def validate_contract(contract: dict) -> None:
    expected = {"device_family", "task", "channels", "classes", "sample_s", "evidence_type", "scope"}
    if set(contract) != expected or contract["task"] not in SUPPORTED_TASKS:
        raise ValueError("UNSUPPORTED_TASK_CONTRACT")
    if not isinstance(contract["device_family"], str) or not contract["device_family"]:
        raise ValueError("MISSING_DEVICE_FAMILY")
    classes = contract["classes"]
    if not 2 <= len(classes) <= 256 or len(set(classes)) != len(classes) or not all(isinstance(x, str) and x for x in classes):
        raise ValueError("INVALID_CLASSES")
    channels = contract["channels"]
    if not channels or len(channels) > 512 or len({c["name"] for c in channels}) != len(channels):
        raise ValueError("INVALID_CHANNELS")
    for c in channels:
        if set(c) != {"name", "unit", "role", "provenance"}:
            raise ValueError("CHANNEL_CONTRACT_INCOMPLETE")
        if c["role"] != "deployment_input" or not c["unit"] or not c["provenance"]:
            raise ValueError("INPUT_ROLE_OR_PROVENANCE_INVALID")
        if c["name"].startswith(FORBIDDEN) or any(s in c["name"] for s in FORBIDDEN_PARTS):
            raise ValueError("TRUTH_INPUT_FORBIDDEN")
    if isinstance(contract["sample_s"], bool) or not np.isfinite(contract["sample_s"]) or contract["sample_s"] <= 0:
        raise ValueError("INVALID_SAMPLE_PERIOD")
    if contract["evidence_type"] not in {"simulation", "software_synthetic", "real_unreviewed"}:
        raise ValueError("UNSUPPORTED_EVIDENCE_TYPE")
    if not contract["scope"]:
        raise ValueError("MISSING_SCOPE")


def sequence_from_csv(path: Path, contract: dict, window: int, *, exact_columns: bool = True) -> np.ndarray:
    names = [c["name"] for c in contract["channels"]]
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        required = {"time_s", *names}
        if len(fields) != len(set(fields)) or not required.issubset(fields):
            raise ValueError(f"MISSING_OR_DUPLICATE_COLUMNS: {path}")
        if exact_columns and set(fields) != required:
            raise ValueError("UNDECLARED_INPUT_COLUMNS")
        rows = []
        for row in reader:
            try:
                rows.append([float(row[n]) for n in ["time_s", *names]])
            except (ValueError, TypeError) as exc:
                raise ValueError("NONNUMERIC_INPUT") from exc
    values = np.asarray(rows, dtype=np.float64)
    if len(values) < window or values.ndim != 2:
        raise ValueError("INSUFFICIENT_SEQUENCE_LENGTH")
    if not np.isfinite(values).all():
        raise ValueError("NONFINITE_INPUT")
    differences = np.diff(values[:, 0])
    if np.any(differences <= 0) or not np.allclose(differences, contract["sample_s"], rtol=1e-6, atol=1e-7):
        raise ValueError("TIME_ORDER_OR_SAMPLE_PERIOD_VIOLATION")
    with np.errstate(over="ignore"):
        selected = values[-window:, 1:].astype(np.float32)
    if not np.isfinite(selected).all():
        raise ValueError("INPUT_OUTSIDE_FLOAT32_RANGE")
    return selected


@dataclass
class Block:
    x: np.ndarray
    y: np.ndarray
    samples: list[dict]

    @property
    def groups(self) -> set[str]:
        return {s["group_id"] for s in self.samples}


class Dataset:
    def __init__(self, manifest: str | Path, *, allowed_splits: set[str]) -> None:
        self.path = Path(manifest)
        self.root = self.path.parent
        self.raw = read_json(self.path)
        if set(self.raw) != {"schema", "dataset_id", "contract", "samples", "provenance"} or self.raw["schema"] != "phm.sequence-dataset.v1":
            raise ValueError("DATASET_SCHEMA_MISMATCH")
        self.contract = self.raw["contract"]
        validate_contract(self.contract)
        self.samples = self.raw["samples"]
        if not self.samples:
            raise ValueError("EMPTY_DATASET")
        ids, groups, files, lineage = set(), {}, {}, {}
        for s in self.samples:
            if set(s) != {"id", "group_id", "lineage_id", "split", "label", "path", "sha256"}:
                raise ValueError("SAMPLE_SCHEMA_MISMATCH")
            if not s["id"] or s["id"] in ids or not s["group_id"] or not s["lineage_id"]:
                raise ValueError("DUPLICATE_OR_MISSING_ID")
            ids.add(s["id"])
            if s["split"] not in allowed_splits:
                raise ValueError("DATA_ROLE_VIOLATION")
            if s["label"] not in self.contract["classes"]:
                raise ValueError("UNKNOWN_LABEL")
            for mapping, key in ((groups, s["group_id"]), (lineage, s["lineage_id"]), (files, s["sha256"])):
                if key in mapping and mapping[key] != s["split"]:
                    raise ValueError("CROSS_SPLIT_LEAKAGE")
                mapping[key] = s["split"]
            safe_child(self.root, s["path"])
        self.hash = digest(self.raw)

    def verify_files(self) -> None:
        for sample in self.samples:
            path = safe_child(self.root, sample["path"])
            if not path.is_file() or sha_file(path) != sample["sha256"]:
                raise ValueError("DATA_FILE_CHANGED")

    def load(self, split: str, window: int) -> Block:
        samples = [s for s in self.samples if s["split"] == split]
        if not samples:
            raise ValueError(f"EMPTY_SPLIT: {split}")
        x = np.stack([sequence_from_csv(safe_child(self.root, s["path"]), self.contract, window) for s in samples])
        y = np.asarray([self.contract["classes"].index(s["label"]) for s in samples], dtype=np.int64)
        if set(y.tolist()) != set(range(len(self.contract["classes"]))):
            raise ValueError(f"CLASS_COVERAGE_INCOMPLETE: {split}")
        return Block(x, y, samples)

    def counts(self) -> dict:
        result = {}
        for split in sorted({s["split"] for s in self.samples}):
            rows = [s for s in self.samples if s["split"] == split]
            result[split] = {"cases": len(rows), "groups": len({s["group_id"] for s in rows}),
                "cases_per_class": dict(Counter(s["label"] for s in rows)),
                "groups_per_class": {c: len({s["group_id"] for s in rows if s["label"] == c}) for c in self.contract["classes"]}}
        return result


def reject_overlap(training_samples: list[dict], test_samples: list[dict]) -> None:
    for key in ("id", "group_id", "lineage_id", "sha256"):
        if {s[key] for s in training_samples} & {s[key] for s in test_samples}:
            raise ValueError(f"HOLDOUT_OVERLAP: {key}")


def reject_numeric_overlap(first: Block, second: Block) -> None:
    a = {digest(x.tolist()) for x in first.x}
    b = {digest(x.tolist()) for x in second.x}
    if a & b:
        raise ValueError("IDENTICAL_OBSERVATIONS_ACROSS_ROLES")


def fit_scaler(x: np.ndarray) -> dict:
    values = x.astype(np.float64)
    mean, scale = values.mean((0, 1)), values.std((0, 1))
    scale = np.maximum(scale, 1e-8)
    return {"mean": mean.tolist(), "scale": scale.tolist(), "fit_role": "train_only"}


def transform(x: np.ndarray, scaler: dict) -> np.ndarray:
    result = ((x.astype(np.float64) - np.asarray(scaler["mean"])) / np.asarray(scaler["scale"])).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("NONFINITE_TRANSFORM")
    return result


def statistics(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    return np.concatenate((x.mean(1), x.std(1), x.min(1), x.max(1), x[:, -1], x[:, -1]-x[:, 0], np.sqrt((x*x).mean(1))), axis=1)
