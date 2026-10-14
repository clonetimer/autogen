"""Research-only causal rolling heat-balance representation for the *family* task.

Unlike multicomponent ``mechanism.v2``, this adds statistics to the single-thermal
family.v1 input contract.  It never reads labels, onset, fault severity, paired
nominal trajectories or hidden simulator states.  Normal design passports must
have been frozen before injection.  No hardware accuracy is implied.
"""
from __future__ import annotations

import copy
import csv
from pathlib import Path

import numpy as np

from .adapters import sat_nonadcs as native
from .data import Dataset, sequence_from_csv, validate_contract
from .family_pipeline import (
    SCHEMA as PASSPORT_SCHEMA, compute_features as family_features,
    feature_contract as family_contract, feature_binding as family_binding,
    validate_passport,
)
from .io import digest, read_json, write_json, sha_file, safe_child, seal, verify

SCHEMA = "phm.family-thermal-rolling.v2"
WINDOWS = (16, 64)
FEATURES = tuple(
    (name + f"_{w}" + suffix, unit)
    for w in WINDOWS
    for name, suffix, unit in (
        ("energy_residual_mean", "_w", "W"),
        ("radiation_loss_slope", "", "1"),
        ("unexplained_heat_intercept", "_w", "W"),
        ("thermal_regression_computable", "", "1"),
    )
)


def feature_contract() -> dict:
    c = copy.deepcopy(family_contract("thermal", "mechanism"))
    c["device_family"] += ".rolling.v2"
    c["scope"] += "; causal per-window trailing regression (16 and 64 samples), no fault label or future observations"
    c["channels"].extend({"name": name, "unit": unit,
        "role": "deployment_input", "provenance": "Causal heat balance from present and past telemetry and pre-injection normal passport"}
        for name, unit in FEATURES)
    validate_contract(c)
    return c


def feature_binding() -> dict:
    b = copy.deepcopy(family_binding("thermal", "mechanism"))
    b["contract"] = feature_contract()
    b["groups"].append({"id": "causal_thermal_rolling", "channels": [name for name, _ in FEATURES]})
    b["feature_pipeline"] = SCHEMA
    b["representation_version"] = "family.thermal.rolling.v2"
    return b


def _rolling_features(raw: np.ndarray, passport: dict) -> np.ndarray:
    x = np.asarray(raw, dtype=np.float64)
    C, solar, rad = validate_passport(passport, "thermal")
    dt = float(native.FAMILIES["thermal"]["sample_s"])
    t = x[:, 2]
    if np.any((t <= 0) | (t > 2000)):
        raise ValueError("ABSOLUTE_TEMPERATURE_REQUIRED")
    qrad = rad * t**4
    q = (qrad[1:] + qrad[:-1]) / 2.0
    r = C * np.diff(t) / dt - (solar * x[1:, 0] + x[1:, 1] - q)
    extras = []
    for width in WINDOWS:
        out = np.zeros((len(t), 4), dtype=np.float64)
        # Row i uses only intervals with right endpoints <= i.
        for i in range(1, len(t)):
            lo = max(0, i-width)
            rr = r[lo:i]
            qq = q[lo:i]
            out[i, 0] = float(np.mean(rr))
            if len(qq) < 8:
                continue
            qm = float(np.mean(qq))
            variance = float(np.var(qq))
            if variance <= max(.01, 1e-4*qm*qm):
                continue
            slope = float(np.mean((qq-qm)*(rr-np.mean(rr)))/variance)
            out[i, 1:] = (slope, float(np.mean(rr))-slope*qm, 1.0)
        extras.append(out)
    return np.concatenate(extras, axis=1)


def compute_features(raw: np.ndarray, passport: dict) -> np.ndarray:
    base = family_features("thermal", raw, passport, "mechanism")
    # family_features handles shape, finiteness, boundary and passport validation.
    extra = _rolling_features(raw, passport)
    out = np.column_stack((base.astype(np.float64), extra)).astype(np.float32)
    if not np.isfinite(out).all():
        raise ValueError("NONFINITE_THERMAL_V2_FEATURES")
    if out.shape[1] != len(feature_contract()["channels"]):
        raise ValueError("THERMAL_V2_CONTRACT_SHAPE")
    return out


def prepare_section(campaign, section: str, output) -> dict:
    """Prepare only the requested sealed role from real campaign raw observations."""
    if section not in {"development", "calibration", "holdout"}:
        raise ValueError("THERMAL_V2_SECTION_INVALID")
    root, target = Path(campaign), Path(output)
    if target.exists():
        raise FileExistsError("THERMAL_V2_OUTPUT_EXISTS")
    verify(root)
    registry = read_json(root / "passport_registry.json")
    if registry.get("schema") != "phm.passport-registry.v1" or registry.get("frozen_before_simulation") is not True:
        raise ValueError("PASSPORT_NOT_FROZEN")
    roles = {"development": {"train", "validation"},
             "calibration": {"calibration"}, "holdout": {"test"}}[section]
    data = Dataset(root / "native" / "thermal" / section / "dataset.json", allowed_splits=roles)
    data.verify_files()
    if data.contract != native.contract("thermal"):
        raise ValueError("RAW_DEVICE_CONTRACT_CHANGED")
    target.mkdir(parents=True)
    contract = feature_contract()
    samples, records = [], []
    for s in data.samples:
        key = "thermal/" + s["lineage_id"]
        pr = registry["passports"][key]
        pp = safe_child(root, pr["path"])
        if sha_file(pp) != pr["sha256"]:
            raise ValueError("PASSPORT_FILE_CHANGED")
        passport = read_json(pp)
        if passport["schema"] != PASSPORT_SCHEMA or passport["instance_lineage"] != s["lineage_id"]:
            raise ValueError("PASSPORT_INSTANCE_MISMATCH")
        raw = sequence_from_csv(safe_child(data.root, s["path"]), data.contract, native.FAMILIES["thermal"]["window"])
        features = compute_features(raw, passport)
        relative = "observations/" + s["id"] + ".csv"
        p = target / relative
        p.parent.mkdir(exist_ok=True)
        with p.open("x", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["time_s", *[entry["name"] for entry in contract["channels"]]])
            for i, row in enumerate(features):
                w.writerow([i * data.contract["sample_s"], *map(float, row)])
        samples.append({**s, "path": relative, "sha256": sha_file(p)})
        records.append({"id": s["id"], "passport_sha256": pr["sha256"], "raw_sha256": s["sha256"]})
    manifest = {"schema": "phm.sequence-dataset.v1",
        "dataset_id": digest([data.hash, SCHEMA, sha_file(Path(__file__))]),
        "contract": contract, "samples": samples,
        "provenance": {"raw_dataset_hash": data.hash, "section": section,
            "transform": SCHEMA, "transform_sha256": sha_file(Path(__file__)),
            "parent_representation": "family.mechanism.v1",
            "rolling_windows": list(WINDOWS), "source_role": "trusted_preparation_no_learning",
            "passport_role": "pre_injection_nominal", "fault_labels_used_for_features": False,
            "fault_onset_used_for_features": False, "pair_baseline_used": False,
            "selected_on_test": False, "production_qualified": False}}
    write_json(target / "dataset.json", manifest)
    write_json(target / "device_binding.json", feature_binding())
    write_json(target / "source_records.json", records)
    Dataset(target / "dataset.json", allowed_splits=roles).verify_files()
    seal(target, [str(p.relative_to(target)) for p in target.rglob("*") if p.is_file()])
    return manifest
