"""Thermal commissioning v2: healthy nuisance regression for residual diagnosis.

This module reuses a *separate healthy commissioning observation* and the frozen
normal-design passport.  It fits only telemetry-observable nuisance structure in
that healthy run.  No fault label, onset, severity, paired diagnosis-time nominal
trajectory, or simulator internal truth is consumed.

The fitted linear health model explains nominal thermal balance residual using
three deployment-observable regressors: nominal absorbed solar input, commanded
heat input, and nominal radiation.  Diagnosis features subtract that frozen
healthy relation before estimating fault-like radiation slope and heat intercept.
This is a research feature transform, not system identification or hardware
qualification.
"""
from __future__ import annotations
import copy
import csv
import math
from pathlib import Path
import numpy as np

from .adapters import sat_nonadcs as native
from . import family_pipeline as base
from . import commissioning as v1
from .data import Dataset, sequence_from_csv, validate_contract, transform
from .engine import deterministic_cpu
from .io import digest, read_json, write_json, sha_file, seal, verify, safe_child

MODEL_SCHEMA = "phm.thermal-health-regression.v1"
REGISTRY_SCHEMA = "phm.thermal-health-regression-registry.v1"
TRANSFORM = "phm.thermal-commissioning-features.v2"
COEFFICIENTS = ("solar_input_gain_error", "heat_command_gain_error", "radiation_gain_error", "intercept_w")
EXTRA = [
    ("health_corrected_energy_residual_w", "W"),
    ("health_corrected_residual_short_mean_w", "W"),
    ("health_corrected_residual_long_mean_w", "W"),
    ("fault_radiation_slope", "1"),
    ("fault_heat_intercept_w", "W"),
    ("fault_regression_computable", "1"),
    ("health_model_fit_rms_w", "W"),
    ("health_model_log10_condition", "1"),
]


def implementation_sha() -> str:
    return digest({"thermal_v2": sha_file(Path(__file__)), "commissioning_v1": v1.implementation_sha(),
                   "base_features": base.implementation_sha()})


def _thermal_terms(raw: np.ndarray, passport: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return interval residual y, regressors X and nominal radiation, all length n-1."""
    values = base.validate_passport(passport, "thermal")
    C, solar, rad = values
    x = np.asarray(raw, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 10 or x.shape[1] != len(native.FAMILIES["thermal"]["channels"]) or not np.isfinite(x).all():
        raise ValueError("INVALID_THERMAL_HEALTH_OBSERVATION")
    if np.any((x[:, 0] < 0) | (x[:, 0] > 1)):
        raise ValueError("BOUNDARY_VALUE_OUT_OF_RANGE")
    T = x[:, 2]
    if np.any((T <= 0) | (T > 2000)):
        raise ValueError("ABSOLUTE_TEMPERATURE_REQUIRED")
    dt = float(native.FAMILIES["thermal"]["sample_s"])
    qrad = rad * T**4
    radiation = .5 * (qrad[1:] + qrad[:-1])
    residual = C * np.diff(T) / dt - (solar * x[1:, 0] + x[1:, 1] - radiation)
    X = np.column_stack((solar * x[1:, 0], x[1:, 1], radiation, np.ones(len(residual))))
    if not np.isfinite(X).all() or not np.isfinite(residual).all():
        raise ValueError("NONFINITE_THERMAL_HEALTH_TERMS")
    return residual, X, radiation


def _fit_linear_health_model(residual: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, float, float, int]:
    residual = np.asarray(residual, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != 4 or residual.shape != (len(X),) or len(X) < 16:
        raise ValueError("THERMAL_HEALTH_REGRESSION_SHAPE")
    # Use the second half to avoid start-up transients, matching v1 commissioning intent.
    start = len(X) // 2
    A, y = X[start:], residual[start:]
    # Column scaling is numerical only; coefficients are converted back to physical units.
    scale = np.maximum(np.sqrt(np.mean(A[:, :3] ** 2, axis=0)), 1e-8)
    As = A.copy(); As[:, :3] /= scale
    coef_s, _, rank, singular = np.linalg.lstsq(As, y, rcond=1e-10)
    if rank < 4 or len(singular) != 4 or singular[-1] <= 0:
        raise ValueError("INSUFFICIENT_THERMAL_COMMISSIONING_EXCITATION")
    cond = float(singular[0] / singular[-1])
    if not math.isfinite(cond) or cond > 1e8:
        raise ValueError("ILL_CONDITIONED_THERMAL_COMMISSIONING")
    coef = coef_s.copy(); coef[:3] /= scale
    pred = A @ coef
    rms = float(np.sqrt(np.mean((y - pred) ** 2)))
    if not np.isfinite(coef).all() or not math.isfinite(rms):
        raise ValueError("NONFINITE_THERMAL_HEALTH_MODEL")
    return coef, rms, cond, len(y)


def model_from_raw(world: str, raw: np.ndarray, passport: dict, source_sha: str) -> dict:
    residual, X, _ = _thermal_terms(raw, passport)
    coef, rms, cond, n = _fit_linear_health_model(residual, X)
    model = {
        "schema": MODEL_SCHEMA,
        "domain": "thermal",
        "instance_lineage": world,
        "source_observation_sha256": source_sha,
        "source_role": "separate_nominal_commissioning_observation",
        "regression": "healthy_balance_residual ~ solar_input + heat_command + nominal_radiation + intercept",
        "fit_samples": n,
        "coefficients": dict(zip(COEFFICIENTS, map(float, coef))),
        "fit_rms_w": rms,
        "condition_number": cond,
        "fault_information_used": False,
        "paired_diagnosis_nominal_used": False,
        "simulator_truth_used": False,
        "scope": "research healthy-nuisance regression; health state must be independently established",
    }
    validate_model(model)
    return model


def validate_model(model: dict) -> np.ndarray:
    fields = {"schema", "domain", "instance_lineage", "source_observation_sha256", "source_role",
              "regression", "fit_samples", "coefficients", "fit_rms_w", "condition_number",
              "fault_information_used", "paired_diagnosis_nominal_used", "simulator_truth_used", "scope"}
    if not isinstance(model, dict) or set(model) != fields or model.get("schema") != MODEL_SCHEMA or model.get("domain") != "thermal":
        raise ValueError("THERMAL_HEALTH_MODEL_SCHEMA")
    if not isinstance(model["instance_lineage"], str) or not model["instance_lineage"]:
        raise ValueError("THERMAL_HEALTH_MODEL_LINEAGE")
    if not isinstance(model["source_observation_sha256"], str) or len(model["source_observation_sha256"]) != 64:
        raise ValueError("THERMAL_HEALTH_MODEL_SOURCE")
    if model["source_role"] != "separate_nominal_commissioning_observation":
        raise ValueError("THERMAL_HEALTH_MODEL_ROLE")
    if type(model["fit_samples"]) is not int or model["fit_samples"] < 8:
        raise ValueError("THERMAL_HEALTH_MODEL_SAMPLE_COUNT")
    if any(model[k] is not False for k in ("fault_information_used", "paired_diagnosis_nominal_used", "simulator_truth_used")):
        raise ValueError("THERMAL_HEALTH_MODEL_FORBIDDEN_INFORMATION")
    if not isinstance(model["coefficients"], dict) or tuple(model["coefficients"]) != COEFFICIENTS:
        raise ValueError("THERMAL_HEALTH_MODEL_COEFFICIENTS")
    coef = np.asarray([model["coefficients"][k] for k in COEFFICIENTS], dtype=np.float64)
    if not np.isfinite(coef).all():
        raise ValueError("THERMAL_HEALTH_MODEL_NONFINITE")
    for k in ("fit_rms_w", "condition_number"):
        v = model[k]
        if type(v) not in (int, float) or isinstance(v, bool) or not math.isfinite(v) or v < 0:
            raise ValueError("THERMAL_HEALTH_MODEL_DIAGNOSTIC")
    if not isinstance(model["scope"], str) or not model["scope"]:
        raise ValueError("THERMAL_HEALTH_MODEL_SCOPE")
    return coef


def fit_registry(campaign, commissioning, output) -> dict:
    campaign, commissioning, output = Path(campaign), Path(commissioning), Path(output)
    verify(campaign); verify(commissioning)
    if output.exists():
        raise FileExistsError("THERMAL_HEALTH_REGISTRY_OUTPUT_EXISTS")
    creg = read_json(commissioning / "registry.json")
    if creg.get("schema") != v1.REGISTRY_SCHEMA or creg.get("source_campaign_manifest_sha256") != sha_file(campaign / "manifest.json"):
        raise ValueError("COMMISSIONING_REGISTRY_SOURCE_CHANGED")
    preg = read_json(campaign / "passport_registry.json")
    output.mkdir(parents=True)
    models = {}
    for key, entry in creg["profiles"].items():
        domain, world = key.split("/", 1)
        if domain != "thermal":
            continue
        obs = safe_child(commissioning, entry["observation"])
        if sha_file(obs) != entry["observation_sha256"]:
            raise ValueError("COMMISSIONING_OBSERVATION_CHANGED")
        pr = preg["passports"][key]; pfile = safe_child(campaign, pr["path"])
        if sha_file(pfile) != pr["sha256"]:
            raise ValueError("PASSPORT_FILE_CHANGED")
        passport = read_json(pfile)
        # Full commissioning series, not diagnosis window truncation.
        with obs.open(newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        expected = ["time_s", *[n for n, _ in native.FAMILIES["thermal"]["channels"]]]
        if not rows or rows[0] != expected:
            raise ValueError("THERMAL_COMMISSIONING_HEADER")
        raw = np.asarray([[float(v) for v in row[1:]] for row in rows[1:]], dtype=np.float64)
        model = model_from_raw(world, raw, passport, sha_file(obs))
        path = output / "models" / (world + ".json")
        write_json(path, model)
        models[key] = {"model": str(path.relative_to(output)), "model_sha256": sha_file(path),
                       "commissioning_observation_sha256": sha_file(obs)}
    if not models:
        raise ValueError("NO_THERMAL_COMMISSIONING_MODELS")
    record = {"schema": REGISTRY_SCHEMA,
        "source_campaign_manifest_sha256": sha_file(campaign / "manifest.json"),
        "source_commissioning_manifest_sha256": sha_file(commissioning / "manifest.json"),
        "models": models, "completed_models": len(models), "implementation_sha256": implementation_sha(),
        "fault_information_used": False, "diagnosis_time_paired_nominal_used": False,
        "production_qualified": False}
    write_json(output / "registry.json", record)
    seal(output, [str(p.relative_to(output)) for p in output.rglob("*") if p.is_file() and p.name != "manifest.json"])
    return record


def thermal_v2_contract() -> dict:
    c = copy.deepcopy(v1.commissioned_contract("thermal"))
    c["device_family"] += ".healthy_regression.v2"
    c["scope"] += "; healthy residual regression removes static commissioning mismatch before fault regression"
    for name, unit in EXTRA:
        c["channels"].append({"name": name, "unit": unit, "role": "deployment_input",
            "provenance": "causal diagnosis statistic using frozen separate healthy commissioning regression; no fault truth"})
    validate_contract(c)
    return c


def thermal_v2_binding() -> dict:
    b = copy.deepcopy(v1.commissioned_binding("thermal"))
    b["contract"] = thermal_v2_contract()
    b["groups"].append({"id": "thermal_health_regression", "channels": [n for n, _ in EXTRA]})
    b["feature_pipeline"] = TRANSFORM
    b["thermal_health_model_required"] = True
    b["thermal_health_model_schema"] = MODEL_SCHEMA
    return b


def _rolling_mean(x: np.ndarray, width: int) -> np.ndarray:
    return v1._rolling_mean_causal(np.asarray(x, dtype=np.float64), width)


def compute_features(raw: np.ndarray, passport: dict, profile: dict, model: dict) -> np.ndarray:
    if profile.get("instance_lineage") != passport.get("instance_lineage") or model.get("instance_lineage") != passport.get("instance_lineage"):
        raise ValueError("THERMAL_COMMISSIONING_INSTANCE_MISMATCH")
    coef = validate_model(model)
    base_features = v1.compute_commissioned_features("thermal", raw, passport, profile).astype(np.float64)
    residual, X, radiation = _thermal_terms(raw, passport)
    corrected = residual - X @ coef
    n = len(raw)
    corrected_full = np.r_[0.0, corrected]
    count = np.arange(1, n, dtype=float)
    sumq = np.cumsum(radiation); sumr = np.cumsum(corrected)
    qmean = sumq / count; rmean = sumr / count
    var = np.cumsum(radiation**2) / count - qmean**2
    cov = np.cumsum(radiation * corrected) / count - qmean * rmean
    good = (count >= 8) & (var > np.maximum(.01, 1e-4 * qmean**2))
    slope = np.divide(cov, var, out=np.zeros_like(var), where=good)
    intercept = np.where(good, rmean - slope * qmean, 0.0)
    regression = np.vstack((np.zeros((1, 3)), np.column_stack((slope, intercept, good.astype(float)))))
    fit_rms = float(model["fit_rms_w"]); logcond = math.log10(max(float(model["condition_number"]), 1.0))
    extra = np.column_stack((corrected_full, _rolling_mean(corrected_full, 4), _rolling_mean(corrected_full, 16),
                             regression[:, 0], regression[:, 1], regression[:, 2],
                             np.full(n, fit_rms), np.full(n, logcond)))
    result = np.concatenate((base_features, extra), axis=1)
    if not np.isfinite(result).all():
        raise ValueError("NONFINITE_THERMAL_V2_FEATURE")
    return result.astype(np.float32)


def _registry(registry: Path, campaign: Path, commissioning: Path) -> dict:
    verify(registry)
    r = read_json(registry / "registry.json")
    if r.get("schema") != REGISTRY_SCHEMA or r.get("source_campaign_manifest_sha256") != sha_file(campaign / "manifest.json") or r.get("source_commissioning_manifest_sha256") != sha_file(commissioning / "manifest.json"):
        raise ValueError("THERMAL_HEALTH_REGISTRY_SOURCE_CHANGED")
    if r.get("implementation_sha256") != implementation_sha():
        raise ValueError("THERMAL_HEALTH_IMPLEMENTATION_CHANGED")
    return r


def prepare_section(campaign, commissioning, health_registry, section, output):
    if section not in {"development", "calibration", "holdout"}:
        raise ValueError("SECTION_INVALID")
    campaign, commissioning, health_registry, output = map(Path, (campaign, commissioning, health_registry, output))
    creg = v1._registry(commissioning, campaign)
    hreg = _registry(health_registry, campaign, commissioning)
    preg = read_json(campaign / "passport_registry.json")
    roles = {"development": {"train", "validation"}, "calibration": {"calibration"}, "holdout": {"test"}}[section]
    data = Dataset(campaign / "native" / "thermal" / section / "dataset.json", allowed_splits=roles)
    data.verify_files()
    if data.contract != native.contract("thermal"):
        raise ValueError("RAW_DEVICE_CONTRACT_CHANGED")
    output.mkdir(parents=True, exist_ok=False)
    samples, records = [], []
    window = native.FAMILIES["thermal"]["window"]
    for s in data.samples:
        key = "thermal/" + s["lineage_id"]
        pr = preg["passports"][key]; pfile = safe_child(campaign, pr["path"])
        hp = creg["profiles"][key]; hfile = safe_child(commissioning, hp["profile"])
        hm = hreg["models"][key]; mfile = safe_child(health_registry, hm["model"])
        if sha_file(pfile) != pr["sha256"] or sha_file(hfile) != hp["profile_sha256"] or sha_file(mfile) != hm["model_sha256"]:
            raise ValueError("THERMAL_V2_SOURCE_CHANGED")
        passport, profile, model = read_json(pfile), read_json(hfile), read_json(mfile)
        raw = sequence_from_csv(safe_child(data.root, s["path"]), data.contract, window)
        y = compute_features(raw, passport, profile, model)
        rel = "observations/" + s["id"] + ".csv"; path = output / rel; path.parent.mkdir(exist_ok=True)
        with path.open("x", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["time_s", *[c["name"] for c in thermal_v2_contract()["channels"]]])
            for i, row in enumerate(y): w.writerow([i * data.contract["sample_s"], *map(float, row)])
        samples.append({**s, "path": rel, "sha256": sha_file(path)})
        records.append({"id": s["id"], "passport_sha256": pr["sha256"], "commissioning_profile_sha256": hp["profile_sha256"],
                        "thermal_health_model_sha256": hm["model_sha256"], "raw_sha256": s["sha256"]})
    manifest = {"schema": "phm.sequence-dataset.v1",
        "dataset_id": digest([data.hash, creg["source_campaign_manifest_sha256"], hreg["source_commissioning_manifest_sha256"], TRANSFORM, implementation_sha()]),
        "contract": thermal_v2_contract(), "samples": samples,
        "provenance": {"raw_dataset_hash": data.hash, "commissioning_registry_sha256": sha_file(commissioning / "manifest.json"),
            "thermal_health_registry_sha256": sha_file(health_registry / "manifest.json"), "transform": TRANSFORM,
            "transform_sha256": implementation_sha(), "source_role": "trusted_preparation; no learning",
            "commissioning_role": "separate_historical_nominal_observation", "selected_on_test": False}}
    write_json(output / "dataset.json", manifest); write_json(output / "device_binding.json", thermal_v2_binding()); write_json(output / "source_records.json", records)
    Dataset(output / "dataset.json", allowed_splits=roles).verify_files()
    seal(output, [str(p.relative_to(output)) for p in output.rglob("*") if p.is_file()])
    return manifest


def predict_selected(run, selection, csv_path, passport_path, profile_path, health_model_path):
    from .route_selection import _selection, selected_scores
    run, dep, _, rec = _selection(run, selection)
    if dep.get("raw_feature_transform") != TRANSFORM or dep.get("raw_feature_transform_sha256") != implementation_sha():
        raise ValueError("THERMAL_V2_DEPLOYMENT_TRANSFORM_CHANGED")
    if dep["contract"] != thermal_v2_contract():
        raise ValueError("THERMAL_V2_DEPLOYMENT_CONTRACT")
    raw = sequence_from_csv(Path(csv_path), native.contract("thermal"), dep["shape"]["window"])
    features = compute_features(raw, read_json(passport_path), read_json(profile_path), read_json(health_model_path))
    x = transform(features[None], read_json(run / "preprocessing.json"))
    with deterministic_cpu(): scores = selected_scores(run, selection, x)[0]
    j = int(scores.argmax()); accepted = bool(scores[j] >= dep["threshold"])
    return {"route": rec["chosen_route"], "status": "PREDICTED" if accepted else "ABSTAIN_LOW_SCORE",
        "prediction": dep["contract"]["classes"][j] if accepted else None, "closed_set_argmax": dep["contract"]["classes"][j],
        "scores": scores.tolist(), "commissioning_profile_required": True, "thermal_health_model_required": True,
        "probability_calibrated": False, "claim_status": "RESEARCH_ONLY"}


def predict_risk(run, selection, risk, csv_path, passport_path, profile_path, health_model_path):
    from .risk_calibration import _risk
    from .route_selection import selected_scores
    run, dep, _, rec, rr = _risk(run, selection, risk)
    if dep.get("raw_feature_transform") != TRANSFORM or dep.get("raw_feature_transform_sha256") != implementation_sha():
        raise ValueError("THERMAL_V2_DEPLOYMENT_TRANSFORM_CHANGED")
    if dep["contract"] != thermal_v2_contract():
        raise ValueError("THERMAL_V2_DEPLOYMENT_CONTRACT")
    raw = sequence_from_csv(Path(csv_path), native.contract("thermal"), dep["shape"]["window"])
    features = compute_features(raw, read_json(passport_path), read_json(profile_path), read_json(health_model_path))
    x = transform(features[None], read_json(run / "preprocessing.json"))
    with deterministic_cpu(): scores = selected_scores(run, selection, x)[0]
    j = int(scores.argmax()); accepted = bool(rr["status"] == "QUALIFIED_THRESHOLD" and scores[j] >= rr["threshold"])
    return {"route": rec["chosen_route"], "risk_status": rr["status"], "status": "PREDICTED" if accepted else "ABSTAIN_RISK_THRESHOLD",
        "prediction": dep["contract"]["classes"][j] if accepted else None, "closed_set_argmax": dep["contract"]["classes"][j],
        "scores": scores.tolist(), "threshold": rr["threshold"], "commissioning_profile_required": True,
        "thermal_health_model_required": True, "probability_calibrated": False, "risk_certified": False,
        "claim_status": "RESEARCH_ONLY"}
