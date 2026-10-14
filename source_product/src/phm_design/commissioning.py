"""Healthy commissioning baselines for non-ADCS PHM research.

A commissioning profile is derived from a separate native NOMINAL run using the
same device instance nuisance parameters but a different boundary phase and noise
seed.  It contains only statistics computed from deployment-observable telemetry
and the already-approved normal-design passport.  Fault label, onset, severity,
paired diagnosis-time nominal telemetry and simulator internal truth are never
stored in the profile.

This remains a simulation research assumption: a real deployment would need a
reviewed process that establishes when a commissioning interval is genuinely
healthy.
"""
from __future__ import annotations
import copy
import csv
import math
from pathlib import Path
import numpy as np

from .adapters import sat_nonadcs as native
from .data import Dataset, sequence_from_csv, validate_contract, transform
from .engine import deterministic_cpu
from .io import digest, read_json, write_json, sha_file, seal, verify, safe_child
from . import family_pipeline as base

PROFILE_SCHEMA = "phm.commissioning-profile.v1"
REGISTRY_SCHEMA = "phm.commissioning-registry.v1"
TRANSFORM = "phm.nonadcs-commissioning-features.v1"

REFERENCE_NAMES = {
    "eps": ["solar_loss_fraction", "load_innovation_w", "solar_loss_running_mean"],
    "thermal": ["energy_residual_running_w", "radiation_loss_slope", "unexplained_heat_intercept_w"],
    "data": ["generation_excess_running_bps", "contact_innovation_running_bps", "service_loss_fraction"],
}

EXTRA = {
    "eps": [
        ("health_centered_solar_loss_fraction", "1"),
        ("health_centered_load_innovation_w", "W"),
        ("health_centered_solar_running_mean", "1"),
        ("solar_loss_short_running_mean", "1"),
        ("solar_loss_long_running_mean", "1"),
        ("load_innovation_short_running_mean_w", "W"),
        ("load_innovation_long_running_mean_w", "W"),
    ],
    "thermal": [
        ("health_centered_energy_residual_w", "W"),
        ("health_centered_radiation_loss_slope", "1"),
        ("health_centered_unexplained_heat_w", "W"),
        ("energy_residual_short_running_mean_w", "W"),
        ("energy_residual_long_running_mean_w", "W"),
        ("unexplained_heat_short_running_mean_w", "W"),
        ("unexplained_heat_long_running_mean_w", "W"),
    ],
    "data": [
        ("health_centered_generation_excess_bps", "bit/s"),
        ("health_centered_contact_innovation_bps", "bit/s"),
        ("health_centered_service_loss_fraction", "1"),
        ("generation_excess_short_running_mean_bps", "bit/s"),
        ("generation_excess_long_running_mean_bps", "bit/s"),
        ("contact_innovation_short_running_mean_bps", "bit/s"),
        ("contact_innovation_long_running_mean_bps", "bit/s"),
    ],
}


def implementation_sha() -> str:
    return digest({"commissioning": sha_file(Path(__file__)), "base_features": base.implementation_sha()})


def _commissioning_parameters(world: str, domain: str, parameters: dict) -> dict:
    p = copy.deepcopy(parameters)
    # The static plant/meter nuisance remains identical. Only the exogenous phase
    # and measurement-noise realization differ from the diagnosis campaign.
    angle = int(digest([world, domain, "commissioning-phase"])[:16], 16) / (16**16) * 2 * math.pi
    p["phase"] = float((float(p["phase"]) + angle) % (2 * math.pi))
    p["measurement_seed"] = int(digest([world, domain, "commissioning-measurement"])[:16], 16)
    return p


def _profile_from_raw(world: str, domain: str, raw: np.ndarray, passport: dict, source_sha: str) -> dict:
    features = base.compute_features(domain, raw, passport, "mechanism")
    nphys = len(base.PHYSICAL[domain])
    physical = features[:, -nphys:].astype(np.float64)
    valid = physical[:, -1] > 0.5
    # Avoid startup prefix artifacts and require actual physical excitation.
    tail = np.arange(len(physical)) >= len(physical) // 2
    mask = valid & tail
    if int(mask.sum()) < 4:
        raise ValueError("INSUFFICIENT_COMMISSIONING_EXCITATION")
    refs = {}
    for i, name in enumerate(REFERENCE_NAMES[domain]):
        value = float(np.median(physical[mask, i]))
        if not math.isfinite(value):
            raise ValueError("NONFINITE_COMMISSIONING_REFERENCE")
        refs[name] = value
    result = {
        "schema": PROFILE_SCHEMA,
        "domain": domain,
        "instance_lineage": world,
        "source_observation_sha256": source_sha,
        "source_role": "separate_nominal_commissioning_observation",
        "reference_method": "median_of_valid_second_half_mechanism_features",
        "valid_reference_samples": int(mask.sum()),
        "references": refs,
        "fault_information_used": False,
        "paired_diagnosis_nominal_used": False,
        "simulator_truth_used": False,
        "scope": "research commissioning assumption; health status must be independently established in real deployment",
    }
    validate_profile(result, domain)
    return result


def validate_profile(profile: dict, domain: str) -> list[float]:
    fields = {"schema", "domain", "instance_lineage", "source_observation_sha256", "source_role",
              "reference_method", "valid_reference_samples", "references", "fault_information_used",
              "paired_diagnosis_nominal_used", "simulator_truth_used", "scope"}
    if not isinstance(profile, dict) or set(profile) != fields or profile.get("schema") != PROFILE_SCHEMA or profile.get("domain") != domain:
        raise ValueError("COMMISSIONING_PROFILE_SCHEMA")
    if not isinstance(profile["instance_lineage"], str) or not profile["instance_lineage"]:
        raise ValueError("COMMISSIONING_PROFILE_LINEAGE")
    if not isinstance(profile["source_observation_sha256"], str) or len(profile["source_observation_sha256"]) != 64:
        raise ValueError("COMMISSIONING_PROFILE_SOURCE")
    if profile["source_role"] != "separate_nominal_commissioning_observation" or profile["reference_method"] != "median_of_valid_second_half_mechanism_features":
        raise ValueError("COMMISSIONING_PROFILE_METHOD")
    if type(profile["valid_reference_samples"]) is not int or profile["valid_reference_samples"] < 4:
        raise ValueError("COMMISSIONING_PROFILE_SAMPLE_COUNT")
    if any(profile[k] is not False for k in ("fault_information_used", "paired_diagnosis_nominal_used", "simulator_truth_used")):
        raise ValueError("COMMISSIONING_PROFILE_FORBIDDEN_INFORMATION")
    names = REFERENCE_NAMES[domain]
    if not isinstance(profile["references"], dict) or set(profile["references"]) != set(names):
        raise ValueError("COMMISSIONING_PROFILE_REFERENCES")
    values = []
    for name in names:
        v = profile["references"][name]
        if type(v) not in (int, float) or isinstance(v, bool) or not math.isfinite(v):
            raise ValueError("COMMISSIONING_PROFILE_REFERENCE_VALUE")
        values.append(float(v))
    if not isinstance(profile["scope"], str) or not profile["scope"]:
        raise ValueError("COMMISSIONING_PROFILE_SCOPE")
    return values


def generate_commissioning(campaign, output, on_profile=None) -> dict:
    campaign, output = Path(campaign), Path(output)
    verify(campaign)
    if output.exists():
        raise FileExistsError("COMMISSIONING_OUTPUT_EXISTS")
    plan = read_json(campaign / "native" / "generation_plan.json")
    registry = read_json(campaign / "passport_registry.json")
    scenarios = plan.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("COMMISSIONING_SOURCE_PLAN")
    output.mkdir(parents=True)
    profiles = {}
    completed = 0
    seen = set()
    for item in scenarios:
        world, domain = item["world_id"], item["domain"]
        key = domain + "/" + world
        if key in seen:
            raise ValueError("DUPLICATE_COMMISSIONING_INSTANCE")
        seen.add(key)
        pr = registry["passports"][key]
        pfile = safe_child(campaign, pr["path"])
        if sha_file(pfile) != pr["sha256"]:
            raise ValueError("PASSPORT_FILE_CHANGED")
        passport = read_json(pfile)
        hp = _commissioning_parameters(world, domain, item["parameters"])
        native_result = native.run_native(domain, hp, 0)
        raw_rows = native_result["observations"]
        obs = output / "observations" / domain / (world + ".csv")
        obs.parent.mkdir(parents=True, exist_ok=True)
        with obs.open("x", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["time_s", *[name for name, _ in native.FAMILIES[domain]["channels"]]])
            w.writerows(raw_rows)
        raw = np.asarray([row[1:] for row in raw_rows], dtype=np.float64)
        profile = _profile_from_raw(world, domain, raw, passport, sha_file(obs))
        ppath = output / "profiles" / domain / (world + ".json")
        write_json(ppath, profile)
        profiles[key] = {"profile": str(ppath.relative_to(output)), "profile_sha256": sha_file(ppath),
                         "observation": str(obs.relative_to(output)), "observation_sha256": sha_file(obs)}
        completed += 1
        if on_profile:
            on_profile({"domain": domain, "world_id": world, "profile": profiles[key]})
    record = {
        "schema": REGISTRY_SCHEMA,
        "source_campaign_manifest_sha256": sha_file(campaign / "manifest.json"),
        "source_generation_plan_sha256": sha_file(campaign / "native" / "generation_plan.json"),
        "profiles": profiles,
        "completed_profiles": completed,
        "separate_nominal_runs": completed,
        "fault_information_used": False,
        "diagnosis_time_paired_nominal_used": False,
        "implementation_sha256": implementation_sha(),
        "production_qualified": False,
    }
    write_json(output / "registry.json", record)
    seal(output, [str(p.relative_to(output)) for p in output.rglob("*") if p.is_file() and p.name != "manifest.json"])
    return record


def commissioned_contract(domain: str) -> dict:
    c = copy.deepcopy(base.feature_contract(domain, "mechanism"))
    c["device_family"] += ".commissioning.v1"
    c["scope"] += "; separate healthy commissioning baseline required; health state not automatically inferred"
    for name, unit in EXTRA[domain]:
        c["channels"].append({"name": name, "unit": unit, "role": "deployment_input",
            "provenance": "causal diagnosis-window statistic centered by separate healthy commissioning telemetry; no fault truth"})
    validate_contract(c)
    return c


def commissioned_binding(domain: str) -> dict:
    b = copy.deepcopy(base.feature_binding(domain, "mechanism"))
    b["contract"] = commissioned_contract(domain)
    b["groups"].append({"id": "commissioning_adjustment", "channels": [n for n, _ in EXTRA[domain]]})
    b["feature_pipeline"] = TRANSFORM
    b["commissioning_profile_required"] = True
    b["commissioning_profile_schema"] = PROFILE_SCHEMA
    return b


def _rolling_mean_causal(x: np.ndarray, width: int) -> np.ndarray:
    if width <= 0:
        raise ValueError("ROLLING_WIDTH")
    x = np.asarray(x, dtype=np.float64)
    cs = np.r_[0.0, np.cumsum(x)]
    out = np.empty_like(x)
    for i in range(len(x)):
        start = max(0, i + 1 - width)
        out[i] = (cs[i + 1] - cs[start]) / (i + 1 - start)
    return out


def compute_commissioned_features(domain: str, raw: np.ndarray, passport: dict, profile: dict) -> np.ndarray:
    refs = np.asarray(validate_profile(profile, domain), dtype=np.float64)
    if profile["instance_lineage"] != passport.get("instance_lineage"):
        raise ValueError("COMMISSIONING_PROFILE_INSTANCE_MISMATCH")
    base_features = base.compute_features(domain, raw, passport, "mechanism").astype(np.float64)
    physical = base_features[:, -len(base.PHYSICAL[domain]):]
    centered = physical[:, :3] - refs[None, :]
    if domain == "eps":
        a, b = centered[:, 0], centered[:, 1]
    elif domain == "thermal":
        a, b = centered[:, 0], centered[:, 2]
    else:
        a, b = centered[:, 0], centered[:, 1]
    extra = np.column_stack((centered[:, 0], centered[:, 1], centered[:, 2],
                             _rolling_mean_causal(a, 4), _rolling_mean_causal(a, 16),
                             _rolling_mean_causal(b, 4), _rolling_mean_causal(b, 16)))
    result = np.concatenate((base_features, extra), axis=1)
    if not np.isfinite(result).all():
        raise ValueError("NONFINITE_COMMISSIONED_FEATURE")
    out = result.astype(np.float32)
    if not np.isfinite(out).all():
        raise ValueError("COMMISSIONED_FEATURE_OUTSIDE_FLOAT32")
    return out


def _registry(commissioning: Path, campaign: Path) -> dict:
    verify(commissioning)
    r = read_json(commissioning / "registry.json")
    if r.get("schema") != REGISTRY_SCHEMA or r.get("source_campaign_manifest_sha256") != sha_file(campaign / "manifest.json"):
        raise ValueError("COMMISSIONING_REGISTRY_SOURCE_CHANGED")
    if r.get("implementation_sha256") != implementation_sha():
        raise ValueError("COMMISSIONING_IMPLEMENTATION_CHANGED")
    return r


def prepare_commissioned_section(campaign, commissioning, domain, section, output):
    if section not in {"development", "calibration", "holdout"}:
        raise ValueError("SECTION_INVALID")
    campaign, commissioning, output = Path(campaign), Path(commissioning), Path(output)
    creg = _registry(commissioning, campaign)
    preg = read_json(campaign / "passport_registry.json")
    roles = {"development": {"train", "validation"}, "calibration": {"calibration"}, "holdout": {"test"}}[section]
    data = Dataset(campaign / "native" / domain / section / "dataset.json", allowed_splits=roles)
    data.verify_files()
    if data.contract != native.contract(domain):
        raise ValueError("RAW_DEVICE_CONTRACT_CHANGED")
    output.mkdir(parents=True, exist_ok=False)
    samples, records = [], []
    window = native.FAMILIES[domain]["window"]
    for s in data.samples:
        key = domain + "/" + s["lineage_id"]
        pr = preg["passports"][key]
        pfile = safe_child(campaign, pr["path"])
        if sha_file(pfile) != pr["sha256"]:
            raise ValueError("PASSPORT_FILE_CHANGED")
        passport = read_json(pfile)
        hp = creg["profiles"][key]
        hfile = safe_child(commissioning, hp["profile"])
        if sha_file(hfile) != hp["profile_sha256"]:
            raise ValueError("COMMISSIONING_PROFILE_CHANGED")
        profile = read_json(hfile)
        raw = sequence_from_csv(safe_child(data.root, s["path"]), data.contract, window)
        y = compute_commissioned_features(domain, raw, passport, profile)
        relative = "observations/" + s["id"] + ".csv"
        path = output / relative
        path.parent.mkdir(exist_ok=True)
        with path.open("x", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["time_s", *[c["name"] for c in commissioned_contract(domain)["channels"]]])
            for i, row in enumerate(y):
                w.writerow([i * data.contract["sample_s"], *map(float, row)])
        samples.append({**s, "path": relative, "sha256": sha_file(path)})
        records.append({"id": s["id"], "passport_sha256": pr["sha256"], "commissioning_profile_sha256": hp["profile_sha256"],
                        "raw_sha256": s["sha256"]})
    manifest = {"schema": "phm.sequence-dataset.v1",
        "dataset_id": digest([data.hash, creg["source_campaign_manifest_sha256"], TRANSFORM, implementation_sha()]),
        "contract": commissioned_contract(domain), "samples": samples,
        "provenance": {"raw_dataset_hash": data.hash, "commissioning_registry_sha256": sha_file(commissioning / "manifest.json"),
            "transform": TRANSFORM, "transform_sha256": implementation_sha(), "source_role": "trusted_preparation; no learning",
            "commissioning_role": "separate_historical_nominal_observation", "selected_on_test": False}}
    write_json(output / "dataset.json", manifest)
    write_json(output / "device_binding.json", commissioned_binding(domain))
    write_json(output / "source_records.json", records)
    Dataset(output / "dataset.json", allowed_splits=roles).verify_files()
    seal(output, [str(p.relative_to(output)) for p in output.rglob("*") if p.is_file()])
    return manifest


def predict_commissioned_selected(run, selection, csv_path, passport_path, profile_path, domain):
    from .route_selection import _selection, selected_scores
    run, dep, _, rec = _selection(run, selection)
    if dep.get("raw_feature_transform") != TRANSFORM or dep.get("raw_feature_transform_sha256") != implementation_sha():
        raise ValueError("COMMISSIONING_DEPLOYMENT_TRANSFORM_CHANGED")
    if dep["contract"] != commissioned_contract(domain):
        raise ValueError("COMMISSIONING_DEPLOYMENT_CONTRACT")
    raw = sequence_from_csv(Path(csv_path), native.contract(domain), dep["shape"]["window"])
    features = compute_commissioned_features(domain, raw, read_json(passport_path), read_json(profile_path))
    x = transform(features[None], read_json(run / "preprocessing.json"))
    with deterministic_cpu():
        scores = selected_scores(run, selection, x)[0]
    j = int(scores.argmax()); accepted = bool(scores[j] >= dep["threshold"])
    return {"route": rec["chosen_route"], "status": "PREDICTED" if accepted else "ABSTAIN_LOW_SCORE",
        "prediction": dep["contract"]["classes"][j] if accepted else None,
        "closed_set_argmax": dep["contract"]["classes"][j], "scores": scores.tolist(),
        "commissioning_profile_required": True, "probability_calibrated": False, "claim_status": "RESEARCH_ONLY"}


def predict_commissioned_risk(run, selection, risk, csv_path, passport_path, profile_path, domain):
    from .risk_calibration import _risk
    from .route_selection import selected_scores
    run, dep, _, rec, rr = _risk(run, selection, risk)
    if dep.get("raw_feature_transform") != TRANSFORM or dep.get("raw_feature_transform_sha256") != implementation_sha():
        raise ValueError("COMMISSIONING_DEPLOYMENT_TRANSFORM_CHANGED")
    if dep["contract"] != commissioned_contract(domain):
        raise ValueError("COMMISSIONING_DEPLOYMENT_CONTRACT")
    raw = sequence_from_csv(Path(csv_path), native.contract(domain), dep["shape"]["window"])
    features = compute_commissioned_features(domain, raw, read_json(passport_path), read_json(profile_path))
    x = transform(features[None], read_json(run / "preprocessing.json"))
    with deterministic_cpu():
        scores = selected_scores(run, selection, x)[0]
    j = int(scores.argmax())
    accepted = bool(rr["status"] == "QUALIFIED_THRESHOLD" and scores[j] >= rr["threshold"])
    return {"route": rec["chosen_route"], "risk_status": rr["status"],
        "status": "PREDICTED" if accepted else "ABSTAIN_RISK_THRESHOLD",
        "prediction": dep["contract"]["classes"][j] if accepted else None,
        "closed_set_argmax": dep["contract"]["classes"][j], "scores": scores.tolist(), "threshold": rr["threshold"],
        "commissioning_profile_required": True, "probability_calibrated": False, "risk_certified": False,
        "claim_status": "RESEARCH_ONLY"}
