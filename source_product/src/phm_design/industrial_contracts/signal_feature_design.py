"""R97 — physical-clock-aware VIBRATION FEATURE + diagnostic pipeline DESIGN.

R95 searches models/sensors; R96 fits numerical preprocessing on each source
TRAIN fold; R97 pre-registers two measurement-derived, explainable feature
recipes and runs their R96 experiments under ONE shared trial budget.

One original recording yields ONE aggregate feature row. Disjoint windows
inside that recording are technical subsamples, NEVER independent machines.
No external VALIDATION, CALIBRATION or FINAL is accepted.

Clock and engineering units are user-declared and merely checked for internal
consistency, NOT authenticated. The raw file SHA in a DatasetCard is not
validated by this in-memory feature interface.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
import math

from .contracts import ContractRejected, validate_equipment, validate_dataset
from .pipeline_design import propose_pipeline_study, evaluate_pipeline_study

SCHEMA = "phm.autogen.physical-feature-design.v1"
REPORT = "phm.autogen.physical-feature-source-study.v1"
FEATURES = ("rms", "crest", "band_50_500", "band_500_8000")
RECIPES = ("time_rms_crest", "time_plus_frequency_bands")
WINDOW_SAMPLES = 4096
MIN_SAMPLE_HZ = 20000
MAX_TOTAL_SOURCE_FITS = 64
MAX_SOURCE_SAMPLES = 1_000_000


class SignalDesignRejected(ContractRejected):
    """A signal or design constraint is unsupported, never silently guessed."""


def _reject(reason):
    raise SignalDesignRejected(reason)


def _sha(obj):
    return hashlib.sha256(json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False).encode("utf-8")).hexdigest()


def _resolve_equipment(equipment, trial):
    validate_equipment(equipment)
    vib = [s for s in equipment["channels"]
           if s["modality"] == "vibration"
           and s["name"] in trial["input_channels"]]
    if len(vib) != 1:
        _reject("R97_EXACTLY_ONE_REGISTERED_VIBRATION_SENSOR_REQUIRED")
    sensor = vib[0]
    clock = sensor["sampling_hz"]
    if (type(clock) not in (int, float) or not math.isfinite(clock)
            or clock < MIN_SAMPLE_HZ or clock > 200_000):
        _reject("R97_PHYSICAL_SAMPLING_RATE_REQUIRED")
    if clock / 2 <= 8000:
        _reject("R97_FREQUENCY_BANDS_EXCEED_NYQUIST")
    name = sensor["name"]
    others = [s for s in equipment["channels"]
              if s["name"] != name and
              s["name"] in trial["input_channels"]]
    if any(s["modality"] not in (
            "current", "pressure", "temperature", "speed", "telemetry", "other")
           for s in others):
        _reject("R97_UNSUPPORTED_NON_VIBRATION_INPUT_MODALITY")
    return name, float(clock), others


def _derived_cards(equipment, trial, signal_name, others, recipe, count):
    if recipe not in RECIPES:
        _reject("R97_UNREGISTERED_FEATURE_RECIPE")
    keys = FEATURES[:2] if recipe == RECIPES[0] else FEATURES
    out = deepcopy(equipment)
    out["channels"] = [c for c in out["channels"]
                       if c["name"] != signal_name]
    names = []
    for key in keys:
        dname = signal_name + "__" + key
        if any(s["name"] == dname for s in out["channels"]):
            _reject("R97_DERIVED_FEATURE_NAME_COLLISION")
        names.append(dname)
        out["channels"].append({
            "name": dname, "modality": "vibration",
            "unit": "raw_source_unit_UNVERIFIED" if key == "rms"
                    else "dimensionless",
            "sampling_hz": None,
            "available_at_decision": True,
        })
    task = deepcopy(trial)
    task["input_channels"] = names + [s["name"] for s in others]
    task["candidate_families"] = [
        f for f in task["candidate_families"] if
        (task["task_type"], f) in {
            ("classification", "linear"), ("classification", "tree"),
            ("classification", "random_forest"),
            ("anomaly_detection", "isolation_forest"),
        }]
    if not task["candidate_families"]:
        _reject("R97_NO_IMPLEMENTED_ALGORITHM_PROVIDER")
    if count < len(task["candidate_families"]):
        _reject("R97_BUDGET_CANNOT_COMPARE_ALL_DECLARED_FAMILIES")
    task["budget"]["max_trials"] = count
    return out, task


def propose_signal_diagnostic_study(equipment, source_train, trial):
    """Pre-register BOTH signal feature recipes before reading sensor values."""
    # R93 source role fence, physical group checks, target restrictions.
    validate_dataset(source_train, equipment, trial, train_only=True)
    if trial["task_type"] not in ("classification", "anomaly_detection"):
        _reject("R97_UNSUPPORTED_TASK")
    signal_name, clock, others = _resolve_equipment(equipment, trial)
    total_trials = trial["budget"]["max_trials"]
    if total_trials < len(RECIPES):
        _reject("R97_INSUFFICIENT_TOTAL_RECIPE_BUDGET")
    budget = [total_trials // 2, total_trials - total_trials // 2]
    arms = []
    for recipe, allocated in zip(RECIPES, budget):
        derived_equipment, derived_trial = _derived_cards(
            equipment, trial, signal_name, others, recipe, allocated)
        nested = propose_pipeline_study(
            derived_equipment, source_train, derived_trial)
        arms.append({
            "feature_recipe": recipe,
            "derived_equipment": derived_equipment,
            "derived_trial": derived_trial,
            "pipeline_plan": nested,
            "max_candidates_allocated": allocated,
            "declared_clock_hz": clock,
            "physical_sensor": signal_name,
        })
    total_fits = sum(x["pipeline_plan"]["planned_source_fit_calls"]
                     for x in arms)
    refusal = None
    if total_fits > MAX_TOTAL_SOURCE_FITS:
        refusal = "R97_EXCEEDED_COMBINED_PIPELINE_FIT_CEILING"
    elif any(a["pipeline_plan"]["screening_refusal"] for a in arms):
        refusal = "R97_NESTED_SOURCE_GROUP_EVIDENCE_INSUFFICIENT"
    doc = {
        "schema": SCHEMA,
        "equipment_sha256": _sha(equipment),
        "train_manifest_sha256": _sha(source_train),
        "trial_sha256": _sha(trial),
        "original_sensor": signal_name,
        "declared_sampling_rate_hz": clock,
        "window_samples": WINDOW_SAMPLES,
        "feature_aggregation": "per-original-recording-median-of-disjoint-windows",
        "feature_recipes": arms,
        "total_candidate_pipelines": sum(
            a["pipeline_plan"]["candidate_count"] for a in arms),
        "total_source_fit_calls_planned": total_fits,
        "fit_calls_hard_limit": MAX_TOTAL_SOURCE_FITS,
        "screening_refusal": refusal,
        "measured_sampling_clock_independently_authenticated": False,
        "units_calibrated_against_instrument": False,
        "original_mat_file_bytes_authenticated": False,
        "protected_validation_final_accessed": False,
        "independent_diagnostic_generalization_proven": False,
        "evidence_scope": "SOURCE_ONLY_DIAGNOSTIC_PIPELINE_RESEARCH",
    }
    return {**doc, "study_sha256": _sha(doc)}


def _series_observation(signal, expected_clock, source_name, other_names):
    if (not isinstance(signal, Mapping) or set(signal) != {
            "record_id", "sample_rate_hz", "samples", "scalars"}):
        _reject("R97_SIGNAL_RECORD_SCHEMA_MISMATCH")
    clock = signal["sample_rate_hz"]
    if (type(clock) not in (int, float) or not math.isfinite(clock) or
            clock != expected_clock):
        _reject("R97_RAW_SAMPLING_RATE_MISMATCH")
    other = signal["scalars"]
    if (not isinstance(other, Mapping) or set(other) != set(other_names)
            or any(type(v) not in (int, float) or
                   not math.isfinite(v) or abs(v) > 1e12
                   for v in other.values())):
        _reject("R97_INVALID_OR_EXTRA_SCALAR_OBSERVATION")
    values = signal["samples"]
    if (not isinstance(values, (tuple, list)) or
            len(values) < WINDOW_SAMPLES or
            len(values) > MAX_SOURCE_SAMPLES):
        _reject("R97_SIGNAL_LENGTH_OUTSIDE_BOUND")
    if any(type(v) not in (int, float) or not math.isfinite(v)
           or abs(v) > 1e12 for v in values):
        _reject("R97_NONFINITE_OR_INVALID_RAW_SIGNAL")
    import numpy as np
    x = np.asarray(values, dtype=np.float64)
    windows = len(x) // WINDOW_SAMPLES
    samples = x[:windows*WINDOW_SAMPLES].reshape(windows, WINDOW_SAMPLES)
    rms = np.sqrt(np.mean(samples*samples, axis=1))
    if np.any(rms <= 1e-10):
        _reject("R97_DEGENERATE_VIBRATION_WINDOW")
    crest = np.max(np.abs(samples), axis=1) / rms
    arr = {
        source_name+"__rms": float(np.median(rms)),
        source_name+"__crest": float(np.median(crest)),
    }
    if not all(math.isfinite(v) for v in arr.values()):
        _reject("R97_INVALID_TIME_DOMAIN_STATISTIC")
    # Hann-weighted true physical frequency bins (Hz); one FFT per
    # non-overlapping technical window, aggregated to ONE recording-level row.
    h = np.hanning(WINDOW_SAMPLES)
    spectra = np.abs(np.fft.rfft(samples*h, axis=1))**2
    hz = np.fft.rfftfreq(WINDOW_SAMPLES, 1. / expected_clock)
    denominator = np.sum(spectra[:, hz >= 50], axis=1)
    if np.any(denominator <= 1e-20):
        _reject("R97_NO_RELIABLE_NON_DC_SPECTRAL_ENERGY")
    for lo, hi, key in (
        (50, 500, "band_50_500"),
        (500, 8000, "band_500_8000"),
    ):
        band = np.sum(spectra[:, (hz >= lo) & (hz < hi)], axis=1)
        arr[source_name+"__"+key] = float(np.median(band/denominator))
    if any(not math.isfinite(v) or v < 0 or
           v > (1. + 1e-10 if "band" in k else 1e12)
           for k, v in arr.items()):
        _reject("R97_INVALID_SPECTRAL_STATISTIC")
    arr.update({name: float(other[name]) for name in other_names})
    raw_numeric_hash = hashlib.sha256(
        np.asarray(x, dtype="<f8").tobytes()).hexdigest()
    return arr, {
        "record_id": signal["record_id"],
        "numeric_signal_f64le_sha256": raw_numeric_hash,
        "numeric_hash_not_raw_file_sha256": True,
        "independent_recording_windows_created": False,
        "technical_windows_aggregated": windows,
        "input_sensor_sample_count": len(x),
        "input_sample_rate_hz_self_declared": expected_clock,
    }


def generate_source_recipe_features(study, equipment, source_train, trial,
                                    signal_records):
    """Return source-only feature rows; reject missing/duplicate/protected data.

    All signal vectors are fully checked prior to provider fitting. A source
    recording (not a window) produces one record_id feature row.
    """
    if study != propose_signal_diagnostic_study(
            equipment, source_train, trial):
        _reject("R97_FROZEN_FEATURE_STUDY_MUTATED")
    if not isinstance(signal_records, (list, tuple)):
        _reject("R97_INVALID_RAW_SOURCE_COLLECTION")
    expected = {r["record_id"] for r in source_train["records"]}
    if len(signal_records) != len(expected):
        _reject("R97_MISSING_OR_EXTRA_SOURCE_RECORD")
    source_name, clock, others = _resolve_equipment(equipment, trial)
    all_observations, receipts = {}, {}
    for signal in signal_records:
        if not isinstance(signal, Mapping):
            _reject("R97_INVALID_SOURCE_RECORD")
        rid = signal.get("record_id")
        if type(rid) is not str or rid not in expected or rid in all_observations:
            _reject("R97_DUPLICATE_OR_UNDECLARED_RECORD_ID")
        values, receipt = _series_observation(
            signal, clock, source_name, [s["name"] for s in others])
        all_observations[rid] = values
        receipts[rid] = receipt
    if set(all_observations) != expected:
        _reject("R97_SOURCE_RECORD_ID_MISMATCH")
    features = {}
    for arm in study["feature_recipes"]:
        names = arm["derived_trial"]["input_channels"]
        features[arm["feature_recipe"]] = [
            {"record_id": record["record_id"],
             "observations": {
                 name: all_observations[record["record_id"]][name]
                 for name in names}}
            for record in source_train["records"]]
    return features, [receipts[r["record_id"]]
                      for r in source_train["records"]]


def evaluate_signal_diagnostic_study(study, equipment, source_train, trial,
                                     signal_records, *, provider=None):
    """Run the R96 pipeline evaluator for each predeclared signal recipe."""
    features, provenance = generate_source_recipe_features(
        study, equipment, source_train, trial, signal_records)
    if study["screening_refusal"]:
        return {
            "schema": REPORT, "study_sha256": study["study_sha256"],
            "status": "ABSTAIN_INSUFFICIENT_SOURCE_EVIDENCE",
            "reason": study["screening_refusal"],
            "recipe_reports": [], "source_fit_calls_actual": 0,
            "raw_input_receipts": provenance,
            "independent_generalization_proven": False,
            "industrial_production_approved": False,
        }
    reports = []
    for arm in study["feature_recipes"]:
        result = evaluate_pipeline_study(
            arm["pipeline_plan"], arm["derived_equipment"],
            source_train, arm["derived_trial"],
            features[arm["feature_recipe"]], provider=provider)
        reports.append({
            "recipe": arm["feature_recipe"],
            "source_exploratory_report": result,
            "qualified": False,
        })
    return {
        "schema": REPORT,
        "study_sha256": study["study_sha256"],
        "status": "SOURCE_TRAIN_ONLY_SIGNAL_PIPELINE_COMPARISON",
        "recipe_reports": reports,
        "source_fit_calls_actual": sum(
            x["source_exploratory_report"].get("actual_fit_calls", 0)
            for x in reports),
        "source_fit_calls_planned": study["total_source_fit_calls_planned"],
        "raw_input_receipts": provenance,
        "physical_group_folds_unchanged_across_recipes": True,
        "feature_representations_compared_under_shared_candidate_trial_budget": True,
        "compute_budget_parity_proven": False,
        "raw_mat_bytes_authenticated": False,
        "independent_generalization_proven": False,
        "industrial_production_approved": False,
    }
