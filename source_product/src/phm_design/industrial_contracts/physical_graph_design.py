"""R98: physically gated, structured vibration diagnostic algorithm DESIGN.

This is a bounded DAG search over fixed, reviewed transformations:
original recording -> signal features -> recording-level optional fusion
-> R96 fold-local numerical transform -> existing R94 provider estimator.
No new detector, untrusted expressions, data-derived feature search, label
feature, external VALIDATION/FINAL access or production qualification.

Order features require a registered rpm channel and ONE tachometer-derived,
self-reported rpm per technical window. A single mean rpm guessed from a
waveform is NOT an order reference. No order tracking for variable speed.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
import math

from .contracts import validate_dataset, ContractRejected
from .signal_feature_design import (
    _resolve_equipment, _series_observation, WINDOW_SAMPLES,
    MAX_SOURCE_SAMPLES, MAX_TOTAL_SOURCE_FITS,
)
from .pipeline_design import propose_pipeline_study, evaluate_pipeline_study

SCHEMA = "phm.autogen.physical-graph-design.v1"
REPORT = "phm.autogen.physical-graph-source-study.v1"
RECIPES = (
    "time_baseline",
    "envelope_demod",
    "rotational_order",
    "recording_level_fusion",
)
ENVELOPE_CARRIER_HZ = (2000, 8000)
ENVELOPE_MODULATION_HZ = (40, 700)
ORDER_BANDS = ((1, 3), (3, 7))
MAX_RPM_VARIATION_FRACTION = 0.03
REGISTERED_PROVIDER = {
    ("classification", "linear"),
    ("classification", "tree"),
    ("classification", "random_forest"),
    ("anomaly_detection", "isolation_forest"),
}


class PhysicalGraphRejected(ContractRejected):
    """Physical metadata, source split or a frozen graph was not admissible."""


def _reject(reason):
    raise PhysicalGraphRejected(reason)


def _digest(obj):
    return hashlib.sha256(json.dumps(
        obj, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ensure_ascii=False).encode("utf-8")).hexdigest()


def _capabilities(equipment, trial):
    vibration, hz, others = _resolve_equipment(equipment, trial)
    speeds = [s for s in others if s["modality"] == "speed"]
    if len(speeds) > 1:
        _reject("R98_MULTIPLE_AMBIGUOUS_SPEED_REFERENCES")
    speed = speeds[0] if speeds else None
    if speed is not None and speed["unit"] != "rpm":
        _reject("R98_REGISTERED_TACHOMETER_UNIT_MUST_BE_RPM")
    companion = [s for s in others if s["modality"] != "speed"]
    if any(not s["available_at_decision"] for s in others):
        _reject("R98_UNAVAILABLE_PHYSICAL_COMPANION_CHANNEL")
    available = [RECIPES[0], RECIPES[1]]
    if speed:
        available.append(RECIPES[2])
        if companion:
            available.append(RECIPES[3])
    return vibration, hz, speed, companion, available


def _features_for_recipe(vibration, recipe):
    base = [vibration+"__rms", vibration+"__crest"]
    envelope = [
        vibration+"__envelope_40_700_fraction",
        vibration+"__envelope_peak_hz",
    ]
    order = [
        vibration+"__order_1_3_fraction",
        vibration+"__order_3_7_fraction",
    ]
    if recipe == RECIPES[0]:
        return base
    if recipe == RECIPES[1]:
        return base + envelope
    if recipe == RECIPES[2]:
        return base + order
    if recipe == RECIPES[3]:
        return base + envelope + order
    _reject("R98_UNREGISTERED_STRUCTURAL_RECIPE")


def _derived_cards(equipment, trial, vibration, recipe, companions,
                   count):
    generated = _features_for_recipe(vibration, recipe)
    names = list(generated)
    fused = recipe == RECIPES[3]
    if fused:
        # Existing registered sensor columns are valid fusion inputs, not
        # newly allocated names. Only generated columns can collide.
        names += [s["name"] for s in companions]
    equipment_out = deepcopy(equipment)
    existing = {s["name"] for s in equipment_out["channels"]}
    if existing & set(generated):
        _reject("R98_DERIVED_FEATURE_COLLIDES_WITH_REGISTERED_CHANNEL")
    # Speed is only conditioning data, not an unexamined ML feature.
    equipment_out["channels"] = [
        s for s in equipment_out["channels"]
        if s["name"] != vibration and s["modality"] != "speed"
    ]
    for name in _features_for_recipe(vibration, recipe):
        is_frequency = name.endswith("_peak_hz")
        is_rms = name.endswith("__rms")
        equipment_out["channels"].append({
            "name": name,
            "modality": "vibration",
            "unit": "Hz" if is_frequency else (
                "UNVERIFIED" if is_rms else "dimensionless"),
            "sampling_hz": None,
            "available_at_decision": True,
        })
    child = deepcopy(trial)
    child["input_channels"] = names
    child["candidate_families"] = [
        family for family in trial["candidate_families"]
        if (trial["task_type"], family) in REGISTERED_PROVIDER
    ]
    if count < len(child["candidate_families"]):
        _reject("R98_NO_FAIR_BUDGET_FOR_ALL_FAMILIES_PER_GRAPH")
    child["budget"]["max_trials"] = count
    return equipment_out, child


def propose_physical_graph_study(equipment, train_only, trial):
    """Return fixed, metadata-eligible DAG arms without reading waveforms."""
    validate_dataset(train_only, equipment, trial, train_only=True)
    vib, hz, speed, companions, eligible = _capabilities(equipment, trial)
    family_count = sum((trial["task_type"], f) in REGISTERED_PROVIDER
                       for f in trial["candidate_families"])
    if family_count == 0:
        _reject("R98_NO_IMPLEMENTED_MODEL_FAMILY")
    total_budget = trial["budget"]["max_trials"]
    if total_budget < len(eligible)*family_count:
        _reject("R98_TOTAL_TRIAL_BUDGET_CANNOT_COMPARE_REGISTERED_GRAPHS")
    budget = [total_budget // len(eligible)]*len(eligible)
    for n in range(total_budget % len(eligible)):
        budget[n] += 1
    graphs = []
    for recipe, slots in zip(eligible, budget):
        card, child = _derived_cards(
            equipment, trial, vib, recipe, companions, slots)
        nested = propose_pipeline_study(card, train_only, child)
        graphs.append({
            "graph_id": "graph_%02d" % (len(graphs)+1),
            "recipe": recipe,
            "stages": [
                "original_recording", "reviewed_physical_features",
                "recording_fusion" if recipe == RECIPES[3] else "no_fusion",
                "fold_local_preprocessing", "existing_provider_estimator",
            ],
            "derived_equipment": card, "derived_trial": child,
            "pipeline_plan": nested,
            "candidate_slots": slots,
        })
    fitting = sum(g["pipeline_plan"]["planned_source_fit_calls"]
                  for g in graphs)
    reason = None
    if fitting > MAX_TOTAL_SOURCE_FITS:
        reason = "R98_TOTAL_SOURCE_FIT_BUDGET_EXCEEDED"
    elif any(g["pipeline_plan"]["screening_refusal"] for g in graphs):
        reason = "R98_SOURCE_PHYSICAL_GROUP_EVIDENCE_INSUFFICIENT"
    body = {
        "schema": SCHEMA,
        "equipment_sha256": _digest(equipment),
        "source_manifest_sha256": _digest(train_only),
        "trial_sha256": _digest(trial),
        "vibration_channel": vib,
        "declared_sampling_rate_hz": hz,
        "order_reference": speed["name"] if speed else None,
        "order_reference_requires_per_window_tachometer": bool(speed),
        "scalar_fusion": [s["name"] for s in companions],
        "envelope_carrier_band_hz": list(ENVELOPE_CARRIER_HZ),
        "envelope_modulation_band_hz": list(ENVELOPE_MODULATION_HZ),
        "order_bands": [list(x) for x in ORDER_BANDS],
        "max_rpm_variation_fraction": MAX_RPM_VARIATION_FRACTION,
        "graphs": graphs,
        "total_pipeline_candidates": sum(
            g["pipeline_plan"]["candidate_count"] for g in graphs),
        "total_source_fit_calls_planned": fitting,
        "total_source_fit_limit": MAX_TOTAL_SOURCE_FITS,
        "screening_refusal": reason,
        "physical_clock_tachometer_and_units_independently_verified": False,
        "recording_level_fusion_is_sample_synchronized": False,
        "external_validation_or_final_read": False,
        "independent_diagnostic_generalization_proven": False,
        "evidence_scope": "SOURCE_TRAIN_INTERNAL_GRAPH_SEARCH_ONLY",
    }
    return {**body, "study_sha256": _digest(body)}


def _record_features(item, *, vib, hz, speed, companions):
    if not isinstance(item, Mapping):
        _reject("R98_INVALID_SOURCE_SIGNAL_RECORD")
    expected = {"record_id", "sample_rate_hz", "samples",
                "scalars", "scalar_alignment"}
    if speed is not None:
        expected |= {"rpm_per_window", "rpm_provenance"}
    if set(item) != expected:
        _reject("R98_PROTECTED_OR_MISSING_RAW_RECORD_FIELDS")
    companions_by_name = [s["name"] for s in companions]
    if (item["scalar_alignment"] != (
            "same_recording_interval_self_declared" if companions else "none")):
        _reject("R98_SCALAR_FUSION_INTERVAL_UNSUPPORTED")
    original = {k: item[k] for k in (
        "record_id", "sample_rate_hz", "samples", "scalars")}
    # R97 verifies exact source-record allowlist, finite original waveform,
    # registered clock and ALL scalar identities before any model gets fitted.
    values, receipt = _series_observation(
        original, hz, vib, companions_by_name)
    import numpy as np
    from scipy.signal import butter, sosfiltfilt, hilbert
    arr = np.asarray(item["samples"], dtype=np.float64)
    n = len(arr) // WINDOW_SAMPLES
    samples = arr[:n*WINDOW_SAMPLES].reshape(n, WINDOW_SAMPLES)
    # Physically bounded carrier and modulation. This is not an inferred
    # BPFO/BPFI signature, bearing geometry is unavailable.
    sos = butter(4, ENVELOPE_CARRIER_HZ, fs=hz,
                 btype="bandpass", output="sos")
    filtered = sosfiltfilt(sos, samples, axis=1)
    envelope = np.abs(hilbert(filtered, axis=1))
    mod = envelope - envelope.mean(axis=1, keepdims=True)
    freqs = np.fft.rfftfreq(WINDOW_SAMPLES, d=1./hz)
    taper = np.hanning(WINDOW_SAMPLES)
    spectrum = np.abs(np.fft.rfft(mod*taper, axis=1))**2
    background = (freqs >= 40) & (freqs < 2000)
    interest = (freqs >= 40) & (freqs <= 700)
    weights = spectrum[:, background].sum(axis=1)
    if np.any(weights <= 1e-18):
        _reject("R98_NO_MEASURABLE_ENVELOPE_MODULATION")
    local = spectrum[:, interest]
    if local.shape[1] == 0:
        _reject("R98_ENVELOPE_RESOLUTION_INADEQUATE")
    peak = freqs[interest][np.argmax(local, axis=1)]
    values[vib+"__envelope_40_700_fraction"] = float(np.median(
        local.sum(axis=1)/weights))
    values[vib+"__envelope_peak_hz"] = float(np.median(peak))
    rpm_trace_digest = None
    if speed is not None:
        if item["rpm_provenance"] != "tachometer_per_window_self_declared":
            _reject("R98_TACHOMETER_PROVENANCE_REQUIRED")
        readings = item["rpm_per_window"]
        if (not isinstance(readings, list) or len(readings) != n or
                any(type(x) not in (int, float) or
                    not math.isfinite(x) or not 120 <= x <= 12000
                    for x in readings)):
            _reject("R98_INVALID_OR_MISSING_PER_WINDOW_RPM")
        rpm = np.asarray(readings, dtype=float)
        middle = float(np.median(rpm))
        if float(np.max(np.abs(rpm-middle))/middle) > MAX_RPM_VARIATION_FRACTION:
            _reject("R98_VARIABLE_SPEED_REQUIRES_ORDER_TRACKING_NOT_IMPLEMENTED")
        # True order-domain coordinates need independent tachometer speed.
        # We do NOT estimate shaft speed using class-discriminative labels
        # nor transform time samples via unverified order tracking.
        power = np.abs(np.fft.rfft(
            (samples-samples.mean(axis=1, keepdims=True))*taper,
            axis=1))**2
        order = freqs[None, :]/(rpm[:, None]/60.)
        overall = (order >= .5) & (order < 10.)
        denominator = np.sum(power*overall, axis=1)
        if np.any(denominator <= 1e-18):
            _reject("R98_NO_MEASURABLE_ROTATIONAL_BAND")
        for lo, hi in ORDER_BANDS:
            region = (order >= lo) & (order < hi)
            if np.any(np.sum(region, axis=1) < 2):
                _reject("R98_ORDER_FREQUENCY_RESOLUTION_INADEQUATE")
            fraction = np.sum(power*region, axis=1)/denominator
            values[vib+f"__order_{lo}_{hi}_fraction"] = float(np.median(fraction))
        rpm_trace_digest = hashlib.sha256(
            np.asarray(rpm, dtype="<f8").tobytes()).hexdigest()
    if any(type(x) not in (int, float) or not math.isfinite(x)
           for x in values.values()):
        _reject("R98_NONFINITE_DERIVED_SIGNAL_FEATURE")
    receipt.update({
        "sensor_fusion_level": "original_recording_only",
        "rpm_from_self_declared_tachometer": speed is not None,
        "rpm_numeric_f64le_sha256": rpm_trace_digest,
        "raw_binary_and_sensor_clock_independently_verified": False,
        "order_tracking_for_variable_speed_implemented": False,
    })
    return values, receipt


def generate_graph_source_features(study, equipment, source_train, trial,
                                  raw_records):
    """Check EVERY TRAIN recording before even one provider fit can occur."""
    if study != propose_physical_graph_study(
            equipment, source_train, trial):
        _reject("R98_FROZEN_DIAGNOSTIC_GRAPH_CHANGED")
    if not isinstance(raw_records, (tuple, list)):
        _reject("R98_INVALID_RAW_RECORD_COLLECTION")
    rows = source_train["records"]
    ids = [r["record_id"] for r in rows]
    if len(raw_records) != len(ids):
        _reject("R98_WRONG_NUMBER_OF_SOURCE_RECORDINGS")
    vib, hz, speed, companions, _ = _capabilities(equipment, trial)
    read = {}
    for row in raw_records:
        if not isinstance(row, Mapping):
            _reject("R98_INVALID_RAW_SOURCE_RECORD")
        rid = row.get("record_id")
        if (type(rid) is not str or rid not in ids or rid in read):
            _reject("R98_DUPLICATE_OR_UNKNOWN_RECORD_ID")
        read[rid] = _record_features(
            row, vib=vib, hz=hz, speed=speed,
            companions=companions)
    all_rows = {}
    for graph in study["graphs"]:
        cols = graph["derived_trial"]["input_channels"]
        all_rows[graph["recipe"]] = [
            {"record_id": rid,
             "observations": {k: read[rid][0][k] for k in cols}}
            for rid in ids]
    return all_rows, [read[rid][1] for rid in ids]


def evaluate_physical_graph_study(study, equipment, source_train, trial,
                                  raw_records, *, provider=None):
    """Evaluate frozen candidate graphs only on R93 SOURCE TRAIN folds."""
    features, receipts = generate_graph_source_features(
        study, equipment, source_train, trial, raw_records)
    if study["screening_refusal"] is not None:
        return {
            "schema": REPORT, "study_sha256": study["study_sha256"],
            "status": "ABSTAIN_SOURCE_EVIDENCE_INSUFFICIENT",
            "reason": study["screening_refusal"],
            "evaluated_graphs": [], "source_fit_calls_actual": 0,
            "recording_receipts": receipts,
            "independent_diagnostic_generalization_proven": False,
            "production_approved": False,
        }
    results = []
    for graph in study["graphs"]:
        result = evaluate_pipeline_study(
            graph["pipeline_plan"], graph["derived_equipment"],
            source_train, graph["derived_trial"],
            features[graph["recipe"]], provider=provider)
        results.append({
            "graph_id": graph["graph_id"],
            "recipe": graph["recipe"],
            "source_only_pipeline_comparison": result,
            "independent_validation_qualified": False,
        })
    actual = sum(r["source_only_pipeline_comparison"]["actual_fit_calls"]
                 for r in results)
    if actual != study["total_source_fit_calls_planned"]:
        _reject("R98_MODEL_FIT_RECEIPT_COUNT_MISMATCH")
    return {
        "schema": REPORT, "study_sha256": study["study_sha256"],
        "status": "SOURCE_TRAIN_INTERNAL_PHYSICAL_GRAPH_COMPARISON",
        "evaluated_graphs": results,
        "source_fit_calls_planned": study["total_source_fit_calls_planned"],
        "source_fit_calls_actual": actual,
        "recording_receipts": receipts,
        "same_physical_source_folds_per_graph": True,
        "physical_preconditions_applied_before_fit": True,
        "raw_mat_bytes_authenticated": False,
        "cpu_gpu_compute_cost_parity_proven": False,
        "independent_diagnostic_generalization_proven": False,
        "production_approved": False,
    }
