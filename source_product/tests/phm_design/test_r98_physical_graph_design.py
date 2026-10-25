"""R98 physics-gated envelope / tachometer order / recording-fusion design.

All test signals, tachometer traces, equipment and records are intentionally
SYNTHETIC fixtures. They prove mathematical routing and refusal guards, NOT
fault classification on independent real industrial bearings.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import json
import math

import numpy as np
import pytest

from phm_design.industrial_contracts.physical_graph_design import (
    PhysicalGraphRejected, propose_physical_graph_study,
    generate_graph_source_features, evaluate_physical_graph_study,
)
from phm_design.industrial_contracts.design_engine import DesignRejected
from phm_design.industrial_contracts.contracts import ContractRejected
from test_r94_torch_like_bridge import SpyTorchLike
from test_r95_diagnostic_design_engine import RealSklearnProviderContract

EX = Path(__file__).resolve().parents[2] / "examples/industrial_contracts"


def _cards(*, speed):
    def read(name):
        return json.loads((EX/("bearing_"+name+".json")).read_text("utf-8"))
    equipment, train, trial = (read("equipment"), read("train_only"), read("trial"))
    if speed:
        equipment["channels"].append({
            "name": "shaft_rpm", "modality": "speed", "unit": "rpm",
            "sampling_hz": None, "available_at_decision": True,
        })
        trial["input_channels"].append("shaft_rpm")
    return equipment, train, trial


def _signals(train, *, speed=True, windows=2):
    fs = 48000.
    t = np.arange(windows*4096)/fs
    result = []
    for i, meta in enumerate(train["records"]):
        # Synthetic 4000 Hz resonance carrier with 200 Hz AM sidebands;
        # constant 3000-rpm shaft with visible order-2=100Hz, order-5=250Hz.
        carrier = (1+.55*np.sin(2*np.pi*200*t))*np.sin(2*np.pi*4000*t)
        order_part = .26*np.sin(2*np.pi*100*t)+.13*np.sin(2*np.pi*250*t)
        x = (.4+.3*i)*carrier + order_part + .01*np.sin(2*np.pi*50*t)
        obs = {
            "record_id": meta["record_id"],
            "sample_rate_hz": 48000.,
            "samples": x.tolist(),
            "scalars": {"motor_current": 1. + i},
            "scalar_alignment": "same_recording_interval_self_declared",
        }
        if speed:
            obs.update({
                "rpm_per_window": [3000.0]*windows,
                "rpm_provenance": "tachometer_per_window_self_declared",
            })
        result.append(obs)
    return result


def test_graph_designer_selects_four_eligible_dags_only_with_speed_sensor():
    eq, train, trial = _cards(speed=True)
    plan = propose_physical_graph_study(eq, train, trial)
    assert plan == propose_physical_graph_study(eq, train, trial)
    assert plan["schema"] == "phm.autogen.physical-graph-design.v1"
    assert len(plan["study_sha256"]) == 64
    assert plan["order_reference"] == "shaft_rpm"
    assert plan["total_pipeline_candidates"] == 8
    assert plan["total_source_fit_calls_planned"] == 32
    assert plan["screening_refusal"] is None
    assert [x["recipe"] for x in plan["graphs"]] == [
        "time_baseline", "envelope_demod", "rotational_order",
        "recording_level_fusion"]
    assert all(x["candidate_slots"] == 2 for x in plan["graphs"])
    assert all(len(x["pipeline_plan"]["folds"]) == 4 for x in plan["graphs"])
    assert not plan["physical_clock_tachometer_and_units_independently_verified"]
    assert not plan["recording_level_fusion_is_sample_synchronized"]
    assert not plan["external_validation_or_final_read"]
    assert not plan["independent_diagnostic_generalization_proven"]
    assert "motor_current" not in plan["graphs"][2]["derived_trial"]["input_channels"]
    assert "motor_current" in plan["graphs"][3]["derived_trial"]["input_channels"]
    assert all("shaft_rpm" not in g["derived_trial"]["input_channels"]
               for g in plan["graphs"])


def test_without_tachometer_order_and_order_fusion_never_proposed():
    eq, train, trial = _cards(speed=False)
    plan = propose_physical_graph_study(eq, train, trial)
    assert plan["order_reference"] is None
    assert [x["recipe"] for x in plan["graphs"]] == [
        "time_baseline", "envelope_demod"]
    assert plan["total_pipeline_candidates"] == 8
    features, receipts = generate_graph_source_features(
        plan, eq, train, trial, _signals(train, speed=False))
    assert len(receipts) == len(train["records"])
    assert len(features["time_baseline"]) == len(train["records"])
    assert "rotational_order" not in features


def test_envelope_peak_hz_and_order_bands_have_physical_hz_coordinates():
    eq, train, trial = _cards(speed=True)
    plan = propose_physical_graph_study(eq, train, trial)
    features, receipts = generate_graph_source_features(
        plan, eq, train, trial, _signals(train))
    assert len(receipts) == 4
    assert all(x["technical_windows_aggregated"] == 2 for x in receipts)
    assert all(x["independent_recording_windows_created"] is False
               for x in receipts)
    assert all(x["rpm_from_self_declared_tachometer"] for x in receipts)
    assert all(len(x["rpm_numeric_f64le_sha256"]) == 64 for x in receipts)
    for rec in features["envelope_demod"]:
        d = rec["observations"]
        assert 180 <= d["vibration__envelope_peak_hz"] <= 220
        assert 0 <= d["vibration__envelope_40_700_fraction"] <= 1
        assert set(d) == {
            "vibration__rms", "vibration__crest",
            "vibration__envelope_40_700_fraction",
            "vibration__envelope_peak_hz",
        }
    for rec in features["rotational_order"]:
        d = rec["observations"]
        assert 0 <= d["vibration__order_1_3_fraction"] <= 1
        assert 0 <= d["vibration__order_3_7_fraction"] <= 1
    assert len(features["recording_level_fusion"][0]["observations"]) == 7
    assert "motor_current" in features["recording_level_fusion"][0]["observations"]


def test_actual_provider_executes_four_graphs_under_same_global_fit_budget():
    eq, train, trial = _cards(speed=True)
    plan = propose_physical_graph_study(eq, train, trial)
    spy = SpyTorchLike()
    result = evaluate_physical_graph_study(
        plan, eq, train, trial, _signals(train), provider=spy)
    assert result["status"] == "SOURCE_TRAIN_INTERNAL_PHYSICAL_GRAPH_COMPARISON"
    assert len(result["evaluated_graphs"]) == 4
    assert len(spy.created) == result["source_fit_calls_actual"] == 32
    assert result["source_fit_calls_actual"] == plan["total_source_fit_calls_planned"]
    assert result["same_physical_source_folds_per_graph"]
    assert result["physical_preconditions_applied_before_fit"]
    assert not result["independent_diagnostic_generalization_proven"]
    assert not result["production_approved"]
    assert not result["cpu_gpu_compute_cost_parity_proven"]
    for graph in result["evaluated_graphs"]:
        r = graph["source_only_pipeline_comparison"]
        assert r["actual_fit_calls"] == 8
        assert len(r["candidates"]) == 2
        assert r["preprocessor_fold_local"]
        assert graph["independent_validation_qualified"] is False


def test_real_sklearn_training_after_envelope_and_order_feature_construction():
    pytest.importorskip("sklearn")
    eq, train, trial = _cards(speed=True)
    trial["candidate_families"] = ["linear", "tree"]
    plan = propose_physical_graph_study(eq, train, trial)
    native = RealSklearnProviderContract()
    report = evaluate_physical_graph_study(
        plan, eq, train, trial, _signals(train), provider=native)
    assert report["source_fit_calls_actual"] == 32
    assert {n for n, kwargs in native.used_real_algorithms} == {
        "LogisticRegression", "DecisionTreeClassifier"}
    assert not report["independent_diagnostic_generalization_proven"]


@pytest.mark.parametrize("fault,reason", [
    ("missing_rpm", "R98_INVALID_OR_MISSING_PER_WINDOW_RPM"),
    ("wrong_rpm_count", "R98_INVALID_OR_MISSING_PER_WINDOW_RPM"),
    ("ramp_rpm", "R98_VARIABLE_SPEED_REQUIRES_ORDER_TRACKING_NOT_IMPLEMENTED"),
    ("bad_rpm_unit", "R98_REGISTERED_TACHOMETER_UNIT_MUST_BE_RPM"),
    ("fake_rpm_source", "R98_TACHOMETER_PROVENANCE_REQUIRED"),
    ("inconsistent_clock", "R97_RAW_SAMPLING_RATE_MISMATCH"),
    ("nan_signal", "R97_NONFINITE_OR_INVALID_RAW_SIGNAL"),
    ("extra_label", "R98_PROTECTED_OR_MISSING_RAW_RECORD_FIELDS"),
    ("wrong_scalar", "R97_INVALID_OR_EXTRA_SCALAR_OBSERVATION"),
    ("bad_alignment", "R98_SCALAR_FUSION_INTERVAL_UNSUPPORTED"),
    ("duplicate_record", "R98_DUPLICATE_OR_UNKNOWN_RECORD_ID"),
])
def test_bad_physics_and_protected_inputs_fail_before_any_provider_fit(
        fault, reason):
    eq, train, trial = _cards(speed=True)
    plan = propose_physical_graph_study(eq, train, trial)
    waves = _signals(train)
    if fault == "missing_rpm":
        waves[0]["rpm_per_window"] = []
    elif fault == "wrong_rpm_count":
        waves[0]["rpm_per_window"] = [3000.]
    elif fault == "ramp_rpm":
        waves[0]["rpm_per_window"] = [3000., 3200.]
    elif fault == "bad_rpm_unit":
        eq["channels"][-1]["unit"] = "Hz"
    elif fault == "fake_rpm_source":
        waves[0]["rpm_provenance"] = "vibration_guessed"
    elif fault == "inconsistent_clock":
        waves[0]["sample_rate_hz"] = 12000.
    elif fault == "nan_signal":
        waves[0]["samples"][0] = float("nan")
    elif fault == "extra_label":
        waves[0]["fault_label"] = "FAULT"
    elif fault == "wrong_scalar":
        waves[0]["scalars"]["future_label"] = 1
    elif fault == "bad_alignment":
        waves[0]["scalar_alignment"] = "interpolated_without_clock"
    elif fault == "duplicate_record":
        waves[-1]["record_id"] = waves[0]["record_id"]
    spy = SpyTorchLike()
    with pytest.raises((PhysicalGraphRejected, ContractRejected), match=reason):
        evaluate_physical_graph_study(plan, eq, train, trial,
                                      waves, provider=spy)
    assert spy.created == []


def test_missing_or_wrong_tachometer_registration_blocks_order_proposal():
    eq, train, trial = _cards(speed=True)
    eq["channels"][-1]["unit"] = "Hz"
    with pytest.raises(PhysicalGraphRejected,
                       match="R98_REGISTERED_TACHOMETER_UNIT_MUST_BE_RPM"):
        propose_physical_graph_study(eq, train, trial)
    eq, train, trial = _cards(speed=True)
    eq["channels"][-1]["available_at_decision"] = False
    with pytest.raises(ValueError, match="R93_PROTECTED_OR_UNAVAILABLE_TRIAL_INPUT"):
        propose_physical_graph_study(eq, train, trial)


def test_all_graphs_must_fit_candidate_family_budget_before_any_raw_read():
    eq, train, trial = _cards(speed=True)
    trial["candidate_families"] = ["linear", "tree"]
    trial["budget"]["max_trials"] = 7  # 4 recipes × 2 families needs 8
    with pytest.raises(PhysicalGraphRejected,
                       match="R98_TOTAL_TRIAL_BUDGET_CANNOT_COMPARE_REGISTERED_GRAPHS"):
        propose_physical_graph_study(eq, train, trial)


def test_mutated_frozen_graph_and_protected_final_roles_never_execute():
    eq, train, trial = _cards(speed=True)
    plan = propose_physical_graph_study(eq, train, trial)
    corrupt = deepcopy(plan)
    corrupt["graphs"][1]["stages"][1] = "eval(fake)"
    with pytest.raises(PhysicalGraphRejected,
                       match="R98_FROZEN_DIAGNOSTIC_GRAPH_CHANGED"):
        generate_graph_source_features(
            corrupt, eq, train, trial, _signals(train))
    complete = json.loads((EX/"bearing_trusted_audit.json").read_text())
    with pytest.raises(ValueError, match="R93_PROTECTED_ROLE_IN_DESIGN_WORKER"):
        propose_physical_graph_study(eq, complete, trial)


def test_no_independent_bearings_results_in_abstention_not_fabricated_scores():
    eq, train, trial = _cards(speed=True)
    train["records"] = train["records"][:3]
    plan = propose_physical_graph_study(eq, train, trial)
    assert plan["screening_refusal"] == "R98_SOURCE_PHYSICAL_GROUP_EVIDENCE_INSUFFICIENT"
    spy = SpyTorchLike()
    result = evaluate_physical_graph_study(
        plan, eq, train, trial, _signals(train), provider=spy)
    assert result["status"] == "ABSTAIN_SOURCE_EVIDENCE_INSUFFICIENT"
    assert result["source_fit_calls_actual"] == 0
    assert not result["evaluated_graphs"]
    assert not spy.created


def test_no_tachometer_rejects_unregistered_order_fields_in_raw_records():
    eq, train, trial = _cards(speed=False)
    plan = propose_physical_graph_study(eq, train, trial)
    signals = _signals(train, speed=False)
    signals[0]["rpm_per_window"] = [3000., 3000.]
    with pytest.raises(PhysicalGraphRejected,
                       match="R98_PROTECTED_OR_MISSING_RAW_RECORD_FIELDS"):
        generate_graph_source_features(plan, eq, train, trial, signals)
