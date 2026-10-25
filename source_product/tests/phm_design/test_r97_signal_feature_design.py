"""R97 physical-signal feature and real diagnosis-design pipeline checks.

All produced test waveforms are SYNTHETIC SINES, not measured equipment.
There is exactly ONE feature vector per original recording, irrespective
of number of technical windows inside a recording.
"""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import json
import math

import numpy as np
import pytest

from phm_design.industrial_contracts.signal_feature_design import (
    RECIPES, SignalDesignRejected, propose_signal_diagnostic_study,
    generate_source_recipe_features, evaluate_signal_diagnostic_study,
)
from phm_design.industrial_contracts.design_engine import DesignRejected
from test_r94_torch_like_bridge import SpyTorchLike
from test_r95_diagnostic_design_engine import RealSklearnProviderContract

EX = Path(__file__).resolve().parents[2] / "examples" / "industrial_contracts"


def _read(name):
    return json.loads((EX/name).read_text(encoding="utf-8"))


def _data():
    return (_read("bearing_equipment.json"),
            _read("bearing_train_only.json"),
            _read("bearing_trial.json"))


def _signals(train, *, windows=3, low=140., high=2600.):
    """Four different synthetic source physical IDs, never real bearings."""
    hz = 48000.
    t = np.arange(4096*windows)/hz
    records = []
    for i, record in enumerate(train["records"]):
        dominant = low if record["label"] == "HEALTHY" else high
        amplitude = .2 if record["label"] == "HEALTHY" else 1.8
        vibration = amplitude*np.sin(2*np.pi*dominant*t)
        # Non-identical signal bytes across records even when class matches.
        vibration = vibration + .015*(i+1)*np.sin(2*np.pi*80*t)
        records.append({
            "record_id": record["record_id"],
            "sample_rate_hz": hz,
            "samples": vibration.tolist(),
            "scalars": {"motor_current": float(1+i)},
        })
    return records


def test_two_physical_feature_recipes_stay_under_one_total_budget():
    eq, train, trial = _data()
    one = propose_signal_diagnostic_study(eq, train, trial)
    two = propose_signal_diagnostic_study(eq, train, trial)
    assert one == two
    assert one["schema"] == "phm.autogen.physical-feature-design.v1"
    assert len(one["study_sha256"]) == 64
    assert one["original_sensor"] == "vibration"
    assert one["declared_sampling_rate_hz"] == 48000
    assert len(one["feature_recipes"]) == 2
    assert [x["feature_recipe"] for x in one["feature_recipes"]] == list(RECIPES)
    assert one["total_candidate_pipelines"] == trial["budget"]["max_trials"] == 8
    assert one["total_source_fit_calls_planned"] == 32
    assert one["screening_refusal"] is None
    assert not one["measured_sampling_clock_independently_authenticated"]
    assert not one["original_mat_file_bytes_authenticated"]
    for arm in one["feature_recipes"]:
        assert arm["max_candidates_allocated"] == 4
        assert arm["pipeline_plan"]["candidate_count"] == 4
        assert len(arm["pipeline_plan"]["folds"]) == 4
    names = one["feature_recipes"][1]["derived_trial"]["input_channels"]
    assert names == [
        "vibration__rms", "vibration__crest", "vibration__band_50_500",
        "vibration__band_500_8000", "motor_current"]


def test_sine_bandpower_is_calculated_using_48khz_physical_frequency_axis():
    eq, train, trial = _data()
    frozen = propose_signal_diagnostic_study(eq, train, trial)
    rows, originals = generate_source_recipe_features(
        frozen, eq, train, trial, _signals(train))
    assert len(originals) == len(train["records"]) == 4
    assert len(rows[RECIPES[0]]) == len(rows[RECIPES[1]]) == 4
    assert all(v["technical_windows_aggregated"] == 3 for v in originals)
    assert all(v["independent_recording_windows_created"] is False
               for v in originals)
    for i, row in enumerate(rows[RECIPES[1]]):
        x = row["observations"]
        assert len(x) == 5
        assert x["vibration__rms"] > 0
        assert x["vibration__crest"] > 1
        assert 0 <= x["vibration__band_50_500"] <= 1
        assert 0 <= x["vibration__band_500_8000"] <= 1
        if train["records"][i]["label"] == "HEALTHY":
            assert x["vibration__band_50_500"] > .95
            assert x["vibration__band_500_8000"] < .05
        else:
            assert x["vibration__band_500_8000"] > .95
            assert x["vibration__band_50_500"] < .05
    assert "vibration__band_50_500" not in rows[RECIPES[0]][0]["observations"]


def test_two_recipe_evaluations_execute_real_provider_shaped_models():
    eq, train, trial = _data()
    study = propose_signal_diagnostic_study(eq, train, trial)
    fake = SpyTorchLike()
    report = evaluate_signal_diagnostic_study(
        study, eq, train, trial, _signals(train), provider=fake)
    assert report["status"] == "SOURCE_TRAIN_ONLY_SIGNAL_PIPELINE_COMPARISON"
    assert report["source_fit_calls_actual"] == 32
    assert report["source_fit_calls_actual"] == report["source_fit_calls_planned"]
    assert len(fake.created) == 32
    assert len(report["recipe_reports"]) == 2
    assert len(report["raw_input_receipts"]) == 4
    assert report["physical_group_folds_unchanged_across_recipes"]
    assert report["feature_representations_compared_under_shared_candidate_trial_budget"]
    assert report["compute_budget_parity_proven"] is False
    assert report["independent_generalization_proven"] is False
    assert report["industrial_production_approved"] is False
    for rec in report["recipe_reports"]:
        result = rec["source_exploratory_report"]
        assert len(result["candidates"]) == 4
        assert result["actual_fit_calls"] == 16
        assert result["preprocessor_fold_local"] is True
        assert rec["qualified"] is False
    for raw in report["raw_input_receipts"]:
        assert len(raw["numeric_signal_f64le_sha256"]) == 64
        assert raw["numeric_hash_not_raw_file_sha256"] is True


def test_real_sklearn_fits_end_to_end_on_both_waveform_feature_recipes():
    pytest.importorskip("sklearn")
    eq, train, trial = _data()
    trial["candidate_families"] = ["linear", "tree"]
    trial["budget"]["max_trials"] = 4  # 2 algorithms each recipe
    study = propose_signal_diagnostic_study(eq, train, trial)
    provider = RealSklearnProviderContract()
    report = evaluate_signal_diagnostic_study(
        study, eq, train, trial, _signals(train), provider=provider)
    assert report["source_fit_calls_actual"] == 16
    assert len(provider.used_real_algorithms) == 16
    assert {n for n, _ in provider.used_real_algorithms} == {
        "LogisticRegression", "DecisionTreeClassifier"}
    assert all(len(r["source_exploratory_report"]["candidates"]) == 2
               for r in report["recipe_reports"])
    assert not report["independent_generalization_proven"]


@pytest.mark.parametrize("fault", ["rate", "too_short", "nan", "extra",
                                    "duplicate", "bad_record", "bool"])
def test_raw_vibration_input_fail_closed_before_any_model_fit(fault):
    eq, train, trial = _data()
    study = propose_signal_diagnostic_study(eq, train, trial)
    signals = _signals(train)
    if fault == "rate":
        signals[0]["sample_rate_hz"] = 12000.
    elif fault == "too_short":
        signals[0]["samples"] = signals[0]["samples"][:10]
    elif fault == "nan":
        signals[0]["samples"][0] = float("nan")
    elif fault == "extra":
        signals[0]["scalars"]["fault_label"] = 1
    elif fault == "duplicate":
        signals[-1]["record_id"] = signals[0]["record_id"]
    elif fault == "bad_record":
        signals[0]["role"] = "VALIDATION"
    elif fault == "bool":
        signals[0]["samples"][0] = True
    fake = SpyTorchLike()
    with pytest.raises(SignalDesignRejected):
        evaluate_signal_diagnostic_study(
            study, eq, train, trial, signals, provider=fake)
    assert fake.created == []


def test_any_change_to_locked_recipe_or_split_is_rejected():
    eq, train, trial = _data()
    study = propose_signal_diagnostic_study(eq, train, trial)
    tamper = deepcopy(study)
    tamper["feature_recipes"][0]["pipeline_plan"]["pipeline_designs"][0][
        "preprocessing"] = "signed_log1p"
    with pytest.raises(SignalDesignRejected,
                       match="R97_FROZEN_FEATURE_STUDY_MUTATED"):
        generate_source_recipe_features(
            tamper, eq, train, trial, _signals(train))
    other = deepcopy(trial)
    other["budget"]["seed"] = 177
    with pytest.raises(SignalDesignRejected):
        evaluate_signal_diagnostic_study(
            study, eq, train, other, _signals(train), provider=SpyTorchLike())


def test_declared_sampling_rate_and_nyquist_gate_are_not_guessed():
    eq, train, trial = _data()
    bad = deepcopy(eq)
    bad["channels"][0]["sampling_hz"] = None
    with pytest.raises(SignalDesignRejected,
                       match="R97_PHYSICAL_SAMPLING_RATE_REQUIRED"):
        propose_signal_diagnostic_study(bad, train, trial)
    bad = deepcopy(eq)
    bad["channels"][0]["sampling_hz"] = 12000
    with pytest.raises(SignalDesignRejected,
                       match="R97_PHYSICAL_SAMPLING_RATE_REQUIRED"):
        propose_signal_diagnostic_study(bad, train, trial)


def test_unsupported_nonvibration_process_data_cannot_become_fake_fft():
    eq = json.loads((EX/"compressor_equipment.json").read_text())
    train = json.loads((EX/"compressor_train_only.json").read_text())
    trial = json.loads((EX/"compressor_trial.json").read_text())
    with pytest.raises(SignalDesignRejected,
                       match="R97_EXACTLY_ONE_REGISTERED_VIBRATION_SENSOR_REQUIRED"):
        propose_signal_diagnostic_study(eq, train, trial)


def test_insufficient_source_group_support_abstains_not_fake_accuracy():
    eq, train, trial = _data()
    train["records"] = train["records"][:3]
    proposal = propose_signal_diagnostic_study(eq, train, trial)
    assert proposal["screening_refusal"] == (
        "R97_NESTED_SOURCE_GROUP_EVIDENCE_INSUFFICIENT")
    fake = SpyTorchLike()
    receipt = evaluate_signal_diagnostic_study(
        proposal, eq, train, trial, _signals(train), provider=fake)
    assert receipt["status"] == "ABSTAIN_INSUFFICIENT_SOURCE_EVIDENCE"
    assert receipt["source_fit_calls_actual"] == 0
    assert not fake.created


def test_total_trial_budget_must_allocate_at_least_one_per_algorithm_per_recipe():
    eq, train, trial = _data()
    trial["candidate_families"] = ["linear", "tree"]
    trial["budget"]["max_trials"] = 3  # R93 permits two; R97 needs two per recipe
    with pytest.raises(SignalDesignRejected,
                       match="R97_BUDGET_CANNOT_COMPARE_ALL_DECLARED_FAMILIES"):
        propose_signal_diagnostic_study(eq, train, trial)


def test_band_energy_rejects_noninformative_constant_waveform():
    eq, train, trial = _data()
    study = propose_signal_diagnostic_study(eq, train, trial)
    raws = _signals(train)
    raws[0]["samples"] = [0.] * 4096
    with pytest.raises(SignalDesignRejected, match="R97_DEGENERATE_VIBRATION_WINDOW"):
        generate_source_recipe_features(study, eq, train, trial, raws)
