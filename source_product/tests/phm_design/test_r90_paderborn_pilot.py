"""R90 real numeric pipeline on explicitly fabricated Paderborn-shaped MAT files.

No downloaded Paderborn files, real bearing measurements or independent
generalization evidence are used in these tests.
"""
from __future__ import annotations

import copy
import hashlib
import json
import shutil

import numpy as np
import pytest
from scipy.io import savemat

from phm_design.industrial.r86_neuro_fuzzy import FuzzyRulePolicy
from phm_design.industrial.r90_paderborn_pilot import (
    TRAIN, VALIDATION, RECORDINGS, _check_roster, protocol_sha256,
    read_vibration_mat, rms_observations, read_pinned_recordings,
    freeze_paderborn_source, evaluate_paderborn_validation)
from phm_design.industrial.taskspec import TaskSpecRejected
from test_r81a_unified_design_engine import EXAMPLES


@pytest.fixture(scope="module")
def paderborn_fixture(tmp_path_factory):
    root = tmp_path_factory.mktemp("r90_paderborn_synthetic")
    rng = np.random.default_rng(90)
    for cond, specimen, label in TRAIN + VALIDATION:
        directory = root / specimen
        directory.mkdir(exist_ok=True)
        amplitude = (2.5 if label == "FAULT" else .3) + .03*len(specimen)
        for index in RECORDINGS:
            waveform = (rng.normal(0, amplitude, 256000)
                        + amplitude*.03*np.sin(np.arange(256000)/23))
            # This is only a named, nested MATLAB structure fixture.
            # Authentic Paderborn v5/v7.3 structure compatibility has NOT
            # been demonstrated with an original laboratory file.
            savemat(directory / f"{cond}_{specimen}_{index}.mat",
                    {"experiment": {
                        "Y": {"Name": "vibration_1", "Data": waveform},
                        "motor": {"Name": "phase_current_1",
                                  "Data": np.ones(16)}}})
    task_path = EXAMPLES / "paderborn_r90_binary_pilot.json"
    spec = json.loads(task_path.read_text(encoding="utf-8"))
    frozen = root / "r90-frozen"
    result = freeze_paderborn_source(spec, root, frozen,
        policy=FuzzyRulePolicy(max_rules=2, premise_steps=2))
    return root, spec, frozen, result


def test_source_trains_only_from_prospectively_locked_three_conditions(
        paderborn_fixture):
    root, spec, frozen, out = paderborn_fixture
    train, digests = read_pinned_recordings(root, role="TRAIN")
    assert _check_roster(train, role="TRAIN") is None
    assert len(train) == 12 == len(digests)
    assert {r["metadata"]["operating_condition"] for r in train} == {
        "N15_M07_F10", "N09_M07_F10", "N15_M01_F10"}
    assert len({r["metadata"]["bearing_instance_id"] for r in train}) == 6
    assert len({r["metadata"]["label"] for r in train}) == 2
    assert all(len(r["measurements"]) == 256 for r in train)
    assert all(x["raw_sha256"] for x in digests)
    assert out["roster_sha256"] == protocol_sha256()
    assert out["validation_accessed"] is False
    assert out["independent_qualification"] is False
    assert len(out["frozen"]["source_oof_report"]["arms"]) == 5
    assert (frozen / "manifest.json").is_file()


def test_frozen_weights_readonly_validate_one_heldout_condition(
        paderborn_fixture):
    root, _, frozen, out = paderborn_fixture
    receipt = evaluate_paderborn_validation(
        frozen, root, expected_manifest_sha256=out["frozen"]["manifest_sha256"])
    assert receipt["validation_role"] == "VALIDATION_ONLY"
    assert receipt["single_heldout_condition"] == "N15_M07_F04"
    assert receipt["healthy_independent_bearings_in_validation"] == 3
    assert receipt["fault_independent_bearings_in_validation"] == 3
    assert receipt["frozen_all_five_models_verified"] is True
    assert receipt["validation_fit_or_selection_performed"] is False
    assert receipt["source_sensor_units_verified"] is False
    assert receipt["source_sampling_clock_verified"] is False
    assert receipt["blind_test_or_final_used"] is False
    assert receipt["neuro_fuzzy_improvement_proven"] is False
    assert receipt["multi_condition_external_generalization_proven"] is False
    assert receipt["eligible_for_stable"] is False
    assert len(receipt["results"]) == 5
    assert len(receipt["raw_mat_receipts"]) == 12
    for metrics in receipt["results"].values():
        assert all(0 <= x <= 1 for x in metrics.values())
    for interval in receipt["paired_physical_group_intervals"].values():
        assert interval["physical_groups"] == 6
        assert interval["replicates"] == 499
        assert interval["confirmatory_significance_claim"] is False
    assert receipt == evaluate_paderborn_validation(
        frozen, root, expected_manifest_sha256=out["frozen"]["manifest_sha256"])


def test_wrong_manifest_hash_or_modified_npz_rejected_before_scoring(
        paderborn_fixture, tmp_path):
    root, _, frozen, result = paderborn_fixture
    with pytest.raises(TaskSpecRejected, match="R90_MANIFEST_HASH_MISMATCH"):
        evaluate_paderborn_validation(frozen, root,
            expected_manifest_sha256="0"*64)
    dest = tmp_path / "model-tampered"
    shutil.copytree(frozen, dest)
    weights = dest / "adaptive_fuzzy_learned_premise.npz"
    weights.write_bytes(weights.read_bytes() + b"modification")
    with pytest.raises(TaskSpecRejected, match="R90_WEIGHT_HASH_MISMATCH"):
        evaluate_paderborn_validation(dest, root,
            expected_manifest_sha256=result["frozen"]["manifest_sha256"])


def test_train_val_roles_cannot_be_reassigned_or_new_fault_group_injected(
        paderborn_fixture):
    root, spec, _, _ = paderborn_fixture
    train, _ = read_pinned_recordings(root, role="TRAIN")
    modified = copy.deepcopy(train)
    modified[0]["metadata"]["role"] = "VALIDATION"
    with pytest.raises(TaskSpecRejected, match="R90_LOCKED_PHYSICAL_BEARING_ROSTER_MISMATCH"):
        _check_roster(modified, role="TRAIN")
    modified = copy.deepcopy(train)
    modified[0]["metadata"]["bearing_instance_id"] = "fake_new_bearing"
    with pytest.raises(TaskSpecRejected, match="R90_LOCKED_PHYSICAL_BEARING_ROSTER_MISMATCH"):
        _check_roster(modified, role="TRAIN")
    with pytest.raises(TaskSpecRejected, match="R90_PROTECTED_ROLE"):
        read_pinned_recordings(root, role="FINAL")


def test_explicit_named_channel_required_and_no_silent_array_index_fallback(
        paderborn_fixture, tmp_path):
    root, _, _, _ = paderborn_fixture
    p = root / TRAIN[0][1] / f"{TRAIN[0][0]}_{TRAIN[0][1]}_1.mat"
    x = read_vibration_mat(p)
    assert len(x) == 256000
    assert len(rms_observations(x)) == 256
    assert set(rms_observations(x)[0]) == {
        "nominal_elapsed_s", "vibration_rms", "vibration_peak_abs"}
    bad = tmp_path / "not-a-real-vibration.mat"
    savemat(bad, {"signal": np.ones(256000)})
    with pytest.raises(TaskSpecRejected, match="R90_ONE_EXPLICIT_NAMED_VIBRATION_CHANNEL_REQUIRED"):
        read_vibration_mat(bad)
    bad2 = tmp_path / "short.mat"
    savemat(bad2, {"experiment": {"Name": "vibration_1",
                                 "Data": np.ones(1000)}})
    with pytest.raises(TaskSpecRejected, match="R90_INVALID_SIGNAL_SAMPLES"):
        read_vibration_mat(bad2)


def test_premise_data_cannot_be_null_nan_or_outside_single_condition(
        paderborn_fixture, tmp_path):
    root, _, _, _ = paderborn_fixture
    with pytest.raises(TaskSpecRejected, match="R90_INVALID_RMS_INPUT"):
        rms_observations(np.full(256000, np.nan))
    with pytest.raises(TaskSpecRejected, match="R90_INVALID_RMS_INPUT"):
        rms_observations(np.ones(2000))
    with pytest.raises(TaskSpecRejected, match="R90_INVALID_SIGNAL_SAMPLES"):
        short = tmp_path / "invalid-raw.mat"
        savemat(short, {"Name": "vibration_1", "Data": np.ones(100)})
        read_vibration_mat(short)
