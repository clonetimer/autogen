"""Synthetic tests of strict CWRU MAT ingestion; no real-source claim."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
from scipy.io import savemat

from phm_design.industrial.r91_cwru_reality_check import (
    REGISTRY, RealSignalRefused, audit_real_cwru_pair,
    main, _read_recording, verify_first_documented_official_bytes,
    FIRST_OBSERVED_OFFICIAL_SHA256)


@pytest.fixture
def simulated_cwru_mat(tmp_path):
    rng = np.random.default_rng(911)
    for file_id in REGISTRY:
        named = REGISTRY[file_id]["field"]
        data = rng.normal(loc=0., scale=(1. if file_id == "97" else 2.),
                          size=80000)
        fields = {named: data}
        if file_id == "97":
            fields["X098_DE_time"] = data * 100000.
        savemat(tmp_path / f"{file_id}.mat", fields)
    return tmp_path


def test_no_pseudoreplication_or_fuzzy_advantage_from_original_mat_ingestion(
        simulated_cwru_mat):
    r = audit_real_cwru_pair(simulated_cwru_mat)
    assert r["publisher_describes_experiment_as_measured"] is True
    assert r["these_local_bytes_authenticated_as_official"] is False
    assert r["manufacturer_provenance_independently_authenticated"] is False
    assert r["number_of_original_mat_files"] == 2
    assert r["number_of_independent_physical_bearings_verified"] == 0
    assert r["windows_are_independent_machines"] is False
    assert r["diagnostic_training_performed"] is False
    assert r["validation_or_test_performed"] is False
    assert r["measured_generalization_accuracy"] is None
    assert r["estimated_false_alarm_rate"] is None
    assert r["fuzzy_vs_baseline_advantage_proven"] is False
    assert r["eligible_for_stable"] is False
    assert r["two_recordings_same_reported_sample_rate"] is True
    for item in r["recordings"]:
        assert item["declared_sampling_rate_hz"] == 48000
        assert item["descriptive_window_count_not_independent_samples"] >= 10
        assert item["sha256"] == hashlib.sha256(
            (simulated_cwru_mat / f"{item['file_id']}.mat").read_bytes()
        ).hexdigest()
    assert r["recordings"][0]["other_de_time_variables_not_used"] == [
        "X098_DE_time"]
    assert r["recordings"][0]["rms_median_original_source_unit"] < 5
    assert r["recordings"][1]["rms_median_original_source_unit"] < 5


def test_wrong_variable_key_cannot_silently_use_other_recording(
        simulated_cwru_mat):
    sample = simulated_cwru_mat / "109.mat"
    savemat(sample, {"X999_DE_time": np.ones(80000)})
    with pytest.raises(RealSignalRefused,
                       match="R91_EXACT_ORIGINAL_SENSOR_FIELD_REQUIRED"):
        audit_real_cwru_pair(simulated_cwru_mat)


def test_nonmat_and_nonfinite_waveform_refusal(simulated_cwru_mat):
    sample = simulated_cwru_mat / "109.mat"
    sample.write_text("this is not a MATLAB binary")
    with pytest.raises(RealSignalRefused,
                       match="R91_INVALID_OR_NON_MATLAB_V5_BINARY"):
        audit_real_cwru_pair(simulated_cwru_mat)
    bad = np.ones(80000)
    bad[0] = np.nan
    savemat(sample, {"X109_DE_time": bad})
    with pytest.raises(RealSignalRefused,
                       match="R91_NONFINITE_SOURCE_MEASUREMENT"):
        audit_real_cwru_pair(simulated_cwru_mat)


def test_nonidentifiable_or_degenerate_waveforms_refused(simulated_cwru_mat):
    sample = simulated_cwru_mat / "109.mat"
    savemat(sample, {"X109_DE_time": np.ones(80000)})
    with pytest.raises(RealSignalRefused,
                       match="R91_DEGENERATE_SOURCE_SIGNAL"):
        audit_real_cwru_pair(simulated_cwru_mat)


def test_cli_receipt_new_only_and_never_claims_detection(
        simulated_cwru_mat, tmp_path):
    out = tmp_path / "signal-audit.json"
    assert main(["--raw-dir", str(simulated_cwru_mat),
                 "--receipt", str(out)]) == 0
    artifact = json.loads(out.read_text(encoding="utf-8"))
    assert artifact["measured_generalization_accuracy"] is None
    assert artifact["diagnostic_training_performed"] is False
    assert artifact["official_https_download_in_this_execution"] is False
    with pytest.raises(SystemExit):
        main(["--raw-dir", str(simulated_cwru_mat),
              "--receipt", str(out)])



def test_first_actual_official_download_sha_pin_refuses_changed_original_files(
        simulated_cwru_mat):
    r = audit_real_cwru_pair(simulated_cwru_mat)
    with pytest.raises(RealSignalRefused,
                       match="R91_OFFICIAL_SOURCE_BYTES_CHANGED_SINCE_FIRST_AUDIT"):
        verify_first_documented_official_bytes(r)
    # This exercises byte-identity pin algebra ONLY, not real provenance.
    for item in r["recordings"]:
        item["sha256"] = FIRST_OBSERVED_OFFICIAL_SHA256[item["file_id"]]
    assert verify_first_documented_official_bytes(r) is True
