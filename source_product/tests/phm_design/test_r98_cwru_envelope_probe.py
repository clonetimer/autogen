"""R98 CWRU original waveform reader structure tests with synthetic MAT bytes.

GitHub separate live downloader (NOT these fixtures) establishes that
official source bytes were processed in a real-data smoke test.
"""
from __future__ import annotations

from io import BytesIO
import hashlib

import numpy as np
import pytest
from scipy.io import savemat

from phm_design.industrial_contracts import r98_cwru_envelope_probe as probe
from phm_design.industrial_contracts.physical_graph_design import PhysicalGraphRejected


def _fixture_originals(monkeypatch):
    saved = {}
    registry = []
    for idx, ident in enumerate(("97", "109")):
        var = f"X{int(ident):03d}_DE_time"
        t = np.arange(2*4096)/48000.
        carrier = (1+.4*np.sin(2*np.pi*200*t))*np.sin(2*np.pi*4000*t)
        source = carrier * (1+.3*idx)
        stream = BytesIO()
        savemat(stream, {var: source})
        blob = stream.getvalue()
        key = f"https://engineering.case.edu/sites/default/files/{ident}.mat"
        saved[key] = blob
        registry.append((ident, f"SYNTHETIC_{ident}", var,
                         hashlib.sha256(blob).hexdigest()))
    monkeypatch.setattr(probe, "INPUTS", tuple(registry))
    return saved


def test_exact_original_field_and_envelope_processed_without_orders(monkeypatch):
    binaries = _fixture_originals(monkeypatch)
    result = probe.inspect_original_cwru_envelopes(
        downloader=lambda url: binaries[url])
    assert len(result["original_recordings"]) == 2
    assert result["original_files_count"] == 2
    assert result["rotational_order_features_computed"] is False
    assert result["independent_tachometer_present"] is False
    assert result["number_of_independent_physical_bearings_authenticated"] == 0
    assert result["model_trained_or_selected"] is False
    assert result["diagnostic_model_accuracy_estimated"] is False
    assert not result["fuzzy_or_neural_benefit_proven"]
    for rec in result["original_recordings"]:
        source = rec["features"]
        assert 180 <= source["vibration__envelope_peak_hz"] <= 220
        assert "vibration__order_1_3_fraction" not in source
        assert rec["recording_receipt"]["rpm_from_self_declared_tachometer"] is False
        assert len(rec["original_mat_sha256"]) == 64


def test_modified_original_bytes_refused_and_no_fake_data_fallback(monkeypatch):
    binaries = _fixture_originals(monkeypatch)
    key = next(iter(binaries))
    binaries[key] = binaries[key] + b"modified"
    with pytest.raises(PhysicalGraphRejected,
                       match="R98_CWRU_ORIGINAL_SHA256_DRIFT"):
        probe.inspect_original_cwru_envelopes(
            downloader=lambda url: binaries[url])


def test_missing_exact_vibration_variable_cannot_select_other_mat_channel(
        monkeypatch):
    binaries = _fixture_originals(monkeypatch)
    key = next(iter(binaries))
    stream = BytesIO()
    savemat(stream, {"X888_DE_time": np.ones(8192)})
    raw = stream.getvalue()
    amended = list(probe.INPUTS)
    amended[0] = tuple([*amended[0][:3], hashlib.sha256(raw).hexdigest()])
    monkeypatch.setattr(probe, "INPUTS", tuple(amended))
    binaries[key] = raw
    with pytest.raises(PhysicalGraphRejected,
                       match="R98_REQUIRED_ORIGINAL_MATLAB_FIELD_MISSING"):
        probe.inspect_original_cwru_envelopes(
            downloader=lambda url: binaries[url])
