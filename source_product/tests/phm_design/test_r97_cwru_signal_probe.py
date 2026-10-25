"""R97 original-CWRU feature probe regression using fabricated MAT files.

This simulates MAT I/O and digest pin behavior. It does NOT substitute
synthetic data for the separate official HTTPS GitHub Actions job.
"""
from __future__ import annotations

from io import BytesIO
import hashlib

import numpy as np
import pytest
from scipy.io import savemat

from phm_design.industrial_contracts import r97_cwru_signal_probe as probe
from phm_design.industrial_contracts.signal_feature_design import SignalDesignRejected


def _two_fake_mats(monkeypatch):
    raws, metadata = {}, []
    for idx, ident in enumerate(("97", "109")):
        field = "X" + ident.zfill(3) + "_DE_time"
        t = np.arange(8192) / 48000.
        source = np.sin(2*np.pi*(200 if idx == 0 else 2400)*t)
        stream = BytesIO()
        savemat(stream, {field: source})
        raw = stream.getvalue()
        sha = hashlib.sha256(raw).hexdigest()
        url = f"https://engineering.case.edu/sites/default/files/{ident}.mat"
        raws[url] = raw
        metadata.append((ident, "TEST_" + ident, field, sha))
    monkeypatch.setattr(probe, "INPUTS", tuple(metadata))
    return raws


def test_original_probe_uses_exact_mat_fields_sha_and_frequencies(monkeypatch):
    originals = _two_fake_mats(monkeypatch)
    report = probe.inspect_two_original_recordings(
        downloader=lambda url: originals[url])
    assert report["status"] == "OFFICIAL_ORIGINAL_MEASURED_SIGNAL_REPRESENTATION_ONLY"
    assert len(report["files"]) == 2
    for entry in report["files"]:
        assert entry["original_mat_sha256"] == hashlib.sha256(
            originals[entry["original_url"]]).hexdigest()
        assert entry["recording_provenance"]["technical_windows_aggregated"] == 2
        assert entry["recording_provenance"]["independent_recording_windows_created"] is False
    x, y = [r["signal_features"] for r in report["files"]]
    assert x["vibration__band_50_500"] > .95
    assert y["vibration__band_500_8000"] > .95
    assert not report["model_trained"]
    assert not report["independent_physical_bearings_verified"]
    assert not report["diagnostic_accuracy_measured"]
    assert not report["stable_release_eligible"]


def test_any_original_mat_bytes_change_must_fail_sha_pin(monkeypatch):
    originals = _two_fake_mats(monkeypatch)
    first = next(iter(originals))
    changed = dict(originals)
    changed[first] = changed[first] + b"modified"
    with pytest.raises(SignalDesignRejected,
                       match="R97_ORIGINAL_CWRU_SHA256_MISMATCH"):
        probe.inspect_two_original_recordings(
            downloader=lambda url: changed[url])


def test_missing_named_vibration_channel_never_replaces_with_random_other_key(
        monkeypatch):
    originals = _two_fake_mats(monkeypatch)
    first = next(iter(originals))
    stream = BytesIO()
    savemat(stream, {"X888_DE_time": np.ones(8192)})
    content = stream.getvalue()
    old = list(probe.INPUTS)
    old[0] = tuple(list(old[0][:3]) + [hashlib.sha256(content).hexdigest()])
    monkeypatch.setattr(probe, "INPUTS", tuple(old))
    originals[first] = content
    with pytest.raises(SignalDesignRejected,
                       match="R97_ORIGINAL_CWRU_FIELD_MISSING"):
        probe.inspect_two_original_recordings(
            downloader=lambda url: originals[url])
