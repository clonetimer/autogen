"""R91: CWRU original measured-waveform access and acquisition sanity gate.

This is NOT a diagnostic model evaluator. Only two independently numbered
recordings are examined: CWRU Normal 97 and 48k Drive-End IR007 109 (0 HP).
Window counts are NOT independent physical-bearing counts. Downloaded bytes
are never committed; URL provenance and hashes do NOT authenticate equipment.
"""
from __future__ import annotations

from pathlib import Path
from collections.abc import Mapping
import argparse
import hashlib
import json
import math
import urllib.request
from urllib.parse import urlparse

import numpy as np
from scipy.io import loadmat

SCHEMA = "phm.r91.real-measured-cwru-acquisition-audit.v1"
REGISTRY = {
    "97": {
        "role": "HEALTHY",
        "experiment": "NORMAL_0_HP",
        "sample_rate_hz_declared": 48000,
        "field": "X097_DE_time",
        "url": "https://engineering.case.edu/sites/default/files/97.mat",
    },
    "109": {
        "role": "INNER_RACE_007",
        "experiment": "DRIVE_END_48K_IR007_0_HP",
        "sample_rate_hz_declared": 48000,
        "field": "X109_DE_time",
        "url": "https://engineering.case.edu/sites/default/files/109.mat",
    },
}
MAX_BYTES = 16 * 1024 * 1024
WINDOW_SAMPLES = 4096


class RealSignalRefused(ValueError):
    """Do not let a guessed or modified CWRU waveform become evidence."""


def _reject(reason):
    raise RealSignalRefused(reason)


def official_download_files(destination, *, timeout_s=30):
    """Download only two locked official URLs, with a strict binary byte cap.

    The repository and CI artifacts must retain only audit receipts, never
    duplicate third-party raw waveform files. URL identifies where bytes were
    retrieved, not laboratory authenticity or signing.
    """
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    downloaded = []
    for file_id, item in REGISTRY.items():
        target = root / (file_id + ".mat")
        if target.exists():
            _reject("R91_OFFICIAL_DOWNLOAD_REFUSES_OVERWRITE")
        request = urllib.request.Request(item["url"], headers={
            "User-Agent": "autogen-r91-source-acquisition/1.0",
            "Accept": "application/octet-stream,*/*",
        })
        buf = bytearray()
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as stream:
                effective = urlparse(stream.geturl())
                if (effective.scheme != "https" or
                        effective.netloc != "engineering.case.edu"):
                    _reject("R91_OFFICIAL_URL_UNEXPECTED_REDIRECT")
                while True:
                    chunk = stream.read(1024 * 1024)
                    if not chunk:
                        break
                    buf.extend(chunk)
                    if len(buf) > MAX_BYTES:
                        _reject("R91_REMOTE_FILE_EXCEEDS_SIZE_LIMIT")
        except (OSError, TimeoutError) as exc:
            raise RealSignalRefused("R91_OFFICIAL_SOURCE_DOWNLOAD_FAILED") from exc
        if (len(buf) < 128 or not buf[:116].startswith(b"MATLAB")):
            _reject("R91_OFFICIAL_RESPONSE_NOT_MATLAB_V5")
        target.write_bytes(buf)
        downloaded.append(target.name)
    return {"source": "CWRU_OFFICIAL_DIRECT_HTTPS",
            "files": downloaded, "local_raw_files_to_commit": False,
            "independent_source_authenticity_verified": False}


def _read_recording(file_id, raw_root):
    entry = REGISTRY[file_id]
    path = Path(raw_root) / (file_id + ".mat")
    if not path.is_file() or path.is_symlink():
        _reject("R91_REQUIRED_ORIGINAL_MAT_NOT_PRESENT")
    data = path.read_bytes()
    if not 128 <= len(data) <= MAX_BYTES or not data[:116].startswith(b"MATLAB"):
        _reject("R91_INVALID_OR_NON_MATLAB_V5_BINARY")
    digest = hashlib.sha256(data).hexdigest()
    try:
        mat = loadmat(path, squeeze_me=True)
    except (OSError, ValueError, NotImplementedError) as exc:
        raise RealSignalRefused("R91_ORIGINAL_MAT_UNREADABLE") from exc
    expected_field = entry["field"]
    if expected_field not in mat:
        _reject("R91_EXACT_ORIGINAL_SENSOR_FIELD_REQUIRED")
    array = np.asarray(mat[expected_field])
    if (array.dtype.kind not in "fiu" or array.ndim != 1 or
            not 60000 <= len(array) <= 1200000):
        _reject("R91_UNSUPPORTED_ORIGINAL_SENSOR_SHAPE")
    x = array.astype(np.float64)
    if not np.isfinite(x).all():
        _reject("R91_NONFINITE_SOURCE_MEASUREMENT")
    # The exact Xnnn_DE_time variable is selected, not the first matching
    # *_DE_time variable: official recordings may contain stale extra keys.
    possible_extraneous = sorted(k for k in mat if
                                k.endswith("_DE_time") and k != expected_field)
    n = len(x) // WINDOW_SAMPLES
    if n < 10:
        _reject("R91_TOO_FEW_NONOVERLAPPING_WINDOWS")
    windows = x[:n * WINDOW_SAMPLES].reshape(n, WINDOW_SAMPLES)
    rms = np.sqrt(np.mean(windows ** 2, axis=1))
    center = windows - windows.mean(axis=1, keepdims=True)
    sd = np.sqrt(np.mean(center ** 2, axis=1))
    peak = np.max(np.abs(windows), axis=1)
    if np.any(sd <= 1e-12):
        _reject("R91_DEGENERATE_SOURCE_SIGNAL")
    crest = peak / np.maximum(rms, 1e-12)
    if not (np.isfinite(crest).all() and np.isfinite(rms).all()):
        _reject("R91_UNSAFE_SIGNAL_STATISTICS")
    return {
        "file_id": file_id, "source": entry["url"],
        "sha256": digest, "binary_bytes": len(data),
        "original_signal_variable": expected_field,
        "other_de_time_variables_not_used": possible_extraneous,
        "diagnostic_class_label_for_metadata_only": entry["role"],
        "experiment_id": entry["experiment"],
        "declared_sampling_rate_hz": entry["sample_rate_hz_declared"],
        "original_signal_samples": len(x),
        "nonoverlapping_window_samples": WINDOW_SAMPLES,
        "descriptive_window_count_not_independent_samples": n,
        "rms_median_original_source_unit": float(np.median(rms)),
        "rms_iqr_original_source_unit": [
            float(v) for v in np.quantile(rms, [.25, .75])],
        "crest_factor_median": float(np.median(crest)),
        "measured_sensor_unit_authenticated": False,
    }


def audit_real_cwru_pair(raw_root):
    items = [_read_recording(i, raw_root) for i in REGISTRY]
    if items[0]["sha256"] == items[1]["sha256"]:
        _reject("R91_SAME_ORIGINAL_BYTES_RELABELED")
    rates = {i["declared_sampling_rate_hz"] for i in items}
    if rates != {48000}:
        _reject("R91_ACQUISITION_RATE_CONFLICT")
    return {
        "schema": SCHEMA,
        "status": "MEASURED_SIGNAL_INGESTION_ONLY",
        "original_source_page": (
            "https://engineering.case.edu/bearingdatacenter/download-data-file"),
        "source_lab_data_are_measured": True,
        "number_of_original_mat_files": len(items),
        "recordings": items,
        "two_recordings_same_reported_sample_rate": True,
        "same_load_operating_condition": True,
        "number_of_independent_physical_bearings_verified": 0,
        "distinct_bearings_cannot_be_established_from_file_ids": True,
        "windows_are_independent_machines": False,
        "original_mat_binary_hashes_logged": True,
        "manufacturer_provenance_independently_authenticated": False,
        "diagnostic_training_performed": False,
        "validation_or_test_performed": False,
        "measured_generalization_accuracy": None,
        "estimated_false_alarm_rate": None,
        "fuzzy_vs_baseline_advantage_proven": False,
        "eligible_for_stable": False,
    }


def main(args=None):
    parser = argparse.ArgumentParser(description=(
        "Audit two official CWRU original real vibration recordings; "
        "never infer diagnostic generalization from raw window counts."))
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--download-official", action="store_true")
    parser.add_argument("--receipt", required=True)
    opt = parser.parse_args(args)
    output = Path(opt.receipt)
    if output.exists():
        parser.error("R91_EXISTING_AUDIT_RECEIPT_REFUSES_OVERWRITE")
    if opt.download_official:
        official_download_files(opt.raw_dir)
    result = audit_real_cwru_pair(opt.raw_dir)
    with output.open("x", encoding="utf-8") as f:
        json.dump(result, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
