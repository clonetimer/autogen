"""R98 official original CWRU: real measured envelope FEATURE audit, no model.

Two original CWRU files have no independently verified per-window
tachometer trace. Therefore this script cannot invoke R98 order/fusion
graphs, fit a classifier or publish cross-bearing accuracy. It only checks
official HTTPS bytes against first-observed R91 digests, reads EXACT original
MATLAB vibration fields and applies R98 time/envelope transform.
"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
import argparse
import hashlib
import json
import urllib.request
from urllib.parse import urlparse

from .r97_cwru_signal_probe import INPUTS, MAX_BYTES
from .physical_graph_design import _record_features, PhysicalGraphRejected


def inspect_original_cwru_envelopes(*, downloader=None):
    from scipy.io import loadmat
    import numpy as np

    if downloader is None:
        def downloader(url):
            request = urllib.request.Request(
                url, headers={"User-Agent": "autogen-r98-research/1.0"})
            with urllib.request.urlopen(request, timeout=30) as stream:
                dest = urlparse(stream.geturl())
                if dest.scheme != "https" or dest.netloc != "engineering.case.edu":
                    raise PhysicalGraphRejected("R98_OFFICIAL_DATA_HOST_CHANGED")
                return stream.read(MAX_BYTES + 1)

    records = []
    for file_id, publisher_category, matlab_field, original_sha in INPUTS:
        uri = f"https://engineering.case.edu/sites/default/files/{file_id}.mat"
        blob = downloader(uri)
        if (type(blob) is not bytes or not 128 <= len(blob) <= MAX_BYTES or
                not blob[:116].startswith(b"MATLAB")):
            raise PhysicalGraphRejected("R98_ORIGINAL_CWRU_MAT_INVALID")
        sha = hashlib.sha256(blob).hexdigest()
        if sha != original_sha:
            raise PhysicalGraphRejected("R98_CWRU_ORIGINAL_SHA256_DRIFT")
        data = loadmat(BytesIO(blob), squeeze_me=True)
        if matlab_field not in data:
            raise PhysicalGraphRejected("R98_REQUIRED_ORIGINAL_MATLAB_FIELD_MISSING")
        source = np.asarray(data[matlab_field])
        if source.ndim != 1 or source.dtype.kind not in "fiu":
            raise PhysicalGraphRejected("R98_ORIGINAL_VIBRATION_SHAPE_INVALID")
        # NO rpm_per_window property: these two official waveform files are
        # NOT shown to have an independent tachometer recording.
        raw = {
            "record_id": "cwru_"+file_id,
            "sample_rate_hz": 48000,
            "samples": source.tolist(),
            "scalars": {},
            "scalar_alignment": "none",
        }
        features, receipt = _record_features(
            raw, vib="vibration", hz=48000., speed=None, companions=[])
        if any("__order_" in k for k in features):
            raise PhysicalGraphRejected("R98_CWRU_MUST_NOT_INFER_SHAFT_ORDER")
        records.append({
            "original_publisher_file_id": file_id,
            "publisher_condition_metadata": publisher_category,
            "source_url": uri, "original_mat_sha256": sha,
            "exact_matlab_channel": matlab_field,
            "features": {k: v for k,v in features.items()
                         if k in {
                             "vibration__rms", "vibration__crest",
                             "vibration__envelope_40_700_fraction",
                             "vibration__envelope_peak_hz"}},
            "recording_receipt": receipt,
        })
    return {
        "schema": "phm.autogen.r98-official-cwru-envelope-audit.v1",
        "status": "PUBLIC_MEASURED_ENVELOPE_FEATURES_ONLY",
        "original_recordings": records,
        "original_files_count": len(records),
        "number_of_independent_physical_bearings_authenticated": 0,
        "independent_tachometer_present": False,
        "rotational_order_features_computed": False,
        "model_trained_or_selected": False,
        "diagnostic_model_accuracy_estimated": False,
        "fuzzy_or_neural_benefit_proven": False,
        "industrial_stable_qualification": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Original CWRU envelope signal audit ONLY; no model or rpm")
    parser.add_argument("--receipt", required=True)
    args = parser.parse_args(argv)
    dst = Path(args.receipt)
    if dst.exists():
        parser.error("R98_ORIGINAL_RECEIPT_OVERWRITE_REFUSED")
    result = inspect_original_cwru_envelopes()
    with dst.open("x", encoding="utf-8") as file:
        json.dump(result, file, sort_keys=True, indent=2,
                  ensure_ascii=False, allow_nan=False)
        file.write("\n")
    print(json.dumps({
        "original_files_count": result["original_files_count"],
        "model_trained_or_selected": False,
        "rotational_order_features_computed": False,
        "receipt": str(dst),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
