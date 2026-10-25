"""R97 *optional* real CWRU vibration-feature smoke, no model training.

Official files 97.mat and 109.mat were already obtained in R91; bytes below
are first-observed GitHub HTTPS hashes, NOT publisher-signed certificates.
Each file is ONE recording. No CWRU split, accuracy or model validation
is attempted. This tool stores ONLY a JSON descriptor, never original MAT.
"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
import argparse
import hashlib
import json
import urllib.request
from urllib.parse import urlparse

from .signal_feature_design import _series_observation, SignalDesignRejected

INPUTS = (
    ("97", "HEALTHY_BASELINE",
     "X097_DE_time",
     "16bf48babcf1c7ac224bc1a81cd9eafdb27e42d5cf559761907e067e8eeadf3c"),
    ("109", "INNER_RACE_007",
     "X109_DE_time",
     "daddef5f784879becdf8fefbe2453f11c3cfc8d061a2a671da2f546d1bb48460"),
)
MAX_BYTES = 8 * 1024 * 1024


def inspect_two_original_recordings(*, downloader=None):
    """Synchronous bounded HTTPS access; callable tests may inject a fake downloader."""
    from scipy.io import loadmat
    import numpy as np

    if downloader is None:
        def downloader(url):
            request = urllib.request.Request(
                url, headers={"User-Agent": "autogen-r97-research/1.0"})
            with urllib.request.urlopen(request, timeout=30) as stream:
                final = urlparse(stream.geturl())
                if (final.scheme != "https" or
                        final.netloc != "engineering.case.edu"):
                    raise SignalDesignRejected("R97_OFFICIAL_HOST_CHANGED")
                raw = stream.read(MAX_BYTES + 1)
            return raw

    checked = []
    for ident, label, name, expected_sha in INPUTS:
        url = f"https://engineering.case.edu/sites/default/files/{ident}.mat"
        content = downloader(url)
        if (type(content) is not bytes or not 128 <= len(content) <= MAX_BYTES
                or not content[:116].startswith(b"MATLAB")):
            raise SignalDesignRejected("R97_INVALID_ORIGINAL_MAT_BYTES")
        sha = hashlib.sha256(content).hexdigest()
        if sha != expected_sha:
            raise SignalDesignRejected("R97_ORIGINAL_CWRU_SHA256_MISMATCH")
        data = loadmat(BytesIO(content), squeeze_me=True)
        if name not in data:
            raise SignalDesignRejected("R97_ORIGINAL_CWRU_FIELD_MISSING")
        raw = np.asarray(data[name])
        if raw.ndim != 1 or raw.dtype.kind not in "fiu":
            raise SignalDesignRejected("R97_BAD_CWRU_ORIGINAL_CHANNEL")
        item = {
            "record_id": "cwru_" + ident,
            "sample_rate_hz": 48000,  # declared in the official metadata
            "samples": raw.tolist(),
            "scalars": {},
        }
        features, original = _series_observation(
            item, 48000., "vibration", [])
        checked.append({
            "id": ident, "publisher_recording_category": label,
            "original_url": url,
            "original_mat_sha256": sha,
            "matlab_channel": name,
            "declared_sample_rate_hz": 48000,
            "signal_features": features,
            "recording_provenance": original,
        })
    return {
        "schema": "phm.autogen.r97-cwru-original-feature-audit.v1",
        "status": "OFFICIAL_ORIGINAL_MEASURED_SIGNAL_REPRESENTATION_ONLY",
        "files": checked,
        "source_files_differ_by_publisher_metadata": True,
        "signal_units_calibrated": False,
        "manufacturer_signed_digest_verified": False,
        "independent_physical_bearings_verified": 0,
        "model_trained": False,
        "diagnostic_accuracy_measured": False,
        "self_organizing_advantage_proven": False,
        "industrial_generalization_proven": False,
        "stable_release_eligible": False,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Real official CWRU R97 FEATURE ONLY, no diagnosis model")
    ap.add_argument("--receipt", required=True)
    args = ap.parse_args(argv)
    output = Path(args.receipt)
    if output.exists():
        ap.error("R97_REFUSE_OVERWRITE_REAL_INPUT_RECEIPT")
    receipt = inspect_two_original_recordings()
    with output.open("x", encoding="utf-8") as f:
        json.dump(receipt, f, sort_keys=True, indent=2,
                  ensure_ascii=False, allow_nan=False)
        f.write("\n")
    print(json.dumps({
        "real_recordings_processed": len(receipt["files"]),
        "receipt": str(output),
        "model_trained": False,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
