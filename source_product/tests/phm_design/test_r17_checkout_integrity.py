"""Integration regression: immutable R16 original CRLF bytes vs Git LF checkout.

This test does not change any historical audit SHA, qualification number,
measurement or data format; different CSV content must still be rejected.
"""
import hashlib
import json
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paired_observation_audit import _trace_matches_frozen_original, load_case


FIX = Path(__file__).resolve().parent/"fixtures"/"r16_opened"


def test_all_original_r16_train_trace_digests_revalidate_on_git_checkout():
    for audit_path in sorted((FIX/"audit").glob("*.json")):
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        trace = FIX/"traces"/(audit["case_id"]+".csv")
        assert trace.is_file(), audit["case_id"]
        assert _trace_matches_frozen_original(trace, audit["trace_sha256"]), audit["case_id"]
    assert len(list((FIX/"audit").glob("*.json"))) == 54


def test_normalized_checkout_does_not_accept_tampered_measurements(tmp_path):
    sample = sorted((FIX/"audit").glob("*.json"))[0]
    audit = json.loads(sample.read_text(encoding="utf-8"))
    raw = (FIX/"traces"/(audit["case_id"]+".csv")).read_bytes()
    modified = raw.replace(b"time_s", b"TIME_S", 1)
    assert modified != raw
    path = tmp_path/"trace.csv"
    path.write_bytes(modified)
    assert not _trace_matches_frozen_original(path, audit["trace_sha256"])


def test_frozen_crlf_original_sha_is_preserved_exactly(tmp_path):
    sample = sorted((FIX/"audit").glob("*.json"))[0]
    audit = json.loads(sample.read_text(encoding="utf-8"))
    source = (FIX/"traces"/(audit["case_id"]+".csv")).read_bytes()
    restored = source.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
    assert hashlib.sha256(restored).hexdigest() == audit["trace_sha256"]
    restored_path = tmp_path/"original-crlf.csv"
    restored_path.write_bytes(restored)
    assert _trace_matches_frozen_original(restored_path, audit["trace_sha256"])
