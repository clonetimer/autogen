"""R101 research pilot: deterministic, audited BYTE-FIXTURE scorer.

Only pre-reviewed JSON linear coefficients and a fixed raw-byte transform.
NEVER executes pickle, Python code, third-party model plugins or dynamic eval.
Raw bytes are deliberately NOT industrial signal physics: an executable
lineage *protocol* smoke test, not torch_like/AAS/native diagnosis validation.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import hashlib
import json
import math
import os
import platform
import time

SCHEMA = "phm.autogen.r101-fixture-executable.v1"
SEAL = "phm.autogen.r101-fixture-execution-seal.v1"
RECEIPT = "phm.autogen.r101-fixture-execution-receipt.v1"
MAX_BYTES = 1024 * 1024

class SealedExecutionRejected(ValueError):
    pass

def _refuse(reason):
    raise SealedExecutionRejected(reason)

def _digest(v):
    try:
        data=json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False)
    except (ValueError,TypeError):
        _refuse("R101_NONCANONICAL_JSON")
    return hashlib.sha256(data.encode("utf-8")).hexdigest()

def _sha(s):
    return (type(s) is str and len(s)==64 and
            all(c in "0123456789abcdef" for c in s))

def _str(v):
    return type(v) is str and 0 < len(v) <= 128

def _validate_model(model):
    if not isinstance(model,Mapping) or set(model)!={"schema","algorithm",
             "transform","classes","coefficients","intercept","source_plan_sha256",
             "trained_on_source_only"}:
        _refuse("R101_UNRECOGNIZED_MODEL_SCHEMA")
    if (model["schema"]!=SCHEMA or model["algorithm"]!="linear_binary_logit"
            or model["transform"]!="raw_byte_mean_div_255"
            or model["trained_on_source_only"] is not True
            or not _sha(model["source_plan_sha256"])):
        _refuse("R101_UNTRUSTED_MODEL_OR_SOURCE")
    labels=model["classes"]
    if (type(labels) is not list or len(labels)!=2
            or not all(_str(s) for s in labels) or labels[0]==labels[1]):
        _refuse("R101_BAD_CLASSES")
    coefficients=model["coefficients"]
    if (type(coefficients) is not list or len(coefficients)!=1
            or not all(type(x) in (float,int) and math.isfinite(x)
                       and abs(x)<=1e4 for x in coefficients)
            or type(model["intercept"]) not in (float,int)
            or not math.isfinite(model["intercept"])
            or abs(model["intercept"])>1e4):
        _refuse("R101_UNSAFE_NUMERIC_COEFFICIENTS")

def seal_fixture_model(model, record_sha256):
    """Caller must seal BEFORE seeing external labels; timing not attested."""
    _validate_model(model)
    if (not isinstance(record_sha256,Mapping) or not record_sha256
            or len(record_sha256)>10000 or not all(
                _str(k) and _sha(v) for k,v in record_sha256.items())):
        _refuse("R101_MISSING_EXACT_RAW_FILE_MANIFEST")
    doc={"schema":SEAL,"model_sha256":_digest(model),
         "source_plan_sha256":model["source_plan_sha256"],
         "original_record_sha256":dict(sorted(record_sha256.items())),
         "predictor":"audited_byte_fixture_linear_v1",
         "no_retraining_or_threshold_tuning":True,
         "seal_chronology_externally_attested":False}
    return {**doc,"seal_sha256":_digest(doc)}

def execute_sealed_fixture(seal,model,raw_paths):
    """Run fixed scorer against EXACT local file bytes; never read label/role.

    The caller cannot provide predictions. No arbitrary model execution.
    Does not independently authenticate claimed source-plan/model provenance.
    """
    _validate_model(model)
    if not isinstance(seal,Mapping):
        _refuse("R101_MISSING_SEAL")
    expected=seal_fixture_model(model,seal.get("original_record_sha256"))
    if seal!=expected:
        _refuse("R101_TAMPERED_MODEL_SEAL_OR_RECORD_SET")
    required=seal["original_record_sha256"]
    if not isinstance(raw_paths,Mapping) or set(raw_paths)!=set(required):
        _refuse("R101_MISSING_OR_EXTRA_RAW_FILE")
    # Validate and hash all raw bytes BEFORE producing any prediction.
    checked={}
    for rid in sorted(required):
        value=raw_paths[rid]
        if type(value) is not str:
            _refuse("R101_RAW_PATH_REQUIRED")
        path=Path(value)
        if path.is_symlink() or not path.is_file():
            _refuse("R101_UNSAFE_OR_MISSING_RAW_FILE")
        if not 128 <= path.stat().st_size <= MAX_BYTES:
            _refuse("R101_RAW_SIZE_OUTSIDE_FIXTURE_LIMIT")
        raw=path.read_bytes()
        digest=hashlib.sha256(raw).hexdigest()
        if digest!=required[rid]:
            _refuse("R101_RAW_FILE_SHA_MISMATCH")
        checked[rid]=(raw,digest)
    wall=time.perf_counter()
    cpu=time.process_time()
    predictions={}
    for rid,(raw,digest) in checked.items():
        value=sum(raw)/(255.0*len(raw))
        score=model["coefficients"][0]*value+model["intercept"]
        label=model["classes"][1] if score>=0 else model["classes"][0]
        predictions[rid]={"predicted_label":label,
                          "raw_sha256":digest,
                          "fixed_feature_sha256":_digest([value])}
    measured={"wall_seconds":max(0.,time.perf_counter()-wall),
              "process_cpu_seconds":max(0.,time.process_time()-cpu),
              "python_implementation":platform.python_implementation(),
              "python_version":platform.python_version(),
              "platform":platform.platform(),
              "cpu_count":os.cpu_count(),
              "memory_peak_authenticated":False,
              "gpu_cost_measured":False,
              "matched_hardware_cost_proven":False}
    result={"schema":RECEIPT,"seal_sha256":seal["seal_sha256"],
            "model_sha256":_digest(model),
            "record_count":len(predictions),
            "predictions":predictions,
            "measured_resources":measured,
            "execution": "fixed_builtin_byte_fixture_only",
            "external_validation_truth_read":False,
            "trained_model_provenance_independently_authenticated":False,
            "original_physical_device_identity_authenticated":False,
            "industrial_feature_graph_executed":False,
            "native_torch_like_provider_executed":False,
            "independent_real_equipment_superiority_proven":False,
            "eligible_for_stable":False}
    return {**result,"receipt_sha256":_digest(result)}
