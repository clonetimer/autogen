"""R89 external physical-cohort preregistration tests.

Fixtures use tiny fabricated MAT signals solely to exercise SHA and cohort
audit. No authentic Paderborn or CWRU signals are supplied; PASS is NOT a
real-world diagnosis result.
"""
from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
import pytest
from scipy.io import savemat

from phm_design.industrial.r89_external_cohort_audit import (
    SCHEMA, CohortAuditError, audit_external_bearing_cohort,
    audit_manifest_file)


def _cohort(root):
    rows = []
    # Three distinct TRAIN healthy + three fault across 3 different
    # environments; four distinct VAL healthy + fault across 2 environments.
    for role, envs, per_class in (
        ("TRAIN", ("operating_a", "operating_b", "operating_c"), 1),
        ("VALIDATION", ("operating_d", "operating_e"), 2),
    ):
        for e in envs:
            for lab in ("healthy", "fault"):
                for n in range(per_class):
                    sid = f"bearing_{role}_{e}_{lab}_{n}"
                    rel = sid + ".mat"
                    path = root / rel
                    savemat(path, {"vibration": np.linspace(n, n + 2, 32) +
                            0.1 * len(rows)})
                    sha = hashlib.sha256(path.read_bytes()).hexdigest()
                    rows.append({
                        "record_id": "record_" + str(len(rows)),
                        "role": role, "diagnosis_label": lab,
                        "specimen_id": sid, "environment_id": e,
                        "raw_file": rel, "raw_sha256": sha,
                        "observation_sha256": hashlib.sha256(
                            ("measurement/" + sid).encode()).hexdigest(),
                    })
    return rows


def test_seven_independent_healthy_fixture_manifest_passes_strict_support(
        tmp_path):
    rows = _cohort(tmp_path)
    assert len(rows) == 14
    receipt = audit_external_bearing_cohort(
        rows, source="EXTERNAL_REVIEWER_DOCUMENTED", raw_root=tmp_path)
    assert receipt["r88_strict_protocol_manifest_eligible"]
    assert receipt["minimum_combined_healthy_bearings"] == 7
    assert receipt["distinct_self_reported_specimens"] == 14
    assert receipt["raw_file_sha256_checked_against_local_bytes"]
    assert receipt["source_registry_is_self_reported"]
    assert receipt["independent_generalization_proven"] is False
    assert receipt["actual_sensor_or_specimen_provenance_authenticated"] is False
    assert receipt["eligible_for_stable"] is False
    assert receipt["blockers"] == []
    p = tmp_path / "external.json"
    p.write_text(json.dumps({"schema": SCHEMA, "source":
                 "EXTERNAL_REVIEWER_DOCUMENTED", "records": rows}))
    assert audit_manifest_file(p, raw_root=tmp_path) == receipt


def test_paderborn_six_healthy_cannot_meet_r88_seven_healthy_even_if_forged(
        tmp_path):
    rows = _cohort(tmp_path)
    receipt = audit_external_bearing_cohort(
        rows, source="PADERBORN_KAT", raw_root=tmp_path)
    assert not receipt["r88_strict_protocol_manifest_eligible"]
    assert "R89_CATALOG_HEALTHY_BEARINGS_BELOW_R88_SUPPORT" in receipt["blockers"]
    assert receipt["catalog"]["declared_unique_healthy_specimens"] == 6
    assert receipt["source_registry_is_self_reported"]


def test_cwru_different_motor_loads_do_not_prove_distinct_normal_bearings(
        tmp_path):
    receipt = audit_external_bearing_cohort(
        _cohort(tmp_path), source="CWRU_PUBLIC")
    assert not receipt["r88_strict_protocol_manifest_eligible"]
    assert "R89_CWRU_DISTINCT_HEALTHY_SPECIMENS_UNVERIFIED" in receipt["blockers"]


def test_cannot_use_windows_from_one_physical_bearing_as_new_groups(tmp_path):
    rows = _cohort(tmp_path)
    copied = copy.deepcopy(rows)
    copied[6]["specimen_id"] = copied[0]["specimen_id"]
    with pytest.raises(CohortAuditError, match="R89_SPECIMEN_ROLE_LABEL_OR_ENVIRONMENT_CHANGED"):
        audit_external_bearing_cohort(
            copied, source="EXTERNAL_REVIEWER_DOCUMENTED")


def test_duplicate_raw_across_specimens_refused(tmp_path):
    rows = _cohort(tmp_path)
    altered = copy.deepcopy(rows)
    altered[1]["raw_sha256"] = altered[0]["raw_sha256"]
    with pytest.raises(CohortAuditError, match="R89_RAW_FILE_REUSED_ACROSS_BEARINGS"):
        audit_external_bearing_cohort(
            altered, source="EXTERNAL_REVIEWER_DOCUMENTED")


def test_modified_raw_bytes_and_observation_relabel_refused(tmp_path):
    rows = _cohort(tmp_path)
    path = tmp_path / rows[0]["raw_file"]
    path.write_bytes(path.read_bytes() + b"changed_after_manifest")
    with pytest.raises(CohortAuditError, match="R89_RAW_FILE_HASH_MISMATCH"):
        audit_external_bearing_cohort(
            rows, source="EXTERNAL_REVIEWER_DOCUMENTED", raw_root=tmp_path)
    altered = copy.deepcopy(rows)
    altered[1]["observation_sha256"] = altered[0]["observation_sha256"]
    with pytest.raises(CohortAuditError, match="R89_RELABELED_OBSERVATION_ACROSS_BEARINGS"):
        audit_external_bearing_cohort(
            altered, source="EXTERNAL_REVIEWER_DOCUMENTED")


def test_protected_final_and_unsafe_source_path_blocked(tmp_path):
    rows = _cohort(tmp_path)
    modified = copy.deepcopy(rows)
    modified[0]["role"] = "FINAL"
    with pytest.raises(CohortAuditError, match="R89_PROTECTED_OR_UNSUPPORTED_ROLE"):
        audit_external_bearing_cohort(
            modified, source="EXTERNAL_REVIEWER_DOCUMENTED")
    modified = copy.deepcopy(rows)
    modified[0]["raw_file"] = "../secret.mat"
    with pytest.raises(CohortAuditError, match="R89_UNSAFE_OR_NONMAT_SOURCE_PATH"):
        audit_external_bearing_cohort(
            modified, source="EXTERNAL_REVIEWER_DOCUMENTED")
    modified = copy.deepcopy(rows)
    modified[0]["raw_sha256"] = "abc"
    with pytest.raises(CohortAuditError, match="R89_INVALID_RAW_SHA256"):
        audit_external_bearing_cohort(
            modified, source="EXTERNAL_REVIEWER_DOCUMENTED")


def test_insufficient_healthy_group_support_refused_without_fabrication(tmp_path):
    rows = _cohort(tmp_path)
    # Retain only one healthy group in the first validation environment.
    pruned = [r for r in rows if r["specimen_id"] !=
              "bearing_VALIDATION_operating_d_healthy_1"]
    receipt = audit_external_bearing_cohort(
        pruned, source="EXTERNAL_REVIEWER_DOCUMENTED")
    assert not receipt["r88_strict_protocol_manifest_eligible"]
    assert "R89_VALIDATION_NEEDS_TWO_BEARINGS_PER_ENV_CLASS" in receipt["blockers"]
    assert "R89_TOO_FEW_VALIDATION_BEARINGS_PER_CLASS" in receipt["blockers"]
