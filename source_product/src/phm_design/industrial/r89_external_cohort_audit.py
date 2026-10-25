"""R89 external-bearing cohort feasibility auditor, before reading protected data.

This audits *self-declared manifest identities* and optional local byte hashes.
It does not authenticate the existence of distinct physical bearings, sensor
calibration or manufacturer provenance. R88's preregistered 3-source-env +
2-VAL-env-with-2-groups/class supports MUST NOT be silently relaxed.

Paderborn KAt has six documented healthy specimens; the present conservative
source+VAL design needs at least seven disjoint healthy specimens (3 + 2*2).
CWRU's normal operating-load files also do not establish multiple independent
healthy bearings. Both must fail this particular R88 evidence protocol rather
than relabeling windows as independent physical groups.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
import hashlib
import json
import re

SCHEMA = "phm.r89.external-physical-cohort.v1"
ROLES = ("TRAIN", "VALIDATION")
LABELS = ("healthy", "fault")
SHA = re.compile(r"^[0-9a-f]{64}$")
IDENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
# These are catalog statements, not filenames/raw data authenticated by us.
CATALOG = {
    "PADERBORN_KAT": {
        "declared_unique_healthy_specimens": 6,
        "basis": "Paderborn KAt BearingDataCenter documents six undamaged bearings",
        "website": "https://mb.uni-paderborn.de/en/kat/research/bearing-datacenter/data-sets-and-download",
        "catalog_verified_on_individual_raw_files": False,
    },
    "CWRU_PUBLIC": {
        "declared_unique_healthy_specimens": None,
        "basis": "Normal files by motor load do not prove distinct installed bearings",
        "website": "https://engineering.case.edu/bearingdatacenter/download-data-file",
        "catalog_verified_on_individual_raw_files": False,
    },
    "EXTERNAL_REVIEWER_DOCUMENTED": {
        "declared_unique_healthy_specimens": None,
        "basis": "External lab inventory and specimen proof must be reviewed",
        "website": None,
        "catalog_verified_on_individual_raw_files": False,
    },
}


class CohortAuditError(ValueError):
    """Fail-closed malformed manifest; not a failed ML model."""


def _require(ok, name):
    if not ok:
        raise CohortAuditError(name)


def _identifier(x, name):
    _require(type(x) is str and IDENT.fullmatch(x) is not None, name)
    return x


def _digest(x, name):
    _require(type(x) is str and SHA.fullmatch(x) is not None, name)
    return x


def audit_external_bearing_cohort(rows: Sequence[Mapping], *,
                                  source: str,
                                  raw_root: str | Path | None = None):
    """Compute strict R88 TRAIN->VALIDATION feasibility from immutable IDs.

    raw_root makes raw SHA checks mandatory. Never issue a validity pass
    without separately reviewing that specimen identities refer to real,
    distinct physical bearings and labels are independently supported.
    """
    _require(source in CATALOG, "R89_UNREGISTERED_EXTERNAL_SOURCE")
    _require(isinstance(rows, (list, tuple)) and bool(rows),
             "R89_EMPTY_COHORT")
    expected = {"record_id", "role", "diagnosis_label", "specimen_id",
                "environment_id", "raw_file", "raw_sha256",
                "observation_sha256"}
    groups = defaultdict(set)
    specimen = {}
    raw_hash_owner = {}
    obs_owner = {}
    records = set()
    checked_raw_paths = {}
    root = None if raw_root is None else Path(raw_root).resolve(strict=True)
    for row in rows:
        _require(isinstance(row, Mapping) and set(row) == expected,
                 "R89_INVALID_COHORT_ROW")
        record_id = _identifier(row["record_id"], "R89_INVALID_RECORD_ID")
        _require(record_id not in records, "R89_DUPLICATE_RECORD_ID")
        records.add(record_id)
        role = row["role"]
        label = row["diagnosis_label"]
        _require(role in ROLES, "R89_PROTECTED_OR_UNSUPPORTED_ROLE")
        _require(label in LABELS, "R89_UNSUPPORTED_DIAGNOSIS_LABEL")
        sid = _identifier(row["specimen_id"], "R89_INVALID_PHYSICAL_SPECIMEN_ID")
        env = _identifier(row["environment_id"], "R89_INVALID_ENVIRONMENT")
        raw_sha = _digest(row["raw_sha256"], "R89_INVALID_RAW_SHA256")
        obs_sha = _digest(row["observation_sha256"], "R89_INVALID_OBSERVATION_SHA256")
        rel = Path(row["raw_file"])
        _require(type(row["raw_file"]) is str and
                 not rel.is_absolute() and rel.suffix.lower() == ".mat"
                 and not any(p in ("..", ".") for p in rel.parts),
                 "R89_UNSAFE_OR_NONMAT_SOURCE_PATH")
        prior = specimen.setdefault(sid, (role, label, env))
        _require(prior == (role, label, env),
                 "R89_SPECIMEN_ROLE_LABEL_OR_ENVIRONMENT_CHANGED")
        prior_owner = raw_hash_owner.setdefault(raw_sha, sid)
        _require(prior_owner == sid, "R89_RAW_FILE_REUSED_ACROSS_BEARINGS")
        prior_obs = obs_owner.setdefault(obs_sha, sid)
        _require(prior_obs == sid, "R89_RELABELED_OBSERVATION_ACROSS_BEARINGS")
        if root is not None:
            actual = (root / rel).resolve(strict=True)
            _require(actual.is_relative_to(root), "R89_PATH_ESCAPES_SOURCE_ROOT")
            _require(actual.is_file(), "R89_SOURCE_NOT_A_FILE")
            key = str(actual)
            if key not in checked_raw_paths:
                sha = hashlib.sha256()
                with actual.open("rb") as f:
                    for chunk in iter(lambda: f.read(1024 * 1024), b""):
                        sha.update(chunk)
                checked_raw_paths[key] = sha.hexdigest()
            _require(checked_raw_paths[key] == raw_sha,
                     "R89_RAW_FILE_HASH_MISMATCH")
        groups[(role, env, label)].add(sid)

    def support(role):
        return {
            env: {label: len(groups[(role, env, label)]) for label in LABELS}
            for env in sorted({e for r, e, _ in groups if r == role})
        }

    train = support("TRAIN")
    validation = support("VALIDATION")
    independent_counts = {
        role: {label: len({sid for sid, (r, l, _) in specimen.items()
                           if r == role and l == label})
               for label in LABELS}
        for role in ROLES
    }
    reasons = []
    if root is None:
        reasons.append("R89_RAW_FILES_NOT_VERIFIED")
    # Train requires three environments, each class represented in all three.
    if len(train) < 3:
        reasons.append("R89_TRAIN_NEEDS_AT_LEAST_THREE_ENVIRONMENTS")
    if any(any(count < 1 for count in cell.values()) for cell in train.values()):
        reasons.append("R89_SOURCE_ENVIRONMENT_CLASS_GAP")
    # R88 protected VALIDATION requires >=2 physical groups per class/cell
    # in at least two separate operating environments.
    if len(validation) < 2:
        reasons.append("R89_VALIDATION_NEEDS_TWO_ENVIRONMENTS")
    if any(any(count < 2 for count in cell.values())
           for cell in validation.values()):
        reasons.append("R89_VALIDATION_NEEDS_TWO_BEARINGS_PER_ENV_CLASS")
    if min(independent_counts["TRAIN"].values()) < 3:
        reasons.append("R89_TOO_FEW_SOURCE_BEARINGS_PER_CLASS")
    if min(independent_counts["VALIDATION"].values()) < 4:
        reasons.append("R89_TOO_FEW_VALIDATION_BEARINGS_PER_CLASS")
    source_catalog = CATALOG[source]
    source_healthy_max = source_catalog["declared_unique_healthy_specimens"]
    if source_healthy_max is not None and source_healthy_max < 7:
        reasons.append("R89_CATALOG_HEALTHY_BEARINGS_BELOW_R88_SUPPORT")
    if source == "CWRU_PUBLIC":
        reasons.append("R89_CWRU_DISTINCT_HEALTHY_SPECIMENS_UNVERIFIED")
    # Even the no-blocker state is an eligibility claim about asserted IDs,
    # NOT confirmation of experimental physical independence.
    return {
        "schema": SCHEMA,
        "source": source,
        "catalog": source_catalog,
        "records": len(rows),
        "distinct_self_reported_specimens": len(specimen),
        "source_environment_support": train,
        "validation_environment_support": validation,
        "independent_specimen_count_by_role_class": independent_counts,
        "minimum_source_healthy_bearings": 3,
        "minimum_validation_healthy_bearings": 4,
        "minimum_combined_healthy_bearings": 7,
        "raw_file_sha256_checked_against_local_bytes": root is not None,
        "source_registry_is_self_reported": True,
        "duplicate_specimen_cross_split_detected": False,
        "sample_window_counts_used_as_independent_groups": False,
        "r88_strict_protocol_manifest_eligible": not reasons,
        "blockers": reasons,
        "actual_sensor_or_specimen_provenance_authenticated": False,
        "independent_generalization_proven": False,
        "test_or_final_accessed": False,
        "eligible_for_stable": False,
    }


def audit_manifest_file(path: str | Path, *, raw_root: str | Path | None = None):
    """Offline external dataset preflight; emits receipt, never loads labels as features."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(doc, dict) and set(doc) == {"schema", "source", "records"}
             and doc["schema"] == SCHEMA, "R89_INVALID_MANIFEST_DOCUMENT")
    return audit_external_bearing_cohort(
        doc["records"], source=doc["source"], raw_root=raw_root)
