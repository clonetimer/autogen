# R89 — physical bearing provenance and independent-validation feasibility gate

**Status:** research Draft. No independent real-data model performance has
been measured in R89. Neither a green test suite nor a plausible SHA digest
establishes an independently generated real laboratory cohort.

## The verification problem found in R88

The existing conservative R88 neuro-fuzzy source/VALIDATION study requires:

- At least **3 separate source TRAIN operating environments**, with at least
  one independent bearing **per class and environment**; the held-out
  environment folds retain 2 other environments for fitting.
- At least **2 separate VALIDATION environments**, with at least **2
  independent physical bearings per class per environment**.
- Physical bearings (not windows, recordings or MAT filenames) must be
  disjoint between SOURCE TRAIN and VALIDATION.
- Therefore, with binary healthy/fault classification, the protocol needs
  at least **3 + 2 × 2 = 7 distinct healthy specimens** before any repeated
  seed or statistical confidence requirement.

**Actual publicly documented source limitation:** [Paderborn KAt Bearing
DataCenter](https://mb.uni-paderborn.de/en/kat/research/bearing-datacenter/data-sets-and-download)
documents **6 undamaged bearings** and 26 damaged bearings across 4
operating conditions and 20 recordings per condition. Its **6 distinct
healthy specimens cannot satisfy this exact 7-healthy R88 split**, regardless
of the number of vibration windows or recordings. Splitting same-bearing
windows across roles would manufacture independence and is forbidden.
Paderborn source materials are [CC BY-NC 4.0](https://mb.uni-paderborn.de/en/kat/research/bearing-datacenter);
check terms before transferring data, and do not commit original raw signals.

[CWRU](https://engineering.case.edu/bearingdatacenter/download-data-file)
has motor-load 0/1/2/3 recordings of normal/faulted conditions, but these
motor-load files **are not evidence of separate physical healthy bearings**.
Do not reinterpret 4 load files as 4 independent laboratory specimens.
Consequently CWRU cannot be prequalified under the conservative R88
physical-specimen protocol without independent physical specimen metadata.

## R89 implemented preflight

`industrial/r89_external_cohort_audit.py` provides a fail-closed source
manifest auditor. It validates:

- Exact schema and original specimen identifier, role TRAIN/VALIDATION,
  source environment, declared healthy/fault truth (metadata **only**) and
  raw/derived observation SHA-256.
- One physical specimen may have multiple recordings, but cannot change
  role, label or environment under the existing R88 leave-environment-out
  design; no raw trace may be assigned to another physical specimen.
- Normalization of allowed `.mat` filenames: reject absolute paths and
  parent-directory traversal.
- With `raw_root`, read **actual supplied local MAT bytes** and verify SHA
  digests with incremental hashing. This is file-integrity verification, not
  physical provenance attestation.
- Missing `raw_root` forces `R89_RAW_FILES_NOT_VERIFIED` and a strict
  **NOT READY** receipt, even if all self-declared specimen IDs look distinct.
- No acceptance unless predeclared TRAIN/VALIDATION physical support holds.
  The Paderborn 6-healthy inventory triggers
  `R89_CATALOG_HEALTHY_BEARINGS_BELOW_R88_SUPPORT`. CWRU triggers
  `R89_CWRU_DISTINCT_HEALTHY_SPECIMENS_UNVERIFIED`.
- Every receipt always declares `independent_generalization_proven=false`,
  `actual_sensor_or_specimen_provenance_authenticated=false` and
  `eligible_for_stable=false`.

### Minimal local manifest

```json
{
  "schema": "phm.r89.external-physical-cohort.v1",
  "source": "EXTERNAL_REVIEWER_DOCUMENTED",
  "records": [
    {
      "record_id": "lab_run_0001",
      "role": "TRAIN",
      "diagnosis_label": "healthy",
      "specimen_id": "bearing_0001",
      "environment_id": "speed1500_load1",
      "raw_file": "bearing_0001.mat",
      "raw_sha256": "REPLACE_WITH_REAL_64_DIGIT_SHA256",
      "observation_sha256": "REPLACE_WITH_64_DIGIT_OBS_SHA256"
    }
  ]
}
```

This is a **syntax example, not an eligible experiment**. The full
manifest requires multiple genuinely distinct bearings in every declared
class/environment cell. Individual file SHA must match actual local
contents and derived-observation SHA must follow a separately frozen sensor
normalization protocol. Role names alone prove nothing.

```python
from phm_design.industrial.r89_external_cohort_audit import audit_manifest_file
receipt = audit_manifest_file("cohort.json", raw_root="C:/real_sensor_mat")
assert receipt["r88_strict_protocol_manifest_eligible"], receipt["blockers"]
# The assertion is about self-reported metadata and actual bytes, NOT
# independent physical authenticity or scientific generalization.
```

### Reproduce R89 CI guards

```shell
cd source_product
python -m pip install numpy scipy pytest
PYTHONPATH=src python -m pytest -q tests/phm_design/test_r89_external_cohort_audit.py
```

Tests produce very small fabricated MATLAB files via SciPy, change their
bytes after fingerprinting, mimic cross-split bearing reuse and show
fail-closed Paderborn/CWRU outcomes. They **do not** download or evaluate
authentic real bearing recordings.

## Next real study choices (preregister; don't change frozen R88 evidence)

**Option A: Keep strict R88 limits.** Obtain a legitimately different
physical-bearing dataset, reviewed specimen/source identifiers and at least
7 independent healthy specimens with balanced class and domain support;
audit sensors, use real source records, freeze, and let a reviewer operate
VALIDATION without model feedback. Eventually add separately governed
blind TEST/FINAL.

**Option B: A new prospective lower-support protocol.** If Paderborn is the
chosen first physical dataset, preregister a *new* protocol with lower
VALIDATION cell support, retain strict bearing-disjoint splits, multiple
predeclared seeds and correctly qualified wide uncertainty intervals.
**Do not silently weaken R88**, turn reused recordings into new bearings,
or claim independently confirmed improvement merely because Paderborn
measurements are real.

Across either option: compare non-fuzzy linear/ML baselines, fixed and
adaptive fuzzy rules, frozen and gradient-learned memberships. Report
training FLOPs, actual CPU/wall time, ridge refits, parameter count and
all negative/abstained trials. Satellite native Basilisk sensor quality
still needs independent attestation; synthetic Python simulation is not
physical satellite evidence.

Do not merge into Stable; do not touch frozen R83f/R83g negative outcomes.
