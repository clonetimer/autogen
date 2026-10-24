# R83c — Frozen, source-independent VALIDATION steward (development only)

**Status:** research feature stacked on R83b PR #21; not Stable or R84 FINAL
qualification. Test and README use synthetic fixtures. There are no new
actual Basilisk simulation or equipment outcomes in this change.

## Why a separate evaluator

R83b's three search arms (fixed TCN/GRU/ExtraTrees, preregistered static
temporal structures, feedback self-organization) use the same TRAIN source
folds. Those leave-one-environment-out folds guided the neural proposal and
chosen architecture, so reporting those OOF metrics as evidence of
generalization would be double-dipping.

R83c adds a **read-only frozen-model VALIDATION pass** for an *external
reviewer's* independently prepared physical groups. It runs all three
previously fitted models on the *same* records. It is a separate API that
does not train, propose topologies, select models, or edit the frozen weights.
The original TRAIN result now records SHA-256 hashes of each physical-group
identity and trajectory as well as the complete source-TRAIN observations
digest. The evaluator refuses new records that reuse those exact group IDs
or observations, as well as missing model receipts or differing source-data
digests across frozen models. Hash agreement is not provenance attestation.

The strict validation contract requires:
- role VALIDATION only: never TRAIN, TEST or FINAL; this evaluator must not be
  given sealed FINAL data. Expected metadata is role, label, registered
  physical group keys and operating regime axes only.
- TaskSpec sensor channels and time must be unchanged; no label/truth column
  may appear in observations. No physical group may straddle environments,
  and no duplicate telemetry can masquerade as another independent group.
- At least three independently declared heldout environments and two
  distinct physical groups in every environment/class cell. Reports flag
  whether the **recommended** five groups per cell were obtained. Five is
  not an automatically statistically sufficient sample.
- Every frozen model's original source-TRAIN metadata and observations
  SHA-256 must match the same R83c study receipt. Frozen artifact manifests
  are checked BEFORE the fixed portfolio's joblib loading.
- Fixed portfolio joblib loading is **explicitly trusted only**
  (trusted_fixed_pickle=True) and should happen only in an isolated,
  reviewer-controlled environment. Hashes alone do not prevent a malicious
  author from modifying the artifact and receipt together.

The evaluator produces per-arm physical-group macro accuracy,
trajectory-weighted accuracy, worst environment/class recall and healthy
false alarm rate. Paired delta in group macro accuracy (adaptive vs static
and adaptive vs fixed) is summarized using a **deterministic paired
group-cluster bootstrap** of 499 replicates and seed 83021. This is a
descriptive interval, not a confirmatory hypothesis test. Exact paired
groups for every architecture avoid differences due to heldout sampling.
No p-value, improvement, engineering calibration or statistical
significance claim is automatically produced.

## Example (trusted reviewer code, not a TRAIN worker)

~~~python
import json
from phm_design.industrial.r83_frozen_validation import (
    evaluate_r83_frozen_validation,
)

source_train_receipt = json.load(open("native_train_study/research_report.json"))
# The benchmark receipt is nested inside its research_report.json
train_report = source_train_receipt["comparison"]
task_spec = json.load(open("source_product/examples/industrial/"
                           "satellite_rw_native_r83_matched_train.json"))
# reviewed_validation_records must be built by a separate source-intake
# steward. Every metadata.role must equal VALIDATION. Do not pass TEST/FINAL.
validation = evaluate_r83_frozen_validation(
    train_report,
    task_spec,
    "native_train_study/research_models",
    reviewed_validation_records,
    trusted_fixed_pickle=True,
)
assert validation["independent_final_test_performed"] is False
assert validation["model_fit_or_selection_performed"] is False
~~~

**Limits:** This API accepts reviewed VALIDATION records in memory, not a
free-form CSV/JSON path; full source-native file review/independent
attestation remains a separate external preparation duty. It does not
automatically open any VAL/TEST/FINAL trees. Independent physical origin
requires an external evidence steward; hash disjointness alone cannot
prove it. Running the evaluator repeatedly and tuning on the results
would destroy its independent-validation interpretation. A sealed R84
FINAL test with separate data custody remains required before release.

## Verification

~~~bash
cd source_product
python -m pytest -q tests/phm_design/test_r83c_frozen_validation.py \
    tests/phm_design/test_r83b_matched_budget.py \
    tests/phm_design/test_r83b_guard.py
~~~

The GitHub Actions job checks Linux Python 3.11/3.12 and Windows 3.11
using synthetic-only TRAIN/VALIDATION records and actual CPU-trained
model artifacts. A passed CI job **is not** evidence that native Basilisk
source was run or that any architecture generalizes better.

## Next gating work

- Establish a reviewed, saved VALIDATION-only manifest and mechanical
  source integrity check; keep it outside the TRAIN-search worker.
- Re-run actual Basilisk-native RW source campaign with balanced operating
  regimes and class groups, preserving native physics provenance, and
  keep a separately stewarded independent validation cohort.
- Record observed CPU wall, inference latency, model capacity and ideally
  FLOPs/GPU-hours; equal fit counts only provide a *search-budget* control.
- Repeat the predeclared experiment across independent random seeds and
  distinct physical sources; use a sealed R84 FINAL steward for independent
  release qualification, not this development evaluator.
