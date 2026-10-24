# R83e: REAL pinned Basilisk small-source pilot

**Status:** stacked research PR #24 on R83d #23. This is a native-physics
integration test, not a diagnostic-model performance claim, cross-equipment
OOD evaluation or R84 FINAL qualification.

## Integration bug resolved

R83d scientifically predeclares ZERO TEST/FINAL cases. The original R82 RW
adapter unconditionally instantiated an empty TEST Dataset and correctly
rejected that input. R83e adds only the explicit
\`prepare(..., allow_empty_holdout=True)\` switch for the R83d study.

- General/default calls without this flag still reject empty TEST data.
- The flag cannot bypass any Basilisk source-gate/paired-physics checks.
- An intentionally empty holdout dataset.json remains on disk and is NEVER
  loaded or presented as a qualified evaluation cohort.
- Six channels remain fixed wheel indices: three wheel speeds and three
  *commanded* (not applied) motor torques; no truth labels or hidden actuator
  effects become model inputs.

## Actual native physical smoke workflow

[Native six-case PR pilot](../../.github/workflows/r83e-native-rw-paired-pilot.yml)
runs only after Linux Python 3.11/3.12 + Windows Python 3.11 software
adapter tests pass. The physical job on Ubuntu 24.04 does all of:

1. Pins SatAgentSimulation exactly to
   \`1f709beeae97baec65db9ca13ae373d88728f87d\`.
2. Downloads the original Basilisk \`bsk==2.11.0\` wheel and rejects any
   mismatch from SHA-256
   \`d51fe1ffa1b0b03dc415034b210497b28a2504428d4d8e71b481542a08c08a0a\`.
   Only wheel *metadata* compatibility fields are adjusted, and the native
   binary/file payload is verified bytewise unchanged.
3. First freezes the **same exact R83d 126-case full task plan**. From
   this original plan, selects the predeclared TRAIN pair with pair index
   \`000\` for each of friction increase, jam and torque authority loss:
   **three physical nominal/fault pairs, six physical cases in total**.
   Each source pair retains the same seed, physical settings and fault
   mechanism. No selection uses simulation metrics or labels for inference.
4. Actually executes the pinned SatAgentSimulation campaign generator with
   \`dry_run=False\`. Checks real runtime manifest, Basilisk source version,
   engineering physics diagnostics, six-channel raw traces, paired source
   parameters and native source-contract hashes. Never substitutes a Python
   kinematic proxy.
5. Prepares a TEST-free six-channel research dataset with explicit opt-in.
   Archives original trace.csv, case specs, runtime reports and the
   complete failure log (even if validation refuses a source).

**Scientific scope:** This pilot checks the genuine physical and data
plumbing, not how accurate any of the models are. It covers only the
low-inclination preregistered source and 1 pair per fault, **not** all
three R83d inclinations, all 126 cases or any independent VAL/TEST. It
must not be used to claim improved diagnosis from self-organization.

The full, costly 126-case study remains a separate manual-only
[research workflow](../../.github/workflows/r83d-real-native-balanced-campaign.yml)
and has a fixed, previously declared search/selection/VAL protocol.
The R83e pilot must NOT cause changes to that benchmark's class quotas,
regimes, candidate cap, acceptance thresholds, seeds, or source-role
boundaries.

## Reproduce software-only adapter regression

~~~bash
cd source_product
python -m pytest -q \
  tests/phm_design/test_r83e_native_integration.py \
  tests/phm_design/test_r82_native_campaign_gate.py
~~~

These tests use fabricated numeric CSV/source receipts ONLY. Passing those
tests is not proof that real Basilisk actually ran. For that claim, require
the separate GitHub Actions native job to finish and inspect its original
real-case evidence artifact. A failed native case, payload mismatch or
diagnostic candidate physics refusal is recorded as a failure, not hidden
as a successful software mock.
