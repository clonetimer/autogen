# R101 pilot — byte-bound executable lineage (research only)

This incremental Draft implements a **real invocation** of a constrained, built-in linear JSON scorer and binds its outputs to the SHA-256 of the exact locally read byte files. It is a protocol fixture, not the trusted industrial model evaluator requested in Issue #56.

## What runs
- An allowlisted fixed one-feature, binary linear classifier with finite coefficients, explicit ordered labels, immutable SOURCE plan hash and model digest.
- Its only feature is `sum(file_bytes)/(255 * byte_length)`. This is intentionally **not** a vibration transform or industrial diagnostic model.
- The evaluator checks exact sealed record ID coverage; refuses symlinks, changed/missing files, SHA mismatches, unsafe model schema/coefficients, and mutated seal identities **before producing any predictions**.
- The evaluator itself creates prediction streams, so callers cannot inject arbitrary prediction arrays into this narrow fixture receipt.
- Wall/CPU process timing and platform information are recorded; GPU, peak memory, hardware parity, model-training provenance, and physical-device authentication are explicitly **not verified**.

## How to run
From `source_product`:

```bash
PYTHONPATH=src python -m pytest -q tests/phm_design/test_r101_sealed_fixture_execution.py
```

CI: [R101 fixture workflow](../../.github/workflows/r101-sealed-fixture.yml). Its raw files are synthetic 1 KiB binary fixtures. The model weights are predefined in a test, **not fitted on a device dataset**, and source-role separation is only self-declared. Seal creation timestamps, independent binary provenance, model training, feature-graph correctness, and immutable process isolation are not attested by this implementation.

## Missing to satisfy R101 Issue #56
1. Real trusted `torch_like` native fit/save/reload and reviewed artifact format, on an independently authenticated source-only training run.
2. Physical vibration/telemetry signal DAG and fold-fitted scaler serialized and replayed identically, with source code/environment hashes.
3. Independent steward reads actual raw device signals under restricted access and executes exact frozen estimator/transform artifacts. The current byte-mean feature is a protocol demonstration **only**.
4. Authenticated physical specimen identity, held-out cohort separation, guarded evaluation permission and cost parity. No artificial extra physical devices.
5. Signed or otherwise externally authenticated execution provenance. An attacker who can mutate both the seal and model can forge another locally consistent receipt.

This incremental work **does not** mean R101 is done, and does not satisfy the existing R99 independent industrial model-superiority gate. Preserve v0.8.0 Stable tag and its original 15 Basilisk qualification cases unchanged.
