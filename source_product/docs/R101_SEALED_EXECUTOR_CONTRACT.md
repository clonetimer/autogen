# R101 — Sealed local execution receipts (experimental stage A)

**2026-10-26 · Draft research.** This increment is stacked on R100 Draft #55 and closes only the **contract-fixture portion** of R101 issue #56. It does **not** qualify real torch_like loading or independent industrial diagnostic performance. Neither \`v0.8.0\` artifacts nor original Basilisk 15-case qualification change.

## The implemented boundary

\`phm_design.industrial_contracts.sealed_executor\` adds:

- \`seal_fixture_execution(frozen_R99, trial, artifact_dirs)\`: verifies R99 frozen comparison identity; admits only three explicitly structured, **nonexecutable JSON fixtures** per contender (\`model.json\`, \`preprocessor.json\`, \`graph.json\`); SHA-256 records exact bytes and refuses extras, changed directories, untrusted joblib/pickle, or identical challenger/baseline bundles. This is a hash manifest, not a signature or a trustworthy freeze-time record.
- \`execute_fixture_pair(..., fixture_runner=..., explicit_fixture_mode=True)\`: trusted-data-steward **test harness** checks the full manifest with R93, hashes each original raw file with R99 and invokes a prewired, in-process fixture callback on the same sorted VALIDATION raw recordings for the pre-registered challenger and baseline. Each invocation binds record ID, raw-file hash, artifact-bundle hash and returned label, then hashes the prediction stream.
- The steward computes R99 paired physical-group metrics using the actual callback outputs, not separately uploaded arrays. Receipts include aggregate process wall/CPU seconds, Python-traced peak allocation and basic self-reported platform context, with limitations.
- Refusal cases include changed/artifact files, unsafe native pickle, swapped IDs, altered raw measurements, cross-role physical asset reuse, invalid/extra prediction output, callback exceptions, and a file mutated mid-execution. No partial score is returned after a failure.

### Crucial trust limitation

**Fixture callbacks are arbitrary in-process Python functions in a test environment**, not sandboxed or cryptographically attested native model executables. They have ambient host privileges. The strict JSON bundles explicitly assert \`not_a_trained_model=true\`; the implementation does not implement or train a classifier and does not deserialize models.

The callback receives only:

\`\`\`python
(artifact_id, record_id, original_measurement_bytes, nonexecutable_bundle_documents)
\`\`\`

It does not receive the full manifest or validation labels as a function argument. **This is an interface-level data-flow restriction only**; it cannot stop an untrusted callback reading other files or secretly fitting a model. Run fixtures only in a controlled CI/research test environment, never with private industrial data or untrusted callback code.

The receipt is deliberately unequivocal:

- \`predictions_cryptographically_bound_to_real_model_execution = false\`
- \`runner_implementation_sha256_authenticated = false\`
- \`fitted_model_native_bytes_safely_loaded = false\`
- \`independent_real_device_superiority_proven = false\`
- \`eligible_for_stable = false\`

Passing synthetic fixture tests proves that the hash/identity/refusal logic is implemented and regression-tested. It **does not** pass R101's later real-provider trusted-execution acceptance.

## Example (for protocol tests only)

\`\`\`python
from phm_design.industrial_contracts.sealed_executor import (
    seal_fixture_execution, execute_fixture_pair)

# bundle_dirs includes TWO frozen R99 artifact IDs and exactly the three
# nonexecutable fixture JSON files per directory.
seal = seal_fixture_execution(frozen_comparison, trial, bundle_dirs)
receipt = execute_fixture_pair(
    seal, frozen_comparison, equipment, full_trusted_manifest, trial,
    original_file_paths, bundle_dirs,
    fixture_runner=approved_in_process_TEST_fixture,
    explicit_fixture_mode=True)
assert receipt["eligible_for_stable"] is False
\`\`\`

Default execution (without \`explicit_fixture_mode=True\`) is **refused**, because a production native execution adapter is not ready.

## Resource interpretation

Only combined **fixture callback** wall and process CPU are sampled. These include any operations performed inside that callback. They do not establish feature extraction, model fit, final inference parity, memory for NumPy/C/GPU, wall-time deadlines or resource enforcement. \`tracemalloc\` only covers Python-traced allocations; the platform fingerprint is a self-reported string, not an independently authenticated machine identity.

## Continued R101 acceptance / source of future work

1. Review and explicitly authorize a real \`torch_like\` provider adapter and **trusted** source model persistence. Existing R94 \`estimator.joblib\` is pickle-based, so it must not be loaded from untrusted bytes by this module. Consider reviewed safe serialization/conversion and allowlisted model-provider loader only in a restricted environment.
2. Produce independent SOURCE-model fit + frozen preprocessing/graph artifacts; tie artifact digests to **actual invoked** estimator class, implementation and predictions. Refuse when native loading, model identity or exact dependency digests cannot be established.
3. Execute a physically distinct, authenticated untouched device cohort on a separately supervised environment, with full per-candidate compute receipts and an authentic fixed baseline. Existing R99 paired sign-test receipts cannot prove physical bearer identity, causal improvement or performance superiority.
4. Keep \`CALIBRATION\`/\`VALIDATION\`/\`FINAL\` sensor observations, labels, model-selection feedback and runtime outside the autogen SOURCE design worker. Avoid automatic retries that leak held-out results to the designer.
5. Keep all unknown/unauthenticated/insufficient evidence **NO-GO**. No safety-critical controls, AAS alert-threshold changes, new classifier implementations or retrospective v0.8.0 requalification.

## Repeatable tests

In \`source_product\` root:

\`\`\`bash
python -m pip install numpy==2.3.5 scipy scikit-learn==1.8.0 pytest==9.0.2
PYTHONPATH=src python -m pytest -q \
  tests/phm_design/test_r99_design_review.py \
  tests/phm_design/test_r100_guarded_revision.py \
  tests/phm_design/test_r101_sealed_executor.py
\`\`\`

GitHub [R101 research workflow](../../.github/workflows/r101-sealed-executor.yml) tests Linux CPython 3.11/3.12 and Windows CPython 3.11. The fixtures are **generated synthetic files** and a deliberately predictable callback, not CWRU/Paderborn measurements or native fitted estimator replay.
