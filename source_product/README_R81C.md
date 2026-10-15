# R81c Development Slice — TRAIN-only calibration and UNKNOWN

**Status:** Draft PR #18 stacked onto R81b PR #17. Version `0.8.0.dev23`.
**Stable `main` remains R60 / 0.7.0. This is not a Stable or qualified release.**

## What actually executes

The R80 TaskSpec compiler supports optional `design.calibration`:

```json
{
  "method": "temperature_selective",
  "max_selective_error": 0.20,
  "minimum_coverage": 0.50,
  "min_calibration_samples": 12
}
```

This contract requires `task.allow_unknown=true`. Older TaskSpecs omit calibration and retain
the previous uncalibrated research behavior, including the original fit counts.

The common R81 executor now performs, **for each predeclared trial**:

1. An **outer** leave-entire-TRAIN-environment-out split, used only for trial ranking.
2. Inside that outer training partition, an **inner** leave-environment-out OOF fit.
   Inner OOF probabilities and inner OOF labels alone fit a temperature and conservative
   selective-risk threshold. Outer holdout labels cannot affect the calibrator.
3. Outer-training model fit and prediction through that frozen calibrator on the outer holdout.
   A rejected prediction is `__UNKNOWN__` (numeric sentinel `-1`), distinct from
   a missing fold prediction (sentinel `-2`).
4. Source TRAIN OOF evaluation of original correctness, worst-environment recall,
   healthy false alarm, UNKNOWN rate, coverage, and accepted-only accuracy.
   UNKNOWN is **not** counted as correct, including for healthy examples.
5. After an eligible trial wins and the entire portfolio is compared, a fresh **inner OOF
   calibration on all TRAIN** is fitted, followed by exactly one full-TRAIN refit.
   The learned calibrator is held inside `IndustrialDesignArtifact`.

Both inner fitting and outer fitting count against `max_fits` (factorized heads count
individually). Exhausting the predeclared budget **refuses delivery**.

`artifact.predict(...)` returns labels including possible `__UNKNOWN__`.
`artifact.predict_decisions(...)` returns label, calibrated confidence, and abstention.
`artifact.export(directory)` records `calibration_state` in `audit.json`, with the
model hash and `independent_qualification=false`.

## R81c second slice — real composed feature programs

The optional `design.feature_programs` field provides explicit bounded
multi-atom programs, each with **1–3 unique registered atoms** and the
`concat` composition. This is the original `open_design.Program` plus
`generic_registry().transform` execution, not a new set of satellite-only
handcrafted features.

```json
{
  "feature_programs": [
    {"atoms": ["temporal_bins", "random_kernels"], "composition": "concat"},
    {"atoms": ["difference_kernels", "temporal_bins"], "composition": "concat"}
  ]
}
```

The TaskSpec compiler compiles these into the *same* `candidate_grid` as
learner/backbone, objective and topology selection. The full grid must fit
`candidate_max` **before** any training. Classical and factorized learners
receive the exact concatenated vectors; fitted preprocessing remains inside
each training fold. The final refitted model uses the identical feature
transform during deployment prediction and records a deterministic program
SHA-256 in `audit.json`.

If `feature_programs` is present and `representations` is omitted, there
are no implicit extra classical feature candidates. Both may be explicitly
declared to compare single-atom and multi-atom programs. Learned sequence
models continue to use their established `learned_sequence` representation.

**Not yet implemented:** availability-aware `available_stack` or
`available_average` requires an inner physical-group-OOF per-atom training
contract, so such programs are explicitly rejected rather than faked by
concatenation. Adaptive `FeedbackPlanner` mutation remains in the older
`open_design` path; this slice integrates the safe, predeclared program
grammar rather than implying closed-loop program proposal has been unified.
Neural learned front-end composition is also not implemented.

## R81c third slice — source TRAIN OOF feedback-driven program search

The optional `design.adaptive_feature_search` enables a **bounded**, genuinely
feedback-guided selection order, integrated into the **same**
`run_design` executor. It extends the existing `open_design.FeedbackPlanner`
with a finite `bounded_feedback` mode. Two declared singleton seeds are
evaluated first; subsequent candidates are chosen based on prior completed
**TRAIN-source LOEO OOF class errors**, rather than a fixed list order.

```json
{
  "design": {
    "candidate_max": 3,
    "compare": ["linear"],
    "objectives": ["ERM"],
    "topologies": ["direct"],
    "adaptive_feature_search": {
      "atoms": ["temporal_bins", "random_kernels", "difference_kernels"],
      "max_programs": 3
    }
  }
}
```

Add this `adaptive_feature_search` block within the existing full TaskSpec,
keeping the other required fields. This declares exactly **3 proposals** from
a fixed **7-program pool** (all 1–3 atom subsets). The pool is fully hashed
before training and accepts only registered existing atoms, a single
classical learner/template, one diagnosis topology, and ERM. Adaptive search
cannot also declare `representations` or static `feature_programs`.
Every proposal actually trains and competes on the same source TRAIN
environment-OOF folds; the next program depends on the preceding OOF loss.

`adaptive_trace` records each proposed program, parent program, source TRAIN
class errors, risk loss, proposal step, immutable candidate-grid index,
rejected trials and untested pool count. Selection and refit are still subject
to the **same** bounded fit/CPU budgets and prior acceptance/refusal policy.

**Scientific restriction:** adaptive reuse of source TRAIN heldout folds makes
their metrics *optimization feedback*, never an unbiased independent test.
Unexplored programs in the declared pool are not certified inferior. Results
are development selection only, with no claim of online causality or real
machine OOD generalization. This slice does **not** yet implement adaptive
learner/topology mutation or availability-aware per-atom ensemble training.

```bash
cd source_product
python -m pytest -q tests/phm_design/test_r81c_adaptive_feature_search.py
```

### Trusted research inference replay (not independent retraining)

The `phm_design.industrial.artifact_io.load_trusted_artifact` API can
rehydrate an exported model with its selected feature program, TaskSpec and
calibration state. It validates TaskSpec/grid/trial hashes and the on-disk
model SHA-256 *before* deserialization:

```python
from phm_design.industrial.artifact_io import load_trusted_artifact

artifact = load_trusted_artifact("./my_train_model", trust_pickle=True)
labels = artifact.predict(deployment_measurements)
```

`trust_pickle=True` is an explicit security boundary: joblib/pickle
deserialization can execute attacker-controlled code. Use it **only** on
artifacts from a source you already trust. Checksums detect corruption,
not adversarial tampering or authorship. This restores trained-model inference
and does not reproduce training from independently stewarded source data;
that would require a separate controlled replay pipeline.

Reproducibility evidence now includes source TaskSpec and TRAIN data hashes,
candidate grid hash, selection-trial hash, selected feature-program hash,
calibration policy hash, acceptance policy hash and binary model hash.
These are audit checksums, **not** independent data-provenance attestation.

```bash
cd source_product
python -m pytest -q tests/phm_design/test_r81c_feature_program_composition.py
```

## Limitations and scientific boundary

- Temperature selection and selective-error bound use **TRAIN source environments only**.
- The bound is descriptive within those source folds, **not** a distribution-free OOD guarantee.
- No independent FINAL labels or revealed R71/R78 target traces may be accessed.
- The demo synthetic data are functional tests only; they do **not** demonstrate diagnosis
  observability or generalization on real industrial machinery or Basilisk physics.
- The deployment serializer is research-only and requires trusted joblib sources.
- The trial audit now measures per-candidate process CPU time, host wall time,
  inference host wall time, charged head-fit count and refit count. Failed
  multi-head fits conservatively charge their full planned head count to avoid
  undercounting partially attempted training. These are
  **best-effort machine-dependent measurements**, not FLOPs-equivalent budgets.
- R81c still does not provide availability-aware feature-program ensembles,
  parameter/FLOPs/peak-memory ledgers, full provenance replay, or a sealed FINAL steward.

## Local regression

```bash
cd source_product
python -m pip install -e '.[dev]'
python -m pytest -q tests/phm_design/test_r81a_unified_design_engine.py \
  tests/phm_design/test_r81b_feature_and_variable_heads.py \
  tests/phm_design/test_r81c_train_only_calibration.py
```

GitHub Dev Guard executes these under Python 3.11 and 3.12. A green software CI does
not imply an eligible physical diagnosis algorithm or independent performance qualification.
