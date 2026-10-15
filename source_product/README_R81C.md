# R81c Development Slice — TRAIN-only calibration and UNKNOWN

**Status:** Draft PR #18 stacked onto R81b PR #17. Version `0.8.0.dev21`.
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

## Limitations and scientific boundary

- Temperature selection and selective-error bound use **TRAIN source environments only**.
- The bound is descriptive within those source folds, **not** a distribution-free OOD guarantee.
- No independent FINAL labels or revealed R71/R78 target traces may be accessed.
- The demo synthetic data are functional tests only; they do **not** demonstrate diagnosis
  observability or generalization on real industrial machinery or Basilisk physics.
- The deployment serializer is research-only and requires trusted joblib sources.
- The trial audit now measures per-candidate process CPU time, host wall time,
  inference host wall time, actual head-fit count and refit count. These are
  **best-effort machine-dependent measurements**, not FLOPs-equivalent budgets.
- This first R81c slice does not yet provide richer feature-program composition,
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
