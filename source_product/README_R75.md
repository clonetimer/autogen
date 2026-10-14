# PHM AutoDesign R75 — environment-exclusive inner model selection

**Version:** `0.8.0.dev14` — DEVELOPMENT ONLY. Inherits the R74 source and retains all R60 Stable qualification records byte-for-byte; nothing here expands qualification.

## Why this exists

R71 previously showed that AutoDesign could overestimate transfer to unseen environments despite excellent validation on known environments. In R74's experimental multi-head design graph, the `worst_environment` option calculated per-environment errors from **world-grouped** inner folds: a held-out physical world could share its environment family with training worlds. This can bias model comparison toward environment-specific cues.

## Changes

- The **inner head selection** stage now uses `LeaveOneGroupOut` across entire *TRAIN* environment families when `selection_mode='worst_environment'`. The outer holdout remains as configured, and the inner folds record environment and world separation in each completed candidate's audit.
- Per-head candidate fit failures retain original backend/feature errors, rather than referencing OOF predictions that never existed.
- Added `localization_all_faults_joint_accuracy`, `mechanism_all_faults_joint_accuracy`, and `all_faults_denominator`; they count upstream detection misses and rejected low-confidence branches as failures instead of dropping these difficult cases. Legacy conditional metrics remain separately named for backwards-compatible research analysis.
- The old `mean` (world-group) search and task schema remain unchanged. No special online, RUL, CUDA or new ontology support is implied.

## Evidence rules

- Only synthetic TRAIN fixtures for the new tests; no FINAL-ID/FINAL-OOD or R71 revealed worlds are consulted for candidate design.
- Both inner and outer scores derived from development data are **selection/screening evidence**, not independent target-domain qualification. Even a nested evaluation is compromised for future independent claims if feedback is repeatedly used for new design choices.
- The 0.7.0 Stable remains on `main`; this R75 branch is intended as a stacked Draft PR targeting R74's development branch.

## Verification

```bash
PYTHONPATH=src python -m pytest -q tests/phm_design/test_r75_environment_exclusive_search.py
PYTHONPATH=src python -m pytest -q tests/phm_design/test_r69_task_graph.py tests/phm_design/test_r72_leakage_and_robust_search.py
```
