# R83b: guarded neural growth and matched-fit TRAIN study

**Status:** development-only, stacked on R83 PR #20. No changes to Stable main.

## Delivered

- phm_design.industrial.self_organizing_guard.run_self_organizing_design_guarded:
  process-isolated Python multiprocessing spawn, supported on Windows/POSIX;
  honors a positive TaskSpec design.budget.max_wall_seconds, including worker
  startup, fitting and artifact return. Parent terminates worker on timeout,
  returns GuardedDesignTimeout, never a partially qualified model. GPU/child
  process quotas, robust sandboxing and independent qualification NOT provided.
- Moves the experimental temporal PyTorch model into a module-level class
  so its live trained state can be transferred out of a spawned worker and
  exported/replayed. The previous nested PyTorch class could not be pickled.
- r83_matched_budget.run_matched_source_train_study: bounded two-arm
  comparison on a single source TRAIN set. Fixed registered candidate
  portfolio (tcn, gru, tree) versus three sequential self-organized
  temporal candidates. Identical source physical-group/environment OOF splits,
  exactly 3 candidate slots, at most 3 * environment_count + 1 estimator
  fits per arm, identical predeclared design acceptance and wall guard.
  PyTorch candidate epochs are 8 in both arms (ExtraTrees has no epochs).
  Source TaskSpec's budget must ALREADY cover all these trials; no automatic
  relaxation. Applies the same fixed post-hoc summarization loss to all
  completed candidate OOF metrics.
- run_native_rw_matched_study: trusted R82 TRAIN-only staging, reviewed
  physical group map and pinned native-source binding verification before
  either model-selection worker can access observations. It does not read
  validation/TEST traces or rewrite fault labels.

## Why this is not a claim of equivalent compute

Estimator-fit limits and candidate slots are equal, but FLOPs, parameter
count, CPU time, tree node building and optimizer steps per fit are not
equal. The report explicitly marks those limitations and measures per-arm
host wall time. R83b is a fit-budget matched design-system comparison, not
a precise compute-normalized NAS benchmark.

The R81 fixed portfolio has a legacy score-ranking policy; R83 evolving
uses a preregistered weighted TRAIN-only loss. Their completed attempts
are also summarized using the SAME preregistered loss to make development
feedback visually comparable. No selection or ranking score from these
folds is an independent generalization estimate. The native RW data source
only tests one simulated apparatus under three predeclared inclination bins;
it is not cross-equipment OOD evidence.

Accept/refuse behavior: A strict refusal remains a legitimate outcome.
The runner records each arm's source TRAIN attempts and rejection rather
than softening acceptance criteria. A hard timeout is explicitly recorded,
not converted into a low diagnostic score. Operational errors fail the
study rather than count as scientific negative evidence.

## Reusing REAL Basilisk source data

R82 source campaign original Action run artifacts from 2026-10-16 are no
longer available in the repository's current Actions artifact listing (as
checked 2026-10-24). No source TRAIN experiment was run or claimed here.

Once a new actual Basilisk-native campaign produces a prepared manifest and
source hashes, execute the same source review/staging path:

~~~bash
cd source_product
python -m phm_design.industrial.r83_matched_budget \
  --task-spec examples/industrial/satellite_rw_native_offline_train.json \
  --development-manifest /source/prepared/development/dataset.json \
  --physical-map /source/physical_map_REVIEWED.json \
  --runtime-binding /source/runtime_binding.json \
  --native-source-checkout /source/SatAgentSimulation \
  --basilisk-wheel-factory /venv/lib/python3.X/site-packages/Basilisk/utilities/simIncludeRW.py \
  --output /output/r83b_native_train_study \
  --validate-only
~~~

Remove --validate-only only after native provenance, configuration
and sufficient per-class physical training-group support have been reviewed.
The runner will execute real fixed and evolving training in separate
process-isolated deadline-enforced workers. It reports TRAIN-only research,
not independent physical source attestation or FINAL performance.

## Tests

~~~bash
cd source_product
python -m pytest -q tests/phm_design/test_r83_self_organizing.py \
  tests/phm_design/test_r83b_guard.py \
  tests/phm_design/test_r83b_matched_budget.py \
  tests/phm_design/test_r81c_adaptive_feature_search.py
~~~

CI uses synthetic-only functional observations and mocked comparators
to test orchestration and impossible-to-bypass preflight constraints. A
separate spawned-worker test trains two actual PyTorch networks, returns
the artifact, and verifies exact-model replay across process boundaries.

## Remaining scientific milestones

1. Regenerate a documented Basilisk native RW TRAIN campaign without changing
   the predeclared physical regimes, labels or held-out roles. Never declare
   a stale artifact's missing bytes recreated without new receipts.
2. Execute the native R83b study under the frozen protocol, preserving
   failures and complete source-TRAIN candidate audits. Do not tune the
   screening threshold to manufacture an eligible model.
3. Register true independent equipment and physical-environment OOD
   experiments, including torque command-response observability.
4. Introduce cost-normalized and properly nested/independent comparisons
   before claiming any architecture-induced generalization advantage.
5. Finish independent R84 qualification before any Stable promotion.
