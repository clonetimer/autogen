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
- r83_matched_budget.run_matched_source_train_study: bounded three-arm
  comparison on a single source TRAIN set. Fixed registered candidate
  portfolio (tcn, gru, tree), three preregistered **static**
  temporal topologies, and three adaptive temporal candidates. Identical source physical-group/environment OOF splits,
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

## R83c fairness audit and new mandatory study contract

**Do not reuse the original R82 two-candidate TaskSpec for R83.** Use the
new frozen research declaration
`examples/industrial/satellite_rw_native_r83_matched_train.json`,
which explicitly declares three candidate slots and a sufficient fit
budget. The benchmark never increases a caller's two-slot search budget
implicitly. All TaskSpecs are compiled before any training.

Both arms now select the fitted champion using **exactly the same**
predeclared source TRAIN loss:
0.45 * (1 - accuracy)
+ 0.40 * (1 - worst regime/class recall)
+ 0.15 * healthy false alarm.
The fixed arm's previous legacy selection ranking remains the default for
every non-R83 caller; this comparison opts in explicitly. Candidate seed
schedule (base seed + candidate index, held constant across all OOF folds
and the final refit) is identical in both arms. This does NOT mean model
initializations are mathematically identical for different architectures.

Independent physical group support is reviewed at the environment×fault-class
level. Every environment/class cell must contain **at least 2 distinct
physical groups**, and within-environment class counts cannot differ by
more than 2×. Data with insufficient class support are refused **before**
either model receives observations. The known R82 high-inclination class
with a single example therefore **cannot be claimed as a fair R83
benchmark**; its distribution must be regenerated rather than post-hoc
reweighted. A minimum of **5 groups per cell** is recorded as the
*recommended confirmatory experiment target*, not a claim of
statistical power or sufficient confidence interval coverage.

The result now contains a `fairness_gate`. If either arm fails to
evaluate all three declared source TRAIN candidates, the report explicitly
sets `source_train_comparison_allowed=false`. Even when true, it is only
an *optimization-feedback comparison*, not unbiased OOD/generalization
evidence. Training both arms under equal fit caps is a baseline software
control, not an equal-GPU-hours, wall-time or FLOPs study.

## Same-family isolation: static versus self-organizing

The three-arm protocol is necessary to distinguish architecture adaptation
from changing the model family:

1. Fixed broad-family portfolio: 1 TCN + 1 GRU + 1 ExtraTrees.
2. **Static neural ablation:** 3 fixed, preregistered temporal genomes.
3. **Feedback neural search:** exactly the same first 2 genomes as the
   static arm; third is chosen *only* from earlier TRAIN OOF feedback.

Both neural arms have the same fold partitions, original source measurements,
8 training epochs per candidate, candidate seed schedule, three proposals
and estimator-fit cap. The predeclared static genome IDs never depend on
labels, accuracy or held-out environment scores. The only intentional
search-control difference is whether previous TRAIN OOF scores guide the
third proposal; the resulting third architectures may still have different
parameter counts and compute cost.

Crucially, this ablation is still **development design-feedback evidence**:
the search reuses the source heldout folds. A separate, sealed evaluation
on entirely unseen physical groups is required for claims that adaptation
actually improves generalization. The broad-family portfolio is a systems
baseline, not a controlled proof of the adaptivity effect.

## Why this is not a claim of equivalent compute

Estimator-fit limits and candidate slots are equal, but FLOPs, parameter
count, CPU time, tree node building and optimizer steps per fit are not
equal. The report explicitly marks those limitations and measures per-arm
host wall time. R83b is a fit-budget matched design-system comparison, not
a precise compute-normalized NAS benchmark.

The R83 research comparison **opts into the same source TRAIN-only
weighted loss for candidate ranking and final refit in every arm**. The
legacy R81 selection policy is unchanged for all other callers. No selection or ranking score from these
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
  --task-spec examples/industrial/satellite_rw_native_r83_matched_train.json \
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
