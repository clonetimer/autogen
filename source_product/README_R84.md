# R84: auto-designed neural diagnostic candidates via Pareto-guided adaptation

**Status**: development-only Draft PR #35, stacked on R83o; not Stable qualification.

## Why this is on the critical path

The aim of `autogen` is **automatic discovery and evaluation of generalizable
diagnostic algorithms for industrial equipment**. Satellite/Basilisk is one
high-integrity demonstration environment, not the scope or product definition.
A preflight physical/sensor gate is necessary to avoid invalid scientific
conclusions, but does not itself satisfy algorithm design. R84 returns to
architecture adaptation and source-feedback-driven algorithm discovery.

## Functional delta

- Original R83 self-organizing architecture search picks the lowest
  fixed weighted TRAIN-OOF loss, then the first unseen one-edit neighbor.
- R84 opt-in `feedback_pareto` uses an observed four-axis Pareto front:
  SOURCE TRAIN out-of-environment accuracy, worst environment/class recall,
  healthy false alarm, **measured** Torch parameter count.
- Among non-dominated *evaluated* parents, the deterministic parent score
  is `selection_loss + 0.03*log2(parent_params/min_observed_params)`.
  This is a predeclared **research heuristic**, not a tuned optimum.
- Neighbor selection favors largest minimum declarative topology difference
  to already tried candidates, then an analytical complexity proxy, then
  SHA tie-break. No performance estimate is generated for an unseen child.
- Existing topology grammar remains bounded (width 8/16/32, depth 1/2/3,
  kernels 3/5, mean/mean-max pooling, residual on/off). Training caps
  enforce actual parameter count and fit/CPU/wall budgets.
- Legacy `feedback` and `predeclared_static` remain default and unchanged.

## Experimental controls

The independent study must compare at least:
1. fixed classical/neural algorithm portfolio;
2. predeclared static neural architecture slate;
3. original R83 TRAIN-OOF-feedback first-neighbor search;
4. R84 TRAIN-OOF Pareto + novelty search.

With `run_matched_source_train_study(..., include_pareto=True)`, the new
fourth arm gets **the same TRAIN records, environment/group folds,
three slots, neural seed schedule, acceptance rule, wall deadline and fit cap**
as the others. Equal fits, equal epochs for neural arms, and equal wall
*limits* are not equal FLOPs, parameter counts, actual runtimes, hardware
usage or statistical independence. The fixed portfolio includes a tree and
GRU, so not all compute budgets are directly comparable; report separately.

Pure policy validation:
```shell
PYTHONPATH=source_product/src python -m pytest -q \
  source_product/tests/phm_design/test_r84_pareto_search.py
```
Full neural and orchestration validation (requires package dev dependencies):
```shell
cd source_product
python -m pip install -e '.[dev]'
python -m pytest -q \
  tests/phm_design/test_r84_matched_ablation.py \
  tests/phm_design/test_r84_pareto_training.py
```

These tests use *synthetic industrial-bearing scenarios*, not real sensor
measurements, new Basilisk native runs or independently sealed blind data.

## What the results mean

CI passing establishes code correctness on the covered synthetic contracts;
it does **not** demonstrate a favorable diagnostic accuracy, calibrated false
alarm rate, fault detection latency or cross-site/domain generalization.

Keep original R83f/R83g NO_ELIGIBLE_DESIGN where sensor/source observability
was not established. Preserve blocked role access and split leakage
restrictions. Do not merge research mechanisms into the limited 0.7.0 Stable
release without statistically independent confirmation.

## Next milestones for the actual autogen product

- R85: separate the sensor-agnostic TaskSpec, algorithm candidate grammar,
  modality-specific encoders and supervised source-TRAIN design controller.
  Expand beyond one TCN mutation family without privileging satellites.
- R86: preregistered multi-domain trials for bearings, motors, gearboxes,
  rotating machinery plus independent native satellite cases when physical
  sensing permits. Report group-wise worst domain recall, false alarms,
  event-level metrics, model size, training resources and repeated seeds.
- R87: make the auto-designed candidate export, replay and diagnostics
  directly usable through a non-programmer front end and stable algorithm
  API, with qualifications explicitly tied to supported equipment/domain.
