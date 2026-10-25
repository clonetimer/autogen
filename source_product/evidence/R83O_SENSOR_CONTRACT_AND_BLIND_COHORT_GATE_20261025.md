# R83o — RW sensor contract and blinded-cohort entry gate (research protocol)

Status: **PROPOSED / NOT QUALIFIED**. Parent: R83n Draft PR #33. No native physics run, new sensor calibration, trained model, unseen cohort or verification is asserted by this document.

## Why the gate exists

R83m's 54 genuine Basilisk cases (27 physically paired comparisons) establish a source-internal RW wheel-speed difference under a 0.9° reference excitation, but not a flight-available measurement or fault detector. R83n's ±epsilon bound is hypothetical; it cannot identify actual encoder resolution, quantization, timestamp uncertainty or onboard noise. The original R83f/R83g `NO_ELIGIBLE_DESIGN` outcome is not overturned.

## Required inputs, prior to model search

1. **Sensor specification**: exact measurand, wheel identity, units, encoder/estimator provenance, absolute error assumptions and supporting source documents, quantizer step/rounding/saturation and firmware variant. Unknown fields are **UNKNOWN**, never zero.
2. **Sampling contract**: onboard acquisition frequency, timestamp semantics, maximum jitter, dropped samples, alignment policy and permitted gap policy. Declare any resampling on TRAIN only.
3. **Operational visibility**: enumerate raw onboard telemetry fields accessible at inference time; disallow fault/healthy twin values, native C++ capacity/readback, injected fault time/identity, oracle healthy counterparts, or offline matching keys as model features.
4. **Excitation contract**: reference command availability at inference, allowed control modes, excitation bounds, pointing/actuation constraints and associated authority. A 0.9° experimental response does not imply that an operational spacecraft may safely execute this excitation.
5. **Cohort contract**: distinct native simulation seeds/initial conditions/wheel assignments/operating modes/fault severities with stable group identifiers. Explicitly distinguish simulated variants from independent spacecraft or physical sensor samples.

## Data and label discipline

- Freeze permitted input telemetry and a public schema **before** training. Publish an explicit allowlist and denylist.
- Split at whole trajectory / campaign / group level, **never by row**. Paired healthy/fault runs and all derivatives must share the same split; no source case or source hash may occur across partitions.
- TRAIN: model architecture search, preprocessing fitting, hyperparameter optimization.
- CAL: threshold / uncertainty / coverage calibration, isolated from TRAIN model selection.
- VAL: one predeclared model selection protocol; independent review of usage. A breached partition is invalidated, not recycled as blind evidence.
- TEST / FINAL: sealed until one-time authorized final execution after model, metrics, and artifact hashes freeze.
- Maintain a SHA-256 ledger for raw input, derived data, split manifest, code, configuration, checkpoint, predictions and metrics. Preserve negative results and every rejected candidate.

## Predeclared scientific questions

Q1: Are real or properly validated simulated **sensor observations** physically distinguishable at the declared excitation under resolution, bias, timestamp and stochastic noise uncertainty?
Q2: Does a deployable-input-only baseline achieve a predeclared and calibrated event-level false alarm / detection-delay tradeoff on separate trajectories?
Q3: Does a self-organizing neural algorithm exceed capacity- and tuning-budget-matched static neural networks, classical ML and statistical baselines on identical partitions and allowed inputs?

**No Q2 or Q3 result can be inferred from the current R83m/R83n oracle paired counterfactual tests.**

## Fairness and ablations

- Match train/cal/val/test partitions, telemetry channels, preprocessing, label availability and domain boundaries across methods.
- Log wall time and accelerator hours; set a common predeclared maximum tuning budget and count failed trials. Do not grant the self-organizing method privileged extra search.
- Report parameter count, inference latency, peak memory, calibration time, random seed distribution and repeated-run dispersion.
- Compare: simple statistical detector; standard ML; fixed-capacity neural baseline; self-organizing mechanism switched off with matched capacity; full self-organizing method. If mechanisms alter model size, additionally report parameter-matched and compute-matched variants.
- Stratify by wheel, reference command/excitation, operating regime, severity, sensor-error assumption and source group, including **unobservable** groups.
- Do not derive classifier thresholds or windowing logic from TEST/FINAL. Any alarm debounce/decision logic must be declared as part of each algorithm, not covertly injected by the evaluation platform.

## Hard acceptance / refusal rules

| Condition | Action |
|---|---|
| Missing actual or defensible synthetic sensor quantization/noise specification | BLOCK real-sensor diagnosability claim |
| Missing deployable input allowlist or any oracle leakage | BLOCK algorithm training/qualification |
| Source-group overlap between any split | BLOCK all measured generalization claims |
| No source-observable signal under allowed operational excitation | NO_ELIGIBLE_DESIGN for that cell |
| No separately sealed blind TEST/FINAL cohort | Research-only; no qualification or Stable promotion |
| CI green but sensor validation / independent cohorts absent | Code contract only; do not promote |

## Implementation tasks (not claimed completed)

1. Define versioned `sensor_contract.json` schema with UNKNOWN semantics and physical-unit validation.
2. Write deterministic validators for telemetry allowlist, group split disjointness, paired-run linkage and raw/derived SHA binding. Add adversarial tests injecting oracle and cross-split case IDs.
3. Build real/synthetic encoder adapter from a documented sensor specification; evaluate quantization, offset, drift, noise, dropped samples and clock skew without tuning to held-out groups.
4. Acquire physically executed, independently seeded Basilisk cohorts; audit C-backed actuator readbacks as **truth audit only**, never as ML inputs.
5. Run preregistered frozen comparisons and one-time blind evaluation; publish all failures.

This protocol is a planning increment on a separate branch, not evidence of Q1–Q3 completion.