# R83n — Original Basilisk wheel-speed bounded-error source sensitivity (2026-10-25)

**Outcome: original-source physics sensitivity audit PASSED. Practical encoder fault detectability / neural diagnosis / flight safety NOT established.** Source remains one Basilisk simulator family, not nine independent spacecraft; no new simulations, random noise, hardware measurements, model fitting, or protected VAL/TEST/FINAL reads occurred in R83n.

## Immutable provenance and actual independent QA

- Original R83m upstream [SatAgentSimulation genuine Basilisk 54-case study #40160788578](https://github.com/clonetimer/SatAgentSimulation/actions/runs/40160788578), [PR #88](https://github.com/clonetimer/SatAgentSimulation/pull/88): PASS, 54 original bsk 2.11.0 native physical cases, three wheels × three fixed initial states × 0/0.5/0.9-degree real FSW guidance × healthy/75%-loss physical actuator. Exactly 27 normal/fault pairs from **nine wheel/initial-state conditions**, not 54 independent flight trials. Original C-backed physical RW torque-capability change \`0.020 → 0.005 N·m\` is verified in original source.
- **Sealed original native JSON SHA-256:** \`635fd4e4cd36da39a15ba1289003ba84aaf4551b02de70c3d063ada878209502\`. Upstream original source ZIP SHA-256 \`db83c8b50a833d2f9c5074cb3f6f9d44c8e0109a2a7be4ddcec5f70bff754796\`. Every R83n analysis calls the full existing R83m physics and provenance gate first, then reads original source ONLY; a synthetic fixture must explicitly override both pins inside tests and can never pass as real physics in production.
- **R83n real original evidence CI [#40172661521](https://github.com/clonetimer/autogen/actions/runs/40172661521): PASS**, Linux Python 3.11/3.12, Windows Python 3.11 synthetic anti-fabrication / scientific-claims tests followed by genuine upstream original artifact retrieval, 54 native cases and 27 matched comparisons revalidated.
- Original independent R83n output artifact [#12590294058](https://github.com/clonetimer/autogen/actions/runs/40172661521/artifacts/12590294058), \`r83n-basilisk-native-original-speed-error-sensitivity\`, contains an aggregate physics-only JSON receipt. It contains no per-case source truth series, simulator-private applied torque, target-wheel ground-truth inference label, or synthetic sensor signals. Original physical source is unchanged.
- R83m source report and R83f/R83g original diagnostic model refusal remain untouched.

## Mathematical assumption (HYPOTHETICAL — no real sensor calibrated)

This is a deterministic *paired physical-oracle* sensitivity calculation, **not a model of an onboard RW encoder**. Assume each of TWO wheel-speed measurements (original healthy twin and original faulted twin) has independently bounded absolute measurement error \`|e| ≤ ε rad/s\`. Then

\`|(\omega_f + e_f) - (\omega_n+e_n)| ≥ |\omega_f-\omega_n| - 2 ε\`.

If **any original post-fault sample** satisfies \`|\omega_f-\omega_n| > max(1e-8, 2 ε)\`, this is a **SUFFICIENT** guarantee that at least one *hypothetical matched healthy/fault pair difference* remains nonzero under any allowed bounded errors. If it does not satisfy the inequality, we do not conclude the difference is actually unobservable; it simply lacks this worst-case guarantee.

**Operational limitation:** There is no synchronized, perfect healthy counterfactual spacecraft on orbit. The source \`rw.native_speed_rad_s_*\` is an internal Basilisk state variable, not verified encoder readout. One guaranteed source-oracle difference does **NOT** imply a realizable FDIR detector, let alone sensitivity, specificity, false-alarm rate or fault-alarm delay. The original source fault wheel, onset and applied torque are never promoted to model inference channels.

## Original results: all nine fixed wheel/initial-state design conditions

Each count is the number of *physical paired comparisons*, not a detection rate, binomial inference, validated sensor resolution or a statistically independent real-equipment cohort.

| Assumed absolute error of EACH speed reading, ε (rad/s) | 0° guidance | 0.5° guidance | 0.9° guidance |
| --- | ---: | ---: | ---: |
| 0 | 0/9 | 0/9 | **9/9** |
| 0.0005 | 0/9 | 0/9 | **9/9** |
| 0.0010 | 0/9 | 0/9 | **9/9** |
| 0.0025 | 0/9 | 0/9 | **6/9** |
| 0.0050 | 0/9 | 0/9 | **1/9** |
| 0.0100 | 0/9 | 0/9 | **0/9** |

All epsilon values were fixed **before R83n source re-analysis** and are *illustrative stress levels*, NOT measured onboard RW-speed encoder accuracy, noise distribution, quantization step or calibration evidence. Actual original 0.9° pulse R83m paired source maxima ranged **0.002894769 to 0.010591663 rad/s**, median **0.005200329 rad/s**. Under ε=0.0010, even the smallest source pair's peak true difference exceeds 2ε. Under ε=0.0050, only one out of nine original physical paired cases achieves the conservative margin.

R83n additionally reports the **time of first source-oracle margin** for each bound, e.g. under ε=0.0025 the median first margin is 3.6 s across six qualifying pairs. This is **not first detection time nor alarm latency**: it uses the source-private matched healthy twin and knows the true target wheel and fault event. There is no model or sampled noise.

## Scientific conclusion and decision

R83m validated that a genuine 0.9° Basilisk-native guidance pulse can drive a 75%-derated wheel past the native 0.005 N·m capability and produce a physically distinct wheel-speed trajectory. R83n reveals an important *magnitude-of-effect limitation*: many of the 9 source differences do **not** retain a guaranteed pairwise separation if hypothetical single-reading absolute wheel-speed error can be as high as ±0.0025 or ±0.005 rad/s.

This alone cannot select an amplifier, encoder, diagnostic threshold or neural architecture. The model-fairness question must remain **closed** pending actual sensor-channel credibility and independent physical groups. A 9/9 source-oracle difference under low hypothetical noise is NOT proof the earlier R83f/R83g \`NO_ELIGIBLE_DESIGN\` outcomes improved.

### Acceptance criteria before R83o / a real sensor-level diagnostic trial

1. Verify exactly which **RW-speed encoder** (or Basilisk native sensor module) provides accessible on-board-style telemetry and its calibration, bias/drift, quantization, noise PSD, sample cadence, latencies, dropout and timing uncertainty. Until measured or modeled from documented equipment assumptions, reject actual detection claims.
2. Keep original FSW command/reference telemetry separate from true C-backed RW \`u_current\`, wheel fault identity and onset. Private simulator truth is physics QA **only**. In particular the native wheel state cannot automatically be called a realistic measured speed sensor.
3. Register a NEW strictly source-disjoint native Basilisk physical cohort with varied wheel hardware, inertia, control operating modes, initial attitudes and independently specified noise before any new trained model; group by physical condition/plant identity, not by contiguous windows or cloned normal/fault twins.
4. Measure realistic false alarms on NORMAL, detection on faults, detection delay and worst-regime performance on a strictly held-out source group, with fair fixed/static/adaptive candidate selection and recorded actual compute budget. Preserve untouched blinded FINAL evidence and forbid retrospective threshold picking from validation outcomes.
5. Include independent mission pointing/current/voltage/wheel-speed/torque-rate constraints; the existing 0.02 N·m request, 900 rad/s wheel speed and 0.005 rad/s body-rate are provisional lab observation limits, **not hardware or flight safety certification**.

**R83n gate: SHA-pinned original Basilisk physics + conservative bounded-error oracle sensitivity PASS; actual sensor noise qualification FAIL / NOT PROVIDED; onboard diagnosability NOT TESTED; no new eligible model; Stable/main unchanged.**
