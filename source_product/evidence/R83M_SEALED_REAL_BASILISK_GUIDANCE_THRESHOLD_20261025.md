# R83m — Independently audited original 54-case Basilisk RW physical threshold evidence

**2026-10-25.** Source-role: **ORIGINAL NATIVE PHYSICS QA ONLY**,
no diagnostic training, model eligibility, hardware qualification or
previous R83f/R83g acceptance-threshold changes.

## Source identity, original real physics and independent verifier

- [Upstream genuine Basilisk native Actions #40160788578](https://github.com/clonetimer/SatAgentSimulation/actions/runs/40160788578):
  **54/54 original physical simulations**, Windows 3.11, Linux 3.11/3.12
  contract checks PASS. Upstream research [PR #88](https://github.com/clonetimer/SatAgentSimulation/pull/88);
  original source executable commit
  \`6e1d54bfef6655a29e0a6c98b135f478fa539a48\`.
- Original source ZIP artifact [#12586915899](https://github.com/clonetimer/SatAgentSimulation/actions/runs/40160788578/artifacts/12586915899),
  ZIP SHA-256 \`db83c8b50a833d2f9c5074cb3f6f9d44c8e0109a2a7be4ddcec5f70bff754796\`;
  original 2,020,615-byte \`r83m_genuine_native_54_original.json\`
  SHA-256 **\`635fd4e4cd36da39a15ba1289003ba84aaf4551b02de70c3d063ada878209502\`**.
- [Downstream independent autogen Actions #40161374796](https://github.com/clonetimer/autogen/actions/runs/40161374796):
  **PASS** Linux 3.11, Linux 3.12, Windows 3.11 synthetic contract regressions
  AND original upstream SHA-locked genuine physics evidence verified.
  Original independent [artifact #12586083694](https://github.com/clonetimer/autogen/actions/runs/40161374796/artifacts/12586083694),
  ZIP SHA-256 \`6ef6d5f58c66e8d2f5839fafa27cbdbb3a5c6ad32ab8330e3916e6aa79c28ddc\`,
  result \`r83m_verified_real_native_54_physics.json\` SHA256
  \`ec2ce9521ad87a745656d0661db502995ee6607a43f4b75737405300e70ff889\`.
- Downstream guard recomputes from 54×31 **true Basilisk**
  physical trace samples: live original actuator C++ \`u_max\` readback,
  actual native RW \`u_current\`, wheel speeds, original FSW reference
  MRP on target axis at scheduler effective 2.0…3.8s, and per-reference
  commanded-wheel torque / body rates. The R83l exporter target-wheel
  metadata shadow issue is absent: R83m original 54/54 native case targets
  were correct and cross-checked against original native fault events.
  No protected VALIDATION/TEST/FINAL source accessed.

## Original-source 27 matched physical pairs

Exactly 9 target-wheel/initial-attitude combinations × 3 guidance
amplitudes × (healthy vs original physically 75%-degraded RW) = 54 actual
native simulations and 27 paired physical comparisons. Not independent
hardware groups; no automatic neural benchmark.

| Native FSW reference | Measurable wheel-speed difference | Median maximum paired post-fault wheel-speed difference |
| --- | ---: | ---: |
| No pulse, 0° | 0/9 | **0 rad/s** |
| Pulse 0.5° | 0/9 | **0 rad/s** |
| Pulse 0.9° | **9/9** | **0.0052003294905 rad/s** |

The original pulse09 post-fault speed difference range is
**0.002894768863 to 0.010591663083 rad/s**; all 9 also have nonzero
**actual** native RW applied-motor-torque difference.
Physical wheel cap changes from **0.020 to 0.005 N·m** at 2.4s paused
native scheduler boundary. The **original native actuator** limited
faulted actual motor torque to 0.005 while nominal C-backed wheel could
obey requests above that level.

Measured peak motor allocator requests across all arms grouped by
pulse level: 0° 0.0020 N·m, 0.5° 0.0048816067 N·m, 0.9°
**0.0083723050 N·m**, which lies *above* the degraded 0.005 but
*below* the original nominal 0.020 cap. Max wheel speed
**31.49274423 rad/s**; max original body rate **0.0012437035 rad/s**.
All provisional research observation limits (request 0.020 N·m,
RW speed 900 rad/s, body angular rate 0.005 rad/s) PASSED. These limits
were predefined but **never certified as mission pointing/current
hardware safety bounds**.

## Scientific interpretation and further independent gate

Under these limited original Basilisk parameters, a 0.9° scheduled
FSW command creates actual closed-loop wheel torque saturation for a
75% degraded RW where the 0.5° command does not. It establishes
*excitation-conditional physical observability*, NOT a valid deployed
fault classification. The healthy/fault aligned trajectory paired
residual is available only for **offline physics QA**; it is not an
on-orbit available counterfactual. Fault target, injected onset,
actual applied motor torque and C-backed capability are physical truth
and may never be silently included in the model's inference telemetry.
A sensor noise floor may make the milliradian-per-second differences
undetectable; **no noise, false-alarm, response-delay, out-of-family
study or blind independent cohort** has been tested.

The 0.9° proposal was informed by R83l; R83m is exploratory, not
independently blinded confirmatory evidence. **R83f/R83g source TRAIN
model result NO_ELIGIBLE_DESIGN remains unchanged**. No model trained
here and nothing authorizes Stable promotion.

Next preregistration must specify wheel-speed measurement quantization,
noise, sample cadence and independent spacecraft regimes (not just
repeat 9 wheel/attitude design conditions), source group splits, mission
actuator current/power/slew constraints, and a separate untouched blind
cohort before a fair fixed/static/adaptive diagnostic benchmark can
honestly be attempted.

**Decision:** real C-backed FSW → RW physics source and original-data
independent audit PASS; *practical fault detectability and algorithm
qualification NOT ESTABLISHED*. Keep R83m research Draft, Stable/main
untouched.
