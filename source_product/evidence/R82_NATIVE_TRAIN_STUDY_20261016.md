# R82.5–R82.6: Real Basilisk RW campaign / source TRAIN OOF audit

**As of 2026-10-16. Development research only. No model delivered.**

## Executed native source

- Producer: [SatAgentSimulation](https://github.com/clonetimer/SatAgentSimulation)
  original fault factory at Git commit
  `1f709beeae97baec65db9ca13ae373d88728f87d`.
- Real execution: [GitHub Actions original source campaign run](https://github.com/clonetimer/autogen/actions/runs/38855256407);
  diagnostics reproduced with full refusal receipt in
  [run #38855706548](https://github.com/clonetimer/autogen/actions/runs/38855706548).
- Runtime: original Linux `bsk==2.11.0` wheel with expected SHA
  `d51fe1ffa1b0b03dc415034b210497b28a2504428d4d8e71b481542a08c08a0a`,
  metadata-only repack to `2.11.0+satfix1`; same payload bytes.
- Native status: source producer, physics-pair validator, diagnostic candidate
  review and R82 native TRAIN candidate gate all passed.
- Task: three classes of reaction-wheel fault
  (friction increase, jam, torque authority loss) versus paired nominal.
- Data: **27 TRAIN physical pairs / 54 trajectories**, plus
  **3 VALIDATION pairs / 6 trajectories** and **3 TEST pairs /
  6 trajectories**, kept out of design selection.
- Sensor inputs: wheel speed [0–2] in rad/s, motor **command** torque
  [0–2] in N*m. Source rows: 61 at 1-second cadence over 60 seconds.
- All six channels are simulated engineering observations; NOT
  flight telemetry. Pair labels are source ground truth and do not enter
  model features. Upstream class `NORMAL` becomes training `NOMINAL`
  through an explicit label-only adapter mapping.
- Expert status: producer still records `training_ready=false`.
  Independent physics attestation and FINAL evaluation are unperformed.

## Source TRAIN grouping and scientific caveat

The fixed source-only research policy groups a physical lineage
(`SHA256(parameters,seed)`) within each fault/nominal pair and assigns an
`operating_regime` by **predeclared source inclination**:
`<38°` low, `[38°,56°)` medium, `>=56°` high.
No label, post-onset truth, TEST sample or model score determines the bin.

| Regime | TRAIN physical pairs | Nominal | Friction | Jam | Torque authority |
|---|---:|---:|---:|---:|---:|
| Inclination low | 10 | 10 | 3 | 2 | 5 |
| Inclination medium | 12 | 12 | 3 | 6 | 3 |
| Inclination high | 5 | 5 | 3 | 1 | 1 |

The bins are **research-only, unreviewed by an independent physics expert**.
In particular, high inclination has one source pair for jam and one for
torque authority; a measured zero class recall in that environment carries
large sampling uncertainty. Inclination variation in one simulator is **not
out-of-domain equipment generalization**.

## Actual unified AutoDesign selection result

Predeclared source TRAIN environment leave-one-out OOF gate:

- `accuracy >= 0.90`
- `worst_environment_class_recall >= 0.80`
- `healthy_false_alarm <= 0.10`

| Trial | Accuracy | Worst environment / class recall | Healthy false alarm | Result |
|---|---:|---:|---:|---|
| Linear + temporal | 0.685185 | 0.0 | 0.0 | Rejected |
| ExtraTrees + temporal | 0.870370 | 0.0 | 0.037037 | Rejected |

All model training, OOF choice and refusal used **TRAIN only**.
The final `IndustrialDesignRejected` decision was
`NO_ELIGIBLE_DESIGN`; no model was refitted, exported or released.

ExtraTrees worst environment/classes include torque-authority loss at high
inclination (0/1), with 4/5 in low inclination and 2/3 in medium;
jam recall at medium inclination is 3/6. The main underperforming signal
is torque-authority change in the six-channel command/speed interface,
not necessarily absence of a physical fault.

## Exploratory reference (NOT the above formal trial)

A separate 42-feature, per-channel summary with 240-tree ExtraTrees and
leave-one-inclination-environment-out validation achieved 48/54
(88.89%) source TRAIN OOF accuracy and 33.33% overall
torque-authority recall. This is *additional source TRAIN exploration*,
not a predeclared equal-budget AutoDesign comparison or a validation
/ FINAL score.

A matched nominal/fault-pair observation check on the real source
revealed maximum wheel-speed differences with medians (rad/s):

- friction increase: **3.624**
- jam: **27.682**
- torque-authority loss: **0.467**

That suggests considerably weaker observed fault effects for torque
authority in this fixed set of outputs. It does NOT establish a physical
model bug without further inspection of the injected mechanism and
sensor/command behavior.

## Decisions

1. **Retain** the refusal and original eligibility thresholds. Never
   lower a gate post hoc to deliver a model.
2. Preserve source TRAIN OOF trial metrics and failed acceptance gates in
   `r82_rw_study.py`'s research report; prior CLI versions emitted only a
   terse refusal code.
3. Future R83 exploration should preregister a **generalizable
   actuator command-response / learned temporal representation comparison**
   with equal fitting budgets, while leaving VALIDATION and TEST
   untouched by model selection.
4. Review the torque-authority fault physics and genuine decision-time
   observables independently of the target label. Validate broader
   physically distinct regimes before making cross-domain claims.

**Important distinction:** the actual native source **simulation**
and source physics candidate contracts passed; the later **AutoDesign model
acceptance failed as designed**. These are not conflicting verdicts.
