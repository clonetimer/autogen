# R83f: Actual 126-case Basilisk RW scientific result, 2026-10-24

**Status: actual Basilisk-native physics completed; ALL THREE diagnostic arms REFUSED.** This is a TRAIN-only development result, not independent OOD or flight qualification.

## Immutable source identification

| Item | Confirmed evidence |
|---|---|
| [GitHub Actions run #40099391872](https://github.com/clonetimer/autogen/actions/runs/40099391872) | Both preflight and full native simulation jobs PASS |
| Tested research commit | 63bca886d585934b6d610c919de7e0980a3a98d4 (PR #25) |
| Native producer | clonetimer/SatAgentSimulation at 1f709beeae97baec65db9ca13ae373d88728f87d |
| Basilisk | Official wheel bsk 2.11.0, installed 2.11.0+satfix1; payload unchanged |
| Preregistered canonical source-plan SHA-256 | c54bfcd154322a620c7d8058ce8a40db6e155602593c5ff5da8bdba43ef3b714 |
| [Original GitHub artifact](https://github.com/clonetimer/autogen/actions/runs/40099391872) | r83f-126-real-basilisk-native-balanced-research |
| Downloaded artifact ZIP SHA-256 | d5d4af3a80120e379d2760a067947b860872b926026009f913ce84451387b173 |
| Archive audit | 602 ZIP entries including native traces, runtime source receipts, 60/24 balanced records and full original TRAIN model-search receipt |

Native campaign index: **126 cases**, **dry_run=false**, campaign **complete**, 126 formal native engineering-training candidates, **zero independently expert approved**. Original case population: **90 TRAIN** (45 normal/fault pairs) and **36 VALIDATION** (18 pairs). Strict class-balanced design cohort: **60 TRAIN** (5 independent physical groups per regime/class) and **24 VALIDATION** (2 groups per regime/class). No TEST/FINAL generated, read, or used. Original excluded nominal twins remain unchanged in raw source.

## Preregistered strict diagnostic acceptance

The source TRAIN-only group-OOF screen required all of: accuracy >= 0.90, minimum environment-class recall >= 0.80, and healthy false alarm <= 0.10. The same source physical-group partitions and 3 predeclared candidate slots, same TRAIN selection rule and same acceptance decision apply in all arms. All arms completed 3 candidates and charged 9 actual OOF estimator fits against fit-cap 10.

| Arm, best TRAIN candidate by shared predeclared loss | Accuracy | Worst environment/class recall | Healthy FAR | Decision |
|---|---:|---:|---:|---|
| Fixed TCN/GRU/ExtraTrees portfolio — ExtraTrees | 0.88333 | 0.40 | 0.06667 | NO_ELIGIBLE_DESIGN |
| Three preregistered static neural topologies — shared width-16/depth-2 TCN | 0.63333 | 0.20 | 0.53333 | NO_ELIGIBLE_SELF_ORGANIZING_DESIGN |
| Self-organizing topology — shared width-16/depth-2 TCN | 0.63333 | 0.20 | 0.53333 | NO_ELIGIBLE_SELF_ORGANIZING_DESIGN |

The adaptive arm's feedback-generated third topology scored accuracy **0.55**, worst-class recall **0.20**, healthy FAR **0.40**. The static third predeclared topology scored accuracy **0.48333**, worst-class recall **0**, healthy FAR **0.06667**. Neither is eligible. The two neural arms' shared first two genomes and scores were identical.

Fairness report: all declared candidates evaluated = true; common loss, seed policy, dataset/folds = true; TRAIN-only comparison permitted = true; independent generalization claim permitted = **false**. Equal FIT counts do **not** mean identical FLOPs, CPU seconds or parameter counts.

**Zero models passed acceptance; zero models were frozen; no read-only VALIDATION was performed.** A green GitHub Actions result means the refusal and correct data handling were implemented, not that any model qualified. The independent 24-case VALIDATION cohort remains untouched for this research run.

## Physical QA from original, actually simulated 45 TRAIN fault/normal pairs

Trusted physics QA read original native trace.csv and engineering event-onset/affected-wheel metadata ONLY for offline physics-quality analysis. Injection truth was NOT an input to model training, features or inference. Every pre-event matched target-wheel-speed discrepancy was exactly zero.

| Injected mechanism | True physical pairs | Median maximum absolute post-event injected-wheel-speed difference | Full range (rad/s) |
|---|---:|---:|---:|
| Friction increase | 15 | 3.761270 rad/s | 2.771406–7.326534 |
| Jam | 15 | 26.835564 rad/s | 12.595952–36.035871 |
| Torque authority loss | 15 | 0.307233 rad/s | 0.038257–1.015184 |

Torque-authority-loss paired medians by preregistered inclination region: low 0.3072, medium 0.4118, high 0.1856 rad/s, 5 physical pairs per region. These numbers show a possible **observability difficulty** for the simulated apparatus, not a mechanism-proven explanation of a classifier error and not real-flight performance.

## Fixed scientific conclusion and next step

- **Native physical pipeline: PASS.** All 126 genuine Basilisk cases completed, source integrity/pair checks passed, the 60/24 class-balanced preparation and three-arm TRAIN candidate search executed.
- **Model eligibility: FAIL as expected.** The strict common worst-class recall gate rejected every arm, independently of Actions job status.
- **Self-organizing superiority: NOT DEMONSTRATED.** No legitimate evidence for improved model generalization or deployment.
- **VALIDATION: NOT RUN.** The frozen model eligibility gate correctly blocked it. TEST/FINAL: NOT CREATED OR ACCESSED.
- **Independent physics attestation and equipment/OOD generalization: NOT AVAILABLE.** Training fit caps were matched, not resource-normalized computational FLOPs.
- Future work may register a NEW, explicitly exploratory TRAIN-only command-response representation experiment, with unchanged protected cohorts and a separate prospective confirmation study. Never silently lower thresholds, reinterpret TRAIN search feedback as held-out performance, or rewrite this frozen R83f negative result.

**Immutable bottom line: NO_ELIGIBLE_DESIGN across all three source-TRAIN arms.**
