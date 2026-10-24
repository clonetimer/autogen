# R83k — verified original 4-arm Basilisk RW fault excitation/observability source (2026-10-24)

**Source evidence:** [SatAgentSimulation R83k research PR #77](https://github.com/clonetimer/SatAgentSimulation/pull/77), original [GITHUB ACTIONS #40114200932](https://github.com/clonetimer/SatAgentSimulation/actions/runs/40114200932), native experiment commit \`74e680e0f86f84707fd69116e7e14c1601ba9343\`. **All four physical Basilisk C-backed runs executed**, plus Windows Python 3.11 and Linux Python 3.11/3.12 source-interface contract tests passed.

Actual source artifact: [original artifact #12567229709](https://github.com/clonetimer/SatAgentSimulation/actions/runs/40114200932/artifacts/12567229709). Frozen original ZIP SHA-256 \`21076917e32671d12cff060d6163e788945e821255571282705e2c0adc47400c\`. Exact source JSON SHA-256 \`4b6fd9b66e48b9be7ff9247eb8d83a36cd0d3977e7616df091a087671e42ce56\` (134075 bytes). Original official bsk 2.11.0 binary SHA-256 \`d51fe1ffa1b0b03dc415034b210497b28a2504428d4d8e71b481542a08c08a0a\`; physical payload unchanged.

**Independent original-source read-only gate:** [autogen R83k CI #40114750594](https://github.com/clonetimer/autogen/actions/runs/40114750594) **PASS**: Linux 3.11/3.12 + Windows 3.11 anti-fabrication tests and exact original real-Basilisk artifact SHA-256 check. Gate independently rederived all metrics using 4×31 exact native samples, three-wheel motor request, native \`rwOutMsgs[0].u_current\` applied torque and original physical fault \`ReactionWheelData[0].u_max\` readback. Original downstream artifact [#12567099404](https://github.com/clonetimer/autogen/actions/runs/40114750594/artifacts/12567099404): \`r83k_genuine_original_native_physics_only.json\`, ZIP SHA-256 \`29789f7b37b88c519d9e323fef1a3c2d477bb58f9dc8c0a81dcb75dff072e6e2\`, original proof JSON SHA-256 \`cf852697c217186560386b10580196b63407bd29544a68b3a69a5ddf47638b6e\`.

## Native physics experiment

Fixed 2×2 factorial: **reference** zero/scheduled 0.5° native \`inertial3D\` FSW pulse, and **RW0 torque capacity** normal 0.0200 N·m vs physical derated 0.0005 N·m at paused native task boundary 2.4 seconds. Three wheels attached to real spacecraft attitude dynamics, 0.2s task period, 6 seconds duration. Exact pre-event equal native wheel speed (difference zero). The native C-backed wheel's applied motor torque is separately recorded, and **physically** limited to the reduced capacity after fault. The original source contains all C-backed motor actuation and actual simulated RW motion, never a Python-proxy numerical sequence.

| Source original paired outcome | No scheduled FSW reference | Scheduled 0.5° native reference |
| --- | ---: | ---: |
| Before-fault max RW0 speed difference | 0 rad/s | 0 rad/s |
| After-fault max RW0 speed difference | **0 rad/s** | **0.03964137678483581 rad/s** |
| After-fault max native RW0 applied torque difference | 0 N·m | **0.003014143316513015 N·m** |
| Peak faulted RW0 actual applied torque | 0 N·m | **0.0005 N·m** |

Max native wheel motor *allocator request* over all four arms/all wheels = **0.00436333005262522 N·m**, below 0.02 N·m preregistered research observer bound; wheel speed max **31.415928985051483 rad/s**, below 900 rad/s research observer bound. These are uncalibrated laboratory thresholds, **NOT equipment, pointing, safety, or flight qualifications**.

**Conclusion:** The original native physics source supports *excitation-dependent observability*: with zero FSW guidance reference the derating is inactive and unobservable in wheel-speed output; with a bounded scheduled reference it changes both native applied wheel torque and genuine C-backed wheel rotational dynamics. This is *one deterministic plant, one RW, one pulse, one extremely severe derating*, not independent sensor-level diagnosis, not a multi-environment benchmark, not confidence intervals and not proof that self-organizing methods work.

The original R83f/R83g **NO_ELIGIBLE_DESIGN** result remains untouched. No diagnostic model trained here, no VALIDATION/TEST/FINAL records read or accessed. Source-private applied motor torque, target wheel, fault onset and actuator cap may be used for trusted offline physics QA only, never as an inferable deployment sensor or label. No Stable/main merges.

## Future research gate

Pre-register a NEW prospective Basilisk-native cohort before generating outcomes, with varied wheels, inclination/pointing regimes, moderate and severe torque capacity loss, multiple independent initial states, actual actuator derating, mission-credible limits (torque, current/power, slew, pointing, speed) and samples allocated strictly TRAIN/VAL/sealed FINAL by physical group. Confirm original-source integrity, actual torque limiter behavior and observability. Only then consider fair matched source TRAIN OOF fixed/static/self-organizing model study, with matched compute, safety and genuinely independent cohorts. Do not adapt R83k parameters to maximize the observed residual or treat a simulator truth variable as a model input.

**Gate classification: GENUINE NATIVE PHYSICAL FAULT MANIFESTATION = VERIFIED; ALGORITHM QUALIFICATION = REFUSED / NOT TESTED; HARDWARE/FLIGHT SAFETY CERTIFICATION = NOT ESTABLISHED.**
