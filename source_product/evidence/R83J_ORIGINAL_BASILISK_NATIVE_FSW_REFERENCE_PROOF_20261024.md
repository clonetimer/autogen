# R83j — REAL Basilisk upstream FSW reference→RW physics accepted (2026-10-24)

## Source proof and identity

**Actual physical-source result, not a diagnostic qualification.** Upstream SatAgentSimulation
[PR #73](https://github.com/clonetimer/SatAgentSimulation/pull/73)
built and executed the original Basilisk 2.11.0+satfix1 source chain
\`inertial3D → attTrackingError → mrpFeedback → rwMotorTorque → reactionWheelStateEffector → spacecraft\`.
The commanded quantity was a bounded **FSW inertial attitude REFERENCE**, not
direct motor torque, a real spacecraft telecommand, or a fault injection.

- Upstream ORIGINAL real execution [#40111820380](https://github.com/clonetimer/SatAgentSimulation/actions/runs/40111820380): PASS on 2 native six-second physical experiments, with all native source probes preserved; upstream original wheel SHA256 \`d51fe1ffa1b0b03dc415034b210497b28a2504428d4d8e71b481542a08c08a0a\`.
- Upstream exact source commit of first accepted real physics \`4c70d2c0fe3ca9165d794c3012509d7bb2646b93\`; original downloaded \`actual_original_native_fsw_schedule.json\` SHA256 \`c93773efed43b8c3a8e68290df53cba966dec9b6e74bf15f6655e0b197730ae8\`. Original source-evidence ZIP SHA256 \`6f221c7fa3202d7b091f16baef78b001a7db015646b542354922116f790e426e\`.
- Actual [autogen original-source gating CI #40112438420](https://github.com/clonetimer/autogen/actions/runs/40112438420): **PASS** on Linux Python 3.11/3.12, Windows 3.11 source-integrity and scientific claim guards; and independently **PASS** using the exact original upstream CI artifact. Original autogen proof artifact: \`r83j-autogen-readonly-original-bsk-fsw-coupling-proof\`.
- Independent autogen proof receipt downloaded ZIP SHA256 \`c09b342f6d665fda095419056cf54df8af7d2ec2b603bbe181bfd734fb6b44e3\`.
- Both upstream and downstream PRs stay **Draft**, untouched Stable/main; no physics job auto-repeats merely because documentation changes.

## Exact original physical measurements

Two 6-second Basilisk simulations with the same original spacecraft, initial attitude/rates,
three RW motor limits and native scheduler, one static-zero reference and one scheduled
0.5-degree x-axis reference setpoint. At the native paused scheduler boundaries the reference
was set to 0°, +0.5°, 0°. The actual native \`inertial3D\` output reports the +0.5° signal first
at **t = 2.0 s**, not 1.8 s; the delayed sampled timestamp is retained in the evidence.

| Metric from ORIGINAL native source | Actual |
|---|---:|
| Native runs / samples each | **2 / 31** |
| Pre-event max paired target-RW speed discrepancy | **0 rad/s** |
| Maximum post-event target-RW speed difference | **0.0666601340 rad/s** |
| Maximum post-event motor allocator request difference | **0.0039194724 N·m** |
| Maximum allocator wheel-motor request over all wheels/arms/times | **0.0043633301 N·m** |
| Provisional wheel-motor command request observation limit | **0.020000 N·m** |
| Maximum native RW wheel speed over both arms | **31.415929 rad/s** |
| Maximum body-axis rate | **0.000519367 rad/s** |
| Preregistered provisional request and wheel-speed observation limit | **PASS** |

This demonstrates **nonzero, time-controlled guidance reference drives real Basilisk FSW and
reaction-wheel physics**, without relaxing the provisional lab request or speed bounds.
It does NOT independently certify a safe flight pointing maneuver or actuator, because the
user hardware, RW current, uncertainty margins and operational rate/pointing limits are
not yet independently calibrated. The motor allocator record is a **requested wheel torque**
and is not proof of a measured physical applied torque sensor.

## Explicit research conclusion and data boundaries

The downstream gate validates original SHA256 and recomputes all physics differences
directly from both source-native 31-sample streams, refusing forged metrics,
synthetic substitutes, source role or claim changes. This is upstream **physics
attestation for experiment feasibility only**, not model correctness.

\`autogen_algorithm_eligible_from_this_evidence = false\`,
\`stable_release_eligible_from_this_evidence = false\`,
\`independent_hardware_safety_certified = false\`.
No RW fault mechanism, no training/model search, no held-out VALIDATION/TEST/FINAL
used. R83f/R83g \`NO_ELIGIBLE_DESIGN\` remains correct and unchanged.

## What R83k must prove BEFORE any self-organizing-network diagnosis claims

1. Replace the test-only model reference setter with an explicit verified **native
   scheduled guidance/telecommand owner** that honors native process and task
   precedence, bounded attitude reference slew, motor torque request limit,
   wheel-speed/current limit, abort and reset, without an implicit simplified
   Python dynamics fallback.
2. Introduce one independently controlled physical **RW torque authority-loss**
   event using a Basilisk-native fault/actuator capability, with matched nominal
   wheel hardware, timestamps, random seeds and initial conditions.
3. Predeclare the exact safe guidance excitation, stop conditions, fault onset
   and severity **before physical outcomes**. An unsafe source must be preserved
   and REFUSED, not tuned by retrospective threshold escalation.
4. Quantify physical *fault/no-fault* separability under native observation-only
   telemetry first; source-internal target wheel, truth, actual applied RW
   torque and onset must not become deployment inference inputs.
5. Only after physical credibility gate passes run a NEW fair fixed/static/
   adaptive modeling protocol with unchanged prospective group definitions,
   budget reporting (fit counts plus measured compute), untouched VAL/FINAL
   and independent blind confirmation.

**Bottom line: original-native FSW reference→RW physical coupling DEMONSTRATED;
safe faulted controllable excitation and superior diagnosis NOT DEMONSTRATED.**
