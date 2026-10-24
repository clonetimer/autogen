# R83i — Prospectively preregistered native static-gain physics pilot

**Research only.** The R83h physics-source audit found small commanded RW torque / rated capability and weak measured torque-authority-loss speed differences. The pinned native producer's official verified controller registry exposes **static** controller_k and controller_p, not an externally verified time-varying torque command schedule. R83i therefore tests a *limited, concrete engineering hypothesis* using supported static gains. Do not describe these as a commanded waveform, guaranteed safe maneuvers, or a new diagnostic model.

## Frozen source and experimental contrast

The same unchanged source: SatAgentSimulation @
\`1f709beeae97baec65db9ca13ae373d88728f87d\`, original wheel
\`bsk==2.11.0\` SHA-256
\`d51fe1ffa1b0b03dc415034b210497b28a2504428d4d8e71b481542a08c08a0a\`
with exact native payload, and frozen R83f original 126-case canonical task plan
\`c54bfcd154322a620c7d8058ce8a40db6e155602593c5ff5da8bdba43ef3b714\`.

The pilot takes exactly one original physical TRAIN pair in each of 3
inclinations (29°, 47°, 65°), for each of 3 original RW fault mechanisms:
**9 original physical seeds** × (normal + fault) × (original static gains +
gain125 static gains) = **36 real Basilisk cases**. There are only 9
independently specified physical parents, not 36 independent samples.
The two controller arms are \`reference\` (original k,p) and \`gain125\`
(both \`controller_k\` and \`controller_p\` multiplied by exactly 1.25).
The RW-rated max torque, motor current limit, original fault parameters,
fault onset, injection wheel, severity, attitude start condition and RNG
seed are unchanged within paired experiments. The 1.25 multiplier is
preregistered before execution and cannot be tuned in response to results.
No hidden TEST or VALIDATION cases are imported or executed.

## Required engineering gates

- Cross-platform tests (Linux Python 3.11/3.12 and Windows 3.11)
  assert exact 36-case plan, paired source seeds, unchanged limits,
  rejection of unknown \`torque_schedule\` interface, original source hash
  and refusal to treat synthetic test fixtures as physics.
- Real GitHub Actions research pilot pins original official Basilisk wheel
  and exact upstream source, regenerates the original 126-case *plan only*,
  checks its canonical SHA, then freezes the 36-case subset.
- The actual upstream \`run_fault_dataset_campaign(..., dry_run=False)\`
  produces genuine Basilisk native RW/attitude dynamics, not simplified
  Python kinematics. Every actual original source-case TaskSpec, hardware
  rating, RNG seed, case role, pair ID, onset, wheel target and raw
  61-sample time series are rechecked in a separate physics auditor.
- Report the *post-onset* nominal commanded torque as a fraction of the
  nominal RW-rated torque, observed applied-minus-command discrepancy,
  original target-wheel speed and observed actuator command/rating extrema.
  Native applied motor torque is a **simulator internal truth** for
  trusted physics QA only, not a deployable diagnostic input.
- A conservative **provisional**, non-certified internal engineering
  envelope requires |target-wheel speed| ≤ 900 rad/s and
  |command torque| / rated torque ≤ 1.0+1e-5 in all physical cases.
  The workflow fails and preserves its original physical evidence if this
  envelope is exceeded. Passing it does NOT certify a safe control law.
- A larger gain is not assumed to cause more excitation and may reduce
  command under closed-loop regulation. The measured result decides
  whether excitation increased, *not* the multiplier name.

**NOT IN SCOPE:** time-varying command profiles, validation of diagnostic
accuracy, new neural topology search, release of models, independent real
equipment faults, physical source flight certification, changing or
overwriting R83f/R83g refusal evidence, and opening R84 FINAL or prior
R83f VALIDATION data.

## Result interpretation

This is one prospective engineering pilot with only three physical
parents per mechanism (one per inclination). Contrasts are paired and
observational. If the gain125 arm increases commanded torque and manifests
a larger target-wheel response without violating provisional limits, that
supports designing an independently preregistered **actual excitation**
experiment through the appropriate FSW command interface. It does not
establish diagnostic class separability or that any self-organizing
network can detect the mechanism. If results are unchanged or worsen,
record the negative evidence and investigate the physical control path
rather than retune retrospectively.
