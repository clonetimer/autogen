# R83i — Real 36-case Basilisk static-gain excitation pilot (2026-10-24)

**Scientific determination:** native source PHYSICS EXECUTED (36/36) and sealed source review PASSED. The preregistered **requested command / original hardware capability envelope FAILED**; the applied-torque and speed physical observations are within the *provisional*, noncertified envelope. Raising static feedback gains 25% did NOT consistently raise mean commanded RW torque. This is a **negative experimental result**, not a failure to find a conveniently passing threshold. No diagnostic classifier/agent was trained.

## Source and governance

- Upstream producer: \`clonetimer/SatAgentSimulation@1f709beeae97baec65db9ca13ae373d88728f87d\` and exact original \`bsk==2.11.0\` native binary wheel (SHA256 \`d51fe1ffa1b0b03dc415034b210497b28a2504428d4d8e71b481542a08c08a0a\`), installed wheel \`2.11.0+satfix1\`, unchanged native payload.
- Original Basilisk R83f 126-case plan canonical digest: \`c54bfcd154322a620c7d8058ce8a40db6e155602593c5ff5da8bdba43ef3b714\`.
- **Prospectively preregistered 36-case plan SHA256:** \`95a189b46db6d73b954b9c15fcb72246a47bcb1b9d9692fe6dc8ea80fac379b2\`. Only supported static controller parameter surfaces \`controller_k\`, \`controller_p\` were modified.
- [Original native physical execution Actions #40108812287](https://github.com/clonetimer/autogen/actions/runs/40108812287): **36/36 real native dynamics/FSW/RW physical cases completed, dry_run=false**, no failed cases; its final job status was RED because the FIRST evidence reviewer incorrectly treated physically valid zero *fault-effective* RW capacity as bad original hardware. Original raw physical outputs retained in [artifact 12565399186](https://github.com/clonetimer/autogen/actions/runs/40108812287/artifacts/12565399186). Original downloaded ZIP SHA256 \`15a6e464cd5914dffd3a3e8aa6e9228a063780cf234fd63f275c4dfa79933461\`.
- [Read-only corrected ORIGINAL source-review run #40109324252](https://github.com/clonetimer/autogen/actions/runs/40109324252): **PASS**, no fresh simulation, 36 real native source case task and contract receipts matched. Sources bound to three original header SHA256 hashes. Corrected [artifact 12564899773](https://github.com/clonetimer/autogen/actions/runs/40109324252/artifacts/12564899773) downloaded ZIP SHA256 \`05b08e92879d103775f2780bb2f47f6b1899bf7747683ee38f544436c26b9975\`.
- Original physical design: 3 faults × 3 inclination regimes (29°, 47°, 65°) × normal/fault twins × 2 controller regimes = **36 observed case files, 18 internal nominal/fault paired groups, 9 distinct original parent conditions**. The same original seed, fault/severity/injection wheel/onset, initial conditions and original rated RW torque/current limits persist across controller settings.
- Strict source roles: **TRAIN physical QA ONLY**. Zero VALIDATION/TEST/FINAL telemetry imported or read. Neither original R83f/R83g algorithm negative result was modified; no model/agent retraining, no source/internal applied torque passed as a deployable inference feature.

## Important physics distinction

The original recorded \`adcs.rw.command_torque_nm_*\` is a **command request, not actual delivered wheel torque**. The nominal design RW maximum torque is in the original task physical parameters \`rw_max_torque_nm\`. When a jam manifests, the trace's \`adcs.rw.effective_max_torque_nm_*\` can genuinely be **zero**, so it is wrong to divide by that fault-degraded effective capability to assess original hardware command loading; the corrected auditor preserves zero capacity as a physical observation and computes request vs ORIGINAL hardware-rated capability.

- **36/36** native source cases mechanically verified in corrected offline reviewer.
- Maximum |requested wheel command| / original hardware rated RW torque: **5.973549** (> preregistered provisional request ratio 1.0). The strict requested-command envelope **FAILED**, despite successful simulator execution.
- Maximum actual **applied** body-control torque / original rated RW capability: **1.000000** (clipped to original limit in observed samples). This is a source simulation observation, not independent actuator safety certification.
- Maximum observed wheel speed: **40.661412 rad/s**, below the provisional 900 rad/s speed envelope, which is not a validated aircraft/spacecraft hardware rating.
- **176 individual wheel-time samples** of zero effective actuator capacity, primarily jam-fault physical consequences, not missing telemetry.
- Overall provisional command+speed research envelope flag remains **FALSE**. The truthful negative flag is NOT changed merely because applied torque was saturated.
- The negative reviewer conclusion was additionally protected by synthetic tests that force over-limit commands on an UNRELATED wheel (the first implementation incorrectly checked only the injected target wheel). Current reviewer inspects all three wheels at all 61 timestamps.

## Actual paired controller-gain contrast

Each cell uses 3 matched parent source conditions, one per preregistered inclination. Listed values are medians of the *per-normal-case mean absolute post-onset commanded torque / original hardware rated RW torque*, not a diagnostic classification metric.

| Original injected mechanism | Reference gains (normal twin) | Both K,P increased 25% (normal twin) | Physical finding |
|---|---:|---:|---|
| RW friction increase | 1.1225% | 1.0735% | Normal-twin command decreased |
| RW jam | 3.9058% | 3.6176% | Normal-twin command decreased |
| RW torque-authority loss | **1.9674%** | **1.8721%** | Normal-twin command decreased |

The observation **does not support** the hypothesis that increasing supported static controller gains by 25% increases the average closed-loop control excitation. This intervention also changes the closed-loop response, so it is not a controlled externally imposed torque waveform. No time-varying command scheduler has been certified in the pinned \`subsystem.adcs_unified_native.v1\` source; no such waveform was injected.

### Next scientific/engineering milestone (not implemented here)

1. Stop gain-only intensification. A real bounded **time-varying excitation** should not be simulated by merely editing metadata or injecting hidden applied torque columns. First establish an actual Basilisk FSW message-chain integration interface that demonstrably switches applied control command/reference at scheduled times and records the request, limiter, actual torque and wheel speed.
2. Define independent wheel-speed, current, slew, attitude pointing error and torque-saturation requirements in engineering units, with physically grounded actuator ratings. Preregister safe limits, waveform, start/stop time and repeatable scheduling and a deterministic abort policy.
3. Repeat a separate prospective **native physical** paired experiment with original and excitation modes under fixed seed/severity, while keeping benchmark TRAIN/VAL/FINAL blind. Separate task-source observability outcome from classification performance.
4. Until a new independently valid diagnostic experiment meets strict gates, R83f/R83g \`NO_ELIGIBLE_DESIGN\` remains in force. Do not merge current research PRs into Stable on the strength of this engineering smoke alone.

**Conclusion:** 36 genuine physical cases, reviewer contract PASS; provisional *requested* torque bound FAIL; observed actual applied actuation clip PASS; raising static gains 25% did not establish stronger nominal command excitation. This result is a design constraint, not a model improvement.
