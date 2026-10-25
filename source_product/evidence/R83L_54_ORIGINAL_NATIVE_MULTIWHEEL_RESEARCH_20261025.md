# R83l immutable original 54-case Basilisk RW physics result — autogen read-only proof

**Date: 2026-10-25. Research only.** Upstream original [SatAgentSimulation PR #84](https://github.com/clonetimer/SatAgentSimulation/pull/84) and [real Basilisk native CI #40156825783](https://github.com/clonetimer/SatAgentSimulation/actions/runs/40156825783), all 54 original real C-backed spacecraft/FSW/RW physical cases completed. Original upstream source artifact [#12585595456](https://github.com/clonetimer/SatAgentSimulation/actions/runs/40156825783/artifacts/12585595456). Exact original source JSON SHA-256 \`45ff210c05828873a4820fd7f2ea51f85e49578b47e88ed6deb8957222d5bc96\`. Original source archive ZIP SHA-256 \`71e841db657e3f17a65a28cc56fc2e2bd6c6de7f06267760614f8cae70ebcf4c\`.

[autogen original-source proof Actions #40157485052](https://github.com/clonetimer/autogen/actions/runs/40157485052) **PASS** on Linux Python 3.11/3.12, Windows 3.11, and separate original upstream source-hash/read-only source audit. Original downstream proof [artifact #12585047504](https://github.com/clonetimer/autogen/actions/runs/40157485052/artifacts/12585047504) downloadable and independently linked to source, proof JSON SHA-256 \`ff958047eed0e1412ebe590b47e2352e5e3c4e6f3f5d234d5a6f5701d4cbd4a4\`, proof ZIP SHA-256 \`3c53723a7129c614199e443076d1587c74916562abd91dc39d941d102ef6d4df\`.

Original 54 native simulations = three wheel identities × three distinct deterministic initial attitudes × two references (zero/scheduled 0.5°) × three motor-capacity states (original 0.020, loss75 0.005, loss95 0.001 N·m). There are **nine target-wheel/initial-state groups**, **18 original healthy cases reused** across severity comparisons, **36 paired healthy/fault contrasts**. The groups do not constitute independent spacecraft/independently procured equipment.

| Original matched physics contrast | Wheel-speed effect >1e-8 rad/s | Median maximum fault-normal RW speed difference |
|---|---:|---:|
| Zero-reference vs 75%-loss | 0/9 | 0 |
| Zero-reference vs 95%-loss | 0/9 | 0 |
| Scheduled 0.5° reference vs 75%-loss | 0/9 | 0 |
| Scheduled 0.5° reference vs 95%-loss | **9/9** | **0.0275243625 rad/s** |

Original all-wheel maximum allocated motor request **0.004881606711 N·m**, below the 75% loss residual capacity 0.005 N·m. Thus there is no physical torque saturation and no effect to diagnose under the predeclared 0.5° reference: **75% torque authority loss was nonmanifesting in this simulator source**. A 95%-loss residual 0.001 N·m is exceeded by the same pulse and produces C-backed actual-applied torque and RW speed differences. Source peak RW speed **31.479712328 rad/s**, source peak hub rate **0.000828166403 rad/s**. Original research observation guards (request ≤0.020 N·m; wheel speed ≤900 rad/s) passed; not actual mission safety certification.

## Material source-record issue found AND DISCLOSED

The original upstream exporter has **54/54 corrupted target-wheel/group-identifier metadata entries** from a Python variable-shadowing mistake: local integer \`wheel\` was overwritten by the last original native **motor request vector**, then accidentally serialized into the source-case metadata. This does **not** change wheel-number in its case ID, the actual native C-backed RW object being derated, the original \`ReactionWheelData[target].u_max\` readback, the event \`physical_fault_target_rw_index_QA_ONLY\`, the **native** applied motor torque source or original RW angular velocities. The original archive SHA remains unchanged.

The downstream auditable verification \`r83l_original_basilisk_multiwheel_physics_gate.py\` expressly enforces the original vector-shadow signature and **reconstructs trust from case ID plus physical C++ event and per-wheel original messages**. It records \`original_case_exporter_metadata_shadowed_count=54\`, \`original_source_metadata_wheel_target_trusted=false\`, and rejects unrecognized substituted metadata. The future upstream exporter has been corrected and a regression test added, but the original native research archive was **not** rerun or retrospectively edited.

**Limitations:** This is real Basilisk *physics source QA*, not fault diagnosis accuracy, noise-robust classification, a generalizable equipment source, proof that self-organizing networks beat the fixed baseline, or flight safety. Zero diagnostic model training; zero protected VALIDATION, TEST, FINAL. Original R83f/R83g \`NO_ELIGIBLE_DESIGN\` decisions remain unchanged. No Stable promotion or research PR merge.

**Next necessary scientific gate:** preregister a NEW safe native FSW reference trajectory with actual motor request above the 75%-loss residual capability, verify original hardware/mission safety bounds and independent physical conditions, then build a new rigorously grouped genuine Basilisk dataset. Only after source integrity and observability gates pass should any fair model comparison be revisited.
