# R83g actual source-TRAIN causal command-response ablation — 2026-10-24

**Decision: NO ELIGIBLE MODEL across both 18-channel representations and all three diagnostic design arms. No independent model generalization claim or production promotion.**

## Source/evidence identity

- GitHub Actions: [#40102279092](https://github.com/clonetimer/autogen/actions/runs/40102279092), completed successfully; code head `5826aa79d0d5a6a37671419ec31cdf8c9332116a`.
- Actual Basilisk source: prior immutable [R83f 126-native-case run #40099391872](https://github.com/clonetimer/autogen/actions/runs/40099391872), not fresh Python simulation, original source producer SatAgentSimulation at `1f709beeae97baec65db9ca13ae373d88728f87d`.
- Source-TRAIN six-channel observation fingerprint: `10d730843fe7488c0fc670bc220176cf4f602a7dedca26aed37f68ee80e7cd36`.
- Source-TRAIN metadata fingerprint: `adac20dd8b4fa7952bcc1b743ba6f8111dd36c1113a507336394f3f0fd5e2f3a`.
- The **60 source TRAIN records are 45 distinct physical groups**, because 15 normal/fault twins legitimately share a lineage ID. This is intentional paired physics, **not** 60 statistically independent groups. Each original regime/class cell has 5 distinct groups.
- No VALIDATION, TEST, FINAL traces accessed for model selection, no simulated physics proxies, no automatic model export or release.
- Original [R83g experimental audit artifact](https://github.com/clonetimer/autogen/actions/runs/40102279092) is `r83g-real-basilisk-train-only-command-response-comparison`; retrieved ZIP SHA-256 `1eb00bc3a4dfa67ae72bee11232e15b565c96baa456081768e30f876f958a156`.

## Locked before TRAIN measurements were compared

Two representations of **identical** R83f six-channel native TRAIN telemetry:
- **zero_padded_raw:** original 6 real wheel-speed/command channels plus 12 all-zero channels.
- **causal_command_response:** same original six channels plus **12 causal derived channels** for all three wheels: speed slope, integrated prior commanded torque, speed delta from trajectory start, and prior commanded torque × prior wheel speed (proxy; NOT true actual motor torque or power).

Both are 18 channels, and each goes through the same fixed TCN/GRU/ExtraTrees, static neural structures, and feedback-adaptive topology portfolios. Same data/group splits, initial two neural genomes, seed policy, 3 candidate slots per design group, **9 actual fold fits per group**, unchanged eligibility thresholds (accuracy >=0.90; worst environment/class recall >=0.80; healthy false alarm <=0.10), and TRAIN-only ranking loss `0.45(1-accuracy)+0.4(1-worst recall)+0.15 FAR`. Two representations × 3 model portfolios × 3 candidates × 3 environment OOF fits = **54 fold fits**. Same fit cap does not mean same FLOPs or statistical power. Self-organizing third topology can differ between representations.

## Measured results (SOURCE TRAIN group-OOF ONLY)

Every row below shows the top-ranked candidate by the common predeclared loss, **not** an independently qualified final model.

| Model-search arm | Representation | Best TRAIN accuracy | Worst environment/class recall | Healthy FAR | TRAIN selection loss | Decision |
|---|---|---:|---:|---:|---:|---|
| Fixed portfolio (ExtraTrees) | 18ch zero-padded raw | 0.88333 | 0.60 | 0.20 | 0.2425 | REFUSED |
| Fixed portfolio (ExtraTrees) | 18ch causal command-response | 0.88333 | 0.40 | 0.06667 | 0.3025 | REFUSED |
| Static registered neural topology | 18ch zero-padded raw | 0.61667 | 0.20 | 0.40 | 0.5525 | REFUSED |
| Static registered neural topology | 18ch causal command-response | 0.60 | 0.00 | 0.00 | 0.5800 | REFUSED |
| Self-organizing topology | 18ch zero-padded raw | 0.61667 | 0.20 | 0.40 | 0.5525 | REFUSED |
| Self-organizing topology | 18ch causal command-response | 0.65 | 0.20 | 0.53333 | 0.5575 | REFUSED |

For the same first two self-organizing/static neural genomes, both ports genuinely had identical candidate/seed contracts across representations; the 3rd topology in the feedback arm was allowed to differ by source TRAIN feedback. Source TRAIN sensitivity of that search process **does not** prove generalization benefit.

End-to-end per-portfolio research wall-clock seconds:

| Representation | Fixed | Static neural | Adaptive |
|---|---:|---:|---:|
| Zero-padded raw | 4.3628 | 3.7719 | 3.7611 |
| Causal command-response | 4.3910 | 3.7653 | 3.7097 |

These times are job environment observations only (not CPU-matched FLOPs or rigorous benchmarking). No candidate satisfied all three unchanged screens. Although the adaptive representation arm's top-ranked TRAIN accuracy increased (0.61667 to 0.65), worst environment/class recall **remained only 0.20** and its FAR rose to **0.53333**; its combined TRAIN selection loss was marginally worse (0.5525 -> 0.5575). Fixed ExtraTrees' best worst recall fell 0.60 -> 0.40 with enhanced features.

### Scientific interpretation and boundaries

1. **No evidence of generalizable improvement from these four algebraic command-response features.** In this specific R83f physical source, the defined features do not resolve the worst-class detection failure at the preregistered candidate budget.
2. **No support for claiming self-organizing search outperforms fixed/static model families.** Neural candidate-search scores are materially below the best fixed-tree arm; none qualifies.
3. **The low post-event torque-authority-loss wheel-speed response in R83f remains an observability hypothesis**, not established as the sole cause of classifier refusal. No target wheel or fault onset was provided to any model.
4. **VALIDATION not accessed.** The 24 unused R83f VAL cases are not a new independent confirmatory experiment after adaptive development; do not tune to them. R84 FINAL and equipment OOD remain sealed/unavailable.
5. Next experiments require predeclared **new** representations or broader independent physically diverse source regimes, fold-local physical parameter calibration, proper repeated seed/cost normalization, and ultimately a new unseen independent physical cohort. Do not retrospectively retune R83f/R83g acceptance, select “best-looking” derived features, or silently rewrite this report.

**Final decision: SOURCE TRAIN comparison executed fairly with a fail-closed refusal across all six arms.**
