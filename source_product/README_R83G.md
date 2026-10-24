# R83g: REAL Basilisk TRAIN-only command-response representation ablation

**Research-only Draft. Source: immutable R83f 126-case Basilisk run, GitHub Actions #40099391872.**

R83f returned **NO_ELIGIBLE_DESIGN** across all three fair model-search arms under predeclared accuracy >=0.90, worst environment-class recall >=0.80 and healthy FAR <=0.10. The full R83f physics and TRAIN-only negative result remains immutable. This new study must not silently reinterpret it.

## Scientific question

Do first-order, physically interpretable command/response features offer better **SOURCE TRAIN group-OOF development scores**, particularly for low-observability torque-authority-loss faults, than an exactly dimension-matched control?

Two representations receive the *same exact* 60 original TRAIN Basilisk records and physical groups:
- **zero_padded_raw**: identical original six channels, with 12 additional identically zero channels.
- **causal_command_response**: identical original six channels, with 12 **past/current only** derived channels (for all three wheels equally).

Each per-wheel derived set is: speed slope Δω/Δt, integral of prior commanded torque ∫u_cmd dt, speed since the first observed sample ω(t)-ω(0), and a lagged command-speed product u_cmd(t-1)ω(t-1). All four are algebraic proxies; *commanded torque is NOT measured actual applied motor torque*, so neither integral nor product is proof of actual physical actuator work. The first time step derives zeros, subsequent values use only samples up to the current time. **No fault-onset time, injected-wheel index, labels, fault severity, true applied torque, simulator hidden states, operating regime or physical lineage enters inference features.** Those are protected metadata used solely for offline supervision/physical-group OOF separation.

Both arms have **18 total input channels**, with the same registered backbones, fixed and feedback topology proposals, first two genomes, random-seed policy, 8 neural epochs, group OOF, 3 candidate slots, 10-fit cap, 1800-second wall cap per model arm and unchanged rejection gates. The command-response arm may nonetheless use more floating-point operations due to nonzero derived features, and self-organization may select a different third genome. Equal candidate-fit cap is not true FLOP-equivalent NAS. Training OOF results are **not** independent performance estimates.

## Data-integrity and no-leakage contract

The trusted R83g source intake stages **only TRAIN CSV bytes** from the previously generated R83f class-balanced development dataset and verifies exact pinned Basilisk native source code/wheel binding. Strict R83f physical-group metadata and raw observation digests are verified before any transformation. Expected TRAIN digests:

- TRAIN metadata SHA-256: `adac20dd8b4fa7952bcc1b743ba6f8111dd36c1113a507336394f3f0fd5e2f3a`
- TRAIN six-channel observations SHA-256: `10d730843fe7488c0fc670bc220176cf4f602a7dedca26aed37f68ee80e7cd36`
- 126-case preregistered source-plan canonical SHA-256: `c54bfcd154322a620c7d8058ce8a40db6e155602593c5ff5da8bdba43ef3b714`

The [R83g actual research workflow](../../.github/workflows/r83g-real-train-causal-ablation.yml) downloads the already completed [original R83f physical evidence archive](https://github.com/clonetimer/autogen/actions/runs/40099391872), verifies its original source manifest and TRAIN report file SHA-256, and reuses original 60 physically generated source TRAIN cases only. It does not run new physics, change a source physical regime, sample severity or train/test split, and never opens 24 VAL source CSVs. The original R83f refusal remains unchanged.

## Comparison procedure

Every representation independently executes the same three-arm R83b portfolio comparison:

| Frozen method family | Zero-padded raw (18 channels) | Causal command-response (18 channels) |
|---|---|---|
| Fixed TCN/GRU/ExtraTrees portfolio | 3 candidates | 3 candidates |
| Static preregistered temporal-network genomes | 3 candidates | 3 candidates |
| OOF-feedback proposed self-organizing topology | 3 candidates | 3 candidates |

Original group/label/environment metadata are identical across all six cells. Every estimate is SOURCE TRAIN-only group OOF and every arm applies exactly the R83f thresholds. All fail-closed refusal results must be retained. Even if one arm becomes eligible, it is not automatically promoted or used for VALIDATION: R83g is expressly exploratory. Confirmatory use requires **a new prospective, independently sequestered physical cohort**, a predeclared protocol and a separate R84 final review.

## Verification

```bash
python -m pytest -q source_product/tests/phm_design/test_r83g_command_response.py
```

Linux Python 3.11/3.12 and Windows Python 3.11 contract tests cover exact causal lag semantics, zero padding, unexpected truth input refusal, protected role exclusion, original unchanged records, and equal candidate/group/seed budgets. Actual model scores must be quoted only after examining the separate real-TRAIN Action and its data-bound audit receipt.

Scientific finding from R83f still stands unless prospective evidence supersedes it: **self-organizing topology superiority NOT DEMONSTRATED**.
