# R78 — Neural Risk Extrapolation for PHM AutoDesign (0.8.0.dev17)

**Research-only / NOT a qualified diagnosis release.**
R78 is stacked on the R77 neural domain-generalization PR and leaves
the frozen R60 / 0.7.0 Stable branch and bundled certified-model hashes intact.

## What changed

- Four new trainable V-REx network variants (`rex_cnn1d`, `rex_gru`,
  `rex_transformer`, `rex_tcn`) using the same underlying encoders as ERM and
  GroupDRO candidates. These are neural networks trained by backpropagation,
  **not** threshold-based fault rules.
- The V-REx objective is mean *training-environment* cross-entropy plus a
  predeclared weighted variance of training-environment risks. The penalty
  starts after a warm-up period. This is an adaptation of Krueger et al.,
  *Out-of-Distribution Generalization via Risk Extrapolation*, ICML 2021:
  https://proceedings.mlr.press/v139/krueger21a.html
- V-REx is selectable inside R75's nested leave-one-environment-family-out
  search alongside standard ERM and GroupDRO models. A new optional
  `neural_design_profile='compact_domain_generalization'` selects three
  predeclared trainable candidates per detection/localization/mechanism head:
  ERM / GroupDRO / V-REx on TCN / CNN / GRU backbones respectively. This is a
  bounded **automatic neural search profile**, not a single hand-selected
  network or an unbounded NAS procedure. It requires
  `selection_mode='worst_environment'`,
  `outer_split_mode='leave_environment_family_out'`, a verified
  `training_lineage`, and corresponding `environment_families`.
  Existing manual candidate specs and automatic profile are mutually exclusive.
  Verified TRAIN lineage
  and environment-only train folds are mandatory for all domain-aware learners.
- Per-fold normalization is fitted **only** on the training fold. Fault labels
  and environment IDs influence training losses but cannot enter deployed
  observation channels. No FINAL/qualified dataset is opened by this feature.

## Default search profile remains conservative

This automatic neural profile is **opt-in**. The old lightweight sklearn
profile continues to work without expensive neural search. The compact neural
profile currently limits each head to three candidate objective variants and
uses CPU training. It does not tune hyperparameters on heldout environments or
make neural model acceptance automatic.

## Stress-test protocol (development-only)

`python research/r78_neural_shift_benchmark.py`

A deliberately challenging synthetic process has a small stable fault waveform
and a larger label-correlated nuisance sensor. Source environments differ in
nuisance amplitude; an isolated target reverses the nuisance sign. ERM-TCN,
GroupDRO-TCN and V-REx-TCN use fixed seeds and roughly equal optimizer-step
budgets. The target is generated *only after* training is complete and is not
used for model selection.

The first frozen run at commit `3d7979d` is available in
[Actions run 38685417982](https://github.com/clonetimer/autogen/actions/runs/38685417982).
Mean unseen synthetic accuracies: ERM-TCN 5.36%, GroupDRO-TCN 1.79%,
V-REx-TCN 1.79%; all three healthy false-alarm rates were 100%. These
are **failures**, not positive OOD evidence. The synthetic target has now
been revealed. The automatically repeated OOD scoring step has been
removed from Dev Guard to prevent repeated feedback from contaminating
subsequent neural architecture design. The original script and audit remain
for reproducing a revealed development result, never for new qualification.

This **is not Basilisk**, is not a faithful satellite-thermal dynamics
simulation, and does not provide independence after a human has examined the
target outcomes. All outputs must be treated as REVEALED_SYNTHETIC_DEV_STRESS
and cannot establish a new qualified result. Different per-step batch sizes
also mean optimizer steps are not identical FLOPs.

## Verification and next work

`python -m pytest -q tests/phm_design/test_r78_neural_risk_extrapolation.py`

The next prerequisite for a meaningful R79 comparison is an independent
Basilisk data-generation protocol with environment-family disjointness, true
sensor observability, minimum weak-fault supports, and no FINAL feedback during
feature/architecture design. Fixed baselines, GroupDRO and V-REx should
compete under identical field, physical lineage and evaluation contracts.
No method should be assumed to win; allow scientific rejection.

**Limitations:** CPU development trainer only, a fixed five-class ontology for
current task graph, no real flight qualification, no CUDA hardware validation,
and no demonstrated OOD performance improvement.
