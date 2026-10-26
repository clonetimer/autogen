# phm-design 0.8.0 Stable Core

**Software platform Stable (core scope), released 2026-10-26.**

This version stabilizes the installable `phm-design` core and retains its previously qualified limited offline SAT/Basilisk capabilities. It includes the R93–R99 industrial diagnostic algorithm-design research APIs **as opt-in research interfaces**, not as independently qualified equipment models.

## Supported core
- Python 3.11 and 3.12 (Linux), Python 3.11 (Windows); a wheel and source archive with published SHA-256 checksums.
- Existing `phm-design` core design and command-line interfaces with frozen 0.7.0 limited simulation-domain qualifications retained but **not expanded**.
- R93 EquipmentCard/DatasetCard/TrialSpec contract checking, train-only design boundaries, and reproducible R95 SOURCE-only candidate-plan generation.
- Preserved research access to preprocessing, physical-signal graph design and source-OOF review helpers, which remain experimental.
- Regression-compatible older APIs, verified on the tested platforms, with the former 0.7.0 artifacts retained for rollback.

## Not part of Stable Core
- Real `clonetimer/torch_like` (private optional provider) native model fit/predict, model persistence and reload: **NOT VERIFIED** and **NOT PROMISED** by this release. The user explicitly requested that this remain nonblocking. Do not mistake provider-shaped synthetic regression for that integration.
- Complete AAS platform host integration, turnkey industrial production deployment, independent multi-bearing algorithm generalization, model-performance superiority, matched CPU/GPU resource budgets, and safety-critical/flight control.
- Other modules beyond the released and declared core contract are research use only.

## Reproducibility and verification
The GitHub `phm-design 0.8.0 Stable Core -- Build, Test and Release` workflow builds, checks and attaches both same-commit artifacts only after 3-platform clean-wheel SOURCE-only smoke tests, 3-platform R93–R99 contract regressions and full historical Python regression pass. Downloaded archives must match the attached `SHA256SUMS`; the GitHub Release tag is bound to the exact main merge commit.

**Known optional skips must be read in workflow logs**; green regression does not by itself confer application-level or model-level scientific quality.

Full scope, commands and rollback requirements: [source_product/docs/STABLE_080_RELEASE.md](source_product/docs/STABLE_080_RELEASE.md).

**Research only. Not for spacecraft flight safety, medical, or other safety-critical equipment control.**
