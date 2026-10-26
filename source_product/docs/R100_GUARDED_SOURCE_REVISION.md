# R100 — SOURCE-only diagnosis design revision admission (research Draft)

**2026-10-26.** Independent research increment built on the merged R93–R99 interfaces. `v0.8.0 Stable Core` is untouched (tag, release assets, 15 Basilisk-qualified offline cases, package version, existing public CLI and dependencies unchanged). **No R100 model is qualified or advertised as Stable.**

## Problem addressed

R99 explains why a vibration graph was proposed and which source-group scores failed, and suggests a frozen alternative. However, blindly following a suggestion can silently accept a regression in the **worst held-out SOURCE group**, healthy false-alarm fraction or a larger provider-fit cost. A fallback to a different algorithm family or to different held-out groups also confounds comparisons.

R100 adds an **admission/refusal receipt**, not a neural topology search, detector, threshold tuner or new evaluation result. It accepts only a frozen R98 graph proposal and SOURCE result, invokes the R99 explanation and checks alternatives already present in the same frozen study.

## API

`phm_design.industrial_contracts.guarded_revision` provides:

```python
from phm_design.industrial_contracts.guarded_revision import (
    freeze_source_revision_policy, audit_guarded_source_revisions)

# Construct before SOURCE fit/OOF scoring; timing must be attested externally.
policy = freeze_source_revision_policy(
    frozen_r98_study, max_fit_time_ratio=1.25,
    min_worst_group_gain=0.0)

# Only SOURCE TRAIN reports. No independent VAL/FINAL argument exists.
receipt = audit_guarded_source_revisions(
    policy, frozen_r98_study, r98_source_only_report)
```

The policy is SHA-locked and must have the exact original study identity. A **self-declared** policy freeze is not proof that its timestamp preceded observing source metrics. Source report bytes are hashed for change detection **but not cryptographically authenticated to an independent runner or real trained model**.

Each R99 failure has an explicit record. A suggested alternative is eligible **only** if all conditions hold:

1. It is a separately registered candidate in the original R98 study, never an invented architecture.
2. Original physical SOURCE fold group hashes **and full fit/scored record-ID membership** match exactly.
3. Both candidates use the **same provider algorithm family**; otherwise physics-graph gains are not identifiable by this contrast.
4. Target SOURCE worst-group accuracy meets the policy minimum gain and SOURCE group-macro accuracy does not decrease.
5. Healthy SOURCE false-alarm fraction does not rise (no threshold tuning).
6. Observed provider fitting seconds do not exceed the frozen relative cap (default 1.25×); this is a limited heuristic, not matched hardware compute.
7. At least one score or provider fitting time strictly improves. All other admissibility conditions remain mandatory.

A candidate that fails *any* gate receives specific reason codes and `REFUSE_AS_UNJUSTIFIED_SOURCE_REVISION`. A passed candidate receives `SOURCE_ONLY_CANDIDATE_FOR_NEXT_SEALED_STUDY`, **not** “improvement verified”. Unsupported source plan remains a refusal/abstention. No additional fits or protected data reads take place.

## Credibility boundaries

- The same SOURCE cohort has been used to rank alternatives. **Post-selection bias remains**, no independent validation has been performed, and the code cannot prove a causal failure mechanism.
- SOURCE candidate fit durations have **no authenticated hardware fingerprint** and exclude independently measured feature extraction, inference time, native memory or GPU/FLOP parity; therefore the time cap is not a scientific resource fairness claim.
- The full R99 independent measured-device steward/validation gate remains separate. This module must not access `VALIDATION`, `CALIBRATION`, `TEST` or `FINAL` truth or sensor recordings.
- This increment is not a production threshold or alarm-decision postprocessor, not an AAS evaluation platform feature, and does not extend real `torch_like` model fit/save/reload coverage.
- Exact per-fold identities come from the frozen plan, **not a trusted executor attestation**. Corrupted/untrusted source reports can still lie about performance; eligible output means merely that *submitted* SOURCE summary metrics meet the registered rule. The next research gate is to bind predictions and resource receipts to genuine frozen executable artifacts before external device review.

## Reproduction

At `source_product` root:

```bash
python -m pip install numpy==2.3.5 scipy scikit-learn==1.8.0 pytest==9.0.2
PYTHONPATH=src python -m pytest -q tests/phm_design/test_r99_design_review.py tests/phm_design/test_r100_guarded_revision.py
```

The [R100 three-platform research workflow](../../.github/workflows/r100-guarded-revision.yml) runs Linux Python 3.11/3.12 and Windows Python 3.11. Fixtures use deliberately synthetic signals and a spy provider, **not original CWRU/Paderborn measured-device scientific validation**.

## Next bounded research

Do not continue adding model families until the trusted evaluator can bind **exact model artifacts, fitted preprocessing artifacts, executed original signal processing, device/session identifiers, prediction streams and runtime cost** by verifiable hashes to the sealed candidate and independent raw measurements. Failure to obtain that evidence is an explicit scientific NO-GO, not a motivation to relax the blind-cohort split.
