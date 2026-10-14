# R76 — environment-and-strength-aware head screening (development)

**Base:** R75 / `phm-design 0.8.0.dev14`. **Current research development version:** `0.8.0.dev15`. This PR is stacked on R75, not a new Stable release or diagnostic qualification.

## Why
R75 prevents the same physical environment from entering both sides of an inner robust candidate fold. A remaining flaw was that candidate ranking by environment loss could still prefer a detection head with intolerable **weak-fault misses** or per-environment false alarms. Aggregate scores and post-hoc gate tuning are not a substitute for a predeclared head feasibility contract.

## New opt-in API

Python API example (TRAIN-only inputs and user-provided, validated world lineage):

```python
from phm_design.task_head_evidence import HeadEvidencePolicy

policy = HeadEvidencePolicy(
    require_fault_strength_strata=True,
    expected_fault_strata=('weak', 'strong'),
    min_per_fault_stratum=2,
)
# design_task_graph(...,
#     training_lineage=train_lineage,
#     environment_families=training_environment_families,
#     selection_mode='worst_environment',
#     outer_split_mode='leave_environment_family_out',
#     head_evidence_policy=policy,
#     fault_strength_strata=train_only_fault_strength_strata)
```


`HeadEvidencePolicy` can be passed to `design_task_graph(head_evidence_policy=...)` with `selection_mode="worst_environment"`, verified TRAIN-only `training_lineage`, and environment-family information. It checks per-environment support, detection fault recall and healthy false alarm rate, and per-head class recall/accuracy. An optional `fault_strength_strata` sequence supports **development-only** strength-stratified sensitivity assessment. When required, `expected_fault_strata` must be predeclared, and each stratum must have minimum support in **every** training environment. Strength metadata never becomes an inference feature.

After nested selection, full-pipeline outer-OOF checks also count weak-fault UNKNOWN, missed detections, incorrect localization and wrong mechanisms as joint misses. Candidates failing any condition receive `REJECTED_BY_HEAD_EVIDENCE`; if all candidates fail, the existing head/graph rejection path retains trial-level failure evidence and does not deliver an eligible model. Without an evidence policy, previous R75 behavior remains supported for regression compatibility.

## Nonclaims and mandatory boundaries
- **No new Basilisk, final-world, or hardware result.** This stage uses synthetic TRAIN-only negative controls.
- The source of fault-strength labels must be audited: simulation injection parameters may define **training strata**, but must never enter deployed observations. Stratum labels or environment names alone cannot establish physical independence.
- It does not quantify unobserved weak fault modes; incomplete/untruthful metadata cannot be fixed by this module.
- Repeated outer OOF design feedback is development evidence, not sealed independent qualification.
- A rejected design is an expected scientific outcome, not a test failure.
- Downstream per-environment checks use conditional fault-head data, so they do not substitute for end-to-end all-fault recall or UNKNOWN accounting; those remain in `TaskGraphPolicy`.

## Test
`python -m pytest -q tests/phm_design/test_r76_head_evidence.py tests/phm_design/test_r75_environment_exclusive_search.py`

Full CPython 3.11/3.12 CI and baseline regression results must be checked before promoting this Draft PR. The frozen 0.7.0 Stable remains on `main`.
