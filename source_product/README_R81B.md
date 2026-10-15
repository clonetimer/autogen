# R81b / 0.8.0.dev20 — Generic feature-program and variable diagnosis-head search

**Status:** Draft research feature, stacked on R81a. No changes to R60 Stable.
The purpose is to move from a task compiler with a nominal multi-head DAG to
an actual *fitted* multi-head alternative, while the same trial engine also
searches several existing physics-neutral time-series representations.

## Implemented (not just a planning diagram)

1. **Selectable feature programs:** R80's `design.representations` is
   optionally a predeclared subset of `temporal_bins`, `random_kernels`
   and `difference_kernels`. The existing `open_design` numerical atoms are
   reused verbatim and passed to actual sklearn linear/extra classifiers.
   Neural CNN/TCN/GRU/Transformer candidates retain learned sequence input.
   Training and deployment compute the exact same deterministic transforms.
2. **Selectable diagnosis topology:** `design.topologies` may contain
   `direct`, `factorized`, or both for `joint_diagnosis`. Direct fits a
   whole-class model. Factorized fits a detection head on all TRAIN samples and
   separate conditional component and mechanism heads on TRAIN faults.
   Inference combines **only legal class tuples from the TaskSpec**. It
   supports an arbitrary number of classes and component/mechanism categories;
   duplicate component–mechanism pairs are rejected *before training*.
   One-valued conditional heads are constants, not fictitious trained nets.
3. **Real architecture × representation × objective × topology trials:**
   one `TrialSpec`, source-only leave-entire-environment-out folds, identical
   end-to-end class-level recall/FAR screen and audited selection for direct
   versus factorized alternatives. Factorized heads are either sklearn or
   real PyTorch neural models, with available ERM/DRO/V-REx training objectives.
4. **Fit accounting:** each trained head consumes a model fit; insufficient
   budget refuses search or refit, including cases where an earlier feasible
   candidate would conceal untested choices.
5. **Compatibility:** If TaskSpec omits both new optional design properties,
   legacy R80/R81a behavior is preserved (direct model and temporal bins).

### Demonstration TaskSpec fragment

```json
{
  "design": {
    "candidate_max": 14,
    "compare": ["linear", "tree", "tcn"],
    "objectives": ["ERM"],
    "representations": [
      "temporal_bins", "random_kernels", "difference_kernels"
    ],
    "topologies": ["direct", "factorized"],
    "allow_refusal": true,
    "budget": {"max_fits": 120, "max_cpu_seconds": 3600, "seed": 819}
  }
}
```

Other fields use R80's required full TaskSpec. The total of 14 includes
12 tabular and 2 learned-sequence candidates. A too-small predeclared trial
budget fails before fitting, rather than silently dropping models.

### Run tests

```bash
PYTHONPATH=src python -m pytest -q \
  tests/phm_design/test_r81b_feature_and_variable_heads.py \
  tests/phm_design/test_r81a_unified_design_engine.py \
  tests/phm_design/test_r80_industrial_taskspec.py
```

Expected controls include: cross-equipment same-executor usage, variable
1/5/6-fault-category task structure, actual factorized PyTorch TCN heads,
source-only OOF grouping, strict model refusal, lineage-role protection,
invalid representations/ontology and per-head model-fit budget enforcement.
Development-only functional tests use predeclared permissive acceptance
thresholds; those are NOT scientific diagnostic qualification criteria.

## Mandatory scientific caveats

- **This is not an independent generalization result.** Data in R81b
  integration tests are synthetic source TRAIN observations only. R71 and R78
  revealed OOD targets are not used as newly sealed tests.
- **Conditional independence is a modeling assumption**, not a physics fact.
  Multiplying component and mechanism head probabilities and renormalizing
  onto the allowed class ontology can underperform direct classification.
  The evaluator must choose using source-only metrics and may reject both.
- More models and representations do not guarantee better weak-fault
  detection. Source data must make the fault physically observable.
- The model is still limited to offline complete trajectories. Physical
  Basilisk and experimental bearing adapter verification is part of R82.
- Full R81 is **not finished**: feedback-driven feature-program composition
  grammar, model calibration/UNKNOWN, measured FLOPs/runtime parity and hard
  process resource enforcement remain R81c.
- Stable 0.7.0's original 15 limited Basilisk simulation cases are unchanged;
  no industrial field approval or certificate is granted here.
