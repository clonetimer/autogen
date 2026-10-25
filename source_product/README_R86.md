# R86 — evaluate *actual* self-organizing neuro-fuzzy learning rather than architecture mutation

**Scientific status:** exploratory, source-TRAIN only, not an independent
fault-diagnosis qualification; negative conclusions from R83f/R83g stand.

## Critical distinction

R83–R85 implemented a feedback-driven **neural architecture/algorithm search**.
They did not implement fuzzy memberships, a fuzzy rule layer, or supervised
growth of fuzzy rules. Therefore those experiments cannot establish a
neuro-fuzzy generalization advantage. ANFIS/TSK neuro-fuzzy systems have
explicit fuzzy antecedents and trained consequents (Jang, 1993,
DOI:10.1109/21.256541); merely changing convolution width is not ANFIS.

R86 provides a **restricted Gaussian TSK neuro-fuzzy pilot**, not a faithful
end-to-end gradient-trained ANFIS reproduction:

- Observed temporal sequences -> deterministic mean/std/end-start features.
- Each antecedent uses a Gaussian membership product (computed safely in
  log-space). Fuzzy rule activations are normalized.
- Each TSK rule has a learned linear class-output consequent, fitted by
  regularized linear least squares on **current training fold only**.
- Fixed-rule control grows centers using unsupervised farthest-point novelty.
- Self-organizing mode proposes centers using *source training residual*
  times novelty and may select/prune the trailing proposed rules based on
  *training in-sample* cross-entropy plus a declared rule-complexity penalty.
- Both fuzzy arms fit the same number of candidate rule models per fold and
  have the same maximum-rule budget. Parameter count and FLOPs can differ
  after pruning. Neither method tunes Gaussian bandwidths by gradient, and
  the current training-only rule selection can overfit.

Source code:
`src/phm_design/industrial/r86_neuro_fuzzy.py`
and `src/phm_design/industrial/r86_fuzzy_source_study.py`.

## Source-only experiment

```python
from phm_design.industrial.r86_neuro_fuzzy import FuzzyRulePolicy
from phm_design.industrial.r86_fuzzy_source_study import run_fuzzy_source_study

# A valid industrial TaskSpec, source TRAIN trajectories and complete
# environment/class support are required. NO TEST/FINAL records.
receipt = run_fuzzy_source_study(
    task_spec, source_train_records,
    policy=FuzzyRulePolicy(max_rules=3),
)
```

Registered development arms:

| Arm | Fuzzy? | Source-group fold fitting | Rule adaptation |
| --- | --- | --- | --- |
| Linear reference | No | Train fold only | None |
| Fixed Gaussian TSK | Yes | Train fold only | Unsupervised farthest centers |
| Adaptive Gaussian TSK | Yes | Train fold only | Residual-guided growth + prefix selection/pruning |

All three see the **same source trajectories**, same folds, same decision
features and healthy/fault labels during *fit* only. They are not compute
matched with the linear reference (1 fit vs K fuzzy candidate fits per fold).
Fixed vs adaptive fuzzy use equal candidate fit counts, **not** equal
parameters or CPU work. The fuzzy rule-growth decision uses only current
fold TRAIN residual, never the held-out fold. Held-out environment OOF
scores are only for development comparison — using them repeatedly to
select designs makes the resulting performance estimate optimistic.

Fail-closed rules: no FINAL/TEST worker entry, source TRAIN metadata and
group isolation enforced, unknown observation channels forbidden, no
truth/healthy counterfactual as predictor, classifier report never
independent-qualified. No confidence-calibrated or event-level delay output.

## What can be concluded today

1. **R83–R85 self-organizing neural topology algorithms are implemented and
   code-tested, but not shown to improve independent generalization.**
2. **R83f original Basilisk source-TRAIN comparison refused all three arms**:
   fixed ExtraTrees 0.8833 accuracy, worst-class recall 0.40, healthy false
   alarm 0.0667; static and feedback-structure TCN both 0.6333 accuracy,
   worst-class recall 0.20 and healthy false alarm 0.5333. These are
   TRAIN-OOF values, not independent blind-test estimates.
3. **The neuro-fuzzy mechanism has not yet received an independent efficacy
   experiment**, and cannot inherit performance claims from R83–R85.
4. R86 tests establish that actual fuzzy-rule learning, deterministic
   inference and leak refusal can run on SYNTHETIC equipment data;
   they do NOT establish practical superiority or physical fidelity.

## What is necessary for a causal contribution claim

Preregister an experiment that isolates:
(A) non-fuzzy neural backbone,
(B) fixed rule fuzzy model,
(C) learnable-premise fixed-rule fuzzy model,
(D) self-organizing rule grow/prune with the same premise tuner.
Control representation, preprocessing, TRAIN/CAL/VAL/TEST groups, seed
schedule, search and fit budgets; report actual FLOPs/wall/runtime and
model size, multiple source domains and independent machines, confidence
intervals and paired effects by physical group. Create a **new sealed FINAL
cohort** unavailable during search. If no consistent improvement appears,
reject the incremental neuro-fuzzy contribution hypothesis rather than
weakening the threshold post hoc.

## Local / CI smoke

```shell
cd source_product
python -m pip install numpy scikit-learn pytest
PYTHONPATH=src python -m pytest -q tests/phm_design/test_r86_neuro_fuzzy.py
```

R86 is **Draft-only**, stacked on R85, and does not alter the limited 0.7.0
Stable main or any frozen native Basilisk study.
