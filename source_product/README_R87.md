# R87 — controlled fuzzy-premise learning and self-organizing rule ablation

**Draft research only, not a positive diagnostic/generalization finding.**
Autogen remains a device-agnostic **automatic industrial diagnosis algorithm
design** project. Original physical Basilisk campaigns are not changed.

## Why R87

R83–R85 perform neural topology/model-family search but were not neuro-fuzzy.
R86 implemented actual Gaussian membership + normalized TSK rule activation,
ridge-trained consequents, supervised rule growth, and prefix pruning, but its
membership **centers and widths were fixed after initialization**. R87
implements this missing mechanism, allowing a scientifically interpretable
factorial ablation rather than treating TCN growth as fuzzy rule learning.

## Implemented

`industrial/r87_premise_learning.py`: analytic gradient of softmax
cross-entropy with respect to all Gaussian antecedent **centers and
log-widths**, through normalized membership firing and a fitted TSK affine
consequent. Bounded alternating optimization:

1. Initialize antecedent centers on SOURCE TRAIN only, widths initially 1 in
   within-fold standardized units.
2. Ridge-fit all affine TSK consequent weights on that same TRAIN fold.
3. Compute analytic d(TRAIN cross entropy)/d(center, log-width) holding
   consequents fixed.
4. Try a bounded backtracked gradient move, ridge-refit consequent weights,
   and keep the move **only** if TRAIN loss decreases.
5. Repeat for at most the declared `premise_steps`; log ridge refit count,
   accepted gradient updates and learned width parameter count.

No held-out environment or protected VAL/TEST/FINAL data enter any parameter
update, line search, pruning, stopping or proposal.

`industrial/r86_neuro_fuzzy.py` now accepts:

```python
FuzzyRulePolicy(
    max_rules=3,
    ridge=0.05,
    complexity_penalty=0.012,
    premise_steps=3,
    premise_learning_rate=0.1,
)
```

Default `premise_steps=0` **preserves R86 behavior**.

## Same TRAIN-source factorial study

```python
from phm_design.industrial.r86_fuzzy_source_study import run_fuzzy_source_study
from phm_design.industrial.r86_neuro_fuzzy import FuzzyRulePolicy

receipt = run_fuzzy_source_study(
    task_spec, source_train_records,
    policy=FuzzyRulePolicy(max_rules=3, premise_steps=3),
    include_learned_premise=True,
)
```

| Mechanism | Gaussian premises | Rules | Model name |
|---|---|---|---|
| F0 | Frozen center/width | Fixed-rule count, unsupervised center proposal | `fixed_fuzzy` |
| F1 | Frozen center/width | Supervised growth and train-only pruning | `adaptive_fuzzy` |
| F2 | Learn center + width | Fixed-rule count, unsupervised center proposal | `fixed_fuzzy_learned_premise` |
| F3 | Learn center + width | Supervised growth and train-only pruning | `adaptive_fuzzy_learned_premise` |
| B | No fuzzy rules | Non-fuzzy linear logistic reference | `linear_reference` |

All five share the **same complete source TRAIN cases, grouped
leave-one-source-environment-out folds, sensor-only representations,
label ontology and predefined maximum fuzzy-rule candidate fits**. Crucial
limitations:

- F2/F3 perform additional ridge refits and gradient calculations compared
  with F0/F1; the source study **does not claim equal runtime, FLOPs,
  optimization opportunities or hardware-normalized compute**.
- The SOURCE TRAIN held-out folds are used for development reporting; repeated
  search use would bias a generalization claim.
- Prefix/pruning decisions still use in-sample TRAIN penalized cross entropy,
  not a separated calibration/validation cohort.
- Fixed vs adaptive centers differ in the proposal heuristic and can differ
  in selected rule count; recorded rule count is a measured *mediator*, not a
  randomized treatment.
- The implementation is a bounded **alternating Gaussian TSK fuzzy
  classifier**, not a reproduction of all classical ANFIS training protocols.

The result includes exact TRAIN observations and manifest digests, candidate
and ridge-fit counts, fold-wise rule counts and parameter accounting, standard
worst-environment-class recall / accuracy / healthy false alarm, and explicit
`fuzzy_mechanism_generalization_benefit_proven=false` and
`eligible_for_stable=false`.

## Regression

```shell
cd source_product
python -m pip install numpy scikit-learn pytest
PYTHONPATH=src python -m pytest -q \
  tests/phm_design/test_r86_neuro_fuzzy.py \
  tests/phm_design/test_r87_fuzzy_premise_ablation.py
```

The unit test differentiates the analytic Gaussian-center and log-width
gradients against central finite differences and tests actual numerical
TRAIN fits. Multi-platform CI covers Linux 3.11, Linux 3.12 and Windows 3.11.

## Scientific decision gate

**No claim of improvement is allowed yet**. To accept the mechanism as
useful, a future preregistered evaluation must:

- Use independent physical groups, multiple industrial equipment domains,
  and a new sealed source-separated FINAL cohort.
- Control supervised feedback and source sample counts; run multiple paired
  seeds; report confidence intervals, worst-class recall and healthy FAR.
- Add a **compute-matched** comparison or a quality-vs-compute Pareto frontier,
  including all ridge refits and gradient steps.
- Separate fuzzy membership value (B vs F0), gradient-updated premise value
  (F2 vs F0), adaptive rule value (F1 vs F0 and F3 vs F2), and interactions.
- Keep strict original R83f/R83g no-eligible-design results unchanged.
- Require independently reviewed sensor observability and data lineage for
  physically simulated satellite studies. Do not substitute synthetic Python
  traces for native Basilisk research data.

Draft PR only; no change to main Stable.
