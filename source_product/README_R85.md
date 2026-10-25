# R85 — adaptive algorithm-family search in industrial AutoDesign

**Development-only; not an independent diagnostic qualification.** The
project objective is **automated design of generalizable industrial fault
diagnosis algorithms**. SAT/Basilisk is only one physical validation domain.

## What was implemented

- `industrial/r85_family_policy.py`: small, registered family search controller.
- An opt-in `family_search_mode` in the **real R81 TRAIN-only design
  executor**, including its spawned wall-time guarded entry.
- A finite predeclared family pool: **linear**, **tree**, **TCN**, **GRU**,
  optionally CNN1d and Transformer, each using registered model builders.
  No arbitrary Python code, network scripts, or untrusted plugins execute.
- Two common anchor trials (linear, TCN). A third or subsequent trial is
  selected using **TRAIN source leave-one-environment-out metrics only**.
  An observed stronger linear anchor leads to a tree family; a stronger
  TCN anchor leads to a GRU family. Other untried families are bounded
  declarative neighbors. A static family order is the controlled ablation.
- Exact source TaskSpec and data/metadata checks, physical-group isolation,
  train-only fold normalization, budget charging, refusal and algorithm
  export remain in the original engine.
- Every source metrics-driven proposal is stamped
  `SOURCE_TRAIN_OOF_ONLY`; no TEST/FINAL, physically independent data, or
  deployed accuracy claims follow from the result.

## Opt-in API

```python
from phm_design.industrial.design_engine import run_design, DesignAcceptance

# Construct a valid TaskSpec with the four registered families, plain direct
# ERM topology, and sufficient source TRAIN support.  NO test records.
task_spec["design"]["compare"] = ["linear", "tcn", "tree", "gru"]
task_spec["design"]["candidate_max"] = 4
task_spec["design"]["budget"]["max_fits"] = 10 # 3 candidate x 3 env folds + refit

model = run_design(
    task_spec, train_records,
    acceptance=DesignAcceptance(),  # do not weaken without preregistration
    selection_rule="r83_matched_train_loss",
    family_search_mode="feedback",
    family_proposals=3,
)
assert not model.audit["independent_qualification"]
```

For a TaskSpec declaring a hard wall budget use the guarded API instead:
`run_design_guarded(..., family_search_mode="feedback", family_proposals=3,
selection_rule="r83_matched_train_loss")`.

To run a controlled same-budget static family sequence change ONLY
`family_search_mode="predeclared_static"`. With three proposals it uses
linear -> TCN -> tree; with feedback, the third trial can instead be GRU
depending on source TRAIN feedback. The seed of each candidate is attached to
its immutable **pool index**, not selection order. This yields equivalent
source groups, folds and maximum estimator fit allocation, **not** equal
FLOPs/parameter counts, wall time, hyperparameter-search complexity or
unbiased selection/test error. Record these differences.

## Automated tests

```shell
PYTHONPATH=source_product/src python -m pytest -q \
  source_product/tests/phm_design/test_r85_family_policy.py

cd source_product
python -m pip install -e '.[dev]'
python -m pytest -q \
  tests/phm_design/test_r85_family_integration.py \
  tests/phm_design/test_r81a_unified_design_engine.py
```

Synthetic industrial-bearing trajectories support **actual sklearn and Torch
training** in the integration test. They are generated fixtures, not real
industrial sensors or independent generalization tests. R83/R84 and main
Stable behaviors are untouched when the new mode is omitted.

## Scientific interpretation and next boundary

R85 improves the **search space** and actually runs feedback-selected
algorithms; it does NOT establish that source-OOF-guided search improves
generalization beyond fixed equal-compute baselines. Required before any
claim: independent equipment/cohort groups; frozen preprocessing and metrics;
multiple industrial domains (bearing, motor, gearbox, and independent native
satellite if physically observable), repeated seeds; same TRAIN/CAL/VAL/TEST
assignments and resource accounting; a sealed unbiased FINAL test.

Do **not** promote the original R83f/R83g negative cases. R85 stays Draft
and does not change the 0.7.0 limited Stable release.
