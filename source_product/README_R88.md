# R88 — frozen source-separated neuro-fuzzy VALIDATION steward

**Status: exploratory Draft PR, no independent industrial efficacy claim.**

The goal of `autogen` is automated discovery, assessment, and delivery of
generalizable industrial fault-diagnosis algorithms. R83–R87 developed
SOURCE TRAIN mechanisms; R88 separates frozen model creation from read-only
reviewer-supplied VALIDATION so that experimental evidence can be collected
without giving a design worker protected labels.

## Protocol

**Phase 1: SOURCE TRAIN (design worker)**

```python
from phm_design.industrial.r86_neuro_fuzzy import FuzzyRulePolicy
from phm_design.industrial.r88_fuzzy_validation_steward import freeze_fuzzy_source_models

freeze = freeze_fuzzy_source_models(
    task_spec, source_train_records, "frozen_r88",
    policy=FuzzyRulePolicy(max_rules=3, premise_steps=3),
    evidence_origin="UNVERIFIED",  # or reviewer documented; never auto-certified
)
# Archive the returned exact manifest SHA-256 before any VALIDATION read.
```

- Requires valid TaskSpec, original TRAIN roles, strict group/metadata contract.
- Executes five-arm source-TRAIN LOEO study, then **fits five predeclared
  models solely on source TRAIN**. Frozen set: linear reference, fixed fuzzy,
  self-organizing fuzzy, fixed fuzzy with learned Gaussian premise, adaptive
  fuzzy with learned premise.
- Saves one NPZ per arm with only numeric coefficient, normalization, learned
  center and width arrays; no pickle and no arbitrary Python code.
- Records source physical-group hashes and exact observation fingerprints,
  file digests, TaskSpec SHA, provenance declaration and underlying TRAIN OOF
  study digest. The output destination must not already exist.
- Source study may report model screening failure. The output is **exploratory
  research only** and must not be promoted based on independent validation.
  The stricter R83 frozen-eligibility protocol remains unchanged.

**Phase 2: reviewer-controlled VALIDATION (read-only)**

```python
from phm_design.industrial.r88_fuzzy_validation_steward import (
    evaluate_frozen_fuzzy_validation)

receipt = evaluate_frozen_fuzzy_validation(
    "frozen_r88", reviewer_validation_records,
    expected_manifest_sha256=freeze["manifest_sha256"],
)
```

- Accepts **VALIDATION only**; TRAIN/CAL/TEST/FINAL are hard rejected.
- Verifies manifest checksum and **all five model file hashes before loading**.
- Rejects source/validation group overlap, source trace replay, within-VAL
  relabeled duplicate traces and cross-environment reused physical groups.
- Requires at least two source-separated validation environments and two
  physical groups per environment/class cell.
- Loads only numeric NPZ with `allow_pickle=False`, does not call fit,
  threshold tuning, model selection or retraining.
- Reports heldout accuracy, worst environment/class recall, healthy false
  alarm, and group-paired 499-bootstrap intervals for five predeclared
  pairwise group-level accuracy contrasts.
- Reporting is *descriptive*. The evaluation function cannot enforce a
  one-time validation across separate invocations, and a caller could
  improperly revise a model after seeing the outputs.

**IMPORTANT provenance limit:** names and hashes do not authenticate real
physical independence, manufacturer IDs, sensor quality or dataset origin.
The caller must obtain reviewer/source attestation separately. A test fixture
with renamed groups is **not** an independently generated dataset.

## Scientific restrictions

- No FINAL/TEST processing code is exposed here.
- No claim of generalizable fuzzy benefit is permitted from synthetic
  smoke tests, self-reported cohort separation or one repeated VALIDATION.
- Even after VALIDATION succeeds, source regime characteristics, sample
  independence, noise, physical sensing, performance confidence and compute
  equality must be reviewed independently.
- Fuzzy learned premises cost extra ridge refits and gradient steps. A matched
  number of rule candidates or trained artifacts is not compute parity.
- Keep R83f/R83g negative Basilisk model results and limited Stable main.
- Before real deployment approval: authentic independent bearing/motor/
  gearbox cohorts, preregistered compute/fold budget and repeated seeds,
  expert-reviewed labels, original sensor/physics contracts, independent
  sealed FINAL steward, and objective acceptance rules.

## Smoke / CI

```shell
cd source_product
python -m pip install numpy scikit-learn pytest
PYTHONPATH=src python -m pytest -q tests/phm_design/test_r88_frozen_fuzzy_validation.py
```

CI runs Linux Python 3.11/3.12 and Windows Python 3.11. The positive test
uses **synthetic generated bearing trajectories**, explicitly relabeled and
shifted for the VALIDATION fixture: this checks software role separation,
not real out-of-distribution generalization.
