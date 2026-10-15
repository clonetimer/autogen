# R81a / 0.8.0.dev19 — Executable industrial AutoDesign vertical slice

**Research-only development stage** (stacked on R80 TaskSpec; NOT a new Stable).
Implements the first executable device-agnostic design loop: multiple candidate
learners share a *single* training record contract, group-wise source validation,
selection objective, risk acceptance policy and auditable model artifact.

## Scope actually implemented

- `phm_design.industrial.design_engine.run_design(task_spec, records)` fits
  **real sklearn and PyTorch models**. Each record contains only a strictly
  schema-checked deployment-visible time-series plus TRAIN-only label and
  group/environment metadata.
- The exact same API supports the R80 illustrative five-label satellite/EPS
  and three-label industrial bearing tasks. The training weights are
  task-specific; no claim of zero-shot cross-machine weight transfer.
- `linear`, `tree`, CNN1D, TCN, GRU and Transformer learners call the existing
  TaskCandidateRegistry. ERM/GroupDRO/V-REx objectives use the **same candidate
  trial framework** where supported; the common `temporal_bins` feature from
  the existing generic `open_design` code is reused for sklearn baselines.
  Other generic program grammars remain pending R81b integration.
- Leave-one-**entire-training-environment-family**-out OOF for all candidates.
  Each source fold fits preprocessing and models only from that fold's training
  examples; physical worlds cannot cross environment partitions. The model
  shortlist is selected from **TRAIN source** OOF metrics alone, with
  per-family class support, worst-family class recall and healthy false alarms.
- If no candidate satisfies a *predeclared* `DesignAcceptance`, raise
  `IndustrialDesignRejected` with all trial reasons and deliver **no model**.
  Otherwise refit only the selected candidate on TRAIN data.
- Export a versioned `model.joblib` and `audit.json` including SHA256 of
  the binary, selected trial, folds and entire evaluation ledger. Import of
  untrusted joblib files is unsafe and not supported.
- Explicit `qualified=False`, `target_feedback_used=False` and
  `final_evaluation_performed=False`. All metrics are *development design
  selection evidence*, NOT independent out-of-distribution qualification.

## Example Python usage

```python
import json
from phm_design.industrial.design_engine import run_design, DesignAcceptance

task_spec = json.load(open("examples/industrial/industrial_bearing_classification_demo.json"))
task_spec["design"]["compare"] = ["linear", "tcn"]
task_spec["design"]["objectives"] = ["ERM"]
task_spec["design"]["candidate_max"] = 2
# TRAIN-only records: measurements = exact rows of deployment-observable fields.
# metadata = {role: TRAIN, label: ..., bearing_instance_id: ...,
#             speed_regime: ..., radial_load_regime: ...}
result = run_design(task_spec, training_records)
print(result.trial)
print(result.audit["decision"])
result.export("/tmp/research_design")
# Predict accepts deployment-only time series, never label/group/fault truth:
label = result.predict(training_records[0]["measurements"])
```

Run test suite from `source_product`:

```bash
python -m pytest -q tests/phm_design/test_r81a_unified_design_engine.py
```

## Scientific and engineering boundaries

1. **NOT a sealed or independent test.** R71/R78 have already been revealed;
   neither their data nor target metrics are used as a new qualification score.
2. **NOT an online implementation.** Causal TaskSpecs are rejected by this
   R81a executor instead of silently using full future trajectories. R85
   carries actual online decision-time evaluation.
3. **NOT a Basilisk adapter or real bearing experiment yet.** The test data
   for both illustrative TaskSpecs are fully synthetic, share a deliberately
   simple injected signal and only verify common *software interfaces*.
   R82 will verify the native physics/sensors and real bearing data and license.
4. **Metadata integrity is not physical independence:** field names, reported
   machine IDs and environment IDs are caller-provided. A separate independent
   steward must audit these identities before FINAL evaluation.
5. **CPU budget is best-effort:** the runner checks process CPU time before fits
   and final refit. A single long-running model fit is not preempted. Hard
   resource enforcement requires an external supervisor.
6. **Not full R81 yet:** adaptive generic program grammar search, generalized
   task-graph head topology, calibration, missing-channel strategies and
   explicit fixed-FLOP comparisons are still pending. No R60 certified model,
   dataset or performance contract is inherited by this development artifact.

## Required next gates

- R81b: unify richer learned and physically derived feature programs and
  configurable diagnostic head DAG within the common TrialSpec/Audit surface.
- R81c: measured compute ledger, no-budget-overrun delivery, calibration and
  unsupported/missing-channel strategy with training-only fitted transforms.
- R82: actual Basilisk-native and independent experimental bearing dataset
  adapters and physical observability/instance split audit.
- R83/R84: frozen strong-baseline experiments and independently stewarded FINAL
  evaluation. No assumed neural gain or generalized industrial qualification.
