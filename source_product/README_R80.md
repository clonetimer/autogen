# R80 / 0.8.0.dev18 — Device-agnostic Industrial TaskSpec compiler (G0)

**Status: developmental contract only.** Does NOT train, rank or qualify a new model.
Built from the R78 research branch; R60 / 0.7 Stable remains unchanged.

## Why this exists

The industrial diagnosis AutoDesign PRODUCT is not a satellite fault classifier.
Basilisk satellite telemetry is the first physically controlled test adapter;
a different industrial device must be expressible without edits to the shared
task compiler or a forced five-class two-component graph.

This module creates a strictly validated, **non-executable** `AutoDesignPlan`
from an equipment-agnostic TaskSpec. It is a preliminary compiler, not yet
the unified R81 learner/trial executor.

## New Python API

```python
import json
from phm_design.industrial import TaskSpec, compile_task_spec, validate_observations

raw = json.load(open("examples/industrial/industrial_bearing_classification_demo.json"))
identity = TaskSpec.from_dict(raw)           # immutable canonical JSON + sha256
plan = compile_task_spec(identity)          # no data read, model fit or search
assert plan["status"] == "PLAN_ONLY_SOURCE_DATA_NOT_AUDITED"
assert plan["qualified"] is False
```

CLI:

```bash
PYTHONPATH=src python -m phm_design.industrial \
  --spec examples/industrial/satellite_eps_joint_demo.json \
  --output /tmp/eps_design_plan.json

PYTHONPATH=src python -m phm_design.industrial \
  --spec examples/industrial/industrial_bearing_classification_demo.json \
  --output /tmp/bearing_design_plan.json
```

The CLI also accepts `--train-manifest` for **TRAIN-only metadata**; data
roles FINAL_ID/FINAL_OOD are explicitly refused. Do not provide sealed files.

## Contracts implemented

- Variable-size ontology, legal component/mechanism mapping and per-task DAG
  output (detection, classification, localization, joint diagnosis).
- Sensor/command/time distinction, optional healthy prefix and offline vs
  causal-past-only task semantics; ground truth, world/environment IDs, split,
  fault injection and decision-future channels cannot be inference input.
- Predeclared bounded architecture/objective grid: linear/tree/CNN1D/TCN/
  GRU/Transformer and ERM/GroupDRO/V-REx where supported; **no model is trained**
  and no SOTA/generality claim is made.
- TRAIN metadata support audit: no protected data roles, physical group
  consistency, at least three independent groups and at least two independent
  groups per class, plus three fully class-supported training regimes before
  domain-robust objectives are fit. Group and environment metadata are for
  partition/loss only, never a neural deployment feature.
- Reproducible SHA-256 TaskSpec identity, explicit stopping conditions and
  `qualified=False` on all plans. No inheritance of R60 certification.

## Current examples are schema fixtures, not verified adapters

- `examples/industrial/satellite_eps_joint_demo.json`: five-category EPS
  dual-component diagnosis. Field names and physical semantics remain
  **illustrative**, not yet verified against the exact Basilisk native
  telemetry schema and message call chain.
- `examples/industrial/industrial_bearing_classification_demo.json`:
  illustrative three-class bearing condition classification using vibration
  and motor current, with no assumed healthy prefix. Official Paderborn
  channel mapping, labels, instance support and research license must be
  audited in R82; **no Paderborn raw data is redistributed in this repository**.

The sample schemas intentionally upgrade R80's planning-stage string-only
sensor lists into explicit typed `{name, role, unit, available_at_decision}`
objects, add explicit `healthy_label` and per-fault `class_attributes`
and a measured fit/CPU budget. These additions close ambiguity present in
the planning draft and must be documented for adapters.

## Verifying

```bash
PYTHONPATH=src python -m pytest -q tests/phm_design/test_r80_industrial_taskspec.py
```

Run the R80 CPython 3.11/3.12 Actions workflow as well. No internet
retrieval, Basilisk installation, GPU or actual experimental data is needed.

## Not yet satisfied by R80

- **R81:** ONE shared full-trial executor for the generic feature-program search
  and task-graph neural model registry. R80 emits *plans* only.
- **R82:** actual Basilisk and experimental bearing device adapters with
  data/acquisition/physics observability evidence.
- **R83:** performance-comparable, budget-matched auto-designed neural algorithms
  on held-out machine/environment families.
- **R84:** separate FINAL_ID/OOD steward and one-shot scientific qualification.
- **R85:** time-to-detect and online industrial deployability. A causal
  TaskSpec compiling does NOT prove an online diagnostic algorithm exists.

Do not reuse revealed R71/R78 evaluations to select R80 features/thresholds
and then claim independent OOD results.
