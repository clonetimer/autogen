# R83d: preregistered class-balanced REAL Basilisk campaign

**State:** opt-in research workflow on a stacked R83d Draft PR. No Stable
promotion. Source production is the **pinned SatAgentSimulation** native
Basilisk generator, not an AutoDesign Python dynamics approximation.

## Why regenerate the R82 source

The previous R82 physical run had a regime×fault cell with just one
independent group and too many nominal examples relative to each individual
fault class. There is not a defensible strict three-arm comparison on that
unbalanced cohort. Additional random draws cannot guarantee coverage.

R83d freezes the new experimental design BEFORE any actual trajectory is
produced. It does not change the RW friction/jam/torque-authority-loss
mechanisms, severity generator, source duration or controller implementations.

| Source split | Physical fault/nominal pairs per mechanism | Pairs per mechanism × regime | Selected samples per regime/class |
| --- | ---: | ---: | ---: |
| TRAIN | 15 | 5 | 5 |
| VALIDATION | 6 | 2 | 2 |
| TEST / FINAL | 0 | 0 | 0 |

Three mechanisms × (15 TRAIN + 6 VAL) pairs × 2 cases = **126 actual
native cases planned**. Every pair's fault and nominal cases share exactly
the same upstream factory source parameters and simulation seed. The
predeclared split_index % 3 selects inclination 29°, 47° or 65°. These
values lie strictly inside the already frozen R82 38° and 56° research
boundaries. This protocol does not use fault labels, metrics or sensor
telemetry to *assign* operating regimes. Other upstream physical
parameters and fault event severities are untouched.

All 45 TRAIN and 18 VALIDATION fault/nominal pairs remain in the raw
simulation campaign. To avoid a **3:1 nominal class excess**, AutoDesign's
trusted preparation keeps ALL 45 TRAIN fault records and ALL 18 VAL fault
records, but subsamples nominal twins deterministically from preexisting
pair IDs (SHA256 order) at the same per-regime count as any fault class.
The balanced cohort is therefore 60 TRAIN (5 × 3 regimes × 4 classes) and
24 VALIDATION (2 × 3 regimes × 4 classes). The excluded normal cases remain
available in the original unmodified native case evidence for audit.

The nominal-twin subsampling is fixed before model fitting, uses no
telemetry values and never selects cases by a model's accuracy. Physical
group IDs and source labels in the prepared manifest are used only by
the separately trusted preparation role to verify class quotas; they are
not model inference inputs. Both TRAIN and VAL physical groups retain their
original upstream lineage identifiers. A source physical regime worksheet
is generated from that preregistered native parameter (inclination) and
is explicitly marked as **NOT expert reviewed**. Source authenticity,
physical case independence and externally confirmed simulation quality
must still be independently checked.

## Available workflows

1. GitHub Actions: **R83d Balanced Native Protocol** (automatic PR
   software/dry-run). Runs real pinned SatAgentSimulation *factory PLAN*
   on Linux and Windows but NEVER runs any physics. The unit tests use
   synthetic metadata and CSV only.
2. GitHub Actions: **R83d REAL Basilisk Balanced RW Campaign**
   (workflow_dispatch only). Costs compute. Verifies the unchanged
   official Basilisk 2.11.0 wheel and pinned producer, freezes 126 source
   cases, actually runs them, checks original physics case contracts,
   prepares six deployment-input channels, creates equal-class views,
   trains all three prespecified models under the fixed screen and,
   only if all are eligible/frozen, runs the read-only VAL steward.
   Artifacts contain real-case evidence and explicit refusal receipts.

The plan receipt also writes frozen_plan_canonical_sha256, computed from
sorted-key UTF-8 JSON rather than platform-specific file bytes. The
automatic CI asserts this semantic digest, prints it in each matrix job,
and archives the exact planned tasks for review. Native source binaries
and full physical campaign receipts must be verified separately.

The manual workflow cannot be started through an ordinary PR CI push.
GitHub normally requires workflow_dispatch definitions to be discoverable
from the default branch; do NOT merge this research workflow into Stable
just to make it dispatchable. Alternatively, after the branch has passed
all gates, execute the documented source commands on an isolated native
Basilisk runner or allow an authorized operator to create a research-only
dispatch entry. No real run should be claimed from a dry-run CI result.

## Commands in a pinned native environment

~~~bash
# Native factory planning; does NOT execute physics.
python -m phm_design.industrial.r83d_native_protocol \
  --mode plan \
  --factory-config satagent/configs/integration/rw_dataset_factory.json \
  --output /runs/r83d_frozen_native_plan.json

# Actually execute the pinned native simulator (see the manual Action for
# official wheel verification, runtime setup, full gate and export steps).
python -c "import json; from sat_sim.fault_dataset_factory import run_fault_dataset_campaign; from pathlib import Path; spec=json.load(open('/runs/r83d_frozen_native_plan.json')); run_fault_dataset_campaign(spec,output_root=Path('/runs/r83d_actual_native'),continue_on_error=False,dry_run=False)"

# TRUSTED PREPARATION, after native campaign review + adapter.prepare()
python -m phm_design.industrial.r83d_native_protocol \
  --mode balance \
  --plan /runs/r83d_frozen_native_plan.json \
  --prepared-manifest /runs/r83d_prepared/development/dataset.json \
  --preparation-report /runs/r83d_prepared/preparation_report.json \
  --output /runs/r83d_balanced
~~~

Use the R83b TRAIN study CLI on the balanced development/dataset.json and
review/train_physical_map.json, with the separately pinned native runtime
binding and source checkout. Use R83c VAL-only staging and frozen evaluation
with review/validation_physical_map.json **only** after the frozen
three-arm models all passed TRAIN eligibility. The experiment never
constructs or inspects a TEST/FINAL split. Separate R84 final data
stewardship is still required for any release qualification.

## What can and cannot be concluded

- PLAN/CI PASS: deterministic native factory task specification and
  software intake/quota logic are operational. **No Basilisk dynamics ran.**
- Native execution PASS: a Basilisk-native candidate source was physically
  produced and mechanically reviewed. Still no independently audited
  real-world fault mechanism, source authenticity or flight claim.
- TRAIN comparison PASS: a bounded, fit-budget-matched *design exercise*.
  Same source and selected physical folds; different architectures can
  have different FLOPs and CPU time.
- VAL result: only frozen-model evaluation on held-out physical lineages
  from the **same simulator family**. No source-independent equipment OOD
  claim. The source regime map is provisional and the VAL 2 groups/cell
  is exploratory, not statistically confirmatory.
- R84 sealed independent FINAL: **not implemented or run here**. Strong
  generalization statements remain prohibited.

The declared five groups per TRAIN cell address R83b's recommended
minimum. The separate VAL cohort has only two per cell; increasing that
sample size would require a *new, preregistered* campaign, not post-hoc
changes. Any failure/refusal must be kept; do not loosen screening
thresholds after viewing performance.

## Software tests

~~~bash
cd source_product
python -m pytest -q tests/phm_design/test_r83d_native_protocol.py \
  tests/phm_design/test_r83b_matched_budget.py \
  tests/phm_design/test_r83c_frozen_validation.py
~~~
