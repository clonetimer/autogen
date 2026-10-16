# R82.0 — reviewed TRAIN-source intake (development-only)

**Status:** stacked draft on R81c; this is a source-contract/gating slice,
not a Basilisk simulation run, real bearing experiment, or independent OOD
qualification.

For the concrete ZIP/file inventory needed to perform an actual physical-data
validation, see [R82_REQUIRED_MATERIALS.md](R82_REQUIRED_MATERIALS.md).
 A declaration such as `source_kind=native_basilisk` is not
physical proof. Source binding hashes check file identity, not physical
model fidelity or independent sensor verification.

## Scope and API

The legacy adapter emits `phm.sequence-dataset.v1`, with physical
`group_id` / `lineage_id`, split roles, labels and CSV telemetry.
R80/R81 AutoDesign instead accepts complete per-trajectory measurement
rows plus TRAIN-only metadata specified in TaskSpec. R82 bridges these
**without giving the AutoDesign worker a FINAL/TEST manifest**.

```python
from phm_design.industrial.source_intake import (
    stage_train_only_view, intake_train_source,
)
from phm_design.industrial.design_engine import run_design

staged = stage_train_only_view(
    "campaign/development/dataset.json", "campaign/R82-train-only"
)
records, receipt = intake_train_source(
    task_spec,
    staged["manifest"],
    "reviewed_physical_map.json",
    native_binding_path="campaign/runtime_binding.json",
    native_source_checkout_root="/reviewed/native-simulation-project",
    native_wheel_factory_file="/reviewed/Basilisk/utilities/simIncludeRW.py",
)
# For bsk-backed datasets, verify physical construction and runtime linkage
# independently. 'receipt' DOES NOT make that verification by itself.
model = run_design(task_spec, records)
```

The staging API is a **trusted data-preparation** activity. It can see
TRAIN/VALIDATION metadata in a development manifest but copies **TRAIN
telemetry only**; it never opens validation measurement files. It refuses
TEST source roles. `intake_train_source` itself refuses any mixed-role
manifest, verifies each declared TRAIN file SHA-256, validates exact input
columns, units, label order, sample period and monotone timestamps.

It expects a separately reviewed, sample-ID-complete physical map:

```json
{
  "schema": "phm.industrial.source-physical-map.v1",
  "records": {
    "sample-id-from-source-manifest": {
      "physical_unit_id": "same-value-as-source-group_id",
      "operating_regime": "regime_1"
    }
  }
}
```

The example keys `physical_unit_id` and `operating_regime` must be
replaced by the exact TaskSpec `data.group_keys` and
`data.environment_axes`. The first R82 slice requires **one group key**
matching the legacy source `group_id`; it does not infer an environment
from case names, labels, injected-fault onset or numeric trajectories.
There must be at least three source environments with every declared fault
class represented. Future cross-domain tests need *new physical equipment
or operating families*, not overlapping windows from one run.

For `native_basilisk` + `simulation`, R82 additionally requires a
`runtime_binding.json` with existing source references and SHA-256
values for the native runtime, fault-effect source, and Basilisk wheel
factory. The reviewer MUST also provide an actual source checkout and the
installed Basilisk `simIncludeRW.py` file. The intake verifies all three
on-disk files against their declared SHA-256 and checks exact expected
source-relative paths. Missing files, bad hashes or substituted paths are
rejected. **Matching source hashes are still not evidence that the
corresponding simulation was executed, that its dynamics are correct, or
that the source was independently authenticated.** `real_equipment_unreviewed`
+ `real_unreviewed` may enter a **research TRAIN** experiment but is
never called qualified equipment evidence merely because it was ingested.
`software_synthetic`, missing binding, unknown roles and oracle-only
channels are rejected.

## R82.3 — native producer execution and evidence collection

R82.3 adds an **opt-in trusted operator-side** launcher for the real
`SatAgentSimulation` RW data generator. Execute it in the same Python
environment as the physically installed `bsk` runtime:

```bash
# Python package sat-sim and the installed Basilisk wheel must already work.
cd /path/to/autogen/source_product
python -m phm_design.industrial.native_campaign_execution \
  --source-checkout /path/to/SatAgentSimulation \
  --config /path/to/SatAgentSimulation/configs/integration/rw_dataset_factory.json \
  --output /separate-data/rw_native_campaign \
  --evidence-dir /separate-data/rw_native_execution_evidence
```

The launch refuses proxy-enabled configurations, unexpected fault labels,
missing source files, modified tracked files, uncommitted source revisions,
missing installed Basilisk or versions outside the producer-approved
`2.11.0` / `2.11.0+satfix1` range. It runs the producer's existing
`generate_astrograph_rw_fault_dataset.py --fail-fast` using the **current
Python interpreter**. No dependency is silently installed or simulated.

The review receipt stores the source Git revision, generator/native-module
and installed `simIncludeRW.py` SHA-256, claimed installed BSK version,
factory configuration hash, generator exit code/log hash, dataset index hash
and source TRAIN candidate-audit results. Runtime candidate review must pass
after the producer exits successfully. Otherwise the receipt records
`REFUSED`; there is no model delivery and no claim of actual physical
qualification.

The default upstream source config requests 9 TRAIN, 3 VALIDATION and
3 TEST pairs **per declared fault**, each with corresponding nominal cases.
This is a producer configuration, not observed completed case counts.
The trusted producer necessarily creates multiple split roles; the later
AutoDesign worker receives **only the isolated TRAIN view**.
`source_intake` and `r82_rw_study` remain separate, explicit steps.

**Important:** This runner does **not** establish independent physical
provenance, provide a sealed FINAL steward, or prove transferability.
The runner is an execution/evidence aid rather than a hardware or flight
model validator. Its GitHub CI tests use mocked subprocesses and fake module
files and are expressly **not** Basilisk execution tests.

For an official-wheel `2.11.0+satfix1` environment, upstream provides
`scripts/build_bsk_compat_wheels.py` to change only compatible wheel
distribution metadata while retaining the original Basilisk payload bytes.
Do not relabel, silently substitute versions or claim compatibility
without inspecting the actual installed environment and wheel build report.

```bash
cd source_product
python -m pytest -q tests/phm_design/test_r82_native_campaign_execution.py
```

## R82.2 — Basilisk candidate physics gate before data preparation

In the connected SatAgentSimulation producer,
`classify_simulation_fidelity` records `A_ENGINEERING_BASILISK_NATIVE`,
actual connected/recorded module counts and runtime truth status. The paired
validation agents separately mark `formal_training_qualified`,
`physics_candidate_ready`, `diagnostic_validation_passed` and
`basilisk_diagnostic_candidate`. **R82.2 now checks these recorded results**
before the fixed-wheel adapter copies any case to a model-ready manifest.

The trusted campaign preflight also checks every TRAIN fault/nominal pair:
matched parameters and seed, the original (not AstroGraph oracle projection)
source trace columns, sampling cadence and row count, and consistency between
the case contract and the simulation runtime report. Failure refuses the
whole native research preparation, not just the affected case.

```bash
cd source_product
python -m phm_design.industrial.native_campaign_gate \
  --campaign /data/actual_basilisk_campaign \
  --report /data/native_train_candidate_review.json

# This repeats the gate automatically before preparing any model-ready CSV:
python -c "from phm_design.adapters.sat_rw import prepare; prepare('/data/actual_basilisk_campaign', '/data/rw_prepared')"
```

The preflight inspects source TRAIN contract/task/report/trace only. It does
not read VALIDATION or TEST telemetry. The trusted `sat_rw.prepare` stage
can then process all declared splits, but **requires the native candidate
flags on every case**. The actual research TRAIN worker remains isolated
from heldout roles.

**Scientific boundary:** producer self-reported fidelity and physics
candidate status do not equal expert approval. `training_ready` and
`formal_diagnostic_qualified` may remain false while independent review
is pending; R82.2 does not override either value. The report explicitly
records `physical_execution_independently_attested=false` and
`independent_qualification=false`. Real Basilisk execution still requires
the campaign files and the correct installed `bsk` version.

```bash
python -m pytest -q tests/phm_design/test_r82_native_campaign_gate.py
```

## R82.1 — executable six-channel native reaction-wheel TRAIN pilot

The six-channel TaskSpec is now supplied at
`examples/industrial/satellite_rw_native_offline_train.json`.
It matches the existing fixed-wheel `sat_rw.prepare()` adapter:
three observable wheel speeds (`rad/s`) followed by three **controller
command** torques (`N*m`). Neither actual wheel torque nor a simulator's
fault-injection truth is an input. The supervised labels are
NOMINAL/FRICTION_INCREASE/JAM/TORQUE_AUTHORITY_LOSS.

This workflow is ready to use **when a real, auditable source campaign has
been produced**. In a trusted data-preparation environment, first run:

```bash
cd source_product
python -c "from phm_design.adapters.sat_rw import prepare; prepare('/campaign/native_rw', '/study/prepared_rw')"
```

The preparer creates separate development (TRAIN/VALIDATION) and TEST
manifests. Only the development manifest is ever supplied to this R82
training workflow. For a review worksheet, first isolate the TRAIN files:

```python
from phm_design.industrial.source_intake import (
    stage_train_only_view, draft_reviewed_physical_map)
from phm_design.io import read_json

staged = stage_train_only_view(
    "/study/prepared_rw/development/dataset.json", "/study/train_for_review")
draft_reviewed_physical_map(
    read_json("examples/industrial/satellite_rw_native_offline_train.json"),
    staged["manifest"], "/study/physical_map_REVIEW.json")
```

The worksheet contains the exact existing native `group_id` and
`operating_regime: REVIEW_REQUIRED` for each TRAIN case. A reviewer **must**
replace each placeholder with a supported, documented operating condition.
The program explicitly rejects any remaining `REVIEW_REQUIRED` value;
neither labels nor array/trajectory indices choose environment bins.

After reviewing the map, generate a native-source binding receipt from
the **actual installed Basilisk environment** (not from a downloaded Python
source file alone):

```bash
python -m phm_design.industrial.native_binding_collector \
  --native-source-checkout /sat/SatAgentSimulation \
  --output /study/runtime_binding.json
```

This collector imports `Basilisk.utilities.simIncludeRW` from the active
Python environment, captures its installed module file SHA-256 and `bsk`
package version, and hashes the local SatAgentSimulation runtime and effect
modules. It does **not** execute a simulation or certify physical fidelity.

The R82 research runner can then preflight and fit:

```bash
python -m phm_design.industrial.r82_rw_study \
  --task-spec examples/industrial/satellite_rw_native_offline_train.json \
  --development-manifest /study/prepared_rw/development/dataset.json \
  --physical-map /study/physical_map_REVIEWED.json \
  --runtime-binding /study/runtime_binding.json \
  --native-source-checkout /sat/SatAgentSimulation \
  --basilisk-wheel-factory /venv/lib/pythonX.Y/site-packages/Basilisk/utilities/simIncludeRW.py \
  --output /study/r82_preflight --validate-only

# Repeat with a NEW --output directory and omit --validate-only to fit.
```

The basic six-channel `sat_rw.prepare` **does not emit a binding
receipt**. Use the collector above or an independently audited source-binding
receipt from the same environment that produced the campaign. Do not copy
hashes from a different checkout simply to pass intake.

The output `research_report.json` records either accepted TRAIN-only
intake, a TRAIN OOF research model selection (with a bounded child process),
or an explicit refusal. It always sets `independent_qualification=false`
and `actual_basilisk_execution_attested=false`, even if source SHA-256
matches. This is a **research executor**, not an independent source steward.

**Important version blocker**: the current connected
`clonetimer/SatAgentSimulation` main branch pins
`bsk==2.11.0+satfix1`, while autogen's original simulation extra uses
`bsk==2.11.0` and earlier platform work used `2.10.2`.
R82 does not silently upgrade/downgrade these environments. A real
campaign must be reproduced in its actual pinned source environment, and
its `simIncludeRW.py` hash verified against that installed environment.

```bash
python -m pytest -q tests/phm_design/test_r82_rw_study.py
```

These tests deliberately synthesize all telemetry and mocked source files
to verify plumbing. They provide **zero evidence** of executed Basilisk
dynamics or field generalization.

## Reproducibility and scientific limits

The receipt contains source manifest hash, physical map hash, optional
Basilisk-binding hash, count and source file fingerprint. All these are
**untrusted/unattested** unless a separate authorized source steward
verifies who generated them and how. The emitted `records` have no
truth, fault states or environment IDs in the measurement fields; those
remain metadata for source TRAIN-only grouped fitting.

A R82.0 functional test uses artificial numeric telemetry and dummy
source claims to exercise contracts; it is explicitly **NOT** Basilisk
runtime evidence or a real bearing dataset. Hardware physics
requirements, sensor observability, cross-device generalization and a
sealed independent FINAL test remain open.

**Basilisk version discrepancy to resolve before an actual native run:**
the current repository simulation extra pins `bsk==2.11.0`. Existing
historical project acceptance used `bsk==2.10.2`. R82.0 changes
neither dependency and claims no completed native run.

```bash
cd source_product
python -m pytest -q tests/phm_design/test_r82_reviewed_source_intake.py
```

A successful software test means *only* the input-contract behavior works;
it does not certify authenticity or model quality.
