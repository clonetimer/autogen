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

After reviewing the map and generating a binding receipt from the **actual**
installed Basilisk environment, the R82 research runner can preflight and fit:

```bash
python -m phm_design.industrial.r82_rw_study \
  --task-spec examples/industrial/satellite_rw_native_offline_train.json \
  --development-manifest /study/prepared_rw/development/dataset.json \
  --physical-map /study/physical_map_REVIEWED.json \
  --runtime-binding /study/prepared_rw/runtime_binding.json \
  --native-source-checkout /sat/SatAgentSimulation \
  --basilisk-wheel-factory /venv/lib/pythonX.Y/site-packages/Basilisk/utilities/simIncludeRW.py \
  --output /study/r82_preflight --validate-only

# Repeat with a NEW --output directory and omit --validate-only to fit.
```

The path `/study/prepared_rw/runtime_binding.json` is illustrative:
the basic six-channel `sat_rw.prepare` alone **does not emit this file**.
It can be produced by the separate mechanism-review workflow or by an
audited native source-binding collector; it must match the actual source
checkout and installed Basilisk wheel-factory module. Do not copy a hash from
some other checkout simply to pass intake.

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
