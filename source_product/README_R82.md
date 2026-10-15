# R82.0 — reviewed TRAIN-source intake (development-only)

**Status:** stacked draft on R81c; this is a source-contract/gating slice,
not a Basilisk simulation run, real bearing experiment, or independent OOD
qualification. A declaration such as `source_kind=native_basilisk` is not
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
factory. It does not execute Basilisk or recheck that those hashes
correspond to physical simulation behavior. `real_equipment_unreviewed`
+ `real_unreviewed` may enter a **research TRAIN** experiment but is
never called qualified equipment evidence merely because it was ingested.
`software_synthetic`, missing binding, unknown roles and oracle-only
channels are rejected.

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
