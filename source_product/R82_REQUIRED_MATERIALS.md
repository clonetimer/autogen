# R82 Physical Data Verification — Required Materials

**Development scope only.** R82.0 software intake tests cannot establish Basilisk
physics fidelity, observational validity, industrial bearing generalization, or
FINAL qualification. Do not upload independent FINAL/TEST datasets to the
TRAIN-side `intake_train_source` worker. An authorized steward must keep those
roles separate.

## Option A — Basilisk native ADCS/reaction-wheel study (preferred first)

Supply one ZIP or a reviewed repository reference with:

1. **Native simulator project sources**
   - `src/sat_sim/bsk_engine/unified_native.py`
   - `src/sat_sim/bsk_engine/project_native_modules.py`
   - Python/environment lock or install report with **installed `bsk` version**.
   - Installed `Basilisk/utilities/simIncludeRW.py` source, if licensing and
     redistribution terms allow; otherwise provide a reproducible installed
     module SHA and environment instructions without redistributing code.
2. **The actual generated TRAIN campaign (not example JSON)**:
   - `cases/<id>/astrograph/dataset_contract.json`
   - corresponding `task_spec.json`
   - every original **source trace CSV** referenced by
     `telemetry.source_trace_file` for selected TRAIN cases.
   - generator configuration (for example
     `configs/phm/rw_campaign_v05.json`), execution command,
     environment/seed record and source revision.
   - `runtime_binding.json` if nine-channel mechanism data are prepared.
3. **Reviewed TRAIN physical-group mapping**:
   - Each sample ID -> **physical group ID** and all operating-environment
     axes declared by the AutoDesign TaskSpec (for example speed/load).
   - Explain what a group represents and whether cases in distinct groups
     actually have independent physical parameters. A renamed trajectory
     or seed alone does **not** establish independent physics.
4. **Correct measurement interpretation**:
   - Specify if wheel torque channels are **controller commands**, rather
     than actual applied torque; disclose ideal gyro measurements,
     friction/torque-authority fault injection mechanism, timing and sensor
     noise model. No oracle/twin fields may enter deployment observations.

In the initial source-intake stage, the model worker may receive only an
isolated **TRAIN-only** manifest. Validation must remain under a separate
trusted steward; TEST/FINAL must not be sent to the training side. To
substantiate actual source execution, the review must inspect logs/results
from a freshly executed Basilisk job, not merely a hashed source tree.

**Environment discrepancy to resolve:** the current autogen simulation
extra pins `bsk==2.11.0`; a historical platform baseline used
`bsk==2.10.2`. Do not silently substitute either before reproducing the
submitted campaign.

## Option B — Industrial bearing experiment

Provide the original dataset or its verified download link, license and
published source (e.g. the exact XJTU-SY/PHM dataset edition used), with:

- Original vibration/other actually measured channel files, labels, units,
  sample rate, cycle/run IDs, load/speed regimes, acquisition setup, fault
  type and physical bearing IDs.
- **Reviewed grouping and split policy at bearing/physical-unit level**,
  made before segmenting windows. Environment axes must be declared based
  on known equipment and operating conditions, not inferred from labels.
- A bounded TRAIN-only research subset, with independent validation and
  FINAL partitions retained by a steward. For RUL data, a separate RUL
  TaskSpec/objective is required; the current R82.0 bridge handles
  offline trajectory classification only and must not silently re-label
  prognostics as classification.
- Sensor availability flags only if actually measured at decision time;
  synthetic validity masks are not evidence of field sensor reliability.

## Machine-checkable R82.0 intake deliverable

- `phm.sequence-dataset.v1` TRAIN-only manifest + source CSVs,
  all with matching SHA-256 and exact declared units/channels.
- `phm.industrial.source-physical-map.v1` with **all** declared TaskSpec
  group/environment fields. The group key matches source `group_id`.
- TaskSpec specifying task ontology, declared observations, group keys,
  environment axes, learner/search bounds and refusal policy.
- For native Basilisk input: `runtime_binding.json` and the exact native
  source checkout / installed wheel factory file (all three file hashes must
  match). This checks source **identity** only, not scientific fidelity.
- For any trained model: record CI hash, resource budget, source receipt,
  software environment, equipment provenance uncertainties, acceptance
  gates and all research-only qualifications/refusals.

## Completion criteria

R82.0 **software intake** passes when the file/role/physical-group contract
holds and its synthetic integration tests pass. **R82 real-data validation**
requires an actual complete TRAIN campaign or experimental dataset, a
reproducible preprocessing mapping, model fit/evaluation under source
environment splits, sanity checks on physically measured variables and a
separate OOD protocol. Neither outcome upgrades Stable or constitutes an
independent FINAL qualification.
