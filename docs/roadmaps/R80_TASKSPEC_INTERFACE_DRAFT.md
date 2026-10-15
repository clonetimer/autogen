# R80 TaskSpec v0: implementation-first interface contract

**This is a draft data/API interface, not shipped code.** It must compile without access to FINAL evaluation data. Validation occurs before neural model search and does not infer physical truth from metadata.

## Design principles

1. The requested *diagnostic task* and deployment data contract are distinct from device names. The engine may inspect task types, class support, time/field availability and compute budgets, not brittle `if equipment == satellite` branches.
2. No fixed five-class ontology, mandatory healthy prefix, fixed detection→location→mechanism topology or hardcoded `satellite EPS` fault list in generic code. Device adapters may map their own legal ontology into shared IDs.
3. Every input declares whether it is a measured sensor, command, metadata or ground truth. No training-only provenance/label/fault injection fields in inference features.
4. The same `TaskSpec` must be sufficient to emit an explicit `AutoDesignPlan` with candidate families, fold policy, objective metrics, cost and refusal reasons.
5. Task is not automatically feasible: independently held-out class support, onset available at decision time, physical observability and grouping requirements are preflight checks.

## Satellite example (schematic, not frozen)

```json
{
  "schema": "phm.industrial.task.v0",
  "id": "satellite.eps.two-component.offline",
  "equipment": {"family": "satellite", "adapter": "basilisk_native_eps", "version": "2.11.0"},
  "task": {"kind": "joint_diagnosis", "decision_mode": "offline_trajectory",
    "allow_unknown": true, "multifault": false,
    "labels": ["NOMINAL", "SOLAR_A_EFFICIENCY_LOSS",
      "SOLAR_B_EFFICIENCY_LOSS", "LOAD_A_EXTRA", "LOAD_B_EXTRA"],
    "structure": {"components": ["solar_a", "solar_b", "load_a", "load_b"],
      "mechanisms": ["efficiency_loss", "extra_load"]}},
  "observations": {"time": "time_s", "healthy_prefix_policy": "verified_if_required",
    "channels": ["illumination", "load_a_command_w", "load_b_command_w",
      "panel_a_meter_w", "panel_b_meter_w", "bus_net_meter_w"]},
  "data": {"role": "TRAIN", "group_keys": ["base_physics_id", "generation_family_id"],
    "environment_axes": ["orbit_illumination", "duty_cycle", "sensor_noise"]},
  "design": {"candidate_max": 12, "compare": ["linear", "tree", "cnn1d", "tcn"],
    "objectives": ["ERM", "GroupDRO"], "allow_refusal": true},
  "deployment": {"causal_online": false, "prototype_only": true}
}
```

## Industrial bearing example (schematic, not frozen)

```json
{
  "schema": "phm.industrial.task.v0",
  "id": "bearing.vibration-current.classification",
  "equipment": {"family": "rolling_bearing", "adapter": "paderborn_experimental"},
  "task": {"kind": "fault_classification", "decision_mode": "offline_trajectory",
    "allow_unknown": true, "multifault": false,
    "labels": ["HEALTHY", "FAULTY"],
    "structure": {"components": ["bearing"], "mechanisms": []}},
  "observations": {"time": "sample_time_s", "healthy_prefix_policy": "not_assumed",
    "channels": ["vibration", "motor_current"]},
  "data": {"role": "TRAIN", "group_keys": ["bearing_instance_id"],
    "environment_axes": ["speed", "radial_load", "torque"]},
  "design": {"candidate_max": 12, "compare": ["linear", "tree", "cnn1d", "tcn"],
    "objectives": ["ERM", "GroupDRO"], "allow_refusal": true},
  "deployment": {"causal_online": false, "prototype_only": true}
}
```

**Important:** The above bearing class mapping is intentionally coarse: real damage categories, fault locations and healthy/fault unit counts must be verified against the downloaded data and official fact sheets *before* a finer ontology is frozen. The listed channels and units need exact official-name/shape mapping by the adapter. The satellite example is likewise illustrative and must be matched to the existing native observation field contract. Both are **proposals, not assertions that the current system already supports them**.

## Compiled AutoDesignPlan minimum

```text
plan_id, task_spec_sha256, eligible_task_nodes, allowed_field_indices,
group_split_strategy, training_role_allowlist, candidate_grid,
fold_pipeline_normalization, early_stopping_source, compute_budget,
acceptance_policy_sha256, refusal_modes, prohibited_target_actions,
artifact_and_predicted_output_contract
```

## Required negative tests

- Arbitrary 2/3/5 label ontologies and component/label hierarchies compile without core edits.
- No forced healthy prefix for the bearing task.
- Detection-only task never invents mechanism/location labels.
- Time axis not strictly increasing, unavailable sensor, undefined output class, missing machine group => reject.
- Fault onset, true fault class, environment ID, `FINAL_ID`/`FINAL_OOD` data and post-decision features cannot become inference inputs.
- No silent upgrade of a fully offline trajectory contract into causal-online validation.
- Missing class support in an environment => report an **unsupported evaluation split**, not invented macro recall.
- No adaptation to FINAL target statistics or automatic use of revealed R71/R78 metrics to change the plan.
- `AUTO_NO_VALID_PLAN` and `NOT_OBSERVABLE_WITH_DECLARED_SENSORS` have explicit audit outcomes.

## Acceptance evidence

R80 is complete **only** when these contract fixtures pass in Python 3.11/3.12 CI, R60 stable tests remain unaffected, and a new industrial equipment label cardinality/topology compiles without branching on its name inside core. Training accuracy is deliberately **not** a G0 metric.
