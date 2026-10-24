# R83 Self-Organizing Neural Designer — research pilot (0.8 development)

**Status:** Experimental branch, stacked on R82 (feature/r82-reviewed-source-intake).
**NOT Stable; NOT independently qualified; NOT online/causal or flight-safe.**
This pilot is a separate opt-in API. It does not modify the Stable model selector,
TaskSpec schema, Basilisk physical source gates, or existing algorithm families.

## Motivation and implemented mechanism

R82 produced 66 actual Basilisk simulated RW trajectories, with the *source*
TRAIN subset consisting of 54 trajectories across 27 nominal/fault physical
pairs. R82 TRAIN-only candidate selection properly refused release: a
conventional ExtraTrees temporal model achieved 0.870370 accuracy but 0.0
worst environment/class recall. This is a research motivation, **not evidence**
that R83 improves those outcomes.

R83 adds a small, finite neural architecture grammar and a feedback-driven
mutation loop:

1. Predeclare two legal starting topologies.
2. For each topology, fit a CPU PyTorch temporal classifier independently
   inside every **source TRAIN leave-one-environment-out** fold.
3. Collect OOF accuracy, minimum environment/class recall and healthy false
   alarm. Calculate a predeclared search loss:
   0.45*(1-accuracy) + 0.40*(1-worst_recall) + 0.15*healthy_false_alarm.
4. Rank evaluated topologies on *TRAIN OOF feedback* and propose an unvisited
   **single-edit neighbor** of the best parent. No hidden architecture code
   generation, unbounded trial loops or FINAL feedback.
5. Compare against a separately predeclared TRAIN OOF acceptance screen.
   If no topology is eligible, fail closed with a refusal audit.
6. On success, refit the selected architecture on *TRAIN only*, charge the
   final estimator fit and emit a research artifact with a numerical NPZ
   model state. Export is NOT a certificate.

The topology genome controls convolution width (8/16/32), block depth
(1/2/3), kernel (3/5), mean or mean+max pooling, and residual skip
(true/false). The finite declared space has 72 topologies; mutations allow
**growth and pruning**. This is bounded evolutionary feedback search, not
NEAT, DARTS, general differentiable NAS, SOINN or online continual learning.

The estimator inherits R68's deterministic CPU Torch fitting, per-fold
training-only normalization and sequence interpolation. Source records are
the same {measurements:..., metadata:...} representation used by R81.
The TaskSpec data roles, declared sensor channels, class ontology, physical
group identity, and environmental axes are checked by the same R80 compiler.

## Sample API (source TRAIN only)

Run from source_product/ after installing its requirements:

~~~python
from phm_design.industrial.self_organizing import (
    EvolutionPolicy, SelfOrganizingArtifact, run_self_organizing_design)
from phm_design.industrial.design_engine import DesignAcceptance

# task_spec: existing R80/R81 offline_trajectory TaskSpec dict
# train_records: native/reviewed source TRAIN-only records
artifact = run_self_organizing_design(
    task_spec, train_records,
    policy=EvolutionPolicy(max_candidates=3, epochs=3,
                           max_parameters=30_000),
    acceptance=DesignAcceptance(accuracy_min=0.90,
        worst_environment_class_recall_min=0.80,
        healthy_false_alarm_max=0.10),
)
audit = artifact.audit
artifact.export("r83_research_artifact")
restored = SelfOrganizingArtifact.load("r83_research_artifact")
decision = restored.predict(deployment_measurements)
# label, confidence_uncalibrated, independent_qualification=False
~~~

Only finite topology configurations are accepted, and every fold plus
final refit counts against declared design.budget.max_fits. Best-effort
CPU use is checked after fitting and before final delivery against
max_cpu_seconds. **This is not a hard CPU or wall-clock preemption
mechanism**; TaskSpecs declaring max_wall_seconds are refused until an
isolated guarded runner has been integrated.

The input must have at least three source environments and every class
represented in every held-out environment. Any physical group crossing
environments, duplicated sample relabeled as an independent physical group,
non-TRAIN role, undeclared truth/label observation column, or invalid model
state must be refused. Training labels exist only in metadata, never as
deployment model features.

Training scores feed architecture proposals *and* candidate selection.
They must **never** be treated as a fresh independent generalization score.
Do not tune acceptance constraints from these scores. The NPZ state
is non-pickle and content-SHA-256-checked, but a file hash alone does not
authenticate authorship or prove data provenance. The output confidence
is not calibrated and cannot be used as a qualified abstention decision.

## Verification

~~~bash
cd source_product
python -m pytest -q tests/phm_design/test_r83_self_organizing.py
# R81 and R82 regression and CI on the branch must also pass before PR review.
~~~

The R83 tests intentionally use synthetic sensor observations generated in
test code, not real equipment data, hidden FINAL labels, or Basilisk claims.
They cover structural mutations, *feedback-dependent* third proposals,
actual grouped Torch training, final refit, exact numerical replay, artifact
tampering, and leakage/budget/refusal paths.

## Explicit next work (not claimed by this pilot)

- **R83.1**: integrate an opt-in TaskSpec extension and process-isolated hard
  timeout in the existing guarded industrial design entry.
- **R83.2**: preregister matched-budget fixed TCN/GRU/ExtraTrees versus evolving
  network trials on the *already generated* native R82 TRAIN source, with
  separate selected-model validation and untouched sealed TEST/FINAL.
- **R83.3**: actuator command-response representation grammar, sensor
  observability/identifiability audit and expert physical-group review.
- **R83.4**: self-supervised/SOINN-like regime representation, continual
  learning/replay and forward-transfer experiments with independently
  partitioned equipment and frozen previous-task regression.
- **R84**: independent release qualification; do not promote simply because
  more candidates were tried.

**Release boundary:** R83 is a tool for designing models, not a physical
diagnosis-capability improvement claim. Failed candidate/model eligibility
must remain a legitimate, reportable scientific outcome.
