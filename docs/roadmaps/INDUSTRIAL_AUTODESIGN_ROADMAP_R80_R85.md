# PHM AutoDesign — Industrial Diagnosis AutoDesign strategic roadmap (R80–R85)

**Status:** PROPOSED STRATEGIC PIVOT / planning contract; no model or release promotion.
**Basis:** [R79 independent read-only review](https://github.com/clonetimer/autogen/pull/7) and [issue #8](https://github.com/clonetimer/autogen/issues/8).
**Code baseline for future development:** R78 `0.8.0.dev17` / `a10469bb419a98cade08a02c0c7ed8e11ab9872a`.
**Stable unaffected:** R60 `0.7.0` (15 explicitly scoped SAT/Basilisk offline simulation-domain qualification cases).
**Owner of product/acceptance:** Product technical representative. Engineering implementation: feature branches. Scientific qualification: independent review and data steward.

## 1. Revised mission / scope

**Primary deliverable:** an *industrial-equipment intelligent diagnostic algorithm design system*. Given a declarative task, lawful deployable signals, a correctly partitioned dataset and resource budget, it must **design**, train, compare, calibrate or reject a reproducible diagnostic algorithm.

**First controlled physics adapter:** SAT/Basilisk 2.11.0. Keep existing native physics call chains and frozen R60 models. Satellite is NOT the definition of the AutoDesign core.

**Second independent equipment adapter:** a non-satellite industrial equipment family. First preferred research dataset: Paderborn University bearing data (bearing vibration, motor current, four operating conditions, distinct bearing instances; verify signal, label, group and license assumptions before use).
- Official source: https://mb.uni-paderborn.de/en/kat/research/bearing-datacenter/data-sets-and-download
- DataCenter license: CC BY-NC 4.0 for non-commercial academic use. Commercial deployment/data redistribution requires a separate legal right; no raw licensed data committed to GitHub.
- Distinguish **operating condition holdout** from **bearing instance holdout**. A window cut from a given bearing belongs to the same split as every other measurement/window from that unit; same fault type does not prove independent generation.
- Evaluate class coverage and train/test feasibility before choosing a protocol. Do not assert that all splits provide adequate samples for every failure subtype.

Two distinct claims must never be conflated:
1. **Design-method transfer:** the **same unmodified core AutoDesign engine** configures and searches diagnostic pipelines for different industrial equipment tasks, training *task-specific model weights* from their lawful datasets.
2. **Trained-model generalization:** a *particular trained network* retains performance on a predefined unseen operating-condition, machine-instance, process or physics family. Zero-shot cross-machine-type reuse of identical weights is NOT required for R80–R84.

**Explicit non-goals for initial release:** universal neural weights, implicit causal root-cause discovery, unqualified physical-hardware diagnosis, inference using simulator truth, covert target-domain adaptation, and automatically claiming RUL support without a distinct censoring/life-task contract.

## 2. Current evidence baseline — no inflated claims

- R60 Stable: 15/15 bounded satellite simulation diagnosis contracts, only offline complete-trajectory and trusted initial healthy segment. Its eight out-of-scope cases remain out of scope.
- Generic `open_design_fit.design`: bounded feature programs and sklearn learners, with provenance-sensitive screening.
- R69–R78 task-graph/neural line: real PyTorch CNN/GRU/Transformer/TCN and ERM/GroupDRO/V-REx, but *fixed* two-component × two-mechanism five-label task ontology in its richest path and explicit healthy-prefix requirement.
- R71 six cross-environment/weak-fault development tasks did not satisfy combined acceptance. Already revealed; cannot be used for new blind qualification.
- R78 synthetic spurious-shift *revealed development* stress: ERM-TCN 5.36%, GroupDRO-TCN 1.79%, V-REx-TCN 1.79% mean target accuracy; all healthy false-alarm rates 100%. This is a useful negative finding, not Basilisk performance.
- Existing CI chiefly proves software regression and source role guards. No end-to-end industrial cross-equipment design-loop qualification yet.

## 3. Target architecture and contracts

```text
TaskSpec (equipment-agnostic task + deployment contract)
    ├── TaskType / arbitrary ontology / decision-time / expected outputs
    ├── SensorSpec (channels, units, sampling, timestamps, availability)
    ├── DatasetRole + lineage + operating regime + fault severity
    └── Budget + safety costs + evaluation objectives
           |
    TaskCompiler and capability/observability preflight
           |
    AutoDesignPlan (validated, bounded, reproducible)
           |
    Unified trial engine
      [representation: raw temporal / spectrum / physical residual / learned]
      × [architecture: trees, linear, CNN, TCN, GRU, Transformer, ...]
      × [objective: ERM / GroupDRO / V-REx when justified]
      × [normalization, augmentation, calibration, rejection policy]
           |
    Nested source-only split + fixed-budget comparison + audit ledger
           |
    NOT_IDENTIFIABLE / NO_ACCEPTABLE_DESIGN / frozen algorithm artifact
           |
    separate FINAL-data steward and one-time qualification
           |
    EquipmentAdapter(satellite/Basilisk | bearing/experimental machine | ...)
```

All selection/search modules use *one* `TrialSpec`, `TrialRecord`, `FitContext`, `EvaluationContract` and `ModelArtifact`. A modular fixed candidate registry is acceptable in v1, but architecture choices cannot be hardcoded to satellite domain names. A new equipment family must not require a new branch inside the generic search engine.

**TaskSpec minimum fields:**
- `task_kind`: `fault_detection`, `fault_classification`, `component_localization`, `mechanism_classification`, or `joint_diagnosis` (separately opt into RUL/prognostics later).
- `decision_mode`: `offline_trajectory` or `causal_window`; window/horizon/latency and health-prefix evidence, if any.
- `ontology`: variable component IDs and fault labels, optional hierarchy, unknown class policy, multi-fault scope and legal outputs.
- `channels`: explicit names, units, sampling/time semantics, measured-vs-commanded availability and missing-channel policy. Fault truth, generator parameters, split IDs and future samples are **never** inference channels.
- `data_contract`: data source, equipment-instance ID, physical generation family, environment axes, fault-severity stratum, sample role and grouping dependencies.
- `metrics`: detection recall, healthy FAR, class/mode conditional and end-to-end joint success, abstention, worst supported environment, late-detection and cost where applicable; support constraints.
- `search_budget`: max trials/fits, wall/cpu/gpu/memory cap, seeds, allowed candidate/objective space; *compute actually measured*, not mere epoch/step count.
- `research_boundary`: development-only or pre-registered sealed-final, target-data adaptation allowlist, reproducibility and exposure ledger.

Invalid or unidentifiable TaskSpecs must **fail before fitting**, with a reason. An unqualified result must never inherit R60 Stable certificates.

## 4. Milestones, implementation order and acceptance gates

### R80 — generic task/ontology contract (G0) | immediate, BLOCKING

**Work:** `industrial/taskspec.py`, `industrial/task_compiler.py`, `industrial/sensor_contract.py` (names provisional); machine-readable schema and two working examples:
- SAT/Basilisk EPS localization with five labels and measured power/command fields.
- Experimental bearing condition diagnosis with a *different* ontology, vibration/current signal shapes, instance/environment IDs, and no assumed initial healthy prefix.

**Gate:** without editing the generic TaskSpec parser/compiler, both examples validate and compile into typed, non-executable `AutoDesignPlan` objects; supports at least two unequal ontology sizes/topologies; tests reject leakage fields, illegal timestamps, unavailable channels, invented fault labels and unsatisfiable task/cost constraints. No model training requirement yet.

**Stop:** new equipment or non-five-class task still requires changing `TaskOntology.from_domain` or the core task compiler.

### R81 — unified design engine (G1) | depends on R80

**Work:** route existing general feature-program learners and task-aware PyTorch search through the **same** trial/evaluation/artifact APIs; compatibility wrappers preserve old APIs. Add bounded co-design of feature space (raw/residual/frequency/learned), architecture, objective and calibration/UNKNOWN; maintain a compute ledger, deterministic seeds, all failed trials, fold provenance and artifact hashes.

**Gate:** a *single* design CLI/API takes each R80 TaskSpec and training manifest; independently selects at least two different algorithm families where supported, records why, can reject ALL candidates without a model, and emits a replayable signed-in-project audit. All preprocessing and augmentation fit only inside training folds. Legacy R60 predictive hashes unchanged. Baseline comparison uses same signals, folds and compute budget.

**Stop:** neural models require a separate satellite-specific design engine, or search uses outer/final feedback to choose hyperparameters.

### R82 — two-domain adapters and physical observability (G2) | depends on R80; may run alongside R81

**Work:** satellite adapter with verified Basilisk native call chain and counterfactual healthy/fault physical response checks; Paderborn bearing real-data adapter with documented license, acquisition checksums, bearing identity, operating regimes, sensor alignment and lawful ontology mapping.

**Gate:** two different equipment families reach a common lawful time-series `DatasetContract` without domain-specific changes to R81 design core; predeclared negative controls catch duplicate units, condition mislabeled as independent machine, fault-truth fields, absent weak signal response, and leakage from windows across splits. Unsupported sensitivity returns `UNIDENTIFIABLE` or `INCONCLUSIVE`.

**Stop:** model predicts synthetic injection labels but the supported measured signals lack reproducible fault response; same bearing appears in training and heldout split.

### R83 — true neural generalization AutoDesign (G3) | depends on R81+R82

**Work:** pretrained candidate architecture registry is kept, but now the system selects representation, network topology class, training objective and validation policy under a fixed **compute-matched** budget. ERM is a mandatory strong baseline, with tree/linear, TCN/CNN/GRU and justified GroupDRO/V-REx arms. For bearing, separately evaluate cross-instance and cross-condition protocols **when class support permits**; for Basilisk, hold out physically disjoint mission/environment families. Add multiple predeclared seeds and confidence intervals resampled by *independent machine/physical-family*, not random windows.

**Gate:** (a) complete machine-readable auto-design comparison for both equipment types; (b) no FINAL feedback used in candidate generation, seed/filter/threshold choice; (c) automatic rejection for unsupported fault-stratum/class counts; (d) report equal-budget relative performance and costs of chosen designs versus fixed baseline, all negative outcomes included. Target numerical uplift/non-inferiority and FAR/recall limits must be pre-registered with statistical support **before** a new final protocol. No retrospective success thresholds.

**Stop:** results rely on R71/R78 revealed stress, GroupDRO/V-REx win is assumed rather than measured, or hidden human champion selection overrides the engine.

### R84 — independent release evaluation (G4) | depends on R83 freeze

**Work:** independent steward with separate file/process permissions and source/model/feature/score freeze; create/acquire FINAL_ID and FINAL_OOD datasets with planned group/condition separation, record access log and one-time fixed predictions.

**Gate:** full pipeline SHA lock, exact input/output audit, one-time sealed evaluation covering design transfer across the two device types and within-equipment generalization. Report fault support, worst operating regime, weak severity, healthy FAR, UNKNOWN, end-to-end error and grouped uncertainty. If fail, mark research result, demote opened data, start a new protocol rather than recertifying by renaming files.

**Stop:** developer/agent receives target-world feedback while still choosing features, model, loss, threshold or early stopping; falsely extends R60 certificate to new equipment.

### R85 — causal/online industrial trial (future, gated, separate scope)

**Work:** task-defined causal windows, streaming inference, missing/late sensor channels, health-prefix-not-available, alarm stabilization, delays, unknown mechanisms, false alarms per hour and deployment cost.

**Gate:** replay is timestamp-causal and new results report detection delay vs FAR trade-off. Do not claim industrial field installation/certification without actual hardware evidence. Prognostics/RUL requires its own censoring and run-to-failure evaluation contract.

## 5. Scientific study and no-leakage policy

1. **Design datasets:** TRAIN fits, DEV_VALID selects, DEV_STRESS diagnoses failures. Already revealed R71 / R78 are DEV_STRESS; they cannot mint fresh blind OOD claims.
2. **Physical groups precede windows.** Same bearing, simulation base world and all augmented/healthy/fault counterfactual family members must remain in same role. Hold out environments at the proper level; make explicit when class support makes leave-one-environment-out impossible.
3. **Never use FINAL in auto-design loops:** no labels, no unlabeled normalization if not predeclared as adaptation, no case studies, no validation feedback paraphrases to another agent. A separate evaluator must handle final metrics.
4. **No meta-overfitting by trial proliferation:** lock architecture/loss search budgets and seed counts; all attempts, rejected hypotheses and target exposures recorded.
5. **Neural architecture choice is not automatically domain generalization:** compare against strong ERM. Physics-compatible representation and nuisance interventions must be tested; an impossibility/identifiability result is a legitimate outcome.
6. **No hardcoded universal transfer claim.** Reusability of a design *method* differs from transferring trained weights across different industrial machines.
7. **Field availability at decision time:** full-trajectory offline and causal online tasks have different `TaskSpec` semantics; new streaming claims need separate evaluation.

## 6. Architecture and repository integration strategy

- `main`: preserve R60 Stable and original limited release status; no dev research refit or reissued Stable.
- Existing stacked PR #1–#5 remain Draft and are preserved for history. The review PR #7 remains a separate read-only governance artifact. Do **not** blindly squash them into main.
- First development implementation branch for R80 should branch from R78 dev head while project-management documents live in this planning-only branch from `main`. No dependency on whether the review PR merges.
- Keep experimental Basilisk and bearing adapters at the boundary; candidate search and task ontology live in shared industrial modules.
- One feature per implementation PR; focused CI and unit/contract matrix first, then full regression and exact dev package builds for candidate promotion.
- Require independent reviewer consultation at architecture gates R80, R82 and R84. The existing read-only reviewer is deterministic, not a distinct model or external human panel.

## 7. Priority and explicit pause decisions

**Now:** R80 TaskSpec + R81 search unification design; R82 dataset/license/observability audit begins after TaskSpec skeleton. Refrain from adding fourth/fifth DG loss simply to improve R78 synthetic revealed stress.

**Defer:** new satellite fault-family count as a success metric, general-purpose NAS claims, cross-machine zero-shot reuse, true online field-ready claims, RUL or unknown/multi-fault expansion before separate contracts.

**Do not discard:** Basilisk-native simulation chain, Stable R60 evidence, existing neural implementations, useful physical features, strict lineage gates, negative development results and all CI infrastructure.

## 8. Definition of first meaningful revised-project success

One reproducible invocation of the **same AutoDesign core** accepts each of:
- a Basilisk satellite sensor/ontology TaskSpec, and
- a real industrial bearing sensor/ontology TaskSpec;

...and autonomously produces **different, justified diagnostic pipelines or explicit refusal**, with complete TRAIN-only search audit and predeclared independent evaluation design. The evaluator, not the designer, then independently scores frozen outputs on never-accessed physical/machine families.

**Successful compilation and code tests alone do not count as proven generalization.** The criterion is transferable *algorithm design process* and reliable, separately validated fault-diagnosis behavior, with negative outcomes preserved.
