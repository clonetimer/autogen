# Independent strategic risk review — PHM AutoDesign as an industrial diagnosis algorithm-design system

**Decision:** `KEEP_DIRECTION_WITH_PIVOT` and `STOP_GENERAL_PRODUCT_CLAIM` until independently demonstrated.
**Risk level:** HIGH direction drift if the next investment is more satellite-specific fault classes, more neural objectives or more gate thresholds without demonstrating transferability of the *design process*.
**Reviewed immutable target:** [R78 commit a10469bb](https://github.com/clonetimer/autogen/tree/a10469bb419a98cade08a02c0c7ed8e11ab9872a).
**Review scope:** read-only static source and published development evidence, not a new trained-algorithm or hardware evaluation.
**Project goal judged:** intelligent industrial-equipment diagnosis **algorithm design**, not a satellite diagnosis model as the final product.
**Satellite/Basilisk role:** a current validation adapter and physical data source, not proof of cross-equipment or simulation-to-real generalization.

## Executive verdict

The **problem selection is valid**: domain-specific fault diagnosis requires systematic choices of evidence, model architectures, training objective, model selection, calibration and uncertainty. The current system contains meaningful engineering components: a general feature-program optimizer, bounded learners, PyTorch neural training, nested environment-group selection, source-lineage checks and refusal when evidence fails.

The **architecture is not yet demonstrated as a general industrial diagnostic AutoDesigner**. The strongest completed evidence proves a small *simulation-domain* Stable product and synthetic code contracts. The richest diagnosis task graph is hard-coded around a two-component/two-mechanism, five-class ontology; the generic feature-program optimizer is a *separate path* from the neural task-graph optimizer. A new industrial device cannot be shown to drive the same entire lifecycle via declarative tasks without domain-specific code. R78 reveals that all three trained neural objectives failed a severe previously unseen synthetic spurious-correlation reversal. Adding GroupDRO/V-REx or tightening rules did not solve domain generalization in that test.

**Recommended action:** redirect the next two milestones toward a device-agnostic TaskSpec, shared AutoDesign graph/search engine, physically observable weak-fault evidence, and a pre-registered *second equipment-family* trial. Continue Basilisk as the physically controlled first adapter. Freeze candidate addition unless an error taxonomy and falsification study justify it.

## Findings and admissible evidence

### P0-1 — Product/validation-target conflation: HIGH

**Source:** [R60 limited Stable release decision](https://github.com/clonetimer/autogen/blob/a10469bb419a98cade08a02c0c7ed8e11ab9872a/RELEASE_DECISION_R60.md); [deployment field inventory](https://github.com/clonetimer/autogen/blob/a10469bb419a98cade08a02c0c7ed8e11ab9872a/source_product/src/phm_design/platform_features.py#L10-L21).

R60 explicitly certifies 15 enumerated **SAT/Basilisk simulation-only, offline complete-trajectory** cases, excluding 8 cases including Thermal and weak fault domains. These are legitimate scoped accomplishments, **not** evidence of automatic industrial machinery design generality. The dominant domain schemas are ADCS/EPS/Thermal/Data-specific.

**Risk:** success may be measured by an ever-expanding satellite case matrix, making the testbed the product. **Counterargument:** the existing `open_design` API has generic features/classes; the project is not solely hard-coded to satellites. **Falsification gate:** a new industrial device family is configured from a portable declarative schema without editing the core AutoDesign source; a predeclared search generates a functioning diagnosis pipeline.

### P0-2 — Task abstraction and algorithm search are split: HIGH

**Source:** [generic search and learner portfolio](https://github.com/clonetimer/autogen/blob/a10469bb419a98cade08a02c0c7ed8e11ab9872a/source_product/src/phm_design/open_design_fit.py#L49-L88), [generic search entry](https://github.com/clonetimer/autogen/blob/a10469bb419a98cade08a02c0c7ed8e11ab9872a/source_product/src/phm_design/open_design_fit.py#L218-L300), [fixed ontology](https://github.com/clonetimer/autogen/blob/a10469bb419a98cade08a02c0c7ed8e11ab9872a/source_product/src/phm_design/task_architectures.py#L29-L57), [neural auto-profile](https://github.com/clonetimer/autogen/blob/a10469bb419a98cade08a02c0c7ed8e11ab9872a/source_product/src/phm_design/task_graph.py#L107-L136).

The generic `open_design_fit.design` searches traditional learners and feature programs in a bounded candidate loop; the separately developed neural task graph requires a **fixed** five-class, two-component/two-mechanism ontology and a fixed detection→localization→mechanism topology. R78's opt-in neural profile provides **three declared candidates per head**, not data-driven free architecture synthesis. The system has not shown it can choose detection-vs-prognostics, closed-vs-open class, online-vs-offline, or varying component granularity from an industrial TaskSpec.

**Risk:** two parallel impressive research prototypes without a unified device-agnostic design product. **Counterargument:** constrained search is often safer than unbounded NAS and should not be discarded. **Falsification gate:** shared dataset, ontology, objective, budget and deployment contracts with a generated, audited and usable plan for two heterogeneous tasks, with no new special-case task-graph branch.

### P0-3 — Lack of established cross-device generalization; known neural failure: HIGH

**Source:** [R78 revealed synthetic negative report](https://github.com/clonetimer/autogen/blob/a10469bb419a98cade08a02c0c7ed8e11ab9872a/source_product/README_R78.md); [fixed synthetic test generator](https://github.com/clonetimer/autogen/blob/a10469bb419a98cade08a02c0c7ed8e11ab9872a/source_product/research/r78_neural_shift_benchmark.py); [CI artifact run](https://github.com/clonetimer/autogen/actions/runs/38685417982); [DomainBed](https://arxiv.org/abs/2007.01434).

The development stress gives ERM-TCN **5.36%**, GroupDRO-TCN **1.79%**, V-REx-TCN **1.79%** accuracy and **100% nominal false alarm** for all three under an extreme source-spurious-correlation reversal. This is a **synthetic counterexample**, not a result for real satellite faults. The failure was correctly retained, and repeated target-scoring was removed from continuous CI after revelation. No locked cross-equipment experiment in reviewed evidence supports the desired generality.

**Risk:** optimizer sophistication mistaken for learned invariants; a 'worst source group' objective need not anticipate target shift. **Counterargument:** stress process deliberately adversarial and small; it does not refute all neural approaches. **Falsification gate:** preregister unseen *physical* environments and an unseen industrial equipment family with identical baseline budgets; require actual gain over strong ERM and refuse qualification on failure.

### P0-4 — Industrial online operational contract is missing: HIGH

**Source:** [R60 deployment scope](https://github.com/clonetimer/autogen/blob/a10469bb419a98cade08a02c0c7ed8e11ab9872a/RELEASE_DECISION_R60.md); [task graph inference](https://github.com/clonetimer/autogen/blob/a10469bb419a98cade08a02c0c7ed8e11ab9872a/source_product/src/phm_design/task_graph.py#L380-L415).

Inference is scoped to offline complete trajectories with a trusted initial healthy prefix, and the current graph checks `healthy_prefix_confirmed=True`. No reviewed validation proves causal streaming, time-to-detect, unknown fault handling for genuinely novel faults, sensor outages or real-time operational cost tradeoffs. This does not make offline design wrong, but it makes unqualified industrial-online claims incorrect.

**Risk:** offline accuracy optimized instead of latency-constrained alarm quality. **Falsification gate:** TaskSpec declares decision timestamp, horizon, allowable sensors, missing-data policy, false alarms per operating hour and time-to-detect; end-to-end causal replay on unseen worlds.

### P1-5 — Physical observability and fault mechanism uncertainty: HIGH

**Source:** [R60 unqualified Thermal/weak cases](https://github.com/clonetimer/autogen/blob/a10469bb419a98cade08a02c0c7ed8e11ab9872a/RELEASE_DECISION_R60.md); [physical feature implementation](https://github.com/clonetimer/autogen/blob/a10469bb419a98cade08a02c0c7ed8e11ab9872a/source_product/src/phm_design/platform_features.py#L30-L95); [leakage and observability lessons](https://github.com/clonetimer/autogen/blob/a10469bb419a98cade08a02c0c7ed8e11ab9872a/source_product/docs/DIAGNOSIS_DESIGN_TESTING_LESSONS_AND_LEAKAGE_PREVENTION.md).

A physically unobservable fault cannot be rescued by architecture search or stricter acceptance thresholds. Current feature programs are physically inspired and useful but not a complete identifiability certificate. Control input excitation, measurement gain/noise/latency and nuisance interventions require explicit engineering evidence.

**Risk:** false auto-design conclusions from weak/non-response fault labels. **Falsification gate:** physical fault/no-fault separation and sensitivity analyses with sufficient excitation across environments; if impossible, issue `UNIDENTIFIABLE` with an observability recommendation.

### P1-6 — Model design cost/selection fidelity: MEDIUM-HIGH

**Source:** [task-graph candidate selection](https://github.com/clonetimer/autogen/blob/a10469bb419a98cade08a02c0c7ed8e11ab9872a/source_product/src/phm_design/task_graph.py#L178-L300); [R78 limitations](https://github.com/clonetimer/autogen/blob/a10469bb419a98cade08a02c0c7ed8e11ab9872a/source_product/README_R78.md).

The present search considers bounded predefined learners and one preselected task graph, not general representation, optimizer, scheduler, augmentation or latency-cost co-design. The reported neural training comparisons have approximately matched optimizer-step counts but not equal FLOPs; candidate ranking under unequal compute can be misleading. **Counterargument:** constrained candidates are a sound v1 engineering approach. **Falsification gate:** fixed architecture/training compute ledger, min-size baselines, and evidence that AutoDesign improves over fixed methods for equal budget.

### P1-7 — Leakage defense has improved, but is not an independent test steward: MEDIUM-HIGH

**Source:** [lineage checks](https://github.com/clonetimer/autogen/blob/a10469bb419a98cade08a02c0c7ed8e11ab9872a/source_product/src/phm_design/leakage_control.py); [project leakage retrospective](https://github.com/clonetimer/autogen/blob/a10469bb419a98cade08a02c0c7ed8e11ab9872a/source_product/docs/DIAGNOSIS_DESIGN_TESTING_LESSONS_AND_LEAKAGE_PREVENTION.md).

The system positively rejects supplied protected roles, duplicate physics identifiers and declared feedback contamination; these are valuable. But caller-supplied metadata and Boolean feedback flags cannot independently authenticate real physics provenance, unseen human/agent feedback, or absence of near duplicates. R78's automatic repeated revealed target evaluation was discontinued; this is good remedial action, not retroactive restoration of blindness.

**Risk:** a clean training code path misrepresented as full evaluation independence. **Falsification gate:** separate evaluation steward, immutable full pipeline hashes, access ledger, one-time release of FINAL metrics; prior revealed data remain development data.

### P2-8 — Development integration and release separation: MEDIUM

**Source:** [separate PR #1–#5](https://github.com/clonetimer/autogen/pulls); [R78 CI](https://github.com/clonetimer/autogen/actions/runs/38685889841).

Latest development capabilities are five stacked PRs above R60 Stable and stay Draft. Scoped GitHub tests pass, but successful code contracts do not establish source-level reconciliation, full product qualification or industrial usefulness. **Counterargument:** preserving Stable while reviewing research is responsible. **Falsification gate:** clean source/product build, consolidated PR lineage, integration tests and explicit non-inheritance of Stable qualification.

## Evidence supporting continuation

1. The Stable product has an unusually explicit *limited* contract and preserves negative out-of-scope cases.
2. The generic algorithm/program search, physical features and grouped evaluation are useful starting infrastructure.
3. R69–R78 added real trainable neural candidates, per-environment selection and explicit refusal.
4. Prior test-feedback contamination was documented rather than concealed; R78 negative results were retained and repeated scoring removed.
5. Some degree of bounded architecture choice is desirable for engineering auditability.

## Recommended revised architecture

```text
Industrial user task / sensor contract / cost / latency / ontology / known uncertainty
                         |
               Generic TaskSpec validation
                         |
             Task capability + observability audit
                         |
        AutoDesign search over algorithm AND data pipeline:
        - weak/unknown fault feasibility
        - representation: temporal, spectral, physical residual, learned
        - architecture: linear, tree, CNN/TCN/GRU/Transformer
        - objective: ERM, GroupDRO, V-REx when justified
        - normalization, calibration, abstention, budget
                         |
        nested per-equipment/per-environment validation
                         |
          reject OR frozen model+provenance+scorecard
                         |
             separately stewarded FINAL evaluation
                         |
                 domain adapter outputs
          satellite/Basilisk | rotating equipment | processes
```

**Satellite remains the first adapter**; do not discard Basilisk or its R60 controlled physical cases. But design decisions belong to the generic layers. Distinguish cross-operating-condition generalization, cross-unit within one equipment family, cross-mechanism, cross-equipment-type transfer and sim-to-real. They are different contracts with potentially different feasibility.

## Steering gates before claiming 'industrial AutoDesign'

| Milestone | Mandatory review result | Stop condition |
|---|---|---|
| G0 Product contract | Independent device-agnostic TaskSpec for two unlike equipment/task types, explicit scope and online/offline | New equipment requires changing search core |
| G1 Design loop | Automated candidate, data-pipeline and objective selection, locked budget, reproducible trails | Only enumerates fixed fault rules or manually picks winner |
| G2 Physics/observability | Basilisk-native verified responses and one real industrial equipment dataset, group/operating-shift distinction | Fault/no-fault observables overlap with insufficient sensitivity |
| G3 Scientific design | Matched ERM baselines + neural variants on TRAIN-only leave-family-out, end-to-end/weak metrics and all failures | Repeats synthetic target tuning or removes failing trials |
| G4 Independent qualification | New, physically disjoint FINAL data steward; hash/role/access audit; no new model selection after reveal | No role separation or target feedback influences design |

The proposed order is **G0 → G1 → G2 → G3 → G4**. Adding more neural architectures without G0/G2 is not the highest-value next action.

## Independence, confidence and limitations

- This is an adversarial *static source and declared evidence* review; it did not train a new model or inspect sealed data.
- Source claims are supported by the pinned snapshot; R71 historical detailed accuracy figures are **not independently re-scored here**.
- No actual third-party LLM or outside human reviewer has executed this audit. The review agent's separate process, Git branch and read-only CI are an independence **control**, not a substitute for a separate model/expert panel.
- Strong confidence in restricted task ontology and offline release scope; medium confidence in overall architectural remediation size and generalization prospects, which require future falsification.
- **Decision:** Continue research with a **strategic pivot to generic industrial task design**, refuse broad industrial product/strong generalization claims until G0–G4 pass. No modification to R60 Stable or to R78 reviewed code is authorized by this report.

## External review anchors

- [ISO 13374-1:2003](https://www.iso.org/standard/21832.html): machine condition-monitoring software/data processing guidelines (confirmed current in 2025).
- [Gulrajani and Lopez-Paz, DomainBed, ICLR 2021](https://arxiv.org/abs/2007.01434): fair domain-generalization comparison requires a model-selection protocol, and strong ERM baselines matter.
- [Xiao et al., domain generalization survey for rotating machinery, 2025](https://www.sciencedirect.com/science/article/abs/pii/S1474034624007146): distribution shift is a primary reliability bottleneck.
- [Data-driven machinery fault diagnosis review, Neurocomputing 2025](https://doi.org/10.1016/j.neucom.2025.129588): sparse faults, portability, representation and evaluation remain nontrivial.
