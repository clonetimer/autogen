# PHM AutoDesign 0.8.0.dev13 — R74 Design Safety Research

**DEVELOPMENT ONLY.** This development branch is based on the complete R69 0.8.0.dev10 codebase; it does not change or extend the 0.7.0 Stable certification. CUDA training and actual GPU hardware validation remain outside this change. R70 interim work is not included because it has no fully verified integrated handoff.

> **R82.0 source intake (Draft):** The TRAIN-only manifest adapter and separately reviewed physical-map gate are described in [README_R82.md](README_R82.md). This does not execute Basilisk or qualify physical equipment.\n\n> **Latest stacked development:** R80/R81a/R81b/R81c are research branches, not R60 Stable. The current R81c slice adds nested TRAIN-only temperature calibration and optional UNKNOWN decisions (see [README_R81C.md](README_R81C.md)). The R81b model/feature search context is in [README_R81B.md](README_R81B.md). Neither has independent FINAL qualification.\n\n## R72–R74: leakage-aware automatic design

R74 additionally rejects synthetic environment splits that relabel the same generation family or canonical base-physics configuration under different environment-family identifiers. This is a metadata-consistency gate, not independent authentication of physical worlds or a new diagnostic qualification.

- `phm_design.leakage_control.WorldLineage` describes a physical-generation family independently of split name, with canonical physical configuration, environment family and optional observation digests.
- `inspect_partition` / `validate_manifest_separation` fail on cross-role base-physics, generation-family or exact-observation collisions. Repeated healthy/fault trajectories within a single physical world are allowed within the **same role**.
- `authorize_design` rejects protected final roles and records feedback taint. It only permits TRAIN metadata to be passed into the development worker.
- `task_graph.design_task_graph` can choose detection, localization and mechanism heads using a **worst-environment OOF loss**, with optional leave-one-environment-family-out outer screening, rather than global mean loss. Robust search requires provenance. Existing R69 average-selection behavior remains available and is explicitly unverified for lineage separation unless a manifest is supplied.
- Thresholds and downstream gate policies remain predeclared; the R69 reject-on-failure mechanism remains in force.

**Limitations:** This is a development-side guard and metadata audit, **not a sealed evaluator, OS access-control sandbox or proof of non-leakage**. A caller could provide inaccurate physics metadata or conceal earlier TEST/OOD feedback. Physical provenance must be authenticated by an independent steward, and FINAL-ID/FINAL-OOD truth must remain outside model-design environments. Hash disjointness cannot prove OOD novelty, near-duplicate absence or causal online validity. Already revealed R71 worlds are **development evidence only**.

Read `docs/DIAGNOSIS_DESIGN_TESTING_LESSONS_AND_LEAKAGE_PREVENTION.md` for the provenance/taint methodology and prior failure history. See `README_R69.md` for existing task graph APIs.

```bash
PYTHONPATH=src python -m pytest -q tests/phm_design/test_r72_leakage_and_robust_search.py
PYTHONPATH=src python -m pytest -q tests/phm_design/test_r69_task_graph.py
```

The five frozen 0.7 model files are included for provenance, but the dev package reports `qualified=false` and must not inherit historical qualification.

### Cross-scenario automatic search (R72)

The existing general `open_design_fit.design` API now accepts optional
`RobustDesignPolicy`, `training_lineage`, and `feedback_used_for_design`.
With a strict policy, it derives physical-group folds from checked
`generation_family_id`, holds environment families out during selection,
checks per-family fault recall, health false acceptance, and overall accuracy,
and raises `RobustDesignRejected.audit` rather than returning a model when no
candidate passes. This is a **training-world development gate**, not an
independent test of generalization. Results from the revealed R71 worlds remain
research diagnostics and are never relabeled as new blind certification.

## R73 lineage-preflight hardening (0.8.0.dev12)

The optional `RobustDesignPolicy` cannot be used to bypass the access policy:
if an AutoDesign call supplies a `training_lineage` manifest, its roles and
world identities are checked **even when robust selection is disabled**.
Base-physics identity metadata also rejects explicit split, output-path and
fault-truth keys, including nested keys. See
`docs/R73_LEAKAGE_REVIEW.md` for limitations, testing and threat boundaries.
This is development-only source and provides **no new independent diagnosis
qualification**. The historical 0.7.0 Stable contract remains unchanged.
