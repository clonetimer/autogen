> **Current software release: `phm-design 0.8.0 Stable Core`.** The supported **core framework** covers frozen package/runtime and command-line interfaces, original `0.7.0` limited offline simulation capability, verified R93 data contracts and **SOURCE-only diagnostic design proposal**, Windows/Linux clean-wheel usage, and regression-tested research APIs. **Actual `torch_like` native fit/save/reload and full AAS host integration are OPTIONAL EXPERIMENTAL integrations, not 0.8.0 Stable guarantees.** No independent real industrial equipment quality/flight-safety qualification. See [0.8.0 Stable release contract](source_product/docs/STABLE_080_RELEASE.md).

# PHM AutoDesign — `phm-design` 0.8.0 Stable Core

> Repository `clonetimer/autogen` hosts the PHM AutoDesign (`phm-design`) source tree and release-delivery records.
>
> **Research-only / NOT for flight safety.** This is a research-grade multi-family compositional network design toolkit for Prognostics and Health Management (PHM). It must **not** be used for real spacecraft flight-safety decisions or online real-time closed-loop control. See [Safety & Scope](#safety--scope).

PHM AutoDesign designs and qualifies multi-family compositional diagnostic networks, targeting the spacecraft (SAT / Basilisk simulation) domain. The prior **`0.7.0 Stable`** release has limited qualification for 15 specified SAT/Basilisk simulation-domain offline diagnostic cases under `RELEASE_DECISION_R60.md`; `0.6.0` is retained as a rollback artifact.

## Package

| | |
|---|---|
| **Name** | `phm-design` |
| **Version** | `0.8.0` |
| **Python** | `>=3.11,<3.14` (tested on CPython 3.11 / 3.12) |
| **Core deps** | `torch==2.10.0`, `numpy==2.3.5`, `scikit-learn==1.8.0` |
| **Optional** | `dev` (pytest), `simulation` (Basilisk `bsk`, `colorama`, `sgp4`), `automl` (optuna) |

## CLI entry points

`phm-design`, `phm-design-platform`, `phm-design-limited`, `phm-design-algorithm-dag`, `phm-design-expert-dag`, `phm-design-open`, `phm-design-readiness`.

## Install (from source)

```bash
python -m pip install --no-build-isolation --no-deps .
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/phm_design
```

For the **0.8.0** wheel/sdist, see the [Stable 0.8.0 release](https://github.com/clonetimer/autogen/releases/tag/v0.8.0) once its GitHub Actions publication completes. The historical **0.7.0** qualified wheel/sdist remain in `artifacts/` and [README_RELEASE_R60.md](README_RELEASE_R60.md); these are not 0.8.0 binaries.

## Safety & Scope

- **Do not use for real flight-safety decisions or online closed-loop control.** Support is limited to offline discrimination over whole-trajectory data within supported simulation distributions.
- The retained `0.7.0` is a **limited simulation-domain Stable** release, not certification for real flight-safety decisions, online diagnosis, or arbitrary datasets. A newly trained `design` model is `DEVELOPMENT_ONLY` and does **not** automatically inherit the bundled 15-cell diagnostic qualification.
- Do not re-label past failed evaluation worlds as new blind tests, and do not expand Stable support scope without fresh independent training / calibration / sealing evidence.

## Repository layout

```
artifacts/      Released wheel & sdist (0.7.0 + 0.6.0 rollback)
source_product/ Core Python package (src/, tests/, scripts/, configs/, pyproject.toml)
evidence/       Validation evidence (small)
rc_evidence/    RC evidence
fixtures/       Test fixtures
testbed/        Test bed
*.json *.md     Release audit, qualification certificate, SHA256 manifests,
*.sh  *.py      build / verify scripts, and release-decision records
README_RELEASE_R60.md   Install / recovery / scope guide for the 0.7.0 Stable release
```

## Excluded from version control

`qualification/` (≈141 MB / 9700+ generated qualification-world artifacts), `__pycache__/`, `failed_attempts/`, and `*.log` build/run logs are excluded via `.gitignore`. The full 900-world qualification evidence package can be regenerated from the release/qualification scripts or obtained as a separate archive; daily install and inference do not require re-running the qualification worlds.

## Cross-equipment algorithm-design research (R93–R99, experimental)

The generic industrial `EquipmentCard`, `DatasetCard`, and `TrialSpec`
research contracts, with strictly TRAIN-only design plans, are documented in
[source_product/docs/R93_EQUIPMENT_DATA_TRIAL_CONTRACTS.md](source_product/docs/R93_EQUIPMENT_DATA_TRIAL_CONTRACTS.md).

R93–R99 introduce **opt-in research-only diagnostic algorithm design tools**:
source-restricted device and data contracts (R93), reusable AAS/torch_like
candidate execution adapters (R94), bounded algorithm/sensor/hyperparameter
search (R95), fold-local preprocessing pipelines (R96), physical frequency
features (R97), envelope/order/fusion graph design with strict physical
prerequisites (R98), and explainable bounded revisions, partial-cost receipts,
and a sealed external-evidence review gate (R99). See
[source_product/docs/R99_EXPLAIN_REVISE_VALIDATE.md](source_product/docs/R99_EXPLAIN_REVISE_VALIDATE.md).

**Merging research code into `main` does not promote it into the frozen
`phm-design 0.7.0 Stable` release or extend the release's 15 qualified
Basilisk simulation-domain offline cases.** Industrial equipment identity,
predictions from frozen executables, authentic cross-equipment external
generalization, end-to-end resource fairness and production approval are
not established by these additions. No deployment/flight-safety claim is made.
The existing limited Stable release artifacts and version remain unchanged.

## License

No license file is included in this delivery. All rights reserved unless otherwise stated.

## References

- Detailed install / recovery / scope guide: [README_RELEASE_R60.md](README_RELEASE_R60.md)
- Source package readme: [source_product/README.md](source_product/README.md)
- Release decision & audit: `RELEASE_DECISION_R60.md`, `FINAL_RELEASE_AUDIT.json`, `QUALIFICATION_CERTIFICATE.json`, `R60_15_CELL_QUALIFICATION.csv`.
