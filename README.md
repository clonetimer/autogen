# PHM AutoDesign — `phm-design` 0.7.0

> Repository `clonetimer/autogen` hosts the PHM AutoDesign (`phm-design`) source tree and release-delivery records.
>
> **Research-only / NOT for flight safety.** This is a research-grade multi-family compositional network design toolkit for Prognostics and Health Management (PHM). It must **not** be used for real spacecraft flight-safety decisions or online real-time closed-loop control. See [Safety & Scope](#safety--scope).

PHM AutoDesign designs and qualifies multi-family compositional diagnostic networks, targeting the spacecraft (SAT / Basilisk simulation) domain. The frozen **`0.7.0 Stable`** release has limited qualification for 15 specified SAT/Basilisk simulation-domain offline diagnostic cases under `RELEASE_DECISION_R60.md`; `0.6.0` is retained as a rollback artifact.

## Package

| | |
|---|---|
| **Name** | `phm-design` |
| **Version** | `0.7.0` |
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

For the released wheel/sdist, see `artifacts/` and [README_RELEASE_R60.md](README_RELEASE_R60.md).

## Safety & Scope

- **Do not use for real flight-safety decisions or online closed-loop control.** Support is limited to offline discrimination over whole-trajectory data within supported simulation distributions.
- `0.7.0` is a **limited simulation-domain Stable** release, not certification for real flight-safety decisions, online diagnosis, or arbitrary datasets. A newly trained `design` model is `DEVELOPMENT_ONLY` and does **not** automatically inherit the bundled 15-cell diagnostic qualification.
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

## License

No license file is included in this delivery. All rights reserved unless otherwise stated.

## References

- Detailed install / recovery / scope guide: [README_RELEASE_R60.md](README_RELEASE_R60.md)
- Source package readme: [source_product/README.md](source_product/README.md)
- Release decision & audit: `RELEASE_DECISION_R60.md`, `FINAL_RELEASE_AUDIT.json`, `QUALIFICATION_CERTIFICATE.json`, `R60_15_CELL_QUALIFICATION.csv`.
