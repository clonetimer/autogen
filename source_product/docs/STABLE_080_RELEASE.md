# phm-design 0.8.0 Stable Core — Release Contract

**Published release: [v0.8.0](https://github.com/clonetimer/autogen/releases/tag/v0.8.0), 2026-10-26; software Core Stable.**
**Approved scope change:** The user explicitly waived native private `torch_like` fit/predict/save/reload from the Stable release gate. It remains an optional, **unqualified** research integration and is not marketed as part of 0.8.0 Stable.

## What "Stable" means here

Stable applies to the **installable and regression-tested core framework**, not to the comparative diagnostic accuracy of arbitrary generated algorithms or their suitability for safety-critical industrial or flight hardware.

| Capability | 0.8.0 status |
|---|---|
| Wheel and sdist packaging; package metadata, CLI entry points | **Core Stable**; built and installed from official release assets |
| Clean-wheel installation, version metadata, SOURCE-only R93/R95 design proposal | **Core Stable**; Linux Python 3.11/3.12 + Windows Python 3.11 |
| Historical 0.7.0 limited simulation-domain offline architecture and compatibility | **Retained**; regression-tested, original 15-cell qualification is **not expanded** |
| R93 source data contracts, R95 proposal; R96–R99 research structural APIs and source-only explainability | **Available experimental APIs**; tested against synthetic/source-only fixtures, **no independent real-equipment effect guarantee** |
| R94 actual private `torch_like` provider fit, four native algorithms, model artifact save/reload | **Optional EXPERIMENTAL; not core Stable support**. Must install/access private provider separately, no cross-repository Actions requirement for v0.8.0 |
| AAS application host execution and end-user deployment UI | **Experimental; out of this release** |
| Paderborn independent multiple physical bearings, complete cost parity, qualified model superiority | **Not proven; out of release**, remains research backlog |
| Live industrial control, real spacecraft flight-safety use, qualified on-orbit autonomy | **Not supported or certified** |

## Frozen main-release gate

1. `source_product/pyproject.toml`, `src/phm_design/__init__.py`, built wheel and installed distribution all report **0.8.0**.
2. Build **both** `phm_design-0.8.0-py3-none-any.whl` and `phm_design-0.8.0.tar.gz`, verify with `python -m twine check --strict`, inspect expected R93–R99 package modules, generate **SHA256SUMS**.
3. Install actual built distribution (not PYTHONPATH source tree) and run `phm-design --help`, `phm_design.industrial_contracts.design_cli propose` from a separate temporary directory. Verify study SHA, no VALIDATION/FINAL read, stable metadata. Run on Ubuntu Python 3.11/3.12 and Windows Python 3.11.
4. Run complete `tests/phm_design` legacy + research Python suite with pinned CPU PyTorch, numpy, sklearn, Optuna and SciPy. Known explicit environment-dependent skips are enumerated in the Actions logs; **no failed tests may be hidden**.
5. Run R93–R99 synthetic/source-only contract suite on Linux 3.11, Linux 3.12 and Windows 3.11. These are not falsely described as independent real-bearing efficacy.
6. Only when all required jobs above succeed **on the same source commit**, generate a GitHub tag `v0.8.0` and an official Release containing wheel, sdist and the matching SHA256SUMS. Tag target must be exactly the main merge commit. No manual publication on a failed gate.
7. Keep the prior `0.7.0` wheel/sdist and historical release evidence as the rollback. `0.8.0` must not overwrite or relabel them.

## Explicitly NONBLOCKING deferred acceptance

The previous RC #52 asked for an exact-commit `clonetimer/torch_like` source checkout through a read-only secret. That repository is PRIVATE and the default `GITHUB_TOKEN` cannot read it. The 3 native provider jobs were **blocked by missing permission, never passed**. The user deliberately removed this integration from current Stable scope. The remaining tests do not claim to substitute for them; to qualify native provider execution later, install a reviewed `torch_like` and exercise the preserved optional smoke with source-only inputs.

## Reproduce locally (Windows)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install numpy==2.3.5 scikit-learn==1.8.0
# Download the audited 0.8.0 release wheel from GitHub Releases, then:
.\.venv\Scripts\python.exe -m pip install --no-deps .\phm_design-0.8.0-py3-none-any.whl
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m phm_design --help
.\.venv\Scripts\python.exe -m phm_design.industrial_contracts.design_cli --help
```

For the SOURCE-only example, run `source_product/scripts/stable_core_wheel_smoke.py` from the source checkout after installing the built wheel; it uses the synthetic `source_product/examples/industrial_contracts` sample cards **only to test reproducibility**, not device accuracy.

## Completed release process and maintenance rule

The actual publication workflow is [`.github/workflows/stable-core-080.yml`](../../.github/workflows/stable-core-080.yml), not the historical `stable-candidate-080.yml` (removed after RC). PR-triggered checks are build/acceptance only. The release run on `main` succeeded: [Actions #40252242491](https://github.com/clonetimer/autogen/actions/runs/40252242491). The `v0.8.0` tag points to **`5248049275167bbcf396fac775bcb71a60c3e76e`**, and the [Release](https://github.com/clonetimer/autogen/releases/tag/v0.8.0) published `phm_design-0.8.0-py3-none-any.whl`, `phm_design-0.8.0.tar.gz`, and `SHA256SUMS`. All nine same-commit check runs succeeded, including release publication.

**Post-release edits MUST NOT retag `v0.8.0`, overwrite released assets, modify the frozen 0.7.0 15-case evidence, or silently expand Stable capability claims.** Maintenance on main after the tag is not automatically part of the published v0.8.0 wheel. For subsequent fixes, use a separate patch release with its own checksum and test evidence. See [post-release maintenance policy](STABLE_080_POSTRELEASE_MAINTENANCE.md).

All artifacts are research-only and **not for flight safety or real equipment safety-critical control**.
