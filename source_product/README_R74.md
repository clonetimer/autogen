# PHM AutoDesign R74 development handoff

Source version: `phm-design==0.8.0.dev13` (development only).

R74 closes a metadata-level loophole in environment-family split integrity. Different names for the same base physical configuration or generation family are no longer accepted as distinct environments. The hard gate runs before AutoDesign chooses or fits candidates. This is **not** protection against forged source metadata, exposure to revealed test results, near-duplicate trajectories, or missing physics parameters.

Stable 0.7.0 remains frozen. R71 is revealed and only suitable for development diagnostics. There is no new final-world qualification and no evidence of actual model performance gains from this change.

To verify: `python -m pytest -q tests/phm_design/test_r74_environment_coherence.py tests/phm_design/test_r72_leakage_and_robust_search.py`.

For GitHub publication, apply the `source_product` overlay to a branch forked from the actual remote `main`. Do not add a second root-level Python project or overwrite the R60 Stable evidence stored at the repository root.
