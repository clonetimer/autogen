# R73 — lineage preflight hardening (development only)

R73 tightens the R72 safety boundary without changing the already-frozen 0.7.0 Stable release or granting any new diagnosis qualification.

## Verified fixes

1. **Optional robust selection no longer disables a provided lineage contract.** `open_design_fit.design()` now validates role, count, world ID, and (when supplied) environment provenance even when `robust_policy=None`. A `FINAL_ID`/`FINAL_OOD` record is rejected before candidate search.
2. **Base-world identity rejects explicit split or injected-truth keys.** Nested mapping/list keys such as `split`, `data_role`, `fault_label`, and `fault_injection` cannot be included in `WorldLineage.base_physics` to generate split-dependent physics hashes. This is a narrow defensive check, not a canonicalization algorithm.
3. Tests exercise both bypasses using synthetic fixtures, without reading sealed or already-revealed test worlds.

## Evidence and constraints

- Targeted lineage and policy tests: 11 passed.
- General search, R69 task graph, R72 lineage/policy regression: 175 passed.
- The previous R72 baseline reported 820 passed in its saved full-regression log. The R73 single-process full regression attempt timed out before completing; therefore **R73 full-suite PASS is not claimed**.
- R71 retrospective results remain **already-revealed development evidence** and must not be called new independent evaluation.
- Manifest values are caller-supplied; this code cannot authenticate provenance, prevent out-of-process access to FINAL data, detect near duplicates, or erase prior test-feedback contamination. A separate trusted data steward must validate the canonical base-physics configuration and enforce access policy.
- The environment is offline for direct `git push` to GitHub. Repository packaging and the Git bundle can be reviewed and pushed by an authorized local Git client; no remote push or PR is represented by this document.

## Engineering guidance

1. For any design run, provide a valid `WorldLineage` for every training example. Prefer `RobustDesignPolicy` and environment-family leave-one-out selection for cross-scenario design.
2. Freeze metric thresholds and preprocessing/feature policies before testing; any exposed FINAL feedback used to change design requires a new independent final cohort.
3. Use fresh physics families and role-separated data access to test modifications. Do not treat distinct filenames, random seeds, or hash values as sufficient proof of domain independence.
4. Keep the complete R71 failure history and rejected candidate trials.
