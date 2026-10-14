# R74 — Environment-family coherence gate (DEVELOPMENT ONLY)

## Motivation
R71 showed that high training-side validation can conceal severe cross-scenario failure. R72/R73 already prohibit protected roles and overlapping generation families **across roles**. However, two TRAIN records from the same physical family could claim distinct environment-family identifiers, creating a bogus leave-environment-family-out estimate. This is a metadata-consistency failure, not proof that OOD generalization is impossible.

## Change
`inspect_partition` now rejects any pair of records sharing either `generation_family_id` or canonical `base_physics` digest while claiming different `environment_family` values, even within the same data role. The check runs through `authorize_design` before the learning/search stage. The output receipt explicitly records `environment_family_coherence` among checked invariants.

## Constraints
- Metadata must still originate with a trusted generator or data steward. Naming consistency does not authenticate physical parameter values or guarantee novelty.
- Identical base-physics configurations across truly different environments must be represented with their actual environmental parameters in the canonical physics configuration; otherwise such a dataset cannot support an 'unseen environment' claim.
- Does not open R60 sealed worlds, requalify R71 revealed evaluation, or create a new qualified model.
- Existing R73 0.8.0.dev12 and Stable 0.7.0 releases remain intact. The new development candidate is 0.8.0.dev13.

## Verification
Synthetic negative controls: same generation family, different reported environments; same canonical physics, different reported environments; independent worlds accepted; healthy/fault paired trajectories accepted. Run `PYTHONPATH=src pytest -q tests/phm_design/test_r74_environment_coherence.py`.

GitHub publication is contingent on the connected GitHub App receiving Contents write (and Pull requests write) scope; a 403 means remote branches and PRs have not been created by this run.
