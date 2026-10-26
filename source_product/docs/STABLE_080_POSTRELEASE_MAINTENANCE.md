# PHM AutoDesign 0.8.0 — Post-release maintenance and research PR ledger

Status: **Stable Core published; research remains experimental**. Date: 2026-10-26 (JST). This document records post-release housekeeping; it does **not** change the released 0.8.0 binaries, release tag or scientific qualification.

## Immutable release baseline

- Official release: https://github.com/clonetimer/autogen/releases/tag/v0.8.0
- Tag: `v0.8.0` → commit `5248049275167bbcf396fac775bcb71a60c3e76e`.
- Original same-commit release run: https://github.com/clonetimer/autogen/actions/runs/40252242491 (**9/9 check runs successful**, including publication).
- Release assets: `phm_design-0.8.0-py3-none-any.whl`, `phm_design-0.8.0.tar.gz`, `SHA256SUMS`.
- Regression evidence in PR #53 / same-run workflow: historical Python suite **848 passed / 47 optional skipped**; R93–R99 tests **150 passed / 4 optional skipped** on each of Linux 3.11, Linux 3.12 and Windows 3.11. A skipped test is not certified execution.
- Retain original 0.7.0 offline SAT/Basilisk **15-case** evidence and rollback artifacts unmodified; the new package does **not** enlarge those qualifications.

`main` may receive maintenance commits after the release commit, but these commits **do not retroactively modify** the downloaded or tagged v0.8.0 release. Do not move the tag, overwrite assets, or claim new research capabilities are included in the Stable support contract. Use a separately gated patch release for any future changes to released code or distributions.

## Stable versus research contract

**Stable Core:** installable `phm-design` Python distribution and CLI; previously qualified limited offline simulation compatibility; R93 equipment/dataset/trial contract checks and R95 SOURCE-only design-plan generation; regression-tested supported platforms (Linux 3.11/3.12, Windows 3.11). A green CI build is evidence of software conformance, **not** proof of a diagnostic model's generalization.

**Experimental/not verified by Stable:** R94 real private `torch_like` native fit/predict/save/reload; full AAS host application integration; R96–R99 diagnostic-pipeline/physics/explain/revise research APIs beyond their source-only contract tests; R80–R92 neural/fuzzy/physical-cohort research; independently measured multiple-device out-of-domain performance; compute-matched GPU/FLOP claims; real spacecraft flight or industrial safety-critical use.

No dependency, runtime, simulator, fitted model, signal physics or algorithm was changed by this maintenance cycle.

## Historical Draft PR triage

At the start of the 2026-10-26 maintenance review, **36 open Draft PRs** existed (no open ready-for-review PR).

### Close as superseded, preserving branches and discussion (7 PRs)

- **#1–#5**: R74–R78 stacked early neural-design experiments. Historical research can still be read on their original branches; do not merge or retrospectively claim 0.8.0 support.
- **#7**: R79 independent direction review; product positioning was resolved through the later R93–R99 integration and scoped Stable Core release.
- **#9**: early R80–R85 roadmap-only PR; retained as planning history and superseded by subsequently agreed Stable/research separation.

Close with an explanatory comment and **do not delete branches or raw experimental evidence**. If a later feature specifically depends on these branches, open a fresh narrowly scoped PR targeting a current branch, with new acceptance gates.

### Keep as research Draft, not automatically mergeable (29 PRs)

- **#15–#19**: R80–R82 equipment-agnostic TaskSpec, trial-engine and source-evidence lineage.
- **#20–#35**: R83–R84 native Basilisk experiments, guarded neural ablations, physical observability and Pareto research. Preserve negative results and original physical evidence hashes.
- **#36–#39**: R85–R88 adaptive cross-family and neuro-fuzzy training / frozen validation research; no generalization superiority established.
- **#40–#42**: R89–R91 original Paderborn/CWRU measurement/cohort research, including preregistration limitations and refusal gates.
- **#43**: R92 industrial cross-domain boundaries and evidence-claim gate; check for overlap with released R93–R99 before proposing any implementation merge.

Because these are **stacked** Draft PRs, their head branches target one another. Closing one downstream prerequisite or rebasing the whole chain without a reviewed lineage plan could obscure evidence and scientific intent. They are **not** outstanding Stable release blockers. Promote a research PR only after verifying a coherent diff against current `main`, test coverage, source/train/validation role separation, and evidence provenance.

After the seven historical closures, the expected inventory is **29 retained historical Draft research PRs**, plus any new maintenance PR while it is open.

## Minimal regression and hygiene rules

1. Documentation changes must identify the *already published* Release URL, correct workflow filename `.github/workflows/stable-core-080.yml`, tag SHA and supported platforms.
2. Check that README no longer calls the v0.8.0 Release unpublished or implies that R93–R99 was not included in the v0.8.0 package. Avoid confusing `0.7.0` historical qualification with `0.8.0` software support.
3. Run the existing Stable and R93–R99 CI checks on any documentation-maintenance PR when triggered. Do **not** run or regenerate a new qualification dataset to justify docs-only changes.
4. Do not add feature branches, new algorithms, training functionality, or optional private `torch_like` secrets merely for release housekeeping.
5. Any later `0.8.1` patch must use a new annotated release decision, new build/tag/checksums and clean install verification; never repoint `v0.8.0`.

## Research continuation

Keep one separately scoped track for R99 **explain → bounded revise → frozen SOURCE comparison → independently measured-device validation** with matched baseline inputs, appropriately isolated source/validation groups and reporting of compute cost. Do not treat software CI passes, synthetic sensor fixtures or simulation response traces as independent proof of improved real-device diagnostic quality.
