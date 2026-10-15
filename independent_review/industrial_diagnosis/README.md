# Industrial Diagnostic AutoDesign — Independent Direction Reviewer

This directory holds the **isolated read-only strategic review agent** for PHM AutoDesign. It evaluates the *algorithm-design product*, treating SAT/Basilisk as the first validation adapter, **not** as the whole product.

- [AGENT_CHARTER.md](AGENT_CHARTER.md): independent mandate, adversarial questions, prohibitions, and optional prompt for a separately provisioned LLM reviewer
- [source_lock.json](source_lock.json): immutable source manifest for R78 commit `a10469bb419a98cade08a02c0c7ed8e11ab9872a`, using exact Git blob identities
- [review_agent.py](review_agent.py): standard-library-only audit runner; checks manifest and source evidence and emits repeatable JSON/Markdown
- [R79_DIRECTION_REVIEW.md](R79_DIRECTION_REVIEW.md): substantial qualitative review, risk ranking, positive counterevidence, falsification tests and redesign gates
- [tests/test_review_agent.py](tests/test_review_agent.py): deliberate evidence-tampering/path-traversal/missing-evidence negative controls

## What 'independent' actually means

The reviewer lives on a Git branch based on R60 Stable `main`, **not** on the R78 developer branch. GitHub Actions checks out a fixed R78 development SHA into a separate directory and has `contents: read` only, with credentials not persisted. It does not train models, open sealed data, modify reviewed code, update network weights or tune thresholds. Its outputs are reports, not deployment certificates.

**This is a deterministic read-only reviewer agent, not a spawned independent LLM or a human panel.** A separate LLM can use the charter prompt and same pinned evidence, if explicitly provisioned and logged. Model judgment independence has *not* been established by CI isolation alone.

## Reproducible local usage

```bash
git clone https://github.com/clonetimer/autogen.git review
cd review
git checkout review/industrial-autodesign-direction-r79

git clone https://github.com/clonetimer/autogen.git target_snapshot
git -C target_snapshot checkout a10469bb419a98cade08a02c0c7ed8e11ab9872a

python -m unittest discover -s independent_review/industrial_diagnosis/tests -v
python independent_review/industrial_diagnosis/review_agent.py \
  --source-root target_snapshot \
  --lock independent_review/industrial_diagnosis/source_lock.json \
  --out-dir review_output
```

The report must be written **outside** the reviewed source checkout. Any mismatched Git commit, Git blob, missing file or unsafe path is an error; it cannot result in a passing audit.

## Decision semantics

- `KEEP_DIRECTION_WITH_PIVOT`: algorithmic auto-design research is worth pursuing, but must become industrial TaskSpec-first.
- `STOP_GENERAL_PRODUCT_CLAIM`: no current basis to certify arbitrary industrial equipment diagnosis/online generalization.
- R60's 15-cell simulation-limited Stable remains valid only within its original bounded scope.
- Synthetic R78 target results are already revealed, including poor OOD accuracy; no later change may reuse them as a new sealed qualification.

## Explicitly not tested

No new Basilisk generation, external industrial sensor data, independent FINAL-ID/FINAL-OOD score, live online response latency, real machine diagnoses, physical hardware or independently trained external reviewer LLM were executed here. Evidence is static + preexisting public project logs. The recommendations and science remain subject to external expert falsification.

## Next governance step

Approve or reject direction **gates** (generic industrial TaskSpec, second equipment family, causal decision contract, strong baselines and independent final steward), not another satellite-specific neural model based on the already-revealed target. Developer fixes must use separate feature branches; do not merge this review into the current neural developer PR to avoid reviewer/developer role collapse.
