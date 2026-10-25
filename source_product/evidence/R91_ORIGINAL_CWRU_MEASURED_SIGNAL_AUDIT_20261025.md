# R91 — Actual original CWRU 48-kHz vibration signals: first verified ingestion

**Date:** 2026-10-25  
**Repository:** clonetimer/autogen  
**Research commit:** `324ab2fe9a98849e3e5625447cd0b7c16294450d`  
**GitHub Actions run:** https://github.com/clonetimer/autogen/actions/runs/40190914686  
**Job:** Official CWRU 97 and 109 raw MAT acquisition (no ML inference)  
**Outcome:** **SUCCESS**; original official binary HTTPS download, exact MATLAB
field extraction, SHA-256 computation and descriptive numeric read all ran in
a temporary GitHub-hosted Linux workspace. JSON-only report uploaded as
`r91-official-cwru-original-receipt` artifact. Raw measurements were **not**
added to git or uploaded as artifacts.

## Observed original measurement identity

| Exact measurement | CWRU `97.mat` | CWRU `109.mat` |
|---|---:|---:|
| Publisher class metadata | healthy 0 HP | 0.007-inch inner race damage 0 HP |
| CWRU experiment | baseline normal | 48k drive-end fault |
| Original source sampling rate declaration | 48,000 Hz | 48,000 Hz |
| Original MATLAB field actually found | `X097_DE_time` | `X109_DE_time` |
| Original source binary bytes | 3,903,344 | 3,903,344 |
| Original waveform sample count | 243,938 | 243,938 |
| Non-overlapping 4,096-point descriptive windows | 59 | 59 |
| RMS median, unit **not yet authenticated** | 0.07376924272035389 | 0.5890906262132836 |
| RMS Q1 | 0.07237733926964318 | 0.5815649313187469 |
| RMS Q3 | 0.07506387596495939 | 0.5925543763665966 |
| Crest-factor median | 3.3131308367242487 | 5.04214688934917 |

SHA-256 of the original MATLAB **bytes received on this first HTTPS download**:

- `97.mat`:
  `16bf48babcf1c7ac224bc1a81cd9eafdb27e42d5cf559761907e067e8eeadf3c`
- `109.mat`:
  `daddef5f784879becdf8fefbe2453f11c3cfc8d061a2a671da2f546d1bb48460`

Exact official binary source URLs:

- https://engineering.case.edu/sites/default/files/97.mat
- https://engineering.case.edu/sites/default/files/109.mat

These hashes identify the **first retrieved files**, not a separately
verified, publisher-signed checksum. If any future retrieval differs,
treat that as a source-version change requiring new review, not silent
substitution.

## Correct and incorrect conclusions

1. **Confirmed:** the study's GitHub-hosted compute environment can access
   two **actual publicly hosted measured CWRU signals**, parse exact DE
   vibration fields and produce finite measurements and hash receipts.
2. **Confirmed:** the published class labels have markedly different
   descriptive RMS/crest distributions in these particular original
   recordings.
3. **Not shown:** independent physical-bearing group identifiers,
   measured sensor calibration/unit accuracy, a statistically independent
   fault diagnosis comparison, test-set accuracy or calibrated false-alarm
   rates. A 4,096-sample window is **not** a new physical bearing.
4. **Not shown:** any superiority of evolving Gaussian/TSK fuzzy networks
   over logistic, conventional ML or neural baselines. Models were **not
   trained** in this R91 acquisition study.
5. **No revision to R83f/R83g:** previous actual native Basilisk experiment
   negative findings and strict source/validation controls remain.
6. **Deployment decision:** NO_ALGORITHM_PROMOTION; Stable main unchanged.

## Reproduction

From `source_product`, follow the Windows/Linux commands in
`README_R91.md`. The online GitHub job downloads the originals into an
ephemeral runner directory; never commit raw files. Use the two SHA-256
values above to check the received bytes. The complete report's explicit
`diagnostic_training_performed=false`,
`measured_generalization_accuracy=null` and
`fuzzy_vs_baseline_advantage_proven=false` fields prevent confusing this
waveform audit with a diagnosis benchmark.

The raw sample statistics and hashes above were copied verbatim from the
**successful original GitHub Actions run log and JSON-only artifact**, not
from a synthetic pytest fixture or a guessed result.
