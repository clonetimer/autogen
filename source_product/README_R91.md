# R91 — original measured CWRU waveform access, before any diagnostic-model claim

**Objective:** Advance general industrial diagnosis algorithm design from synthetic
smoke tests towards public *actual laboratory measurements*, while resisting
the common mistake of splitting windows from one physical recording into
fictitious independent bearing examples.

This experiment does **not** further enlarge the fuzzy architecture family.
R83f/R83g original negative results stand. Stable main is unchanged.

## Frozen measured signal sample

Two original Case Western Reserve University (CWRU) Bearing Data Center
0-HP drive-end vibration MAT files, both published as 48-kHz acquisition:

| File | Source experiment | Official download | Exact MATLAB sensor field |
|---|---|---|---|
| `97.mat` | Normal baseline, motor 0 HP | [97.mat](https://engineering.case.edu/sites/default/files/97.mat) | `X097_DE_time` |
| `109.mat` | 48 kHz drive-end, 0.007-in inner-race fault, motor 0 HP | [109.mat](https://engineering.case.edu/sites/default/files/109.mat) | `X109_DE_time` |

Data center: [original downloadable dataset](https://engineering.case.edu/bearingdatacenter/download-data-file).
We selected the **48-kHz** drive-end defect file 109 rather than the
**12-kHz** defect file 105 to avoid a built-in sample-rate/class confound.
This does NOT certify equivalence of physical bearing, accelerometer
calibration, fault manufacturing process or independent specimens.

## Execution on an ordinary Windows or Linux PC

With `source_product` as the working directory:

PowerShell:

```powershell
python -m pip install numpy scipy
$env:PYTHONPATH="src"
python -m phm_design.industrial.r91_cwru_reality_check --raw-dir "D:/data/cwru-original" --download-official --receipt "D:/data/r91-original-waveform-audit.json"
```

Bash:

```bash
python -m pip install numpy scipy
export PYTHONPATH=src
python -m phm_design.industrial.r91_cwru_reality_check --raw-dir "$HOME/cwru-original" --download-official --receipt "$HOME/r91-original-waveform-audit.json"
```

If the original links are unavailable, manually obtain exactly
`97.mat` and `109.mat` from the linked CWRU official source; omit
`--download-official` and audit the local files. The CLI refuses an
existing audit receipt and refuses overwriting a raw file on automatic
download. No user-owned equipment is required.

## What this audits

- Raw MATLAB v5 binary guard and original file SHA-256, printed/saved.
- **Exact** `X097_DE_time` / `X109_DE_time` field identity; does not select
  the first variable ending in `_DE_time`. Some CWRU MAT files contain
  other recording keys.
- Original sample count, source acquisition declarations, nonoverlapping
  descriptive 4096-sample RMS and crest-factor summaries, and nonfinite /
  constant-signal refusal.
- The resulting receipt names two *original recordings*, **not** the
  number of independent physical bearings. Each window is technical
  subsampling from one original source file, never its own training group.
- GitHub Actions CI includes actual official HTTPS download into an
  **ephemeral runner directory** and saves only the JSON audit report
  (no third-party vibration files in the repository/artifact). If the
  download fails, CI must report failure rather than silently fall back
  to synthetic data.

## Scientific rejection boundaries

- **No diagnosis training, VAL, TEST, source selection or accuracy** is
  performed in R91. A pair of CWRU recording IDs is insufficient to
  establish independent specimens, generalization, or superiority of any
  fuzzy algorithm. Consequently `measured_generalization_accuracy=null`,
  `fuzzy_vs_baseline_advantage_proven=false` and `eligible_for_stable=false`.
- A waveform SHA-256 verifies byte integrity of the *received file*, not
  its original authenticity against a publisher-signed checksum.
  Downloading official HTTPS strengthens source traceability but is not a
  laboratory certificate.
- Sampling frequency is a **published metadata declaration**; any
  purported 48-kHz original file with unknown/unmatched scientific
  provenance requires independent review.
- Actual sensor calibration/units, independent specimen IDs, and
  error-cost uncertainty are unknown.
- Do not cherry-pick favorable measurements after inspecting results.
  Any future trained comparison must preregister independent equipment
  and acquisition handling, use same source group/fold budgets and account
  for premise-gradient refit costs.

## Test command

```bash
cd source_product
python -m pip install numpy scipy pytest
PYTHONPATH=src python -m pytest -q tests/phm_design/test_r91_cwru_reality_check.py
```

Test-generated MATLAB binaries are explicitly fake. The *separate* live
download CI job is what determines whether actual official CWRU waveforms
were accessed in this research branch. Never call green synthetic tests
real measurements.

## Decision after this probe

If official acquisition succeeds and exact field keys are present,
the next experiment may study sensor preprocessing / spectral engineering
on real signals, but **must not** promise class generalization from two
recordings. A proper causal effectiveness comparison of fuzzy self-organization
will require an independent physical-specimen cohort; otherwise it remains
a lower-support descriptive comparison only.
