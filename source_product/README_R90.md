# R90 — preregistered Paderborn single-unseen-condition exploratory diagnosis

**Status: research Draft. Does not modify or qualify R88's stricter two-VALIDATION-condition protocol.**

## Why a new protocol?

The [Paderborn KAt Bearing DataCenter](https://mb.uni-paderborn.de/en/kat/research/bearing-datacenter/data-sets-and-download)
documents only **6 physically distinct undamaged bearings**, four
measurement conditions and 26 damaged bearings. R88 requires at least 7
disjoint healthy bearings for 3 source TRAIN environments and two separate
two-healthy-specimen VALIDATION conditions. That requirement must not be
weakened retroactively.

R90 proposes **one prospectively frozen unseen operating condition** with
3 different heldout healthy and 3 different heldout damaged bearings. This
is a useful but materially weaker exploratory single-condition transfer
study. It is NOT multi-condition, multi-machine or flight qualification.
It must not be used to claim sensor-invariant fuzzy-neural superiority.

## Locked cohort assignment — no outcome-driven changes

Records are expected under `<raw-root>/<bearing-code>/<condition>_<bearing-code>_<recording>.mat`.
Exactly recordings **1 and 2** per specimen are permitted (24 files total).

| Stage | Operating condition | Healthy bearing | Fault bearing(s) |
|---|---|---|---|
| TRAIN | N15_M07_F10 | K001 | KI03 |
| TRAIN | N09_M07_F10 | K002 | KA01 |
| TRAIN | N15_M01_F10 | K003 | KI05 |
| **VALIDATION** | **N15_M07_F04** | **K004, K005, K006** | **KA03, KA04, KI04** |

Different fault origins/mechanisms may be present in the source and
validation faults. This is a **scientific confound**: performance changes
cannot be attributed solely to speed/load change or the fuzzy mechanism.
No retrospectively chosen sample substitutions are authorized. Any
different assignment requires a new named prospective protocol and tests.

Paderborn source is [CC BY-NC 4.0](https://mb.uni-paderborn.de/en/kat/research/bearing-datacenter).
Original archives are about 160 MB per physical bearing. **No raw
Paderborn files are uploaded or committed**, no original recordings have
been read in this PR. The isolated GitHub CI builds entirely synthetic
MATLAB files with the same *named signal* API and approximately four-second
sample length.

## Real-data execution, after obtaining the official raw archives

Use `source_product/examples/industrial/paderborn_r90_binary_pilot.json`.
It intentionally restricts diagnosis to **HEALTHY vs FAULT** and uses only
the explicitly named `vibration_1` source channel, from which two fixed
sensor-derived features (RMS and peak-absolute vibration per bin) are computed.
The second model feature is **not** a second sensor.

```python
import json
from pathlib import Path
from phm_design.industrial.r86_neuro_fuzzy import FuzzyRulePolicy
from phm_design.industrial.r90_paderborn_pilot import (
    freeze_paderborn_source, evaluate_paderborn_validation)

spec = json.loads(Path(
    "examples/industrial/paderborn_r90_binary_pilot.json"
).read_text(encoding="utf-8"))

# Stage 1: TRAIN ONLY. Validation raw files are NOT opened here.
train = freeze_paderborn_source(
    spec, "D:/datasets/Paderborn", "D:/research/r90-frozen",
    policy=FuzzyRulePolicy(max_rules=2, premise_steps=2),
)
# Independently record the exact manifest SHA and freeze the entire output.

# Stage 2: review-only single-condition VALIDATION. No fitting/selection.
report = evaluate_paderborn_validation(
    "D:/research/r90-frozen", "D:/datasets/Paderborn",
    expected_manifest_sha256=train["frozen"]["manifest_sha256"],
)
print(report["results"])
```

### Signal verification limits

- SciPy MAT reader **requires exactly one channel named `vibration_1`**
  under MATLAB `Name` / `Data` fields. It never assumes a fixed array
  index or replaces a missing sensor with a synthetic signal.
- If authentic MAT structure differs (MAT 7.3/HDF5, nested channel
  wrappers, time axis), the reader fails closed; native signal parsing
  compatibility must then be established by an actual recording-level
  review. CI tests are **not** proof of authentic-file compatibility.
- Raw sample length must be 200,000–300,000, reflecting approximately
  four seconds at the nominal 64 kHz; the first 256 equally sized bins
  become **RMS and peak-absolute vibration amplitudes**. The existing
sequence encoder requires two numerical channels besides the time axis. This deliberately discards phase
  and most frequency detail. Sensor units are **unverified**; the time
  axis uses the *nominal* sampling rate, not a measured timestamp.
- Each raw MAT file gets SHA-256 logged. Records are pinned to two
  original recordings per specimen and split by **physical bearing**,
  not window. No raw vibration is committed.
- Any change to channel, feature representation, sample selection,
  source conditions, specimen IDs, labels or thresholds must define a
  **new preregistered protocol**, not silently mutate this experiment.

### Interpretation / rejection

The read-only reviewer loads hash-checked, non-pickle frozen weights and
reports five predeclared arms: linear, fixed Gaussian TSK, adaptive Gaussian
TSK, and both learned-premise versions. Reports overall accuracy,
worst-condition/class recall, healthy false alarm and group-paired
bootstrap intervals using **six distinct heldout bearings**, not twelve
independent recordings. It cannot enforce single-use validation in a
mutable local environment and cannot authenticate original archive
provenance independently.

Source LOEO is TRAIN OOF only. Heldout results are single-condition
exploratory evidence; with just 3 heldout specimens/class, uncertainty
will be large. The result ALWAYS has:
`multi_condition_external_generalization_proven=false`,
`neuro_fuzzy_improvement_proven=false`, `eligible_for_stable=false`.

### CI and proof status

```shell
cd source_product
python -m pip install numpy scipy scikit-learn pytest
PYTHONPATH=src python -m pytest -q tests/phm_design/test_r90_paderborn_pilot.py
```

CI deliberately synthesizes MAT bytes and protects protocol invariants.
**No real Paderborn sensor data or measured neuro-fuzzy benefit has yet been
produced.** The source archives must be independently downloaded and
reviewed before true real-machine measurements may be claimed. The
R89 strict 7-healthy rejection and R83f/R83g negative Basilisk experiments
remain unchanged.


## Windows PowerShell / Linux CLI for non-developers

From the source_product directory, install the scientific dependencies
(numpy, scipy, scikit-learn). Before invoking, set PYTHONPATH to the src
directory (PowerShell: $env:PYTHONPATH="src"; Bash: export PYTHONPATH=src).

Stage 1, read TRAIN recordings only and produce new-only audit outputs:

    python -m phm_design.industrial.r90_cli freeze --raw-root "D:/datasets/Paderborn" --frozen-dir "D:/research/r90-frozen" --task-spec "examples/industrial/paderborn_r90_binary_pilot.json" --receipt "D:/research/r90-train-receipt.json" --max-rules 2 --premise-steps 2

Archive the train receipt and its frozen.manifest_sha256 field before any
heldout reading. Stage 2 runs only once for the study:

    python -m phm_design.industrial.r90_cli review --raw-root "D:/datasets/Paderborn" --frozen-dir "D:/research/r90-frozen" --manifest-sha "<exact sha256 from train receipt>" --receipt "D:/research/r90-validation-receipt.json"

The CLI refuses to overwrite a previous receipt, and wrong hashes refuse
the VALIDATION read. This does NOT enforce a globally one-time blind:
the user can still manually rerun the command under a new output name,
so independent review must govern the original experiment. The CLI does
not download original files or change scientific eligibility.
