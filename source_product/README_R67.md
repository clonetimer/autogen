# PHM AutoDesign R67 — Equal-energy crossed-phase native Thermal pilot

**Version:** 0.8.0.dev8 (development only). **Baseline:** R66 / 0.8.0.dev7. **0.7.0 Stable unchanged.**

## Research decision

The original R66 finding that the A component decreased and the B component increased its oracle mechanism-separation residual could be confounded by different energy distribution, phase and thermal parameters. R67 crossed the timing phase for A/B and held total commanded energy **equal in each 7,200 s half** to isolate timing within one pre-registered schedule family. Neither model training nor qualification was attempted.

## Frozen experimental design and leakage boundary

- Four new development worlds (`r67-thermal-phase-dev-001` to `004`), three profiles X/Y/C, fault labels 0–4, and two noise constructions: **4×3×5×2 = 120 native SensorThermal trajectories**.
- Both phase arms use the same six 2,400 s plateaus with normalized levels `[0.5, 1.5, 1.0, 1.5, 0.5, 1.0]`. Phase X uses shifts `(A=0,B=3)`, phase Y `(A=3,B=0)`. C applies constant base heating `(A=1,B=1)`.
- Every profile, every component and **both** half-horizons integrate to 7,200 watt-seconds *per watt of base heating*, independently of the fault. Equal half-horizon energy was checked in the command planner, in every written time series, and by the evidence verifier. It does not imply equal instantaneous temperature or identical dynamic state.
- Basilisk 2.11.0 native `SensorThermal` integrates temperature. Full horizon 14,400 s, solver 2 s, sample 20 s, fault onset 7,200 s. No pure-Python thermal surrogate was substituted.
- Pair-oracle CSVs contain the same noise realization for research counterfactual subtraction. Independently-noised deployment-shaped CSVs retain only `time_s, illumination, heat_a_command_w, heat_b_command_w, temperature_a_k, temperature_b_k`. No fault label, injection timing or counterfactual enters the deployment CSV fields. Oracle results and simulator parameters are kept separately and are **not** fit or inferred from.
- Frozen protocol `research/R67_DEV_PROTOCOL.json`; source commit and protocol SHA256 are in evidence `R67_DEV_RESULTS.json`.

## Measured results — development only

**Primary metric:** world-paired `(phase3 scaled-response-residual − phase0 residual)`, stratified by physical component; a larger residual means the two fault-signature response vectors are less alike after optimal scalar fit. The statistic comes from research oracle paired curves, **not from a diagnostic predictor**.

| Component | World 001 | World 002 | World 003 | World 004 | Median | Consistent sign | Predefined 0.02/3-of-4 threshold |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| A | +0.074706 | +0.046240 | +0.049792 | +0.034347 | **+0.048016** | 4/4 positive | Met |
| B | +0.053445 | +0.037926 | +0.070541 | +0.075480 | **+0.061993** | 4/4 positive | Met |

The constant-energy control shows intermediate residuals (median A 0.073049, B 0.068423). The positive effects for **both** components mean that crossed phase affects this oracle measure under this schedule but does **not** support the stronger hypothesis that R66's opposite-direction responses were explained by component phase alone. `both_components_material_opposite_sign=false`; retrospective causal attribution is not allowed.

Four physical worlds, not eight component observations, define the repeated experimental units. The thresholds are exploratory development criteria, not formal confidence intervals, generalized claims, classification accuracy, new qualification, or hardware/flight certification.

## Validation and reproducibility

```bash
cd source_product
PYTHONPATH=src:. python research/verify_r67_dev_evidence.py --root ../evidence/r67_native_dev
# Development re-run only, new non-existing path:
PYTHONPATH=src:. python research/run_r67_dev_experiment.py --output /new/development-only/path
```

- Evidence verifier checks all 124 SHA256-listed files, 120 CSV trajectories, chronology, bounds, **half-horizon energy** and recomputes the original mechanism residuals from paired physical CSVs.
- Frozen existing 0.7 qualification is not inherited: 0.8 reports `DEVELOPMENT_NOT_QUALIFIED`.
- See `R67_TEST_STATUS.json` for the actual software regression results and environmental limitations. CPython 3.11/3.12 × wheel/sdist 4-grid certification is **not** established.

## Design significance and next step

R65 (many short plateaus) and R66 (longer plateaus but changed average energy) were insufficient to isolate a mechanism. R67 supplies evidence that **timing distribution**, even at equal overall energy, materially influences *paired physics-response dissimilarity*. Yet the two thermal nodes remain uncoupled and their thermal response may not yield deployable discrimination under independent noise and unknown fault time.

R68 should first evaluate observability from single independently noised measurement streams with a no-truth feature contract, and use separated, predeclared development training/calibration and independent assessment worlds before suggesting diagnosis improvement. No prior development world may be relabeled as a new blind test.
