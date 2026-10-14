# R68 proposal (NOT frozen; NOT executed)

**Question:** Does R67's phase3 improvement in same-noise, known-onset physics response dissimilarity translate to single-stream observational discrimination with realistic noise, unknown onset and no counterfactual?

1. Freeze a permitted-observation contract first. Inputs can include time, illumination, commanded heat and two thermal channels. Explicitly prohibit true fault label, injection time, phase-arm ID as a label proxy, physics parameters, same-noise counterfactual residuals, and simulator truth.
2. Use *new* development worlds to construct a group-disjoint training set; choose nominal/fault classes and class balance prior to running. A classifier-free residual baseline (thermal command→temperature forward model learned from healthy commissioning) should be compared with a constrained supervised temporal baseline. Both must be calibrated using fresh, physically independent calibration worlds.
3. Predict on separately generated independent-noise worlds with unknown fault time; measure healthy false positives, every-fault recall, detection and mechanism+component localization, UNKNOWN rate and worst world. Freeze thresholds and code **before** opening a reserved test world.
4. Compare X/Y/C arm effects at the **world** level, not by treating 5 fault trajectories or 2 components in one world as independent. Reserve a no-excitation or legacy baseline to identify the incremental cost of controlled heating and report power/temperature bounds.
5. R67 worlds are already revealed and may only guide development. They cannot be reused as new calibration or qualification worlds. Until independent prospective evidence exists, R68 remains DEVELOPMENT_ONLY.

**Current state:** draft only. No R68 model, experiments, certification, or numerical results.
