# PHM AutoDesign R68 — Mixed task candidate backends (development only)

Version: `0.8.0.dev9`; parent R67 `0.8.0.dev8`. The 0.7.0 Stable release remains untouched. No new physics worlds were opened and no new qualification was claimed.

## Delivered scope

The **task-aware design** API `phm_design.task_architectures.design` now selects from one explicit allowlisted candidate registry: sklearn `linear`/`extra` over `temporal`/`physical` feature vectors, plus **opt-in** PyTorch CPU `torch_cnn1d`/`torch_gru`/`torch_transformer` over `sequence` observations. Each candidate is evaluated using existing GroupKFold physical-world isolation, the same five-class ontology, the same direct/factorized/fusion architectures, OOF acceptance gate and UNKNOWN threshold. Default 12 candidates are unchanged; use `candidate_specs` to opt into neural candidates.

`sequence` consists of 48 time-interpolated samples of exact deployment sensor columns, excluding time itself. No label, world ID, injected-fault truth, onset or counterfactual is passed as a predictor. Standardization parameters are fit only on the training fold; predicted probabilities follow the same estimator contract. Neural training is fixed CPU / 8 epochs / AdamW with a 200k parameter guard. Failure to meet the predeclared acceptance policy still refuses to deliver a qualified model. This is a small, unoptimized feasibility baseline.

## API example

```python
from phm_design.task_architectures import design, TaskAcceptancePolicy
model = design(
    domain='Thermal_localization', observations=design_train_observations,
    fields=exact_deployment_fields, labels=training_labels,
    world_ids=design_train_world_ids,
    roles=['design_train'] * len(training_labels),
    healthy_prefix_confirmed=[True] * len(training_labels),
    candidate_specs=[('temporal','linear','direct'),
                     ('sequence','torch_cnn1d','direct'),
                     ('sequence','torch_gru','factorized')],
)
# Raises TaskDesignRejected if none meets conservative grouped-OOF gates.
# This is not a certified deployment or online prediction API.
```

### Candidate combinations

| Feature | Learner | Backend | Diagnosis structures |
|---|---|---|---|
| `temporal`/`physical` | `linear`/`extra` | sklearn CPU | direct, factorized, fusion |
| `sequence` | `torch_cnn1d`/`torch_gru`/`torch_transformer` | PyTorch CPU | direct, factorized, fusion |

## Local validation

Run `python -m pip install -e '.[dev]'` then `pytest -q tests/phm_design/test_r68_unified_candidates.py`. Existing R67 tests can also be run. Native Basilisk tests need a compatible `bsk==2.11.0` environment; Python 3.11/3.12 × wheel/sdist and full dependency closure are not qualified in this dev cycle. Hash manifests and test results appear in the R68 handoff.

## Explicit limits

- The separate `open_design` general-purpose search engine is **not yet wired to this registry**. This is the first common backend selection contract in the *task-aware* designer, not an end-to-end platform unification.
- A heterogenous per-head search space, arbitrary ontology, cascaded risk gates, continuous prognostics, time-to-detection and RUL are not supported by this bridge.
- No CUDA or mixed precision training, Mamba / S4 / state-space models or GPU memory planning.
- `TorchTaskEstimator` reuses the project's existing CNN/GRU/Transformer components with a deliberately limited training recipe; no paper-reproduction claim.
- Synthetic fixture integration proves executable interoperability, not actual Thermal localization improvements. 0.8.0.dev9 remains DEVELOPMENT_NOT_QUALIFIED.

## Follow-up

R69: task DAG and per-head model/metric contracts. R70: opt-in CUDA training and resource/ABI guards. R71: controlled architecture family expansion including state-space models. R72: prospective independent-world calibration and end-to-end gate evaluation. The older R68 physical observation proposal from R67 is retained as a **deferred** study, not presented as run or superseded evidence.
