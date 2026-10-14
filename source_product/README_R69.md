# PHM AutoDesign 0.8.0.dev10 — R69 Independent PHM task DAG

**Development research only.** R68 `0.8.0.dev9` is the frozen predecessor. The qualified `0.7.0` Stable bundle, old evaluation data, model hashes and qualification certificate are not modified. No new physical-world experiments, calibrations or sealed-world evaluations are performed in R69.

## Implemented

New public Python API `phm_design.task_graph.design_task_graph` models the explicit offline task graph:

`observations → fault detection → component localization → mechanism classification → legal ontology projection / UNKNOWN`.

Each head has its **own** candidate `(feature, learner)` list and can choose distinct sklearn feature models or opt-in PyTorch CPU CNN1D / GRU / Transformer via the pre-existing R68 registry. `TaskGraphGate` declares stage-specific confidence cutoffs, and `TaskGraphPolicy` separately verifies:

- detection recall on true faults and healthy false-alarm rate;
- localization accuracy conditional on *correctly detected* fault examples;
- mechanism classification accuracy conditional on *correctly detected and localized* fault examples;
- full-pipeline accuracy, per-fault worst recall, per-world worst accuracy and UNKNOWN fraction.

Model selection is nested **inside outer GroupKFold** by physical world: each outer fold chooses the three heads on its training worlds using inner 3-fold group OOF; then fits those heads without viewing outer-held-out worlds. Fixed (predeclared) cascade confidence gates are scored only on outer-held-out predictions. A fresh final selection uses all `design_train` worlds if and only if the graph meets the predeclared policy. There is no train/eval/physical-world relabeling. Truth labels for conditional supervised training and metrics do not enter predictors.

```python
from phm_design.task_graph import design_task_graph, TaskGraphPolicy, TaskGraphGate
from phm_design.platform_features import FIELDS
from phm_design.task_architectures import TaskDesignRejected

try:
    graph = design_task_graph(
        domain="Thermal_localization",
        observations=design_train_observations,
        fields=FIELDS["Thermal_localization"],
        labels=design_train_labels,
        world_ids=design_train_world_ids,
        roles=["design_train"] * len(design_train_labels),
        healthy_prefix_confirmed=[True] * len(design_train_labels),
        candidate_specs={
            "detection": [("temporal", "linear"), ("sequence", "torch_cnn1d")],
            "localization": [("physical", "extra")],
            "mechanism": [("sequence", "torch_gru"), ("temporal", "linear")],
        },
        gate=TaskGraphGate(detector_min_confidence=0.55,
                           localization_min_confidence=0.60,
                           mechanism_min_confidence=0.60),
        acceptance_policy=TaskGraphPolicy(),
    )
except TaskDesignRejected as exc:
    # Inspect exc.audit['nested_outer_oof_metrics'] / ['acceptance'].
    # No fit-to-deliver artifact is returned for a rejected graph.
    raise

result = graph.diagnose(deployment_observation,
                        fields=FIELDS['Thermal_localization'],
                        healthy_prefix_confirmed=True)
assert result['qualified'] is False
```

## Reproduce

`PYTHONPATH=src:tests/phm_design OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 pytest -q tests/phm_design/test_r69_task_graph.py`

R69's integration uses synthetic training fixtures, not a new Thermal diagnosis benchmark. The nested OOF scores are **design_train screening estimates, not an independent held-out qualification**. The schema is still the strict R62 5-class, 2-component, 2-mechanism ontology; the input remains an entire offline telemetry sequence with a confirmed healthy prefix. Confidence gates are fixed by the user, not optimized across new evaluation data. Arbitrary task topologies, calibrated statistical risk bounds, independent per-stage hardware constraints, online early detection, RUL and CUDA are still outside this scope. The earlier `task_architectures.design()` is preserved as a compatibility interface; `open_design` is still separate.

The 0.8 developer wheel must not inherit Stable qualification: all R69 diagnoses are `qualified=False`. Wheel/sdist Python 3.11/3.12 with full Basilisk dependency closure remains a release gate until explicitly executed.
