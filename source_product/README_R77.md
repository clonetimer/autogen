# R77 — Neural Domain-Generalization AutoDesign (research candidate)

Version `0.8.0.dev16`. The model **learns neural features from deployable telemetry** and is **not** a rule classifier or an independently qualified PHM model.

- Candidate families: standard ERM CNN/GRU/Transformer and new residual dilated TCN; corresponding GroupDRO variants (`dg_cnn1d`, `dg_gru`, `dg_transformer`, `dg_tcn`).
- Training: environment/class-balanced minibatches, adaptive group loss weighting, mild gain/offset/noise augmentation and consistency regularization, finite epochs and weight decay. **Group IDs and fault labels are used solely in current TRAIN fold for loss computation, never as inference features.**
- Selection: the existing nested environment-exclusive task graph searches both ERM and DG candidates with identical outer environment holdouts and predeclared acceptance rules. Domain-aware DG search requires verified TRAIN-only physical lineage and whole-environment outer/inner splits.
- A failure to achieve required worst-environment / weak-fault recall is a legitimate **reject**, not permission to inspect FINAL worlds and tune again.
- CPU development pathway only; CUDA training and any SSM/Mamba architecture remain future work. GroupDRO can underperform ERM; compare both before claiming improvements.
- No final, R71-revealed, or qualification data used in R77 synthetic tests. No claim of improved satellite diagnostic accuracy.

Test: `PYTHONPATH=src pytest -q tests/phm_design/test_r77_neural_domain_generalization.py tests/phm_design/test_r75_environment_exclusive_search.py tests/phm_design/test_r69_task_graph.py`

Research rationale: Sagawa et al., *Distributionally Robust Neural Networks for Group Shifts*, ICLR 2020. This project is an adaptation and does not reproduce the cited experiments.
