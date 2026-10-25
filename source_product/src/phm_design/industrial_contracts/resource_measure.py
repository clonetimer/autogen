"""R99 end-to-end resource receipt for ONE frozen R98 SOURCE study.

Unlike prior model-fit-only times, this measurement wraps physical envelope /
FFT extraction, R96 fold-local normalization and native-provider fit/predict
together. Uses perf_counter and process_time, which are platform-compatible.

It is NOT a FLOP meter, GPU profiler, peak native allocation monitor,
process-enforced memory budget or hardware-matched benchmark. The output
intentionally never claims full resource fairness or external accuracy.
"""
from __future__ import annotations

import os
import time
import tracemalloc

from .physical_graph_design import evaluate_physical_graph_study

SCHEMA = "phm.autogen.r99-end-to-end-resource.v1"


def measure_source_graph_end_to_end(study, equipment, source_train, trial,
                                    raw_records, *, provider=None):
    """Instrument full train-only run; provenance is unchanged.

    Peak tracemalloc bytes cover traced Python allocations only, not NumPy
    allocations in native extensions or device GPU memory.
    """
    was_tracking = tracemalloc.is_tracing()
    if not was_tracking:
        tracemalloc.start()
    _, peak_before = tracemalloc.get_traced_memory()
    start_wall = time.perf_counter()
    start_cpu = time.process_time()
    try:
        report = evaluate_physical_graph_study(
            study, equipment, source_train, trial, raw_records,
            provider=provider)
        elapsed_wall = max(0., time.perf_counter()-start_wall)
        elapsed_cpu = max(0., time.process_time()-start_cpu)
        _, peak_after = tracemalloc.get_traced_memory()
    finally:
        if not was_tracking:
            tracemalloc.stop()
    if report["study_sha256"] != study["study_sha256"]:
        raise ValueError("R99_RESOURCE_STUDY_RECEIPT_IDENTITY_CHANGED")
    return {
        "schema": SCHEMA,
        "study_sha256": study["study_sha256"],
        "source_report": report,
        "measured_resources": {
            "complete_feature_pipeline_and_model_fit_wall_seconds": elapsed_wall,
            "complete_feature_pipeline_and_model_fit_process_cpu_seconds": elapsed_cpu,
            "traced_python_allocation_peak_bytes_observed": max(
                peak_after, peak_before),
            "tracemalloc_started_by_runner": not was_tracking,
            "all_candidate_graphs_in_one_source_study_included": True,
            "envelope_fft_and_fold_preprocessing_included_in_timing": True,
            "platform_cpu_count_reported": os.cpu_count(),
            "hardware_environment_fingerprint_authenticated": False,
            "numpy_c_extension_peak_allocation_tracked": False,
            "gpu_memory_measured": False,
            "cpu_gpu_flops_measured": False,
            "provider_subprocess_resources_measured": False,
            "hard_resource_limits_enforced": False,
            "same_hardware_and_software_comparison_proven": False,
            "external_independent_validation_performed": False,
        },
        "independent_generalization_proven": False,
        "qualified_for_industrial_deployment": False,
    }
