"""R81c combinatorial safety contracts, stdlib-only compile coverage."""
import copy

import pytest

from phm_design.industrial.taskspec import TaskSpecRejected, compile_task_spec
from test_r81a_unified_design_engine import task
from test_r81c_available_late_fusion import available_spec
from test_r81c_adaptive_feature_search import spec as adaptive_spec


def test_static_composed_program_and_train_calibration_can_coexist():
    s = task("industrial_bearing_classification_demo.json", compare=("linear",))
    s["design"]["feature_programs"] = [
        {"atoms": ["temporal_bins", "difference_kernels"],
         "composition": "concat"}]
    s["design"]["calibration"] = {
        "method": "temperature_selective", "max_selective_error": .2,
        "minimum_coverage": .5, "min_calibration_samples": 12}
    plan = compile_task_spec(s)
    assert len(plan["candidate_grid"]) == 1
    assert plan["calibration_policy"] is not None
    assert plan["candidate_grid"][0]["representation"] == "feature_program"


def test_adaptive_search_calibration_is_explicitly_same_train_budget():
    s = adaptive_spec()
    s["design"]["calibration"] = {
        "method": "temperature_selective", "max_selective_error": .2,
        "minimum_coverage": .5, "min_calibration_samples": 12}
    plan = compile_task_spec(s)
    assert plan["adaptive_feature_search"]["max_programs"] == 3
    assert plan["calibration_policy"] is not None
    assert plan["budget"]["max_fits"] == 80
    assert plan["qualified"] is False


def test_availability_cannot_claim_train_calibration_on_unsupported_probability():
    s = available_spec()
    s["design"]["calibration"] = {
        "method": "temperature_selective", "max_selective_error": .2,
        "minimum_coverage": .5, "min_calibration_samples": 12}
    with pytest.raises(TaskSpecRejected, match="UNSUPPORTED_AVAILABILITY_CALIBRATION"):
        compile_task_spec(s)


def test_adaptive_cannot_mix_with_unmeasured_or_available_static_programs():
    s = available_spec()
    s["design"]["adaptive_feature_search"] = {
        "atoms": ["temporal_bins", "random_kernels", "difference_kernels"],
        "max_programs": 3}
    s["design"]["candidate_max"] = 3
    with pytest.raises(TaskSpecRejected, match="ADAPTIVE_SEARCH_REQUIRES_SINGLE_CLASSICAL_TEMPLATE"):
        compile_task_spec(s)


def test_wall_deadline_does_not_implicitly_enable_independent_qualification():
    s = adaptive_spec()
    s["design"]["budget"]["max_wall_seconds"] = 15
    plan = compile_task_spec(s)
    assert plan["budget"]["max_wall_seconds"] == 15
    assert plan["qualified"] is False
    assert plan["independent_qualification"] is False
    assert plan["target_data_used"] is False
