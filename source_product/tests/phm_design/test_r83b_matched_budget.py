"""R83b matched-fit protocol: synthetic contracts, no Basilisk claims."""
from types import SimpleNamespace
import copy

import pytest

from phm_design.industrial.design_engine import IndustrialDesignRejected
from phm_design.industrial.r83_matched_budget import (
    run_matched_source_train_study, run_native_rw_matched_study,
)
from phm_design.industrial.taskspec import TaskSpecRejected
from test_r81a_unified_design_engine import task, synthetic_records
from test_r82_rw_study import rw_fixture


def _spec():
    result = task("industrial_bearing_classification_demo.json",
                  compare=("tcn",))
    result["design"]["candidate_max"] = 3
    result["design"]["budget"]["max_fits"] = 11
    result["design"]["budget"]["max_cpu_seconds"] = 120
    result["design"]["budget"]["max_wall_seconds"] = 180
    return result


def _metrics():
    return {
        "accuracy": .85,
        "worst_environment_class_recall": .5,
        "healthy_false_alarm": .10,
    }


def _fixed_model(*args, **kwargs):
    spec = args[0]
    assert spec["design"]["compare"] == ["tcn", "gru", "tree"]
    assert spec["design"]["candidate_max"] == 3
    assert spec["design"]["budget"]["max_fits"] == 10
    assert spec["design"]["budget"]["max_wall_seconds"] == 180
    return SimpleNamespace(audit={
        "decision": "TRAIN_ONLY_SELECTION_REFITTED", "model_delivered": True,
        "total_model_fits": 10,
        "attempts": [{
            "trial": {"backbone": name, "objective": "ERM"},
            "metrics": _metrics(), "model_fits": 3,
            "acceptance": {"eligible": True},
        } for name in ("tcn", "gru", "tree")],
    })


def _evolving_model(*args, **kwargs):
    spec = args[0]
    assert spec["design"]["compare"] == ["tcn"]
    assert spec["design"]["candidate_max"] == 3
    assert spec["design"]["budget"]["max_fits"] == 10
    assert spec["design"]["budget"]["max_wall_seconds"] == 180
    policy = kwargs["policy"]
    assert (policy.max_candidates, policy.epochs) == (3, 8)
    return SimpleNamespace(audit={
        "decision": "RESEARCH_TRAIN_OOF_ELIGIBLE_NOT_QUALIFIED",
        "model_delivered": True, "charged_fits": 10,
        "attempts": [{
            "genome": {"width": 8, "depth": 1, "kernel": 3,
                       "pooling": "mean", "residual": False},
            "genome_sha256": f"candidate_{index}",
            "metrics": _metrics(), "eligible": True, "parameter_count": 300,
        } for index in range(3)],
    })


def test_both_arms_equal_fit_cap_and_identical_train_records(monkeypatch):
    import phm_design.industrial.r83_matched_budget as benchmark
    received = []
    def fixed(spec, records, **kwargs):
        received.append(("fixed", records))
        return _fixed_model(spec, records, **kwargs)
    def evolving(spec, records, **kwargs):
        received.append(("evolved", records))
        return _evolving_model(spec, records, **kwargs)
    monkeypatch.setattr(benchmark, "run_design_guarded", fixed)
    monkeypatch.setattr(benchmark, "run_self_organizing_design_guarded", evolving)
    spec = _spec()
    records = synthetic_records(spec, environment_count=3)
    out = run_matched_source_train_study(spec, records)
    assert out["schema"].endswith("matched-source-train-study.v1")
    assert out["candidate_budget_per_arm"] == 3
    assert out["fit_budget_per_arm"] == 10
    assert out["source_environment_count"] == 3
    assert out["source_data_role"] == "TRAIN_ONLY"
    assert out["source_train_test_or_final_accessed"] is False
    assert out["independent_qualification"] is False
    assert out["model_delivered_to_deployment"] is False
    assert out["fairness"]["equal_flops"] is False
    assert out["fairness"]["equal_estimator_fit_cap"] is True
    assert received == [("fixed", records), ("evolved", records)]
    assert out["arms"]["fixed_portfolio"]["charged_fits"] == 10
    assert out["arms"]["self_organizing"]["charged_fits"] == 10
    assert out["arms"]["self_organizing"]["best_eligible_source_train_candidate"]
    assert out["arms"]["fixed_portfolio"]["best_eligible_source_train_candidate"]
    assert spec["design"]["compare"] == ["tcn"]  # original immutable to search


def test_refusal_of_one_arm_does_not_magically_qualify_other(monkeypatch):
    import phm_design.industrial.r83_matched_budget as benchmark
    def reject(*args, **kwargs):
        raise IndustrialDesignRejected({
            "decision": "NO_ELIGIBLE_DESIGN", "model_delivered": False,
            "fold_model_fits": 9,
            "attempts": [{"trial": {"backbone": "tree"}, "metrics": _metrics(),
                          "acceptance": {"eligible": False}, "model_fits": 3}],
        })
    monkeypatch.setattr(benchmark, "run_design_guarded", reject)
    monkeypatch.setattr(benchmark, "run_self_organizing_design_guarded", _evolving_model)
    spec = _spec()
    out = run_matched_source_train_study(
        spec, synthetic_records(spec, environment_count=3))
    fixed = out["arms"]["fixed_portfolio"]
    assert fixed["decision"] == "NO_ELIGIBLE_DESIGN"
    assert fixed["model_delivered"] is False
    assert fixed["best_eligible_source_train_candidate"] is None
    assert out["arms"]["self_organizing"]["model_delivered"] is True
    assert out["independent_qualification"] is False


def test_invalid_budget_non_train_or_search_grammar_refused_before_fit(monkeypatch):
    import phm_design.industrial.r83_matched_budget as benchmark
    def forbid(*args, **kwargs):
        raise AssertionError("TRAINER_MUST_NOT_RUN")
    monkeypatch.setattr(benchmark, "run_design_guarded", forbid)
    monkeypatch.setattr(benchmark, "run_self_organizing_design_guarded", forbid)
    spec = _spec()
    rows = synthetic_records(spec, environment_count=3)
    too_small = copy.deepcopy(spec)
    too_small["design"]["budget"]["max_fits"] = 9
    with pytest.raises(TaskSpecRejected, match="FIT_BUDGET_TOO_SMALL"):
        run_matched_source_train_study(too_small, rows)
    bad_role = copy.deepcopy(rows)
    bad_role[0]["metadata"]["role"] = "FINAL_OOD"
    with pytest.raises(TaskSpecRejected, match="DESIGN_WORKER_PROTECTED_ROLE"):
        run_matched_source_train_study(spec, bad_role)
    bad_signal = copy.deepcopy(rows)
    bad_signal[0]["measurements"][0]["injected_truth"] = 1
    with pytest.raises(TaskSpecRejected, match="UNDECLARED_OR_PROTECTED_OBSERVATION"):
        run_matched_source_train_study(spec, bad_signal)
    duplicate = copy.deepcopy(rows)
    duplicate[1]["measurements"] = copy.deepcopy(duplicate[0]["measurements"])
    with pytest.raises(TaskSpecRejected, match="DUPLICATE_TRAJECTORY"):
        run_matched_source_train_study(spec, duplicate)
    altered = copy.deepcopy(spec)
    altered["design"]["objectives"] = ["GroupDRO"]
    with pytest.raises(TaskSpecRejected, match="PLAIN_ERM_DIRECT"):
        run_matched_source_train_study(altered, rows)


def test_r82_native_staging_and_binding_before_benchmark(tmp_path):
    args = rw_fixture(tmp_path)
    out = run_native_rw_matched_study(
        **args, output=tmp_path / "native_train_study", validate_only=True)
    assert out["status"] == "TRAIN_SOURCE_ACCEPTED"
    assert out["model_delivered_to_deployment"] is False
    assert out["independent_qualification"] is False
    assert out["validation_telemetry_accessed"] is False
    assert out["source_intake"]["records"] == 24
    assert not (tmp_path / "native_train_study" / "comparison.json").exists()
