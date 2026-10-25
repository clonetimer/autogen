"""R84 matched-ablation contracts: source-only, synthetic; no qualification."""
from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from phm_design.industrial.r83_matched_budget import run_matched_source_train_study
from phm_design.industrial.taskspec import TaskSpecRejected
from test_r81a_unified_design_engine import synthetic_records
from test_r83b_matched_budget import _spec, _fixed_model, _evolving_model


def test_four_arm_budget_and_fold_equivalence_with_pareto_opt_in(monkeypatch):
    import phm_design.industrial.r83_matched_budget as benchmark
    calls = []
    def fixed(spec, records, **kwargs):
        calls.append(("fixed", id(records), spec["design"]["budget"]["max_fits"],
                      kwargs.get("selection_rule")))
        return _fixed_model(spec, records, **kwargs)
    def neural(spec, records, **kwargs):
        calls.append((kwargs["proposal_mode"], id(records),
                      spec["design"]["budget"]["max_fits"],
                      kwargs["seed_schedule"]))
        return _evolving_model(spec, records, **kwargs)

    monkeypatch.setattr(benchmark, "run_design_guarded", fixed)
    monkeypatch.setattr(benchmark, "run_self_organizing_design_guarded", neural)
    spec = _spec()
    rows = synthetic_records(spec, environment_count=3)
    result = run_matched_source_train_study(
        spec, rows, include_pareto=True)
    assert [x[0] for x in calls] == [
        "fixed", "predeclared_static", "feedback", "feedback_pareto"]
    assert len({x[1] for x in calls}) == 1
    assert [x[2] for x in calls] == [10, 10, 10, 10]
    assert [x[3] for x in calls] == [
        "r83_matched_train_loss", "r83_matched_train",
        "r83_matched_train", "r83_matched_train"]
    assert result["complete_four_arm_pareto_comparison"]
    assert result["complete_three_arm_comparison"]
    assert result["fairness_gate"]["source_train_comparison_allowed"]
    assert result["fairness_gate"]["r84_pareto_vs_static_neural_source_comparison"]
    assert result["r84_pareto_ablation_enabled"]
    assert all(a["charged_fits"] == 10 for a in result["arms"].values())
    assert not result["fairness"]["equal_flops"]
    assert not result["independent_qualification"]
    assert not result["fairness_gate"]["independent_generalization_claim_allowed"]


def test_legacy_three_arms_still_identical_without_opt_in(monkeypatch):
    import phm_design.industrial.r83_matched_budget as benchmark
    seen = []
    monkeypatch.setattr(benchmark, "run_design_guarded", _fixed_model)
    def neural(spec, records, **kwargs):
        seen.append(kwargs["proposal_mode"])
        return _evolving_model(spec, records, **kwargs)
    monkeypatch.setattr(benchmark, "run_self_organizing_design_guarded", neural)
    spec = _spec()
    study = run_matched_source_train_study(
        spec, synthetic_records(spec, environment_count=3))
    assert set(study["arms"]) == {"fixed_portfolio", "static_topology",
                                  "self_organizing"}
    assert seen == ["predeclared_static", "feedback"]
    assert study["complete_three_arm_comparison"]
    assert not study["complete_four_arm_pareto_comparison"]
    assert not study["r84_pareto_ablation_enabled"]


def test_reject_final_or_malformed_option_without_model_training(monkeypatch):
    import phm_design.industrial.r83_matched_budget as benchmark
    def impossible(*args, **kwargs):
        raise AssertionError("TRAINING_SHOULD_NOT_START")
    monkeypatch.setattr(benchmark, "run_design_guarded", impossible)
    monkeypatch.setattr(benchmark, "run_self_organizing_design_guarded", impossible)
    spec = _spec()
    data = synthetic_records(spec, environment_count=3)
    protected = copy.deepcopy(data)
    protected[0]["metadata"]["role"] = "FINAL_OOD"
    with pytest.raises(TaskSpecRejected, match="DESIGN_WORKER_PROTECTED_ROLE"):
        run_matched_source_train_study(spec, protected, include_pareto=True)
    with pytest.raises(ValueError, match="INVALID_R84_PARETO_ABLATION_SWITCH"):
        run_matched_source_train_study(spec, data, include_pareto=1)
