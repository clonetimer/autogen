"""R85 SOURCE-TRAIN ablation bookkeeping only; no blind/FINAL evidence."""
from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from phm_design.industrial.design_engine import IndustrialDesignRejected
from phm_design.industrial.r85_source_ablation import run_r85_source_ablation
from phm_design.industrial.resource_guard import run_design_guarded
from phm_design.industrial.taskspec import TaskSpecRejected
from test_r81a_unified_design_engine import synthetic_records
from test_r85_family_integration import r85_task


def spec_and_data():
    spec = r85_task()
    spec["design"]["budget"]["max_wall_seconds"] = 180
    data = synthetic_records(spec, environment_count=3)
    return spec, data


def mock_audit(spec, mode):
    from phm_design.industrial.taskspec import TaskSpec
    digest = TaskSpec.from_dict(spec).sha256
    return {
        "decision": "TRAIN_ONLY_SELECTION_REFITTED",
        "source_environment_count": 3,
        "source_task_spec_sha256": digest,
        "training_manifest_sha256": "a"*64,
        "training_observations_sha256": "b"*64,
        "selected_trial_sha256": "c"*64,
        "fold_model_fits": 9,
        "total_model_fits": 10,
        "attempts": [
            {"trial_index": i, "trial_sha256": ("c"*64 if i == 0 else str(i)),
             "metrics": {"accuracy": .8, "worst_environment_class_recall": .6,
                         "healthy_false_alarm": .1}}
            for i in range(3)],
        "independent_qualification": False,
    }


def test_both_arms_equal_fits_same_input_never_claim_independent(monkeypatch):
    import phm_design.industrial.r85_source_ablation as study
    spec, data = spec_and_data()
    calls = []
    def fake(spec_arg, records, **kwargs):
        calls.append((records, kwargs.copy()))
        return SimpleNamespace(audit=mock_audit(spec_arg, kwargs["family_search_mode"]))
    monkeypatch.setattr(study, "run_design_guarded", fake)
    result = run_r85_source_ablation(spec, data)
    assert [x[1]["family_search_mode"] for x in calls] == [
        "predeclared_static", "feedback"]
    assert all(x[0] is data for x in calls)
    assert all(x[1]["selection_rule"] == "r83_matched_train_loss" for x in calls)
    assert all(x[1]["family_proposals"] == 3 for x in calls)
    assert result["source_train_descriptive_comparison_allowed"]
    assert result["equal_actual_fit_count"]
    assert result["same_train_records"]
    assert result["all_declared_candidates_evaluated"]
    assert set(result["arms"]) == {"feedback", "predeclared_static"}
    assert result["arms"]["feedback"]["selected_train_oof_loss"] == pytest.approx(.265)
    assert not result["independent_generalization_claim_allowed"]
    assert not result["stable_promotion_eligible"]
    assert not result["equal_flops"]


def test_partial_or_refused_arm_must_not_produce_comparison(monkeypatch):
    import phm_design.industrial.r85_source_ablation as study
    spec, data = spec_and_data()
    def fake(spec_arg, records, **kwargs):
        audit = mock_audit(spec_arg, kwargs["family_search_mode"])
        if kwargs["family_search_mode"] == "feedback":
            audit["decision"] = "NO_ELIGIBLE_DESIGN"
            audit.pop("selected_trial_sha256")
            audit.pop("total_model_fits")
            raise IndustrialDesignRejected(audit)
        return SimpleNamespace(audit=audit)
    monkeypatch.setattr(study, "run_design_guarded", fake)
    result = run_r85_source_ablation(spec, data)
    assert not result["source_train_descriptive_comparison_allowed"]
    assert not result["all_declared_candidates_evaluated"]
    assert not result["equal_actual_fit_count"]
    assert not result["arms"]["feedback"]["model_delivered"]


def test_preflight_final_role_blocks_without_spawning(monkeypatch):
    import phm_design.industrial.r85_source_ablation as study
    spec, data = spec_and_data()
    def forbidden(*a, **k):
        raise AssertionError("INVALID_DATA_MUST_NOT_REACH_WORKER")
    monkeypatch.setattr(study, "run_design_guarded", forbidden)
    protected = copy.deepcopy(data)
    protected[0]["metadata"]["role"] = "FINAL_OOD"
    with pytest.raises(TaskSpecRejected, match="DESIGN_WORKER_PROTECTED_ROLE"):
        run_r85_source_ablation(spec, protected)
    poisoned = copy.deepcopy(data)
    poisoned[0]["measurements"][0]["fault_label"] = 1
    with pytest.raises(TaskSpecRejected, match="UNDECLARED_OR_PROTECTED_OBSERVATION"):
        run_r85_source_ablation(spec, poisoned)


def test_requires_hard_wall_and_proposal_cap(monkeypatch):
    import phm_design.industrial.r85_source_ablation as study
    spec, data = spec_and_data()
    def forbidden(*a, **k):
        raise AssertionError("BAD_POLICY_MUST_NOT_START")
    monkeypatch.setattr(study, "run_design_guarded", forbidden)
    del spec["design"]["budget"]["max_wall_seconds"]
    with pytest.raises(TaskSpecRejected, match="R85_ABLATION_REQUIRES_HARD_WALL"):
        run_r85_source_ablation(spec, data)
    spec["design"]["budget"]["max_wall_seconds"] = 180
    with pytest.raises(ValueError, match="R85_INVALID_ABLATION_PROPOSAL_BUDGET"):
        run_r85_source_ablation(spec, data, family_proposals=7)


def test_guard_transfers_family_parameters_into_spawn_target(monkeypatch):
    import phm_design.industrial.resource_guard as guard
    spec, data = spec_and_data()
    recorded = []
    def worker_stub(fn, args, limit):
        recorded.append((fn, args, limit))
        return "refused", {"decision": "CONTROLLED_REFUSAL",
                           "independent_qualification": False}
    monkeypatch.setattr(guard, "_invoke_worker", worker_stub)
    with pytest.raises(IndustrialDesignRejected):
        run_design_guarded(spec, data, selection_rule="r83_matched_train_loss",
                           family_search_mode="feedback", family_proposals=3)
    assert recorded[0][0] is guard._worker
    assert recorded[0][1][-2:] == ("feedback", 3)
    assert recorded[0][2] == 180
