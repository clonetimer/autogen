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
    assert kwargs.get("selection_rule") == "r83_matched_train_loss"
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
    assert kwargs["seed_schedule"] == "r83_matched_train"
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
    assert out["fairness"]["shared_candidate_seed_schedule"] == (
        "base_seed_plus_candidate_index")
    assert out["fairness"]["shared_predeclared_candidate_selection_rule"] == (
        "r83_matched_train_loss")
    supports = out["independent_physical_class_group_support"]
    assert len(supports) == 3
    assert all(counts == {label: 2 for label in spec["task"]["labels"]}
               for counts in supports.values())
    assert received == [("fixed", records), ("evolved", records), ("evolved", records)]
    assert out["ablation"]["same_neural_family"] is True
    assert out["ablation"]["same_two_initial_genomes"] is True
    assert out["fairness_gate"]["static_neural_ablation_present"] is True
    assert out["arms"]["static_topology"]["charged_fits"] == 10
    assert out["arms"]["fixed_portfolio"]["charged_fits"] == 10
    assert out["arms"]["self_organizing"]["charged_fits"] == 10
    assert out["fairness_gate"]["source_train_comparison_allowed"] is True
    assert out["fairness_gate"]["independent_generalization_claim_allowed"] is False
    assert out["fairness_gate"]["minimum_observed_physical_groups_per_regime_class"] == 2
    assert out["fairness_gate"]["confirmatory_sample_support_met"] is False
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
    assert out["fairness_gate"]["source_train_comparison_allowed"] is False
    assert out["arms"]["self_organizing"]["model_delivered"] is True
    assert out["arms"]["static_topology"]["model_delivered"] is True
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
    too_few_trials = copy.deepcopy(spec)
    too_few_trials["design"]["candidate_max"] = 2
    with pytest.raises(TaskSpecRejected, match="SOURCE_TRIAL_SLOTS_BELOW_THREE"):
        run_matched_source_train_study(too_few_trials, rows)
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
    no_support = copy.deepcopy(rows)
    label = spec["task"]["labels"][-1]
    no_support = [r for r in no_support if not (
        r["metadata"]["label"] == label and
        r["metadata"]["speed_regime"] == "regime_2" and
        r["metadata"]["bearing_instance_id"].endswith("_1"))]
    with pytest.raises(TaskSpecRejected, match="INSUFFICIENT_PHYSICAL_CLASS_GROUPS"):
        run_matched_source_train_study(spec, no_support)
    altered = copy.deepcopy(spec)
    altered["design"]["objectives"] = ["GroupDRO"]
    with pytest.raises(TaskSpecRejected, match="PLAIN_ERM_DIRECT"):
        run_matched_source_train_study(altered, rows)


def test_r82_native_staging_and_binding_before_benchmark(tmp_path):
    args = rw_fixture(tmp_path)
    from pathlib import Path
    from phm_design.io import read_json, write_json
    new_spec = read_json(args["task_spec"])
    new_spec["id"] += ".r83matched"
    new_spec["design"]["candidate_max"] = 3
    new_spec["design"]["compare"] = ["tcn", "gru", "tree"]
    new_spec["design"]["budget"]["max_fits"] = 10
    new_spec_path = tmp_path / "r83_matched_spec.json"
    write_json(new_spec_path, new_spec)
    args["task_spec"] = new_spec_path
    out = run_native_rw_matched_study(
        **args, output=tmp_path / "native_train_study", validate_only=True)
    assert out["status"] == "TRAIN_SOURCE_ACCEPTED"
    assert out["model_delivered_to_deployment"] is False
    assert out["independent_qualification"] is False
    assert out["validation_telemetry_accessed"] is False
    assert out["source_intake"]["records"] == 24
    assert not (tmp_path / "native_train_study" / "comparison.json").exists()


def test_legacy_two_candidate_rw_taskspec_refused_even_for_validate_only(tmp_path):
    args = rw_fixture(tmp_path)
    with pytest.raises(TaskSpecRejected, match="SOURCE_TRIAL_SLOTS_BELOW_THREE"):
        run_native_rw_matched_study(
            **args, output=tmp_path / "incorrect_r82_contract",
            validate_only=True)
    assert not (tmp_path / "incorrect_r82_contract").exists()


def test_real_torch_and_extra_trees_run_both_arms_under_same_train_protocol():
    """Actually fit both arms; not a Basilisk study or an independent score."""
    spec = _spec()
    rec = synthetic_records(spec, environment_count=3)
    from phm_design.industrial.design_engine import DesignAcceptance
    study = run_matched_source_train_study(
        spec, rec, acceptance=DesignAcceptance(
            accuracy_min=0.0,
            worst_environment_class_recall_min=0.0,
            healthy_false_alarm_max=1.0))
    assert study["complete_two_arm_comparison"] is True
    assert study["fairness_gate"]["source_train_comparison_allowed"] is True
    assert study["source_train_test_or_final_accessed"] is False
    assert study["independent_qualification"] is False
    assert study["arms"]["fixed_portfolio"]["charged_fits"] == 10
    assert study["arms"]["static_topology"]["charged_fits"] == 10
    assert study["arms"]["self_organizing"]["charged_fits"] == 10
    assert len(study["arms"]["fixed_portfolio"]["attempts"]) == 3
    static = study["arms"]["static_topology"]["attempts"]
    adaptive = study["arms"]["self_organizing"]["attempts"]
    assert len(static) == len(adaptive) == 3
    assert all(x["candidate"]["genome_sha256"] == y["candidate"]["genome_sha256"]
               for x, y in zip(static[:2], adaptive[:2]))
    assert len(study["arms"]["self_organizing"]["attempts"]) == 3
    assert study["arms"]["fixed_portfolio"]["best_eligible_source_train_candidate"]
    assert study["arms"]["self_organizing"]["best_eligible_source_train_candidate"]
