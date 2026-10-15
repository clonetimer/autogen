"""R81c bounded TRAIN-only feedback search, not FINAL generalization evidence."""
import copy
import json

import pytest

from phm_design.open_design import FeedbackPlanner, Program
from phm_design.industrial.artifact_io import load_trusted_artifact
from phm_design.industrial.design_engine import (
    DesignAcceptance, IndustrialDesignRejected, run_design,
)
from phm_design.industrial.feature_program import decode_program
from phm_design.industrial.taskspec import TaskSpecRejected, compile_task_spec
from test_r81a_unified_design_engine import task, synthetic_records

SCREEN = DesignAcceptance(accuracy_min=0,
                          worst_environment_class_recall_min=0,
                          healthy_false_alarm_max=1)


def spec(*, source="industrial_bearing_classification_demo.json",
         learner="linear", max_programs=3, topology="direct"):
    s = task(source, compare=(learner,))
    s["design"]["topologies"] = [topology]
    s["design"]["candidate_max"] = max_programs
    s["design"]["adaptive_feature_search"] = {
        "atoms": ["temporal_bins", "random_kernels", "difference_kernels"],
        "max_programs": max_programs,
    }
    s["design"]["budget"]["max_fits"] = 80
    return s


def test_adaptive_pool_compiles_without_unbounded_code_or_extra_trial_arms():
    compiled = compile_task_spec(spec())
    assert compiled["adaptive_feature_search"]["mode"] == "bounded_feedback"
    assert compiled["adaptive_feature_search"]["candidate_pool_size"] == 7
    assert compiled["adaptive_feature_search"]["max_programs"] == 3
    assert len(compiled["candidate_grid"]) == 7
    assert all(item["representation"] == "feature_program"
               for item in compiled["candidate_grid"])
    assert all(item["topology"] == "direct" for item in compiled["candidate_grid"])
    assert compiled["independent_qualification"] is False


def test_feedback_changes_third_proposal_without_changing_declared_pool():
    atoms = ("temporal_bins", "random_kernels", "difference_kernels")
    first = Program(("temporal_bins",), "linear")
    second = Program(("random_kernels",), "linear")
    pool = [Program((a,), "linear") for a in atoms] + [
        Program(("temporal_bins", "difference_kernels"), "linear"),
        Program(("random_kernels", "difference_kernels"), "linear"),
        Program(("random_kernels", "temporal_bins"), "linear"),
    ]
    def third(first_error, second_error):
        planner = FeedbackPlanner(
            atoms, mode="bounded_feedback", seeds=(first, second),
            program_pool=pool)
        a = planner.propose()
        assert a.key == first.key
        planner.observe(a, [first_error] * 3)
        b = planner.propose()
        assert b.key == second.key
        planner.observe(b, [second_error] * 3)
        p = planner.propose()
        return p.atoms, planner.proposals[-1], planner.archive()

    from_first, event_a, archive_a = third(.01, .8)
    from_second, event_b, archive_b = third(.8, .01)
    assert from_first != from_second
    assert event_a["reason"] == event_b["reason"] == (
        "source_train_feedback_nearest_program")
    assert event_a["parent"] == first.key
    assert event_b["parent"] == second.key
    assert event_a["feedback_count"] == event_b["feedback_count"] == 2
    assert archive_a and archive_b


def test_real_source_oof_adapts_third_trial_and_deploys_selected_program(tmp_path):
    s = spec()
    records = synthetic_records(s, environment_count=3)
    model = run_design(s, records, acceptance=SCREEN)
    trace = model.audit["adaptive_trace"]
    assert len(model.audit["attempts"]) == 3
    assert trace["proposal_count"] == 3
    assert trace["feedback_count"] == 3
    assert trace["candidate_pool_count"] == 7
    assert trace["untested_pool_candidate_count"] == 4
    assert model.audit["fold_model_fits"] == 9
    assert model.audit["total_model_fits"] == 10
    first, second, third = model.audit["attempts"]
    assert first["adaptive_proposal"]["reason"] == "predeclared_shared_seed"
    assert second["adaptive_proposal"]["reason"] == "predeclared_shared_seed"
    assert third["adaptive_proposal"]["reason"] == (
        "source_train_feedback_nearest_program")
    assert all(attempt["adaptive_feedback"]["role"] == "SOURCE_TRAIN_OOF_ONLY"
               for attempt in model.audit["attempts"])
    assert all(attempt["adaptive_feedback"]["independent_qualification"] is False
               for attempt in model.audit["attempts"])
    assert model.audit["target_feedback_used"] is False
    assert model.audit["final_evaluation_performed"] is False
    assert model.audit["source_train_feedback_guided_proposal"] is True
    assert model.predict(records[0]["measurements"])[0] in s["task"]["labels"]

    folder = tmp_path / "adaptive"
    model.export(folder)
    restored = load_trusted_artifact(folder, trust_pickle=True)
    assert restored.predict_decisions(records[0]["measurements"]) == (
        model.predict_decisions(records[0]["measurements"]))
    manifest = json.loads((folder / "audit.json").read_text())
    assert manifest["trial_selection_audit"]["adaptive_trace"] == json.loads(json.dumps(trace))
    assert manifest["independent_qualification"] is False


def test_joint_factorized_uses_same_proposer_and_charges_every_head():
    s = spec(source="satellite_eps_joint_demo.json", topology="factorized")
    model = run_design(s, synthetic_records(s, environment_count=3),
                       acceptance=SCREEN)
    assert model.audit["fold_model_fits"] == 27
    assert model.audit["total_model_fits"] == 30
    assert model.audit["adaptive_trace"]["proposal_count"] == 3
    assert model.trial.topology == "factorized"


def test_predeclared_budget_exhaustion_refuses_even_after_eligible_candidate():
    s = spec()
    s["design"]["budget"]["max_fits"] = 4
    with pytest.raises(IndustrialDesignRejected) as err:
        run_design(s, synthetic_records(s, environment_count=3),
                   acceptance=SCREEN)
    audit = err.value.audit
    assert audit["model_delivered"] is False
    assert audit["decision"] == "BUDGET_EXHAUSTED_BEFORE_FINAL_REFIT"
    assert audit["fold_model_fits"] == 4
    assert audit["adaptive_trace"]["proposal_count"] == 2


@pytest.mark.parametrize("change,expected", [
    (lambda x: x["design"].update(compare=["tcn"]), "ADAPTIVE_SEARCH_REQUIRES_SINGLE_CLASSICAL_TEMPLATE"),
    (lambda x: x["design"].update(objectives=["ERM", "GroupDRO"]), "ADAPTIVE_SEARCH_REQUIRES_SINGLE_CLASSICAL_TEMPLATE"),
    (lambda x: x["design"].update(feature_programs=[{"atoms": ["temporal_bins"], "composition": "concat"}]), "ADAPTIVE_SEARCH_REQUIRES_SINGLE_CLASSICAL_TEMPLATE"),
    (lambda x: x["design"].update(representations=["random_kernels"]), "ADAPTIVE_SEARCH_REQUIRES_SINGLE_CLASSICAL_TEMPLATE"),
    (lambda x: x["design"]["adaptive_feature_search"].update(atoms=["temporal_bins", "unknown_atom", "random_kernels"]), "INVALID_ADAPTIVE_FEATURE_ATOMS"),
    (lambda x: x["design"]["adaptive_feature_search"].update(max_programs=8), "ADAPTIVE_SEARCH_TRIAL_BUDGET_MISMATCH"),
    (lambda x: x["design"].update(candidate_max=5), "ADAPTIVE_SEARCH_TRIAL_BUDGET_MISMATCH"),
])
def test_invalid_search_grammars_and_bounds_fail_before_training(change, expected):
    s = spec()
    change(s)
    with pytest.raises(TaskSpecRejected, match=expected):
        compile_task_spec(s)


def test_protected_role_and_truth_remain_blocked_in_feedback_search():
    s = spec()
    records = synthetic_records(s, environment_count=3)
    records[0]["metadata"]["role"] = "FINAL_OOD"
    with pytest.raises(TaskSpecRejected, match="DESIGN_WORKER_PROTECTED_ROLE"):
        run_design(s, records, acceptance=SCREEN)
    records = synthetic_records(s, environment_count=3)
    records[0]["measurements"][0]["fault_label"] = 5
    with pytest.raises(TaskSpecRejected, match="UNDECLARED_OR_PROTECTED_OBSERVATION"):
        run_design(s, records, acceptance=SCREEN)
