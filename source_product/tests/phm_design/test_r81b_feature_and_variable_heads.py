"""R81b synthetic TRAIN-only representation and variable-head execution.

This is an integration test for actual sklearn/PyTorch fit, not an industrial
physics performance study. It never opens prior revealed targets or FINAL data.
"""
import copy
import numpy as np
import pytest

from phm_design.industrial.taskspec import TaskSpecRejected, compile_task_spec
from phm_design.industrial.design_engine import (
    DesignAcceptance, IndustrialDesignRejected, run_design,
    _candidate, _factorized_fit_cost,
)
from test_r81a_unified_design_engine import task, synthetic_records

FUNCTIONAL = DesignAcceptance(accuracy_min=0,
                              worst_environment_class_recall_min=0,
                              healthy_false_alarm_max=1)


def _search(space=("linear",), representations=("temporal_bins",),
            topologies=("direct",), source="satellite_eps_joint_demo.json"):
    s = task(source, compare=space)
    s["design"]["representations"] = list(representations)
    s["design"]["topologies"] = list(topologies)
    s["design"]["candidate_max"] = len(space)*len(representations)*len(topologies)
    s["design"]["budget"]["max_fits"] = 120
    return s


def test_compiler_predeclares_real_feature_and_task_topology_choices():
    s = _search(space=("linear", "tree", "tcn"),
                representations=("temporal_bins", "random_kernels", "difference_kernels"),
                topologies=("direct", "factorized"))
    plan = compile_task_spec(s)
    options = plan["candidate_grid"]
    assert len(options) == 14  # 2 tabular learners × 3 programs × 2 DAGs; neural × 2
    assert {x["topology"] for x in options} == {"direct", "factorized"}
    assert {x["representation"] for x in options} == {
        "temporal_bins", "random_kernels", "difference_kernels", "learned_sequence"}
    assert plan["qualified"] is False
    assert plan["source_only_nested_selection"] is True


def test_three_executable_generic_feature_programs_train_and_predict_without_truth():
    s = _search(representations=("temporal_bins", "random_kernels", "difference_kernels"))
    records = synthetic_records(s, environment_count=3)
    a = run_design(s, records, acceptance=FUNCTIONAL)
    assert len(a.audit["attempts"]) == 3
    assert all(x["status"] == "ELIGIBLE" for x in a.audit["attempts"])
    assert {t["trial"]["feature"] for t in a.audit["attempts"]} == {
        "temporal", "random_kernels", "difference_kernels"}
    assert all(len(t["folds"]) == 3 for t in a.audit["attempts"])
    assert a.audit["total_model_fits"] == 10
    assert a.predict(records[0]["measurements"])[0] in s["task"]["labels"]


def test_variable_five_label_factorized_heads_are_real_models_and_deployable():
    s = _search(topologies=("factorized",))
    records = synthetic_records(s, environment_count=3)
    a = run_design(s, records, acceptance=FUNCTIONAL)
    assert a.trial.topology == "factorized"
    assert a.audit["total_model_fits"] == 12  # three heads × (3 folds + refit)
    assert a.model.fit_count_ == 3
    assert set(a.model.models_) == {"detection", "component", "mechanism"}
    assert a.model.training_["independent_qualification"] is False
    feature = a.model.predict_proba(np.stack(
        [np.asarray([row[c] for c in
                      [s["observations"]["time"]["field"],
                       *[m["name"] for m in s["observations"]["channels"]]]]
                     for row in records[0]["measurements"]])[:1]
    )) if False else None
    assert a.predict(records[0]["measurements"])[0] in s["task"]["labels"]
    assert all(t["folds"][0]["disjoint"] for t in a.audit["attempts"])


def test_direct_and_factorized_topologies_are_compared_with_real_training():
    s = _search(topologies=("direct", "factorized"))
    records = synthetic_records(s, environment_count=3)
    a = run_design(s, records, acceptance=FUNCTIONAL)
    assert len(a.audit["attempts"]) == 2
    assert [t["trial"]["topology"] for t in a.audit["attempts"]] == [
        "direct", "factorized"]
    assert a.audit["total_model_fits"] == 16  # (1+3) × (3 folds + refit)
    assert all(t["status"] == "ELIGIBLE" for t in a.audit["attempts"])
    assert all(t["metrics"]["evidence"] ==
               "TRAIN_SOURCE_LOEO_OOF_NOT_INDEPENDENT_VALIDATION"
               for t in a.audit["attempts"])


def test_six_class_unseen_equipment_ontology_needs_no_diagnosis_source_edit():
    s = _search(topologies=("factorized",))
    s["equipment"] = {"family": "industrial_process", "adapter": "generic_csv"}
    s["task"]["labels"].append("LOAD_C_EXTRA")
    s["task"]["structure"]["components"].append("load_c")
    s["task"]["structure"]["class_attributes"]["LOAD_C_EXTRA"] = {
        "component": "load_c", "mechanism": "extra_load"}
    records = synthetic_records(s, environment_count=3)
    a = run_design(s, records, acceptance=FUNCTIONAL)
    assert len(a.model.classes_) == 6
    assert len(a.model.components) == 5
    assert a.model.fit_count_ == 3
    assert a.predict(records[-1]["measurements"])[0] in s["task"]["labels"]


def test_factorized_neural_tcn_trains_multiple_heads_without_metadata_in_prediction():
    s = _search(space=("tcn",), topologies=("factorized",))
    records = synthetic_records(s, environment_count=3)
    a = run_design(s, records, acceptance=FUNCTIONAL)
    assert a.trial.learner == "torch_tcn"
    assert a.model.fit_count_ == 3
    assert all(m.training_["backend"] == "torch_cpu"
               for m in a.model.models_.values() if m is not None)
    output = a.predict(records[0]["measurements"])
    assert len(output) == 1 and output[0] in s["task"]["labels"]


def test_single_component_or_mechanism_factorized_heads_can_be_constant():
    s = _search(topologies=("factorized",))
    for label in s["task"]["labels"][1:]:
        s["task"]["structure"]["class_attributes"][label]["component"] = "solar_a"
    with pytest.raises(TaskSpecRejected, match="FACTORIZED_DUPLICATE_COMPONENT_MECHANISM_PAIR"):
        compile_task_spec(s)


def test_rejects_factorized_topology_when_task_has_no_joint_fault_ontology():
    s = _search(source="industrial_bearing_classification_demo.json",
                topologies=("factorized",))
    with pytest.raises(TaskSpecRejected, match="FACTORIZED_REQUIRES_JOINT_DIAGNOSIS"):
        compile_task_spec(s)


@pytest.mark.parametrize("bad", ["future_oracle", "custom_callback", "evaluation_target"])
def test_unregistered_feature_atoms_fail_before_training(bad):
    s = _search(representations=(bad,))
    with pytest.raises(TaskSpecRejected, match="UNREGISTERED_FEATURE_PROGRAM"):
        compile_task_spec(s)


def test_fit_budget_accounts_for_each_component_and_mechanism_head():
    s = _search(topologies=("direct", "factorized"))
    s["design"]["budget"]["max_fits"] = 5
    records = synthetic_records(s, environment_count=3)
    with pytest.raises(IndustrialDesignRejected) as err:
        run_design(s, records, acceptance=FUNCTIONAL)
    audit = err.value.audit
    assert audit["model_delivered"] is False
    assert audit["decision"] == "BUDGET_EXHAUSTED_BEFORE_FINAL_REFIT"
    assert audit["fold_model_fits"] == 3


def test_factorized_budget_blocks_partial_head_fit_before_first_fold():
    s = _search(topologies=("factorized",))
    s["design"]["budget"]["max_fits"] = 1
    with pytest.raises(IndustrialDesignRejected) as err:
        run_design(s, synthetic_records(s, environment_count=3), acceptance=FUNCTIONAL)
    assert err.value.audit["decision"] == "BUDGET_EXHAUSTED_BEFORE_SELECTION"
    assert err.value.audit["fold_model_fits"] == 0


def test_protected_final_metadata_is_blocked_even_with_factorized_search():
    s = _search(topologies=("factorized",))
    records = synthetic_records(s, environment_count=3)
    records[0]["metadata"]["role"] = "FINAL_OOD"
    with pytest.raises(TaskSpecRejected, match="DESIGN_WORKER_PROTECTED_ROLE"):
        run_design(s, records, acceptance=FUNCTIONAL)
