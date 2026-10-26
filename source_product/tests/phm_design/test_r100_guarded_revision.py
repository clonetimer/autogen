"""R100 SOURCE-only revision triage; protocol fixtures are not scientific proof."""
from __future__ import annotations

from copy import deepcopy
import pytest

from phm_design.industrial_contracts.guarded_revision import (
    RevisionGateRejected, freeze_source_revision_policy,
    audit_guarded_source_revisions)
from phm_design.industrial_contracts.design_review import _digest
from phm_design.industrial_contracts.physical_graph_design import (
    propose_physical_graph_study, evaluate_physical_graph_study)
from test_r99_design_review import _source_report
from test_r94_torch_like_bridge import SpyTorchLike
from test_r98_physical_graph_design import _cards, _signals


def _modified_report():
    study, report = _source_report()
    candidates = report["evaluated_graphs"][0][
        "source_only_pipeline_comparison"]["candidates"]
    for c in candidates:
        c["source_worst_group_accuracy"] = 0.01
        c["source_group_macro_accuracy"] = 0.02
        c["source_healthy_false_alarm_fraction"] = 0.90
        c["sum_observed_fit_seconds"] = 0.2
    baseline = candidates[0]
    baseline["source_worst_group_accuracy"] = 0.90
    baseline["source_group_macro_accuracy"] = 0.95
    baseline["source_healthy_false_alarm_fraction"] = 0.01
    baseline["sum_observed_fit_seconds"] = 0.45
    suspect = report["evaluated_graphs"][1][
        "source_only_pipeline_comparison"]["candidates"][0]
    suspect["source_worst_group_accuracy"] = 0.10
    suspect["source_group_macro_accuracy"] = 0.15
    suspect["source_healthy_false_alarm_fraction"] = 0.8
    suspect["sum_observed_fit_seconds"] = 0.4
    return study, report, baseline, suspect


def _get(decisions, key):
    return next(x for x in decisions if x["failed_candidate_key"] == key)


def test_r100_source_failure_has_only_frozen_family_fold_and_cost_guarded_fallback():
    study, report, baseline, suspect = _modified_report()
    policy = freeze_source_revision_policy(study)
    assert policy == freeze_source_revision_policy(study)
    result = audit_guarded_source_revisions(policy, study, report)
    row = _get(result["decisions"], "graph_02/" + suspect["pipeline_id"])
    assert row["admission"] == "SOURCE_ONLY_CANDIDATE_FOR_NEXT_SEALED_STUDY"
    assert row["predeclared_target_candidate_key"] == (
        "graph_01/" + baseline["pipeline_id"])
    assert row["same_original_source_fold_membership"]
    assert row["reason_codes"] == []
    assert result["eligible_source_fallback_count"] >= 1
    assert len(result["receipt_sha256"]) == 64
    assert result == audit_guarded_source_revisions(policy, study, report)
    assert result["source_post_selection_bias_present"]
    assert result["independent_improvement_proven"] is False
    assert result["eligible_for_stable"] is False
    assert not row["new_fit_performed"]


@pytest.mark.parametrize("field,value,reason", [
    ("sum_observed_fit_seconds", 1.0,
     "OBSERVED_PROVIDER_FIT_TIME_BUDGET_EXCEEDED"),
    ("source_worst_group_accuracy", 0.05,
     "WORST_SOURCE_GROUP_REGRESSES_OR_GAIN_TOO_SMALL"),
    ("source_group_macro_accuracy", 0.01,
     "GROUP_MACRO_ACCURACY_REGRESSES"),
    ("source_healthy_false_alarm_fraction", 0.95,
     "HEALTHY_FALSE_ALARM_REGRESSES"),
])
def test_r100_source_dominance_or_fit_cost_regression_refused(field, value, reason):
    study, report, baseline, suspect = _modified_report()
    baseline[field] = value
    policy = freeze_source_revision_policy(study)
    result = audit_guarded_source_revisions(policy, study, report)
    row = _get(result["decisions"], "graph_02/" + suspect["pipeline_id"])
    assert row["admission"] == "REFUSE_AS_UNJUSTIFIED_SOURCE_REVISION"
    assert reason in row["reason_codes"]
    assert not result["independent_improvement_proven"]


def test_r100_cannot_compare_different_model_families_as_graph_only_gain():
    study, report, baseline, _ = _modified_report()
    complex_rows = report["evaluated_graphs"][1][
        "source_only_pipeline_comparison"]["candidates"]
    distinct = next(c for c in complex_rows
                    if c["algorithm_family"] != baseline["algorithm_family"])
    distinct["source_worst_group_accuracy"] = 0.10
    distinct["source_group_macro_accuracy"] = 0.15
    distinct["source_healthy_false_alarm_fraction"] = 0.8
    distinct["sum_observed_fit_seconds"] = 0.4
    result = audit_guarded_source_revisions(
        freeze_source_revision_policy(study), study, report)
    row = _get(result["decisions"], "graph_02/" + distinct["pipeline_id"])
    assert "PROVIDER_FAMILY_DIFFERS" in row["reason_codes"]


def test_r100_refuses_when_proposed_graphs_use_different_original_source_folds():
    study, report, baseline, suspect = _modified_report()
    mutated = deepcopy(study)
    other_fold = mutated["graphs"][1]["pipeline_plan"]["folds"][0]
    other_fold["score_ids"] = list(other_fold["score_ids"]) + [
        "fake_extra_source_record"]
    mutated["study_sha256"] = _digest({
        k: v for k, v in mutated.items() if k != "study_sha256"})
    report["study_sha256"] = mutated["study_sha256"]
    policy = freeze_source_revision_policy(mutated)
    result = audit_guarded_source_revisions(policy, mutated, report)
    row = _get(result["decisions"], "graph_02/" + suspect["pipeline_id"])
    assert "SOURCE_GROUP_FOLDS_DIFFER" in row["reason_codes"]


def test_r100_refuses_policy_tamper_or_invalid_policy_parameters():
    study, report, _, _ = _modified_report()
    policy = freeze_source_revision_policy(study)
    changed = deepcopy(policy)
    changed["max_fit_time_ratio"] = 100
    with pytest.raises(RevisionGateRejected, match="R100_POLICY_SHA256_CHANGED"):
        audit_guarded_source_revisions(changed, study, report)
    for value in (-1, 0, 5, float("nan"), True, "2"):
        with pytest.raises(RevisionGateRejected,
                           match="R100_INVALID_PREDECLARED_COST_OR_GAIN_GUARD"):
            freeze_source_revision_policy(study, max_fit_time_ratio=value)


def test_r100_stricter_registered_gain_requirement_prevents_unearned_claim():
    study, report, baseline, suspect = _modified_report()
    baseline["source_worst_group_accuracy"] = .11
    policy = freeze_source_revision_policy(study, min_worst_group_gain=.2)
    row = _get(audit_guarded_source_revisions(
        policy, study, report)["decisions"], "graph_02/" + suspect["pipeline_id"])
    assert "WORST_SOURCE_GROUP_REGRESSES_OR_GAIN_TOO_SMALL" in row["reason_codes"]


def test_r100_source_plan_cannot_claim_heldout_access_or_generalization():
    study, _, _, _ = _modified_report()
    for attribute in ("external_validation_or_final_read",
                      "independent_diagnostic_generalization_proven"):
        altered = deepcopy(study)
        altered[attribute] = True
        altered["study_sha256"] = _digest({
            k: v for k, v in altered.items() if k != "study_sha256"})
        with pytest.raises(RevisionGateRejected,
                           match="R100_PROTECTED_ROLE_OR_EXTERNAL_CLAIM_IN_SOURCE_STUDY"):
            freeze_source_revision_policy(altered)


def test_r100_r99_source_abstention_remains_abstention():
    equipment, train, trial = _cards(speed=True)
    train["records"] = train["records"][:3]
    study = propose_physical_graph_study(equipment, train, trial)
    source = evaluate_physical_graph_study(
        study, equipment, train, trial, _signals(train),
        provider=SpyTorchLike())
    result = audit_guarded_source_revisions(
        freeze_source_revision_policy(study), study, source)
    assert result["status"] == "ABSTAIN_SOURCE_NOT_EVALUATED"
    assert result["decisions"] == []
    assert result["eligible_source_fallback_count"] == 0
    assert not result["independent_improvement_proven"]
