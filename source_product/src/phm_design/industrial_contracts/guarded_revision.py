"""R100 source-only revision gate: do not promote an attractive R99 fallback blindly.

No new fits, protected validation inputs or model changes. A policy may be
prepared before SOURCE evaluation, but its timestamp is NOT attested. Results
are descriptive triage of already evaluated frozen candidates, not an unbiased
estimate of improvement or independent-device generalization.
"""
from __future__ import annotations

from collections.abc import Mapping
import math

from .design_review import _digest, explain_source_design, DesignReviewRejected

SCHEMA = "phm.autogen.source-revision-policy.v1"
RESULT = "phm.autogen.guarded-source-revision-audit.v1"


class RevisionGateRejected(DesignReviewRejected):
    """Malformed source policy, study or candidate lineage."""


def _fail(reason):
    raise RevisionGateRejected(reason)


def _ratio(value):
    return type(value) in (int, float) and math.isfinite(value) and 1 <= value <= 4


def _gain(value):
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1


def _validate_plan(study):
    if not isinstance(study, Mapping) or not isinstance(study.get("graphs"), list):
        _fail("R100_REGISTERED_SOURCE_STUDY_REQUIRED")
    if (study.get("evidence_scope") != "SOURCE_TRAIN_INTERNAL_GRAPH_SEARCH_ONLY"
            or study.get("external_validation_or_final_read") is not False
            or study.get("independent_diagnostic_generalization_proven") is not False):
        _fail("R100_PROTECTED_ROLE_OR_EXTERNAL_CLAIM_IN_SOURCE_STUDY")
    base = {k: v for k, v in study.items() if k != "study_sha256"}
    if study.get("study_sha256") != _digest(base):
        _fail("R100_STUDY_IDENTITY_NOT_FROZEN")


def freeze_source_revision_policy(study, *, max_fit_time_ratio=1.25,
                                  min_worst_group_gain=0.0):
    """Freeze a conservative decision rule WITHOUT viewing the SOURCE report.

    Caller must provide external chronological proof if preregistration timing
    matters; this function does not authenticate a timestamp.
    """
    _validate_plan(study)
    if not _ratio(max_fit_time_ratio) or not _gain(min_worst_group_gain):
        _fail("R100_INVALID_PREDECLARED_COST_OR_GAIN_GUARD")
    doc = {
        "schema": SCHEMA,
        "study_sha256": study["study_sha256"],
        "max_fit_time_ratio": max_fit_time_ratio,
        "min_worst_group_gain": min_worst_group_gain,
        "require_same_source_group_folds": True,
        "require_same_provider_algorithm_family": True,
        "require_no_group_macro_accuracy_regression": True,
        "require_no_healthy_false_alarm_regression": True,
        "source_only": True,
        "post_selection_independent_validation": False,
        "policy_created_before_source_scores_externally_attested": False,
    }
    return {**doc, "policy_sha256": _digest(doc)}


def _validated_policy(policy, study):
    if not isinstance(policy, Mapping):
        _fail("R100_POLICY_NOT_FROZEN")
    body = {k: v for k, v in policy.items() if k != "policy_sha256"}
    if policy.get("policy_sha256") != _digest(body):
        _fail("R100_POLICY_SHA256_CHANGED")
    expected = freeze_source_revision_policy(
        study, max_fit_time_ratio=policy.get("max_fit_time_ratio"),
        min_worst_group_gain=policy.get("min_worst_group_gain"))
    if policy != expected:
        _fail("R100_POLICY_SCOPE_OR_STUDY_CHANGED")


def _source_fold_id(graph):
    plan = graph.get("pipeline_plan")
    if not isinstance(plan, Mapping):
        _fail("R100_MISSING_SOURCE_FOLD_PLAN")
    folds = plan.get("folds")
    if not isinstance(folds, list) or not folds:
        _fail("R100_NO_SOURCE_GROUP_FOLDS")
    keyed = []
    for fold in folds:
        if not isinstance(fold, Mapping):
            _fail("R100_MALFORMED_SOURCE_FOLD")
        try:
            group, fit, score = fold["group_sha256"], fold["fit_ids"], fold["score_ids"]
        except KeyError:
            _fail("R100_SOURCE_FOLD_MISSING_LINEAGE")
        if (type(group) is not str or not isinstance(fit, list)
                or not isinstance(score, list) or not fit or not score
                or set(fit) & set(score) or len(fit) != len(set(fit))
                or len(score) != len(set(score))):
            _fail("R100_INVALID_SOURCE_FOLD")
        keyed.append({"group_sha256": group, "fit_ids": sorted(fit),
                      "score_ids": sorted(score)})
    if len({x["group_sha256"] for x in keyed}) != len(keyed):
        _fail("R100_DUPLICATE_SOURCE_GROUP_FOLD")
    return _digest(sorted(keyed, key=lambda v: v["group_sha256"]))


def audit_guarded_source_revisions(policy, study, source_report):
    """Audits R99 recommendations and refuses risky SOURCE-only fallback.

    This does not retrain, read VAL/FINAL, use hidden information, or claim
    compute-matched superiority. SOURCE OOF reuse is explicitly selective.
    """
    _validate_plan(study)
    _validated_policy(policy, study)
    # R99's frozen study/report checks and explanation provenance are retained.
    explanation = explain_source_design(study, source_report)
    report_digest = _digest(source_report)
    if explanation["status"] != "DESCRIPTIVE_SOURCE_EXPLANATION_ONLY":
        receipt = {
            "schema": RESULT, "status": "ABSTAIN_SOURCE_NOT_EVALUATED",
            "study_sha256": study["study_sha256"],
            "policy_sha256": policy["policy_sha256"],
            "source_report_sha256": report_digest,
            "explanation_sha256": explanation["explanation_sha256"],
            "decisions": [], "eligible_source_fallback_count": 0,
            "source_post_selection_bias_present": True,
            "independent_improvement_proven": False,
            "eligible_for_stable": False,
        }
        return {**receipt, "receipt_sha256": _digest(receipt)}

    plans = {g["graph_id"]: g for g in study["graphs"]}
    by_key = {}
    for entry in explanation["failure_audit"]:
        key = entry["candidate_key"]
        if key in by_key:
            _fail("R100_DUPLICATE_FROZEN_CANDIDATE_KEY")
        by_key[key] = entry
    if not any(g["recipe"] == "time_baseline" for g in plans.values()):
        _fail("R100_NO_FROZEN_BASELINE_GRAPH")
    # Guard same original physical-group fold membership, not just a
    # shared label or number of windows. SOURCE cohorts must be identical.
    folds = {gid: _source_fold_id(g) for gid, g in plans.items()}
    decisions = []
    for revision in explanation["bounded_revisions"]:
        candidate = by_key.get(revision["failed_candidate_key"])
        if candidate is None:
            _fail("R100_UNKNOWN_FAILED_CANDIDATE")
        proposed = revision["candidate_already_in_frozen_study"]
        target = by_key.get(proposed) if proposed is not None else None
        blocked = []
        if target is None:
            blocked.append("NO_PREDECLARED_TARGET")
        else:
            if proposed == candidate["candidate_key"]:
                blocked.append("SELF_COMPARISON_NOT_A_REVISION")
            cg = candidate["candidate_key"].split("/", 1)[0]
            tg = target["candidate_key"].split("/", 1)[0]
            if cg not in folds or tg not in folds or folds[cg] != folds[tg]:
                blocked.append("SOURCE_GROUP_FOLDS_DIFFER")
            if candidate["algorithm_family"] != target["algorithm_family"]:
                blocked.append("PROVIDER_FAMILY_DIFFERS")
            if (target["source_worst_group_accuracy"] + 1e-12 <
                    candidate["source_worst_group_accuracy"] +
                    policy["min_worst_group_gain"]):
                blocked.append("WORST_SOURCE_GROUP_REGRESSES_OR_GAIN_TOO_SMALL")
            if (target["source_group_macro_accuracy"] + 1e-12 <
                    candidate["source_group_macro_accuracy"]):
                blocked.append("GROUP_MACRO_ACCURACY_REGRESSES")
            if (target["source_healthy_false_alarm_fraction"] >
                    candidate["source_healthy_false_alarm_fraction"] + 1e-12):
                blocked.append("HEALTHY_FALSE_ALARM_REGRESSES")
            src_time = candidate["cost_scope"]["sum_provider_fit_seconds_observed"]
            target_time = target["cost_scope"]["sum_provider_fit_seconds_observed"]
            if target_time > src_time * policy["max_fit_time_ratio"] + 1e-12:
                blocked.append("OBSERVED_PROVIDER_FIT_TIME_BUDGET_EXCEEDED")
            if not (target["source_worst_group_accuracy"] >
                    candidate["source_worst_group_accuracy"] + 1e-12
                    or target["source_group_macro_accuracy"] >
                    candidate["source_group_macro_accuracy"] + 1e-12
                    or target["source_healthy_false_alarm_fraction"] + 1e-12 <
                    candidate["source_healthy_false_alarm_fraction"]
                    or target_time + 1e-12 < src_time):
                blocked.append("NO_STRICT_SOURCE_METRIC_OR_COST_IMPROVEMENT")
        admitted = not blocked
        decisions.append({
            "failed_candidate_key": candidate["candidate_key"],
            "predeclared_target_candidate_key": proposed,
            "allowed_action": revision["allowed_action"],
            "admission": ("SOURCE_ONLY_CANDIDATE_FOR_NEXT_SEALED_STUDY"
                          if admitted else "REFUSE_AS_UNJUSTIFIED_SOURCE_REVISION"),
            "reason_codes": blocked,
            "same_original_source_fold_membership": (
                target is not None
                and candidate["candidate_key"].split("/", 1)[0] in folds
                and target["candidate_key"].split("/", 1)[0] in folds
                and folds[candidate["candidate_key"].split("/", 1)[0]] ==
                    folds[target["candidate_key"].split("/", 1)[0]]),
            "new_fit_performed": False,
            "validated_on_new_independent_device": False,
            "source_model_selection_bias_corrected": False,
        })
    receipt = {
        "schema": RESULT, "status": "SOURCE_ONLY_REVISION_TRIAGE_NOT_VALIDATION",
        "study_sha256": study["study_sha256"],
        "policy_sha256": policy["policy_sha256"],
        "source_report_sha256": report_digest,
        "explanation_sha256": explanation["explanation_sha256"],
        "decisions": decisions,
        "eligible_source_fallback_count": sum(
            x["admission"] == "SOURCE_ONLY_CANDIDATE_FOR_NEXT_SEALED_STUDY"
            for x in decisions),
        "source_post_selection_bias_present": True,
        "source_prediction_stream_authenticated": False,
        "fit_time_hardware_matched": False,
        "feature_extraction_and_inference_cost_included": False,
        "independent_improvement_proven": False,
        "eligible_for_stable": False,
    }
    return {**receipt, "receipt_sha256": _digest(receipt)}
