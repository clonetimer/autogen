"""R99: auditable reasons, bounded revisions, and partial-cost Pareto analysis.

DESIGN WORKER ONLY: accepts an R98 frozen graph plan and a SOURCE TRAIN
internal R98 OOF result. Never accepts VAL/FINAL labels, accesses a raw
recording, changes model weights, executes arbitrary expressions or upgrades
the independent-validation scope. Structural reasoning is deterministic
from observable hardware prerequisites and measured SOURCE failure modes.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math

from .physical_graph_design import SCHEMA as GRAPH_SCHEMA, REPORT as GRAPH_REPORT

SCHEMA = "phm.autogen.explained-diagnostic-design.v1"
BASELINE = "time_baseline"
REASONS = {
    "time_baseline": (
        "registered_vibration_sensor_and_sample_clock",
        "low_complexity_time_domain_reference",
    ),
    "envelope_demod": (
        "vibration_resonance_2_to_8_khz_with_appropriate_nyquist",
        "amplitude_modulation_may_encode_impulses_not_fault_identification",
    ),
    "rotational_order": (
        "separate_registered_tachometer_with_per_window_rpm",
        "steady_speed_required_to_interpret_shaft_order_bands",
    ),
    "recording_level_fusion": (
        "same_recording_rpm_and_registered_scalar_companion",
        "only_recording_level_alignment_self_attested",
    ),
}
FAILURE_THRESHOLDS = {
    "source_worst_group_accuracy": .5,
    "source_group_macro_accuracy": .65,
    "source_healthy_false_alarm_fraction": .2,
}


class DesignReviewRejected(ValueError):
    """Frozen SOURCE report or cost evidence is malformed or inconsistent."""


def _fail(reason):
    raise DesignReviewRejected(reason)


def _digest(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        allow_nan=False, ensure_ascii=False).encode("utf-8")).hexdigest()


def _number(x, minimum=None, maximum=None):
    return (type(x) in (float, int) and math.isfinite(x)
            and (minimum is None or x >= minimum)
            and (maximum is None or x <= maximum))


def _validate(study, source_report):
    if (not isinstance(study, Mapping) or study.get("schema") != GRAPH_SCHEMA
            or not isinstance(source_report, Mapping)
            or source_report.get("schema") != GRAPH_REPORT
            or study.get("study_sha256") != source_report.get("study_sha256")):
        _fail("R99_FROZEN_GRAPH_OR_REPORT_IDENTITY_MISMATCH")
    body = {k: v for k, v in study.items() if k != "study_sha256"}
    if study["study_sha256"] != _digest(body):
        _fail("R99_FROZEN_GRAPH_SHA256_MISMATCH")
    if study.get("evidence_scope") != "SOURCE_TRAIN_INTERNAL_GRAPH_SEARCH_ONLY":
        _fail("R99_NON_SOURCE_DESIGN_NOT_ALLOWED")
    if source_report.get("independent_diagnostic_generalization_proven") is not False:
        _fail("R99_EXTERNAL_VALIDATION_CLAIM_FROM_SOURCE_ONLY_REPORT")
    if source_report.get("production_approved") is not False:
        _fail("R99_SOURCE_REPORT_CANNOT_APPROVE_PRODUCTION")
    graphs = study["graphs"]
    if not isinstance(graphs, list) or not graphs:
        _fail("R99_EMPTY_SOURCE_DESIGN")
    for graph in graphs:
        if graph.get("recipe") not in REASONS:
            _fail("R99_UNREGISTERED_STRUCTURE")
    if study.get("screening_refusal"):
        if (source_report.get("status") != "ABSTAIN_SOURCE_EVIDENCE_INSUFFICIENT"
                or source_report.get("source_fit_calls_actual") != 0
                or source_report.get("evaluated_graphs") != []):
            _fail("R99_INVALID_SOURCE_ABSTENTION")
        return []
    if source_report.get("status") != "SOURCE_TRAIN_INTERNAL_PHYSICAL_GRAPH_COMPARISON":
        _fail("R99_NOT_A_COMPLETED_SOURCE_COMPARISON")
    actual = source_report.get("evaluated_graphs")
    if (not isinstance(actual, list) or len(actual) != len(graphs)
            or source_report.get("source_fit_calls_actual")
            != study["total_source_fit_calls_planned"]):
        _fail("R99_SOURCE_REPORT_NOT_MATCHING_FROZEN_DESIGNS")
    flattened = []
    for expected, observed in zip(graphs, actual):
        if (expected["graph_id"] != observed.get("graph_id")
                or expected["recipe"] != observed.get("recipe")
                or observed.get("independent_validation_qualified") is not False):
            _fail("R99_GRAPH_SOURCE_REPORT_MISMATCH")
        receipt = observed.get("source_only_pipeline_comparison")
        if (not isinstance(receipt, Mapping)
                or receipt.get("study_sha256")
                != expected["pipeline_plan"]["study_sha256"]
                or receipt.get("status") != "SOURCE_GROUP_OOF_PIPELINE_SHORTLIST_ONLY"
                or receipt.get("independent_model_validated") is not False):
            _fail("R99_MISMATCHED_OR_UNTRUSTED_SOURCE_PIPELINE")
        expected_proposals = expected["pipeline_plan"]["pipeline_designs"]
        candidates = receipt.get("candidates")
        if (not isinstance(candidates, list)
                or len(candidates) != len(expected_proposals)
                or receipt.get("actual_fit_calls")
                != expected["pipeline_plan"]["planned_source_fit_calls"]):
            _fail("R99_MISSING_OR_EXTRA_SOURCE_PIPELINE_RESULT")
        for design, candidate in zip(expected_proposals, candidates):
            for key in ("pipeline_id", "algorithm_family",
                        "input_channels", "preprocessing", "hyperparameters"):
                if candidate.get(key) != design.get(key):
                    _fail("R99_MODEL_OR_PREPROCESSOR_UNREGISTERED_IN_REPORT")
            for k in FAILURE_THRESHOLDS:
                if not _number(candidate.get(k), 0, 1):
                    _fail("R99_BAD_SOURCE_PERFORMANCE_METRIC")
            if (not _number(candidate.get("sum_observed_fit_seconds"), 0)
                    or type(candidate.get("total_provider_fit_calls")) is not int
                    or candidate["total_provider_fit_calls"] < 1
                    or candidate.get("source_oof_only") is not True):
                _fail("R99_BAD_COST_OR_SOURCE_PROVENANCE")
            flattened.append((expected, candidate))
    return flattened


def _failure_codes(candidate):
    found = []
    for metric, minimum in list(FAILURE_THRESHOLDS.items())[:2]:
        if candidate[metric] < minimum:
            found.append("POOR_"+metric.upper())
    if candidate["source_healthy_false_alarm_fraction"] > (
            FAILURE_THRESHOLDS["source_healthy_false_alarm_fraction"]):
        found.append("HIGH_HEALTHY_FALSE_ALARM")
    return found


def _frontier(items):
    """Dominance on descriptive accuracy, source healthy FAR and FIT TIME.

    Excludes FFT/feature time, memory, inference cost and mismatched hardware:
    never call it a matched total-cost Pareto frontier.
    """
    frontier = []
    for row in items:
        cost = row["cost_scope"]["sum_provider_fit_seconds_observed"]
        dominated = False
        for other in items:
            if row is other:
                continue
            better_equal = (
                other["source_worst_group_accuracy"] >= row["source_worst_group_accuracy"]
                and other["source_healthy_false_alarm_fraction"] <=
                row["source_healthy_false_alarm_fraction"]
                and other["cost_scope"]["sum_provider_fit_seconds_observed"] <= cost
            )
            strictly_better = (
                other["source_worst_group_accuracy"] > row["source_worst_group_accuracy"]
                or other["source_healthy_false_alarm_fraction"] <
                row["source_healthy_false_alarm_fraction"]
                or other["cost_scope"]["sum_provider_fit_seconds_observed"] < cost
            )
            if better_equal and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(row["candidate_key"])
    return sorted(frontier)


def explain_source_design(study, report):
    """Explain WHY predeclared physical graphs exist and WHY a model failed.

    Revisions may only reference already proposed alternatives, and must be
    independently validated later. They do not train a revised candidate or
    consume hidden validation labels. Costs are explicitly partial.
    """
    flattened = _validate(study, report)
    rationale = [{
        "graph_id": graph["graph_id"],
        "recipe": graph["recipe"],
        "physical_prerequisite_reasons": list(REASONS[graph["recipe"]]),
        "inputs": list(graph["derived_trial"]["input_channels"]),
        "algorithm_families": list(graph["derived_trial"]["candidate_families"]),
        "physical_conditions_independently_authenticated": False,
        "empirical_failure_causality_proven": False,
    } for graph in study["graphs"]]
    if not flattened:
        result = {
            "schema": SCHEMA, "study_sha256": study["study_sha256"],
            "status": "ABSTAIN_SOURCE_UNSUPPORTED",
            "why_generated": rationale,
            "failure_audit": [], "partial_cost_pareto": [],
            "bounded_revisions": [],
            "source_failure_reason": study["screening_refusal"],
            "separate_external_validation_required": True,
            "evidence_scope": "SOURCE_ONLY_NOT_INDEPENDENT",
            "algorithm_improvement_independently_proven": False,
        }
        return {**result, "explanation_sha256": _digest(result)}
    ledger = []
    for graph, candidate in flattened:
        key = graph["graph_id"]+"/"+candidate["pipeline_id"]
        failures = _failure_codes(candidate)
        ledger.append({
            "candidate_key": key,
            "recipe": graph["recipe"],
            "algorithm_family": candidate["algorithm_family"],
            "preprocessing": candidate["preprocessing"],
            "source_worst_group_accuracy": candidate["source_worst_group_accuracy"],
            "source_group_macro_accuracy": candidate["source_group_macro_accuracy"],
            "source_healthy_false_alarm_fraction":
                candidate["source_healthy_false_alarm_fraction"],
            "failures": failures,
            "causal_explanation": "not_identified_from_source_oof",
            "cost_scope": {
                "source_fit_calls": candidate["total_provider_fit_calls"],
                "sum_provider_fit_seconds_observed":
                    candidate["sum_observed_fit_seconds"],
                "feature_extraction_cost_included": False,
                "hardware_matched": False,
                "peak_memory_measured": False,
                "end_to_end_inference_cost_measured": False,
            },
        })
    baseline = [x for x in ledger if x["recipe"] == BASELINE]
    if not baseline:
        _fail("R99_MISSING_REGISTERED_BASELINE")
    best_base = sorted(baseline, key=lambda x: (
        len(x["failures"]), -x["source_worst_group_accuracy"],
        x["source_healthy_false_alarm_fraction"], x["candidate_key"]))[0]
    revisions = []
    for entry in ledger:
        if not entry["failures"]:
            continue
        if entry["recipe"] != BASELINE:
            action = "consider_predeclared_lower_complexity_baseline"
            target = best_base["candidate_key"]
        elif "HIGH_HEALTHY_FALSE_ALARM" in entry["failures"]:
            action = "compare_predeclared_alternative_with_lower_source_far"
            options = [x for x in ledger if
                       x["source_healthy_false_alarm_fraction"] <
                       entry["source_healthy_false_alarm_fraction"]]
            target = (sorted(options, key=lambda x: (
                x["source_healthy_false_alarm_fraction"],
                x["candidate_key"]))[0]["candidate_key"] if options else None)
        else:
            action = "seek_new_independent_source_groups_before_complexity_growth"
            target = None
        revisions.append({
            "failed_candidate_key": entry["candidate_key"],
            "observed_failures": entry["failures"],
            "allowed_action": action,
            "candidate_already_in_frozen_study": target,
            "new_model_architecture_implemented": False,
            "threshold_or_alarm_postprocessing_changed": False,
            "counts_as_new_independent_evidence": False,
            "must_refreeze_and_collect_new_external_groups_before_claim": True,
        })
    result = {
        "schema": SCHEMA, "study_sha256": study["study_sha256"],
        "status": "DESCRIPTIVE_SOURCE_EXPLANATION_ONLY",
        "why_generated": rationale,
        "failure_audit": ledger,
        "partial_cost_pareto": _frontier(ledger),
        "bounded_revisions": revisions,
        "source_failure_reason": None,
        "fair_compute_cost_measured": False,
        "feature_extract_time_measured": False,
        "separate_external_validation_required": True,
        "evidence_scope": "SOURCE_ONLY_NOT_INDEPENDENT",
        "algorithm_improvement_independently_proven": False,
    }
    return {**result, "explanation_sha256": _digest(result)}
