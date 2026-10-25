"""R85 matched source-TRAIN static-vs-feedback industrial family study.

Two actual guarded R81 model searches on identical TRAIN records, frozen
TaskSpec, acceptance, trial/fold cap and per-arm wall deadline.
Results cannot establish independent generalization or model superiority.
"""
from __future__ import annotations

from dataclasses import asdict
from time import monotonic
from typing import Mapping, Sequence

from .design_engine import DesignAcceptance, IndustrialDesignRejected
from .resource_guard import GuardedDesignTimeout, run_design_guarded
from .r85_family_policy import validate_pool
from .taskspec import (TaskSpec, TaskSpecRejected, compile_task_spec,
                       validate_observations)


def _capture(spec, records, *, mode, slots, acceptance):
    begun = monotonic()
    try:
        artifact = run_design_guarded(
            spec, records, acceptance=acceptance,
            selection_rule="r83_matched_train_loss",
            family_search_mode=mode, family_proposals=slots)
        audit = artifact.audit
        delivered = True
    except (IndustrialDesignRejected, GuardedDesignTimeout) as ex:
        audit, delivered = ex.audit, False
    selected = audit.get("selected_trial_sha256") if delivered else None
    selection_loss = None
    if selected is not None:
        for attempt in audit.get("attempts", []):
            if attempt.get("trial_sha256") == selected:
                metrics = attempt.get("metrics", {})
                if all(name in metrics for name in (
                    "accuracy", "worst_environment_class_recall",
                    "healthy_false_alarm")):
                    selection_loss = (
                        .45 * (1-metrics["accuracy"]) +
                        .40 * (1-metrics["worst_environment_class_recall"]) +
                        .15 * metrics["healthy_false_alarm"])
                break
    return {
        "mode": mode, "model_delivered": delivered,
        "decision": audit.get("decision"),
        "source_attempts": len(audit.get("attempts", [])),
        "source_oof_fit_count": audit.get("fold_model_fits"),
        "source_environment_count": audit.get("source_environment_count"),
        "total_fits": audit.get("total_model_fits"),
        "selected_trial_sha256": selected,
        "selected_train_oof_loss": selection_loss,
        "source_task_spec_sha256": audit.get("source_task_spec_sha256"),
        "train_manifest_sha256": audit.get("training_manifest_sha256"),
        "train_observations_sha256": audit.get("training_observations_sha256"),
        "actual_end_to_end_wall_seconds": round(monotonic() - begun, 4),
        "independent_test_performed": False, "deployable_qualification": False,
    }


def run_r85_source_ablation(task_spec: Mapping | TaskSpec,
                            train_records: Sequence[Mapping], *,
                            family_proposals: int = 3,
                            acceptance: DesignAcceptance = DesignAcceptance()) -> dict:
    """Neither compares on a protected test nor changes trained artifacts."""
    if not isinstance(acceptance, DesignAcceptance):
        raise ValueError("R85_INVALID_ACCEPTANCE")
    if type(family_proposals) is not int or not 3 <= family_proposals <= 6:
        raise ValueError("R85_INVALID_ABLATION_PROPOSAL_BUDGET")
    raw = (task_spec.to_dict() if isinstance(task_spec, TaskSpec)
           else TaskSpec.from_dict(task_spec).to_dict())
    if not isinstance(train_records, (tuple, list)) or not train_records:
        raise TaskSpecRejected("EMPTY_R85_ABLATION_TRAIN_RECORDS")
    metadata = []
    for record in train_records:
        if (not isinstance(record, Mapping) or
                set(record) != {"measurements", "metadata"} or
                not isinstance(record["metadata"], Mapping)):
            raise TaskSpecRejected("INVALID_R85_ABLATION_SOURCE_RECORD")
        validate_observations(raw, record["measurements"])
        metadata.append(dict(record["metadata"]))
    # Validate roles/group support before either spawned training worker.
    plan = compile_task_spec(raw, train_manifest=metadata)
    validate_pool(plan["candidate_grid"])
    if not plan["budget"].get("max_wall_seconds"):
        raise TaskSpecRejected("R85_ABLATION_REQUIRES_HARD_WALL")
    if len(plan["candidate_grid"]) < family_proposals:
        raise TaskSpecRejected("R85_ABLATION_PROPOSALS_EXCEED_POOL")
    arms = {}
    for mode in ("predeclared_static", "feedback"):
        arms[mode] = _capture(raw, train_records, mode=mode,
                              slots=family_proposals, acceptance=acceptance)
    same_input_hashes = (
        bool(arms["predeclared_static"]["train_manifest_sha256"])
        and all(arms["predeclared_static"][key] == arms["feedback"][key]
                for key in ("source_task_spec_sha256", "train_manifest_sha256",
                            "train_observations_sha256")))
    env_counts = [a["source_environment_count"] for a in arms.values()]
    same_fits = bool(
        all(type(count) is int and count >= 3 for count in env_counts)
        and len(set(env_counts)) == 1
        and all(a["source_oof_fit_count"] == family_proposals * env_counts[0]
                and a["total_fits"] == family_proposals * env_counts[0] + 1
                for a in arms.values()))
    all_completed = all(
        a["model_delivered"] and a["source_attempts"] == family_proposals
        for a in arms.values())
    comparison_valid = bool(same_input_hashes and same_fits and all_completed)
    return {
        "schema": "phm.r85.source-family-ablation.v1",
        "equipment_family": raw["equipment"]["family"],
        "candidate_pool_count": len(plan["candidate_grid"]),
        "candidate_slots_per_arm": family_proposals,
        "same_train_records": same_input_hashes,
        "equal_actual_fit_count": bool(same_fits),
        "all_declared_candidates_evaluated": all_completed,
        "source_train_descriptive_comparison_allowed": comparison_valid,
        "arms": arms,
        "max_wall_seconds_per_arm": plan["budget"]["max_wall_seconds"],
        "acceptance_policy": asdict(acceptance),
        "equal_flops": False, "equal_parameter_count": False,
        "independent_generalization_claim_allowed": False,
        "stable_promotion_eligible": False,
    }
