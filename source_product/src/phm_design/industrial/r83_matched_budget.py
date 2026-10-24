"""R83b matched-fit SOURCE TRAIN research comparison (not an OOD/FINAL test).

Fixed portfolio: TCN, GRU and ExtraTrees, one registered candidate each.
Static neural ablation: three predeclared TCN topologies.
Evolving portfolio: two shared seeds plus a feedback-proposed third topology.
All portfolios use exactly three
source environment OOF folds per candidate plus (only if eligible) one
TRAIN refit, with the SAME fit cap and acceptance policy. Parameter counts,
epochs/gradient steps for trees, CPU cost, and FLOPs are NOT equivalent.
"""
from __future__ import annotations

from dataclasses import asdict
import copy
import hashlib

import numpy as np
import json
from pathlib import Path
import time
from typing import Mapping, Sequence

from .design_engine import DesignAcceptance, IndustrialDesignRejected, _sha
from .resource_guard import GuardedDesignTimeout, run_design_guarded
from .self_organizing import EvolutionPolicy, _risk
from .self_organizing_guard import run_self_organizing_design_guarded
from .taskspec import TaskSpec, TaskSpecRejected, compile_task_spec, validate_observations
from .source_intake import stage_train_only_view, intake_train_source
from ..io import read_json, write_json, sha_file

SCHEMA = "phm.r83.matched-source-train-study.v1"
FIXED = ("tcn", "gru", "tree")
ARM_COUNT = 3


def _preflight(task_spec, records, policy):
    if not isinstance(records, (list, tuple)) or not records:
        raise TaskSpecRejected("EMPTY_MATCHED_STUDY_TRAIN")
    raw = (task_spec.to_dict() if isinstance(task_spec, TaskSpec)
           else TaskSpec.from_dict(task_spec).to_dict())
    if policy != EvolutionPolicy(max_candidates=ARM_COUNT, epochs=8,
                                 max_parameters=30_000):
        raise ValueError("MATCHED_STUDY_REQUIRES_PREDECLARED_POLICY")
    if (raw["task"]["decision_mode"] != "offline_trajectory" or
            raw["task"]["kind"] not in ("fault_classification", "joint_diagnosis")):
        raise TaskSpecRejected("MATCHED_STUDY_OFFLINE_CLASSIFICATION_ONLY")
    design = raw["design"]
    if (design["objectives"] != ["ERM"] or
            design.get("topologies", ["direct"]) != ["direct"] or
            any(k in design for k in ("feature_programs",
                                      "adaptive_feature_search", "calibration")) or
            design.get("representations", ["temporal_bins"]) != ["temporal_bins"]):
        raise TaskSpecRejected("MATCHED_STUDY_REQUIRES_PLAIN_ERM_DIRECT")
    if "max_wall_seconds" not in design["budget"]:
        raise TaskSpecRejected("MATCHED_STUDY_REQUIRES_HARD_WALL_GUARD")
    if design["candidate_max"] < ARM_COUNT:
        raise TaskSpecRejected("MATCHED_STUDY_SOURCE_TRIAL_SLOTS_BELOW_THREE")
    metadata = []
    groups_by_environment = {}
    fingerprints = {}
    fields = [raw["observations"]["time"]["field"]] + [
        channel["name"] for channel in raw["observations"]["channels"]]
    for record in records:
        if not isinstance(record, Mapping) or set(record) != {"measurements", "metadata"}:
            raise TaskSpecRejected("INVALID_MATCHED_STUDY_RECORD")
        if not isinstance(record["metadata"], Mapping):
            raise TaskSpecRejected("INVALID_MATCHED_STUDY_METADATA")
        validate_observations(raw, record["measurements"])
        item = dict(record["metadata"])
        metadata.append(item)
        group = "|".join(str(item.get(k)) for k in raw["data"]["group_keys"])
        env = "|".join(str(item.get(k)) for k in raw["data"]["environment_axes"])
        if group in groups_by_environment and groups_by_environment[group] != env:
            raise TaskSpecRejected("PHYSICAL_GROUP_CROSSES_ENVIRONMENTS")
        groups_by_environment[group] = env
        matrix = np.asarray([[r[key] for key in fields]
                             for r in record["measurements"]], dtype="<f8")
        fingerprint = hashlib.sha256(np.ascontiguousarray(matrix).tobytes()).hexdigest()
        if fingerprint in fingerprints and fingerprints[fingerprint] != group:
            raise TaskSpecRejected("DUPLICATE_TRAJECTORY_ACROSS_PHYSICAL_GROUPS")
        fingerprints[fingerprint] = group
    plan = compile_task_spec(raw, train_manifest=metadata)
    group_cols = plan["train_data_contract"]["group_keys"]
    env_cols = plan["train_data_contract"]["environment_axes"]
    class_support = {}
    for item in metadata:
        environment = "|".join(item[key] for key in env_cols)
        physical_group = "|".join(item[key] for key in group_cols)
        class_support.setdefault(environment, {}).setdefault(
            item["label"], set()).add(physical_group)
    for environment, counts in class_support.items():
        if set(counts) != set(raw["task"]["labels"]):
            raise TaskSpecRejected("MATCHED_STUDY_INCOMPLETE_ENVIRONMENT_CLASSES")
        values = [len(counts[label]) for label in raw["task"]["labels"]]
        if min(values) < 2:
            raise TaskSpecRejected("MATCHED_STUDY_INSUFFICIENT_PHYSICAL_CLASS_GROUPS")
        if max(values) > 2 * min(values):
            raise TaskSpecRejected("MATCHED_STUDY_IMBALANCED_SOURCE_CLASS_GROUPS")
    cols = plan["train_data_contract"]["environment_axes"]
    envs = {"|".join(m[column] for column in cols) for m in metadata}
    if len(envs) < 3:
        raise TaskSpecRejected("MATCHED_STUDY_REQUIRES_THREE_SOURCE_REGIMES")
    fit_cap = ARM_COUNT * len(envs) + 1
    if plan["budget"]["max_fits"] < fit_cap:
        raise TaskSpecRejected("MATCHED_STUDY_SOURCE_FIT_BUDGET_TOO_SMALL")
    return raw, len(envs), fit_cap, {
        env: {label: len(groups) for label, groups in counts.items()}
        for env, counts in sorted(class_support.items())
    }


def _candidate_rows(audit, *, evolved):
    rows = []
    for attempt in audit.get("attempts", []):
        metrics = attempt.get("metrics")
        if not isinstance(metrics, Mapping):
            continue
        eligible = (attempt.get("eligible") if evolved else
                    attempt.get("acceptance", {}).get("eligible"))
        trial = ({"genome_sha256": attempt["genome_sha256"],
                  "genome": attempt["genome"]}
                 if evolved else attempt["trial"])
        rows.append({
            "candidate": trial,
            "metrics": {name: metrics[name] for name in
                        ("accuracy", "worst_environment_class_recall",
                         "healthy_false_alarm")},
            "predeclared_loss": _risk(metrics),
            "eligible": bool(eligible),
            "parameter_count": (attempt.get("parameter_count") if evolved else None),
            "model_fits": (None if evolved else attempt.get("model_fits")),
        })
    return rows


def _result(audit, *, evolved):
    rows = _candidate_rows(audit, evolved=evolved)
    eligible = [row for row in rows if row["eligible"]]
    winner = (min(eligible, key=lambda row: (
        row["predeclared_loss"],
        row["candidate"]["genome_sha256"] if evolved else _sha(row["candidate"])))
        if eligible else None)
    if winner is not None and audit.get("model_delivered") is True:
        selected = (audit.get("selected_genome_sha256") if evolved
                    else audit.get("selected_trial_sha256"))
        if selected is not None:
            expected = (winner["candidate"]["genome_sha256"] if evolved
                        else _sha(winner["candidate"]))
            if selected != expected:
                raise ValueError("MATCHED_STUDY_SELECTED_MODEL_DIFFERS_FROM_COMMON_RANK")
    return {
        "decision": audit.get("decision"),
        "source_train_candidate_count": len(rows),
        "attempts": rows,
        "best_eligible_source_train_candidate": winner,
        "model_delivered": audit.get("model_delivered") is True,
        "independent_qualification": False,
        "evaluation_is_independent": False,
        "charged_fits": (audit.get("charged_fits") if evolved
                         else audit.get("total_model_fits",
                                        audit.get("fold_model_fits"))),
    }


def run_matched_source_train_study(
    task_spec: Mapping | TaskSpec,
    records: Sequence[Mapping],
    *,
    policy: EvolutionPolicy = EvolutionPolicy(
        max_candidates=ARM_COUNT, epochs=8, max_parameters=30_000),
    acceptance: DesignAcceptance = DesignAcceptance(),
    artifacts_root: str | Path | None = None,
) -> dict:
    """Run two portfolio designs on verified TRAIN-only records.

    Both arms consume identical input records and source operating-regime
    folds, the same FIT CAP, separate identical wall budgets and exactly the
    same predeclared refusal screen. Shared candidate slots do NOT establish
    identical computational cost or unbiased generalization performance.
    """
    if not isinstance(acceptance, DesignAcceptance):
        raise ValueError("INVALID_ACCEPTANCE_POLICY")
    raw, env_count, fits, class_support = _preflight(task_spec, records, policy)
    frozen_root = Path(artifacts_root) if artifacts_root is not None else None
    if frozen_root is not None and frozen_root.exists():
        raise FileExistsError("R83_FROZEN_ARTIFACT_ROOT_EXISTS")
    fixed = copy.deepcopy(raw)
    fixed["design"]["compare"] = list(FIXED)
    fixed["design"]["candidate_max"] = ARM_COUNT
    fixed["design"]["budget"]["max_fits"] = fits
    evolved = copy.deepcopy(raw)
    evolved["design"]["compare"] = ["tcn"]
    evolved["design"]["candidate_max"] = ARM_COUNT
    evolved["design"]["budget"]["max_fits"] = fits
    # Both new TaskSpecs are validated BEFORE any training.
    train_manifest = [dict(row["metadata"]) for row in records]
    fixed_plan = compile_task_spec(fixed, train_manifest=train_manifest)
    evolving_plan = compile_task_spec(evolved, train_manifest=train_manifest)
    if (len(fixed_plan["candidate_grid"]) != ARM_COUNT or
            len(evolving_plan["candidate_grid"]) != 1):
        raise TaskSpecRejected("MATCHED_STUDY_UNEXPECTED_CANDIDATE_GRAMMAR")
    output = {
        "schema": SCHEMA,
        "source_task_spec_sha256": TaskSpec.from_dict(raw).sha256,
        "source_train_manifest_sha256": _sha(train_manifest),
        "source_data_role": "TRAIN_ONLY",
        "source_environment_count": env_count,
        "independent_physical_class_group_support": class_support,
        "minimum_physical_group_per_environment_class": 2,
        "maximum_within_environment_class_support_ratio": 2.0,
        "source_physical_group_axes": raw["data"]["group_keys"],
        "source_environment_axes": raw["data"]["environment_axes"],
        "candidate_budget_per_arm": ARM_COUNT,
        "fit_budget_per_arm": fits,
        "max_wall_seconds_per_arm": fixed_plan["budget"]["max_wall_seconds"],
        "acceptance_policy": asdict(acceptance),
        "fixed_search_spec_sha256": fixed_plan["task_spec_sha256"],
        "evolving_search_spec_sha256": evolving_plan["task_spec_sha256"],
        "loss_definition": "0.45*(1-accuracy)+0.40*(1-worst_env_class_recall)+0.15*healthy_false_alarm",
        "interpretation": "SOURCE_TRAIN_SEARCH_FEEDBACK_ONLY",
        "fairness": {
            "same_records": True, "same_group_folds": True,
            "equal_estimator_fit_cap": True, "equal_actual_fit_count_if_failed": False,
            "equal_candidate_slots": True,
            "equal_neural_epochs": True,
            "shared_predeclared_candidate_selection_rule": "r83_matched_train_loss",
            "shared_candidate_seed_schedule": "base_seed_plus_candidate_index",
            "same_independent_group_support_gate": True,
            "search_reuse_is_unbiased_evaluation": False,
            "equal_flops": False, "equal_cpu_seconds": False,
            "equal_parameter_count": False,
            "baseline_tree_neural_epochs_applicable": False,
            "independent_validation": False,
        },
        "source_train_test_or_final_accessed": False,
        "independent_qualification": False,
        "model_delivered_to_deployment": False,
        "arms": {},
        "ablation": {
            "comparison": "STATIC_TOPOLOGY_VS_FEEDBACK_TOPOLOGY",
            "same_neural_family": True,
            "same_two_initial_genomes": True,
            "same_neural_training_epochs": True,
            "equal_flops_or_parameter_count": False,
            "only_adaptive_arm_uses_source_oof_to_propose_third_genome": True,
        },
    }
    if frozen_root is not None:
        frozen_root.mkdir(parents=True)
    for name, spec, executor in (
        ("fixed_portfolio", fixed, lambda: run_design_guarded(
            fixed, records, acceptance=acceptance,
            selection_rule="r83_matched_train_loss")),
        ("static_topology", evolved, lambda: run_self_organizing_design_guarded(
            evolved, records, policy=policy, acceptance=acceptance,
            seed_schedule="r83_matched_train",
            proposal_mode="predeclared_static")),
        ("self_organizing", evolved, lambda: run_self_organizing_design_guarded(
            evolved, records, policy=policy, acceptance=acceptance,
            seed_schedule="r83_matched_train",
            proposal_mode="feedback")),
    ):
        started = time.monotonic()
        artifact = None
        try:
            artifact = executor()
            audit = artifact.audit
        except IndustrialDesignRejected as exc:
            audit = exc.audit
        except GuardedDesignTimeout as exc:
            audit = exc.audit
        arm_result = _result(audit, evolved=name != "fixed_portfolio")
        if artifact is not None and frozen_root is not None:
            archive_dir = frozen_root / name
            receipt = artifact.export(archive_dir)
            manifest_path = archive_dir / (
                "audit.json" if name == "fixed_portfolio" else "manifest.json")
            arm_result["frozen_research_artifact"] = {
                "relative_directory": name,
                "manifest_sha256": sha_file(manifest_path),
                "qualified": False,
            }
        arm_result["research_wall_seconds"] = round(time.monotonic() - started, 4)
        output["arms"][name] = arm_result
    output["complete_two_arm_comparison"] = set(output["arms"]) == {
        "fixed_portfolio", "static_topology", "self_organizing"}
    # A finished pair of processes is not automatically a fair comparison:
    # one arm might time out before its predeclared candidate slate completes.
    source_complete = all(
        (arm["source_train_candidate_count"] == ARM_COUNT and
         arm["charged_fits"] is not None and
         arm["charged_fits"] >= ARM_COUNT * env_count and
         arm["decision"] not in (
             "HARD_WALL_TIMEOUT", "SEARCH_FIT_BUDGET_EXHAUSTED",
             "SEARCH_CPU_BUDGET_EXHAUSTED", "BUDGET_EXHAUSTED_BEFORE_SELECTION"))
        for arm in output["arms"].values())
    output["fairness_gate"] = {
        "static_neural_ablation_present": "static_topology" in output["arms"],
        "all_declared_candidates_evaluated": bool(source_complete),
        "same_predeclared_selection_loss": True,
        "same_candidate_seed_schedule": True,
        "equal_fit_budget_but_not_equal_compute": True,
        "statistically_independent_performance_estimate": False,
        "source_train_comparison_allowed": bool(source_complete),
        "independent_generalization_claim_allowed": False,
        "minimum_observed_physical_groups_per_regime_class": min(
            min(classes.values()) for classes in class_support.values()),
        "minimum_recommended_groups_for_confirmatory_experiment": 5,
        "confirmatory_sample_support_met": all(
            min(classes.values()) >= 5 for classes in class_support.values()),
    }
    output["independent_qualification"] = False
    return output


def run_native_rw_matched_study(
    *, task_spec, development_manifest, physical_map, runtime_binding,
    native_source_checkout, basilisk_wheel_factory, output,
    validate_only=False,
):
    """Trusted R82 source staging + intake BEFORE either model worker sees data."""
    destination = Path(output)
    if destination.exists():
        raise FileExistsError("R83_MATCHED_STUDY_OUTPUT_EXISTS")
    source = read_json(development_manifest)
    if (source.get("provenance", {}).get("source_kind") != "native_basilisk" or
            source.get("contract", {}).get("device_family") !=
            "satellite.adcs.reaction_wheel.fixed_three_wheels" or
            source.get("contract", {}).get("evidence_type") != "simulation"):
        raise ValueError("R83_REQUIRES_REVIEWED_NATIVE_RW_SOURCE")
    raw = read_json(task_spec)
    if raw["equipment"]["adapter"] != "sat_rw.fixed_index.v1":
        raise ValueError("R83_RW_ADAPTER_MISMATCH")
    # Validate and freeze the three-slot study contract even in validate-only
    # mode; never silently grow an R82 two-candidate TaskSpec after intake.
    preliminary = compile_task_spec(raw)
    if (raw["design"]["candidate_max"] < ARM_COUNT or
            preliminary["budget"]["max_fits"] < ARM_COUNT * 3 + 1):
        raise TaskSpecRejected("MATCHED_STUDY_SOURCE_TRIAL_SLOTS_BELOW_THREE")
    destination.mkdir(parents=True)
    report = {
        "schema": SCHEMA,
        "status": "STARTED",
        "development_manifest_sha256": sha_file(Path(development_manifest)),
        "physical_map_sha256": sha_file(Path(physical_map)),
        "source_task_spec_sha256": TaskSpec.from_dict(raw).sha256,
        "model_delivered_to_deployment": False,
        "independent_qualification": False,
        "actual_basilisk_execution_independently_attested": False,
    }
    try:
        staged = stage_train_only_view(
            development_manifest, destination / "isolated_train"
        )
        records, intake = intake_train_source(
            raw, staged["manifest"], physical_map,
            native_binding_path=runtime_binding,
            native_source_checkout_root=native_source_checkout,
            native_wheel_factory_file=basilisk_wheel_factory,
        )
        if not intake.get("native_source_file_hashes_verified"):
            raise ValueError("R83_NATIVE_RUNTIME_SOURCE_BINDING_NOT_VERIFIED")
        report["source_intake"] = intake
        report["validation_telemetry_accessed"] = False
        report["status"] = "TRAIN_SOURCE_ACCEPTED"
        if not validate_only:
            report["comparison"] = run_matched_source_train_study(
                raw, records, artifacts_root=destination / "research_models")
            report["status"] = "TRAIN_SOURCE_SEARCH_RESEARCH_ONLY"
    except Exception as exc:
        report["status"] = "REFUSED"
        report["refusal_code"] = getattr(exc, "code", None) or type(exc).__name__
        write_json(destination / "research_report.json", report)
        raise
    write_json(destination / "research_report.json", report)
    return report


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    for field in (
        "task-spec", "development-manifest", "physical-map", "runtime-binding",
        "native-source-checkout", "basilisk-wheel-factory", "output",
    ):
        parser.add_argument("--" + field, required=True, type=Path)
    parser.add_argument("--validate-only", action="store_true")
    options = vars(parser.parse_args(argv))
    return run_native_rw_matched_study(
        task_spec=options["task_spec"],
        development_manifest=options["development_manifest"],
        physical_map=options["physical_map"],
        runtime_binding=options["runtime_binding"],
        native_source_checkout=options["native_source_checkout"],
        basilisk_wheel_factory=options["basilisk_wheel_factory"],
        output=options["output"],
        validate_only=options["validate_only"],
    )


if __name__ == "__main__":
    main()
