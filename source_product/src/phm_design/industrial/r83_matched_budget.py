"""R83b matched-fit SOURCE TRAIN research comparison (not an OOD/FINAL test).

Fixed portfolio: TCN, GRU and ExtraTrees, one registered candidate each.
Evolving portfolio: three proposed TCN topologies. Both use exactly three
source environment OOF folds per candidate plus (only if eligible) one
TRAIN refit, with the SAME fit cap and acceptance policy. Parameter counts,
epochs/gradient steps for trees, CPU cost, and FLOPs are NOT equivalent.
"""
from __future__ import annotations

from dataclasses import asdict
import copy
import json
from pathlib import Path
import time
from typing import Mapping, Sequence

from .design_engine import DesignAcceptance, IndustrialDesignRejected, _sha
from .resource_guard import GuardedDesignTimeout, run_design_guarded
from .self_organizing import EvolutionPolicy, _risk
from .self_organizing_guard import run_self_organizing_design_guarded
from .taskspec import TaskSpec, TaskSpecRejected, compile_task_spec
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
    metadata = []
    for record in records:
        if not isinstance(record, Mapping) or set(record) != {"measurements", "metadata"}:
            raise TaskSpecRejected("INVALID_MATCHED_STUDY_RECORD")
        if not isinstance(record["metadata"], Mapping):
            raise TaskSpecRejected("INVALID_MATCHED_STUDY_METADATA")
        metadata.append(dict(record["metadata"]))
    plan = compile_task_spec(raw, train_manifest=metadata)
    cols = plan["train_data_contract"]["environment_axes"]
    envs = {"|".join(m[column] for column in cols) for m in metadata}
    if len(envs) < 3:
        raise TaskSpecRejected("MATCHED_STUDY_REQUIRES_THREE_SOURCE_REGIMES")
    fit_cap = ARM_COUNT * len(envs) + 1
    if plan["budget"]["max_fits"] < fit_cap:
        raise TaskSpecRejected("MATCHED_STUDY_SOURCE_FIT_BUDGET_TOO_SMALL")
    return raw, len(envs), fit_cap


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
        _sha(row["candidate"]))) if eligible else None)
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
) -> dict:
    """Run two portfolio designs on verified TRAIN-only records.

    Both arms consume identical input records and source operating-regime
    folds, the same FIT CAP, separate identical wall budgets and exactly the
    same predeclared refusal screen. Shared candidate slots do NOT establish
    identical computational cost or unbiased generalization performance.
    """
    if not isinstance(acceptance, DesignAcceptance):
        raise ValueError("INVALID_ACCEPTANCE_POLICY")
    raw, env_count, fits = _preflight(task_spec, records, policy)
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
            "equal_flops": False, "equal_cpu_seconds": False,
            "equal_parameter_count": False,
            "baseline_tree_neural_epochs_applicable": False,
            "independent_validation": False,
        },
        "source_train_test_or_final_accessed": False,
        "independent_qualification": False,
        "model_delivered_to_deployment": False,
        "arms": {},
    }
    for name, spec, executor in (
        ("fixed_portfolio", fixed, lambda: run_design_guarded(
            fixed, records, acceptance=acceptance)),
        ("self_organizing", evolved, lambda: run_self_organizing_design_guarded(
            evolved, records, policy=policy, acceptance=acceptance)),
    ):
        started = time.monotonic()
        try:
            artifact = executor()
            audit = artifact.audit
        except IndustrialDesignRejected as exc:
            audit = exc.audit
        except GuardedDesignTimeout as exc:
            audit = exc.audit
        arm_result = _result(audit, evolved=name == "self_organizing")
        arm_result["research_wall_seconds"] = round(time.monotonic() - started, 4)
        output["arms"][name] = arm_result
    output["complete_two_arm_comparison"] = set(output["arms"]) == {
        "fixed_portfolio", "self_organizing"}
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
            report["comparison"] = run_matched_source_train_study(raw, records)
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
