"""R82 ADCS reaction-wheel TRAIN-only native-source study CLI.

Consumes the *prepared* PHM six-channel development manifest, never opens a
TEST/FINAL tree or runs Basilisk itself. It makes a model-selection experiment
on TRAIN folds only and explicitly refuses to claim independent qualification.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..io import read_json, write_json, sha_file
from .design_engine import IndustrialDesignRejected
from .resource_guard import GuardedDesignTimeout, run_design_guarded
from .source_intake import intake_train_source, stage_train_only_view
from .taskspec import compile_task_spec

_SCHEME = "phm.industrial.r82-rw-train-study.v1"


def run_rw_train_study(*, task_spec, development_manifest, physical_map,
                       runtime_binding, native_source_checkout,
                       basilisk_wheel_factory, output, validate_only=False):
    """Return research receipt, or persist explicit refusal before raising.

    A success receipt means source-contract acceptance or research model fit,
    *not* independently verified Basilisk physics. The external steward must
    separately audit the actual campaign and installed Basilisk runtime.
    """
    output = Path(output)
    if output.exists():
        raise FileExistsError("R82_STUDY_OUTPUT_EXISTS")
    source = read_json(development_manifest)
    if (source.get("provenance", {}).get("source_kind") != "native_basilisk" or
            source.get("contract", {}).get("evidence_type") != "simulation" or
            source.get("contract", {}).get("device_family") !=
            "satellite.adcs.reaction_wheel.fixed_three_wheels"):
        raise ValueError("R82_REQUIRES_PREPARED_SIX_CHANNEL_NATIVE_RW")
    spec = read_json(task_spec)
    if spec["equipment"]["adapter"] != "sat_rw.fixed_index.v1":
        raise ValueError("R82_RW_TASKSPEC_ADAPTER_MISMATCH")
    plan = compile_task_spec(spec)
    if "max_wall_seconds" not in plan["budget"]:
        raise ValueError("R82_RW_HARD_WALL_BUDGET_REQUIRED")
    output.mkdir(parents=True)
    report = {
        "schema": _SCHEME, "status": "STARTED",
        "development_manifest_sha256": sha_file(Path(development_manifest)),
        "task_spec_sha256": plan["task_spec_sha256"],
        "physical_map_sha256": sha_file(Path(physical_map)),
        "native_source_binding_reviewed": False,
        "model_delivered": False,
        "independent_qualification": False,
        "actual_basilisk_execution_attested": False,
        "independent_final_evaluation": False,
    }
    try:
        staged = stage_train_only_view(
            development_manifest, output / "isolated_train")
        records, intake = intake_train_source(
            spec, staged["manifest"], physical_map,
            native_binding_path=runtime_binding,
            native_source_checkout_root=native_source_checkout,
            native_wheel_factory_file=basilisk_wheel_factory)
        if intake["source_kind_claimed"] != "native_basilisk":
            raise ValueError("R82_RW_NATIVE_ONLY")
        if intake["native_source_file_hashes_verified"] is None:
            raise ValueError("R82_RW_MISSING_SOURCE_HASH_RECEIPT")
        report["intake"] = intake
        report["native_source_binding_reviewed"] = bool(
            intake.get("native_source_file_hashes_verified"))
        report["status"] = "TRAIN_INTAKE_ACCEPTED"
        if not validate_only:
            # Use the declared nonzero acceptance policy in DesignAcceptance;
            # no test-only relaxed gate is injected by this CLI.
            model = run_design_guarded(spec, records)
            trained = output / "research_model"
            model.export(trained)
            report["status"] = "RESEARCH_TRAIN_FIT_ONLY"
            report["model_delivered"] = True
            report["research_model_audit_sha256"] = sha_file(
                trained / "audit.json")
            chosen = next(
                item for item in model.audit["attempts"]
                if item.get("trial") == model.audit["selected_trial"])
            report["source_oof_selection"] = {
                "selected_trial": model.audit["selected_trial"],
                "decision": model.audit["decision"],
                "source_heldout_metrics": chosen["metrics"],
                "independent_qualification": False,
            }
    except Exception as exc:
        report["status"] = "REFUSED"
        report["refusal_code"] = (getattr(exc, "code", None) or
                                  getattr(exc, "audit", {}).get("decision") or
                                  type(exc).__name__)
        report["model_delivered"] = False
        if isinstance(exc, IndustrialDesignRejected):
            # Preserve all SOURCE TRAIN OOF candidate metrics and refusal
            # diagnostics. The caller must not soften policy/selection gates
            # to turn a physical-domain refusal into an artificial PASS.
            report["design_refusal_audit"] = exc.audit
            report["source_train_oof_only"] = True
        elif isinstance(exc, GuardedDesignTimeout):
            report["resource_guard_audit"] = exc.audit
        write_json(output / "research_report.json", report)
        raise
    write_json(output / "research_report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("task-spec", "development-manifest", "physical-map",
                 "runtime-binding", "native-source-checkout",
                 "basilisk-wheel-factory", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--validate-only", action="store_true")
    options = vars(parser.parse_args(argv))
    validate = options.pop("validate_only")
    run_rw_train_study(
        task_spec=options["task_spec"],
        development_manifest=options["development_manifest"],
        physical_map=options["physical_map"],
        runtime_binding=options["runtime_binding"],
        native_source_checkout=options["native_source_checkout"],
        basilisk_wheel_factory=options["basilisk_wheel_factory"],
        output=options["output"], validate_only=validate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
