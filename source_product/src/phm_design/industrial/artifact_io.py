"""Research artifact integrity checks and explicitly trusted inference replay.

SHA-256 checks detect accidental corruption. They are NOT signatures, FINAL
provenance validation, or protection against an attacker replacing all files.
joblib/pickle is unsafe for untrusted files; load is opt-in.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from .taskspec import compile_task_spec
from .feature_program import program_digest


def load_trusted_artifact(directory, *, trust_pickle=False):
    """Restore a previously exported TRAIN-research model, not retrain or certify."""
    if trust_pickle is not True:
        raise ValueError("EXPLICIT_TRUST_REQUIRED_FOR_PICKLE")
    from .calibration import TemperatureSelectiveCalibrator
    from .design_engine import (
        IndustrialDesignArtifact, TrialSpec, _candidate, _sha,
    )

    directory = Path(directory)
    manifest_path = directory / "audit.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        audit = manifest["trial_selection_audit"]
        source = audit["source_task_spec"]
        trial_data = manifest["trial"]
        trial = TrialSpec(**trial_data)
        plan = compile_task_spec(source)
        policy = audit["acceptance_policy"]
        if manifest["trial_selection_audit_sha256"] != _sha(audit):
            raise ValueError("ARTIFACT_FULL_AUDIT_HASH_MISMATCH")
        if (audit["decision"] != "TRAIN_ONLY_SELECTION_REFITTED" or
                audit["model_delivered"] is not True or
                audit["model_qualified"] is not False or
                audit["target_feedback_used"] is not False or
                audit["final_evaluation_performed"] is not False or
                audit["independent_qualification"] is not False):
            raise ValueError("ARTIFACT_RELEASE_STATE_MISMATCH")
        if (audit["training_observations_hash_contract"] !=
                "LENGTH_PREFIXED_FLOAT64_LE_V1" or
                not isinstance(audit["training_record_count"], int) or
                audit["training_record_count"] <= 0):
            raise ValueError("ARTIFACT_TRAIN_SOURCE_DIGEST_CONTRACT")
        if (audit["search_budget"] != plan["budget"] or
                audit["budget_enforcement"]["wall_deadline"] != (
                    "PROCESS_ISOLATED_PARENT_TERMINATION"
                    if "max_wall_seconds" in plan["budget"] else "NOT_CONFIGURED")):
            raise ValueError("ARTIFACT_BUDGET_CONTRACT_MISMATCH")
        if (manifest["schema"] != "phm.industrial.design-artifact.v0" or
                manifest["model_file"] != "model.joblib" or
                manifest["independent_qualification"] is not False or
                manifest["development_selection_only"] is not True or
                manifest["trained"] is not True):
            raise ValueError("UNSUPPORTED_RESEARCH_ARTIFACT")
        if (manifest["task_spec_sha256"] != plan["task_spec_sha256"] or
                manifest["candidate_grid_sha256"] != _sha(plan["candidate_grid"]) or
                manifest["acceptance_policy_sha256"] != _sha(policy) or
                manifest["calibration_policy_sha256"] != (
                    _sha(plan["calibration_policy"])
                    if plan["calibration_policy"] is not None else None)):
            raise ValueError("ARTIFACT_AUDIT_HASH_MISMATCH")
        if (audit["selected_trial"] != trial_data or
                audit["selected_trial_sha256"] != _sha(trial_data) or
                audit["source_task_spec_sha256"] != plan["task_spec_sha256"] or
                audit["candidate_grid_sha256"] != manifest["candidate_grid_sha256"] or
                audit["training_manifest_sha256"] != manifest["training_manifest_sha256"] or
                audit["training_observations_sha256"] !=
                manifest["training_observations_sha256"]):
            raise ValueError("ARTIFACT_SOURCE_AUDIT_INCONSISTENT")
        if not any(_sha(asdict(_candidate(candidate, plan["budget"]["seed"] + i)))
                   == _sha(trial_data)
                   for i, candidate in enumerate(plan["candidate_grid"])):
            raise ValueError("ARTIFACT_TRIAL_NOT_IN_DECLARED_SEARCH_SPACE")
        expected_program_hash = (
            program_digest(trial.feature)
            if trial.feature.startswith("program:") else None)
        runtime = manifest["runtime_fingerprint"]
        from importlib.metadata import PackageNotFoundError, version
        for package in ("scikit-learn", "numpy", "joblib"):
            try:
                installed = version(package)
            except PackageNotFoundError as exc:
                raise ValueError("ARTIFACT_RUNTIME_DEPENDENCY_MISSING") from exc
            if runtime["packages"][package] != installed:
                raise ValueError("ARTIFACT_RUNTIME_DEPENDENCY_MISMATCH:" + package)
        if runtime["provenance_authenticated"] is not False:
            raise ValueError("ARTIFACT_PROVENANCE_UNSUPPORTED")
        if manifest["feature_program_sha256"] != expected_program_hash:
            raise ValueError("ARTIFACT_FEATURE_PROGRAM_HASH_MISMATCH")
        calibration = manifest["calibration_state"]
        if calibration != audit["calibration_state"]:
            raise ValueError("ARTIFACT_CALIBRATION_AUDIT_MISMATCH")
        if bool(calibration is not None) != bool(plan["calibration_policy"] is not None):
            raise ValueError("ARTIFACT_CALIBRATION_POLICY_MISMATCH")
        cal = (TemperatureSelectiveCalibrator(**calibration)
               if calibration is not None else None)
        if cal is not None and not (
                0 < cal.temperature <= 10 and 0 <= cal.threshold <= 1
                and cal.calibration_samples >= plan["calibration_policy"]["min_calibration_samples"]):
            raise ValueError("INVALID_ARTIFACT_CALIBRATION_STATE")
        model_path = directory / "model.joblib"
        observed_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
        if observed_sha != manifest["model_sha256"]:
            raise ValueError("ARTIFACT_MODEL_HASH_MISMATCH")
    except (KeyError, TypeError, json.JSONDecodeError, UnicodeError) as exc:
        raise ValueError("CORRUPTED_RESEARCH_ARTIFACT_MANIFEST") from exc

    # Only after declared trusted input and all pre-deserialization checks.
    import joblib
    model = joblib.load(model_path)
    return IndustrialDesignArtifact(
        model=model, trial=trial, plan=plan, acceptance=policy,
        audit=audit, calibrator=cal)
