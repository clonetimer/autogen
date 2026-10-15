"""R81c executed multi-atom feature-programs: synthetic TRAIN-only regressions.

Feature-program composition is part of the SAME bounded candidate selection.
No simulator metadata, truth, or FINAL data are used as deployment inputs.
"""
import copy
import json

import numpy as np
import pytest

from phm_design.open_design import (
    Program, ObservationSpec, generic_registry,
    temporal_bins, random_temporal_kernels,
)
from phm_design.industrial.taskspec import TaskSpecRejected, compile_task_spec
from phm_design.industrial.artifact_io import load_trusted_artifact
from phm_design.industrial.design_engine import (
    DesignAcceptance, IndustrialDesignRejected, run_design, _features,
)
from phm_design.industrial.feature_program import (
    decode_program, encode_program, execute_program, program_digest,
)
from test_r81a_unified_design_engine import task, synthetic_records


FUNCTIONAL = DesignAcceptance(
    accuracy_min=0., worst_environment_class_recall_min=0.,
    healthy_false_alarm_max=1.,
)


def combined_spec(source="industrial_bearing_classification_demo.json",
                  *, topology="direct", atoms=("temporal_bins", "random_kernels")):
    spec = task(source, compare=("linear",))
    spec["design"]["feature_programs"] = [
        {"atoms": list(atoms), "composition": "concat"},
    ]
    spec["design"]["topologies"] = [topology]
    spec["design"]["budget"]["max_fits"] = 64
    return spec


def test_program_compiler_emits_bounded_shared_trial_with_canonical_identity():
    spec = combined_spec(atoms=("random_kernels", "temporal_bins"))
    plan = compile_task_spec(spec)
    assert len(plan["candidate_grid"]) == 1
    trial = plan["candidate_grid"][0]
    assert trial["representation"] == "feature_program"
    assert trial["feature_program"] == encode_program(("temporal_bins", "random_kernels"))
    assert decode_program(trial["feature_program"]) == ("random_kernels", "temporal_bins")
    assert plan["registered_feature_programs"] == [
        {"atoms": ["random_kernels", "temporal_bins"],
         "composition": "concat",
         "identity": trial["feature_program"]}
    ]
    assert len(program_digest(trial["feature_program"])) == 64
    assert plan["independent_qualification"] is False


def test_executor_reuses_actual_existing_generic_feature_registry():
    rows = synthetic_records(combined_spec(), environment_count=3)[0]["measurements"]
    sequence = np.asarray([[r["sample_time_s"], r["vibration"], r["motor_current"]]
                           for r in rows])
    program_id = encode_program(("random_kernels", "temporal_bins"))
    computed = execute_program(sequence, program_id)
    expected = np.concatenate([random_temporal_kernels(sequence),
                               temporal_bins(sequence)])
    np.testing.assert_allclose(computed, expected)
    np.testing.assert_allclose(_features(sequence, program_id), expected)
    reference = generic_registry().transform(
        Program(("random_kernels", "temporal_bins"), learner="linear"),
        sequence, ObservationSpec("test", ("time_s", "vibration", "current")))
    np.testing.assert_allclose(reference, computed)


def test_combined_program_runs_real_fold_fits_and_deployment_without_labels(tmp_path):
    spec = combined_spec()
    rows = synthetic_records(spec, environment_count=3)
    artifact = run_design(spec, rows, acceptance=FUNCTIONAL)
    assert artifact.trial.feature == encode_program(("temporal_bins", "random_kernels"))
    assert artifact.audit["total_model_fits"] == 4
    assert artifact.audit["fold_model_fits"] == 3
    assert artifact.audit["attempts"][0]["status"] == "ELIGIBLE"
    assert artifact.audit["attempts"][0]["feature_program_sha256"] == program_digest(
        artifact.trial.feature)
    assert artifact.predict(rows[1]["measurements"])[0] in spec["task"]["labels"]
    artifact.export(tmp_path / "feature_program_model")
    audit = json.loads((tmp_path / "feature_program_model" / "audit.json").read_text())
    assert audit["feature_program_sha256"] == program_digest(artifact.trial.feature)
    assert len(audit["trial_selection_audit"]["selected_trial_sha256"]) == 64
    assert audit["candidate_grid_sha256"] == artifact.audit["candidate_grid_sha256"]
    assert audit["training_manifest_sha256"] == artifact.audit["training_manifest_sha256"]
    assert audit["independent_qualification"] is False
    assert audit["runtime_fingerprint"]["packages"]["numpy"]
    assert audit["runtime_fingerprint"]["packages"]["scikit-learn"]
    assert audit["runtime_fingerprint"]["provenance_authenticated"] is False
    with pytest.raises(ValueError, match="EXPLICIT_TRUST_REQUIRED"):
        load_trusted_artifact(tmp_path / "feature_program_model")
    restored = load_trusted_artifact(tmp_path / "feature_program_model",
                                     trust_pickle=True)
    assert restored.predict(rows[1]["measurements"]) == artifact.predict(
        rows[1]["measurements"])
    artifact_folder = tmp_path / "feature_program_model"
    manifest_path = artifact_folder / "audit.json"
    before = manifest_path.read_text()
    altered = json.loads(before)
    altered["candidate_grid_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="ARTIFACT_AUDIT_HASH_MISMATCH"):
        load_trusted_artifact(artifact_folder, trust_pickle=True)
    manifest_path.write_text(before)
    with (artifact_folder / "model.joblib").open("ab") as handle:
        handle.write(b"altered")
    with pytest.raises(ValueError, match="ARTIFACT_MODEL_HASH_MISMATCH"):
        load_trusted_artifact(artifact_folder, trust_pickle=True)


def test_factorized_models_take_exact_same_composed_features_and_budget():
    spec = combined_spec("satellite_eps_joint_demo.json", topology="factorized",
                         atoms=("temporal_bins", "difference_kernels"))
    rows = synthetic_records(spec, environment_count=3)
    artifact = run_design(spec, rows, acceptance=FUNCTIONAL)
    assert artifact.trial.topology == "factorized"
    assert artifact.model.fit_count_ == 3
    assert artifact.audit["total_model_fits"] == 12
    assert artifact.predict(rows[0]["measurements"])[0] in spec["task"]["labels"]


@pytest.mark.parametrize("config,code", [
    ({"atoms": ["future_oracle"], "composition": "concat"},
     "UNREGISTERED_FEATURE_PROGRAM"),
    ({"atoms": ["temporal_bins", "temporal_bins"], "composition": "concat"},
     "INVALID_OR_DUPLICATE_ITEMS"),
    ({"atoms": ["temporal_bins"] * 4, "composition": "concat"},
     "INVALID_OR_DUPLICATE_ITEMS"),
    ({"atoms": ["temporal_bins"], "composition": "available_stack"},
     "UNSUPPORTED_INDUSTRIAL_PROGRAM_COMPOSITION"),
    ({"atoms": ["temporal_bins"], "composition": "available_average"},
     "UNSUPPORTED_INDUSTRIAL_PROGRAM_COMPOSITION"),
    ({"atoms": ["temporal_bins"], "composition": "concat", "unsafe": "eval"},
     "UNKNOWN_TASKSPEC_FIELDS"),
])
def test_nonregistered_or_unimplemented_program_shapes_fail_before_fitting(config, code):
    spec = combined_spec()
    spec["design"]["feature_programs"] = [config]
    with pytest.raises(TaskSpecRejected, match=code):
        compile_task_spec(spec)


def test_duplicate_programs_and_too_small_grid_fail_closed():
    spec = combined_spec()
    spec["design"]["feature_programs"].append({
        "atoms": ["random_kernels", "temporal_bins"], "composition": "concat"})
    with pytest.raises(TaskSpecRejected, match="DUPLICATE_INDUSTRIAL_FEATURE_PROGRAM"):
        compile_task_spec(spec)
    spec = combined_spec()
    spec["design"]["representations"] = ["difference_kernels"]
    spec["design"]["candidate_max"] = 1
    with pytest.raises(TaskSpecRejected, match="SEARCH_SPACE_EXCEEDS_PREDECLARED_TRIAL_BUDGET"):
        compile_task_spec(spec)


def test_final_role_or_truth_never_enters_composed_training_or_inference():
    spec = combined_spec()
    records = synthetic_records(spec, environment_count=3)
    records[0]["metadata"]["role"] = "FINAL_OOD"
    with pytest.raises(TaskSpecRejected, match="DESIGN_WORKER_PROTECTED_ROLE"):
        run_design(spec, records, acceptance=FUNCTIONAL)
    records = synthetic_records(spec, environment_count=3)
    records[0]["measurements"][0]["truth"] = 42
    with pytest.raises(TaskSpecRejected, match="UNDECLARED_OR_PROTECTED_OBSERVATION"):
        run_design(spec, records, acceptance=FUNCTIONAL)


def test_program_identifier_and_runtime_matrix_must_be_exact():
    with pytest.raises(ValueError, match="NONCANONICAL_INDUSTRIAL_FEATURE_PROGRAM"):
        decode_program("program:concat:temporal_bins+random_kernels")
    with pytest.raises(ValueError, match="UNREGISTERED_OR_DUPLICATE_INDUSTRIAL_ATOMS"):
        encode_program(("calculated_final_label",))
    with pytest.raises(ValueError, match="INVALID_FEATURE_PROGRAM_MATRIX"):
        execute_program(np.ones((8, 1)), encode_program(("temporal_bins",)))
