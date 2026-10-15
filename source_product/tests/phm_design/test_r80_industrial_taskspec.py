"""R80 synthetic-only multi-equipment TaskSpec compiler/guard negative controls.

No R71/R78 opened target, hardware signals, or protected FINAL data accessed.
"""
import copy
import json
from pathlib import Path

import pytest

from phm_design.industrial import (
    TaskSpec, TaskSpecRejected, audit_train_manifest,
    compile_task_spec, validate_observations,
)

EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "industrial"


def case(name="satellite_eps_joint_demo.json"):
    return json.loads((EXAMPLES / name).read_text(encoding="utf-8"))


def bearer():
    return case("industrial_bearing_classification_demo.json")


def reject(source, error, **kwargs):
    with pytest.raises(TaskSpecRejected, match=error):
        compile_task_spec(source, **kwargs)


def rows(s, n=5):
    t = s["observations"]["time"]["field"]
    return [
        {t: float(i), **{c["name"]: float(i + j / 10)
                         for j, c in enumerate(s["observations"]["channels"])}}
        for i in range(n)
    ]


def manifest(spec, env_count=3, reps=2):
    groups = spec["data"]["group_keys"]
    axes = spec["data"]["environment_axes"]
    out = []
    for e in range(env_count):
        for label in spec["task"]["labels"]:
            for j in range(reps):
                gid = f"u{e}-{label}-{j}"
                row = {"role": "TRAIN", "label": label}
                for key in groups:
                    row[key] = gid
                for key in axes:
                    row[key] = f"regime{e}"
                out.append(row)
    return out


def test_satellite_joint_compiles_to_heterogeneous_plan_without_training():
    s = case()
    plan = compile_task_spec(s)
    assert plan["schema"] == "phm.industrial.autodesign-plan.v0"
    assert plan["task_kind"] == "joint_diagnosis"
    assert len(plan["ontology"]["labels"]) == 5
    assert plan["eligible_task_nodes"] == [
        "detection", "localization", "mechanism", "decision_gate"]
    assert plan["feature_contract"]["healthy_prefix_policy"] == "verified_if_required"
    assert len(plan["candidate_grid"]) == 6  # 2 baselines + 2 neural × 2 objectives
    assert plan["status"] == "PLAN_ONLY_SOURCE_DATA_NOT_AUDITED"
    assert plan["qualified"] is False and plan["target_data_used"] is False
    assert set(plan["train_data_contract"]["role_allowlist"]) == {"TRAIN"}
    assert "calibrate" in plan["prohibited_target_actions"]


def test_real_industrial_bearing_contract_compiles_without_healthy_prefix():
    s = bearer()
    plan = compile_task_spec(s)
    assert plan["equipment_adapter"] == "paderborn_experimental"
    assert plan["eligible_task_nodes"] == ["detection", "fault_classification"]
    assert len(plan["ontology"]["labels"]) == 3
    assert plan["feature_contract"]["healthy_prefix_policy"] == "not_assumed"
    assert "mechanism" not in plan["eligible_task_nodes"]
    assert plan["qualified"] is False


def test_binary_fresh_machine_ontology_needs_no_source_code_change():
    s = bearer()
    s["equipment"] = {"family": "hydraulic_actuator", "adapter": "generic_csv"}
    s["task"].update(kind="fault_detection", labels=["HEALTHY", "FAULTY"])
    s["task"]["structure"] = {"components": [], "mechanisms": [],
                              "class_attributes": {"FAULTY": {}}}
    plan = compile_task_spec(s)
    assert plan["eligible_task_nodes"] == ["detection"]
    assert len(plan["ontology"]["labels"]) == 2


def test_taskspec_canonical_hash_is_order_independent_and_source_not_mutable():
    a = case()
    b = dict(reversed(list(a.items())))
    x, y = TaskSpec.from_dict(a), TaskSpec.from_dict(b)
    assert x.sha256 == y.sha256
    a["task"]["labels"].append("EXTERNALLY_CHANGED")
    assert "EXTERNALLY_CHANGED" not in x.to_dict()["task"]["labels"]
    assert compile_task_spec(x)["task_spec_sha256"] == x.sha256


def test_strict_schema_and_unknown_nested_fields_fail_closed():
    s = case()
    s["extra"] = "ignored?"
    reject(s, "UNKNOWN_TASKSPEC_FIELDS")
    s = case()
    s["schema"] = "legacy"
    reject(s, "UNKNOWN_TASKSPEC_SCHEMA")
    s = case()
    s["observations"]["channels"][0]["infer_fault_class"] = True
    reject(s, "UNKNOWN_TASKSPEC_FIELDS")


@pytest.mark.parametrize("kind", [
    "fault_detection", "fault_classification", "component_localization",
    "mechanism_classification", "joint_diagnosis"])
def test_all_explicit_task_kinds_compile_without_custom_device_code(kind):
    s = case()
    s["task"]["kind"] = kind
    p = compile_task_spec(s)
    assert p["task_kind"] == kind
    assert p["eligible_task_nodes"][0] == "detection"


def test_rejects_duplicate_labels_unknown_healthy_label_missing_fault_mapping():
    s = case()
    s["task"]["labels"][1] = "NOMINAL"
    reject(s, "INVALID_OR_DUPLICATE_ITEMS")
    s = case()
    s["task"]["healthy_label"] = "NOT_IN_ONTOLOGY"
    reject(s, "INVALID_TASK_LABEL_ONTOLOGY")
    s = case()
    del s["task"]["structure"]["class_attributes"]["LOAD_A_EXTRA"]
    reject(s, "INCOMPLETE_OR_EXTRANEOUS_CLASS_ATTRIBUTES")


def test_fails_component_and_mechanism_semantics_before_training():
    s = case()
    s["task"]["structure"]["class_attributes"]["LOAD_A_EXTRA"]["component"] = "UNKNOWN"
    reject(s, "ILLEGAL_CLASS_COMPONENT")
    s = case()
    s["task"]["structure"]["class_attributes"]["LOAD_A_EXTRA"]["mechanism"] = "UNKNOWN"
    reject(s, "ILLEGAL_CLASS_MECHANISM")
    s = case()
    s["task"]["structure"]["class_attributes"]["LOAD_A_EXTRA"].pop("mechanism")
    reject(s, "MISSING_CLASS_MECHANISM")
    s = case()
    s["task"]["structure"]["components"] = []
    reject(s, "COMPONENT_ONTOLOGY_REQUIRED")


@pytest.mark.parametrize("field", [
    "label", "fault_label", "ground_truth", "target_next", "final_metric",
    "generation_family_id", "fault_onset", "world_id", "split_name"])
def test_inference_feature_truth_and_provenance_are_prohibited(field):
    s = bearer()
    s["observations"]["channels"][0]["name"] = field
    reject(s, "PROTECTED_FIELD_CANNOT_BE_INFERENCE_INPUT")


def test_command_is_a_valid_deployable_channel_but_metadata_role_is_not():
    assert compile_task_spec(case())["candidate_grid"]
    s = case()
    s["observations"]["channels"][0]["role"] = "metadata"
    reject(s, "UNSUPPORTED_TASKSPEC_VALUE")
    s = case()
    s["observations"]["channels"][0]["available_at_decision"] = False
    reject(s, "POST_DECISION_CHANNEL")


def test_protected_role_does_not_enter_design_worker():
    for role in ("FINAL_ID", "FINAL_OOD", "DEV_STRESS", "TEST"):
        s = case()
        s["data"]["role"] = role
        reject(s, "DESIGN_WORKER_PROTECTED_ROLE")


def test_design_rejects_no_refusal_and_unregistered_architecture():
    s = case()
    s["design"]["allow_refusal"] = False
    reject(s, "DESIGN_REFUSAL_MUST_BE_AVAILABLE")
    s = case()
    s["design"]["compare"] = ["oracle_fault"]
    reject(s, "UNREGISTERED_ARCHITECTURE_OR_TRAINING_OBJECTIVE")
    s = case()
    s["design"]["candidate_max"] = 1
    reject(s, "SEARCH_SPACE_EXCEEDS_PREDECLARED_TRIAL_BUDGET")
    s = case()
    s["design"]["budget"]["max_fits"] = 2
    reject(s, "SEARCH_BUDGET_FITS_BELOW_TRIALS")


def test_dg_not_allowed_on_non_differentiable_candidates():
    s = bearer()
    s["design"]["compare"] = ["linear", "tree"]
    s["design"]["objectives"] = ["GroupDRO", "V-REx"]
    reject(s, "AUTO_NO_VALID_PLAN")


def test_offline_mode_cannot_claim_causal_online_deployment():
    s = bearer()
    s["deployment"]["causal_online"] = True
    reject(s, "DECISION_MODE_DEPLOYMENT_MISMATCH")


def test_causal_window_requires_explicit_past_only_budget_and_samples():
    s = bearer()
    s["task"]["decision_mode"] = "causal_window"
    s["deployment"]["causal_online"] = True
    reject(s, "INVALID_INTEGER")
    s["observations"]["time"].update(window_samples=8, horizon_samples=1)
    reject(s, "FUTURE_SAMPLES_IN_CAUSAL_INPUT")
    s["observations"]["time"]["horizon_samples"] = 0
    plan = compile_task_spec(s)
    assert plan["feature_contract"]["causal_past_only"]
    assert plan["decision_mode"] == "causal_window"


def test_checked_observation_batches_require_only_allowed_measured_inputs():
    s = bearer()
    b = rows(s, n=5)
    assert validate_observations(s, b)["ground_truth_consumed"] is False
    b[1]["label"] = 1
    with pytest.raises(TaskSpecRejected, match="UNDECLARED_OR_PROTECTED_OBSERVATION"):
        validate_observations(s, b)
    b = rows(s)
    del b[1]["motor_current"]
    with pytest.raises(TaskSpecRejected, match="MISSING_DEPLOYMENT_CHANNEL"):
        validate_observations(s, b)


def test_nonmonotone_time_and_nonfinite_sensor_are_rejected():
    s = bearer()
    b = rows(s)
    b[2]["sample_time_s"] = b[1]["sample_time_s"]
    with pytest.raises(TaskSpecRejected, match="NONMONOTONE_TIMESTAMPS"):
        validate_observations(s, b)
    b = rows(s)
    b[1]["vibration"] = float("nan")
    with pytest.raises(TaskSpecRejected, match="NONFINITE_OR_INVALID_DEPLOYMENT_VALUE"):
        validate_observations(s, b)


def test_online_inference_cannot_see_future_or_overrun_window():
    s = bearer()
    s["task"]["decision_mode"] = "causal_window"
    s["deployment"]["causal_online"] = True
    s["observations"]["time"]["window_samples"] = 4
    b = rows(s, n=5)
    with pytest.raises(TaskSpecRejected, match="FUTURE_OR_UNBOUNDED_CAUSAL_OBSERVATIONS"):
        validate_observations(s, b)
    with pytest.raises(TaskSpecRejected, match="FUTURE_OR_UNBOUNDED_CAUSAL_OBSERVATIONS"):
        validate_observations(s, b, decision_timestamp=3.)
    with pytest.raises(TaskSpecRejected, match="INVALID_CAUSAL_DECISION_WINDOW"):
        validate_observations(s, b, decision_timestamp=4.)
    assert validate_observations(s, b[:4], decision_timestamp=3.)["causal"] is True


def test_healthy_prefix_is_per_task_not_a_global_product_assumption():
    s = case()
    s["observations"]["healthy_prefix_policy"] = "required"
    with pytest.raises(TaskSpecRejected, match="HEALTHY_PREFIX_NOT_VERIFIED"):
        validate_observations(s, rows(s))
    assert validate_observations(s, rows(s), healthy_prefix_verified=True)["rows"] == 5
    assert validate_observations(bearer(), rows(bearer()))["rows"] == 5


def test_train_manifest_support_two_equipment_ontologies_without_new_core():
    for s in [case(), bearer()]:
        m = manifest(s)
        audit = audit_train_manifest(s, m)
        assert audit["eligible"] and audit["dg_supported"]
        assert audit["environment_families"] == 3
        p = compile_task_spec(s, train_manifest=m)
        assert p["status"] == "TRAIN_PREFIT_READY_NOT_TRAINED"
        assert p["manifest_audit"]["physical_groups"] == len(m)
        assert p["qualified"] is False


def test_train_manifest_refuses_final_role_labels_and_physical_identity_drift():
    s = bearer()
    m = manifest(s)
    m[0]["role"] = "FINAL_OOD"
    reject(s, "DESIGN_WORKER_PROTECTED_ROLE", train_manifest=m)
    m = manifest(s)
    m[0]["label"] = "FUTURE_FAILURE"
    reject(s, "UNKNOWN_TRAINING_LABEL", train_manifest=m)
    m = manifest(s)
    m[1]["bearing_instance_id"] = m[0]["bearing_instance_id"]
    m[1]["speed_regime"] = "other_speed"
    reject(s, "PHYSICAL_GROUP_CROSSES_ENVIRONMENTS", train_manifest=m)


def test_bad_source_class_support_refuses_training_and_is_not_fabricated():
    s = bearer()
    m = [r for r in manifest(s) if r["label"] != "OUTER_DAMAGE"]
    reject(s, "UNSUPPORTED_TRAINING_CLASS_OR_GROUP_SUPPORT", train_manifest=m)


def test_insufficient_domain_support_refuses_groupdro_without_final_feedback():
    s = bearer()
    m = manifest(s, env_count=2)
    with pytest.raises(TaskSpecRejected, match="UNSUPPORTED_ENVIRONMENT_CLASS_SUPPORT_FOR_DG"):
        compile_task_spec(s, train_manifest=m)


def test_manifest_requires_group_metadata_and_consistent_role():
    s = bearer()
    m = manifest(s)
    m[0].pop("bearing_instance_id")
    reject(s, "MISSING_TASKSPEC_FIELDS", train_manifest=m)
    m = manifest(s)
    m[0]["untrusted_score"] = .99
    reject(s, "UNKNOWN_TASKSPEC_FIELDS", train_manifest=m)


def test_json_serialization_of_compiled_plan_and_no_final_access():
    p = compile_task_spec(case())
    assert json.loads(json.dumps(p, allow_nan=False))["qualified"] is False
    assert set(p["prohibited_target_actions"]) >= {
        "fit", "normalize", "select", "calibrate", "threshold_tune"}
    assert p["source_only_nested_selection"] is True


def test_rejects_unsupported_multifault_task_instead_of_faking_joint_diagnosis():
    s = case()
    s["task"]["multifault"] = True
    reject(s, "UNSUPPORTED_MULTIFAULT_TASK")


def test_rejects_unqualified_field_deployment_flag():
    s = bearer()
    s["deployment"]["prototype_only"] = False
    reject(s, "INDUSTRIAL_FIELD_QUALIFICATION_UNSUPPORTED")
