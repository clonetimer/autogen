"""R81a real fit/predict/source-only selection tests, using SYNTHETIC TRAIN data.

Two different equipment ontologies run the exact same executor. No final or
revealed R71/R78 data, no real bearing downloads and no Basilisk run.
"""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from phm_design.industrial import TaskSpecRejected
from phm_design.industrial.design_engine import (
    DesignAcceptance, IndustrialDesignRejected, _candidate, run_design,
)

EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "industrial"


def task(file_name, *, compare=("linear", "tree"), objectives=("ERM",)):
    obj = json.loads((EXAMPLES / file_name).read_text(encoding="utf-8"))
    obj["design"]["compare"] = list(compare)
    obj["design"]["objectives"] = list(objectives)
    obj["design"]["candidate_max"] = len(compare) * len(objectives)
    obj["design"]["budget"]["max_fits"] = 48
    return obj


def synthetic_records(spec, *, environment_count=4):
    rng = np.random.default_rng(819)
    labels = spec["task"]["labels"]
    cols = [x["name"] for x in spec["observations"]["channels"]]
    time_name = spec["observations"]["time"]["field"]
    out = []
    for env in range(environment_count):
        for class_id, label in enumerate(labels):
            for repeat in range(2):
                group_id = f"physical_{env}_{class_id}_{repeat}"
                metadata = {"role": "TRAIN", "label": label}
                for g in spec["data"]["group_keys"]:
                    metadata[g] = group_id
                for e in spec["data"]["environment_axes"]:
                    metadata[e] = f"regime_{env}"
                points = []
                for i in range(48):
                    t = float(i) / 24.
                    readings = {time_name: t}
                    for j, name in enumerate(cols):
                        # Stable fault-class-specific sensor response; nuisance
                        # environment offset is balanced across all labels.
                        readings[name] = float(
                            0.60 * class_id + .02 * env + .025 * j
                            + .007 * np.sin(2 * np.pi * t * (j + 1))
                            + rng.normal(0, .004))
                    points.append(readings)
                out.append({"measurements": points, "metadata": metadata})
    return out


def test_bearing_runs_actual_two_classical_models_and_predicts():
    spec = task("industrial_bearing_classification_demo.json")
    rec = synthetic_records(spec)
    artifact = run_design(spec, rec)
    assert artifact.audit["decision"] == "TRAIN_ONLY_SELECTION_REFITTED"
    assert artifact.audit["model_qualified"] is False
    assert artifact.audit["source_data_role"] == "TRAIN_ONLY"
    assert len(artifact.audit["attempts"]) == 2
    assert artifact.audit["total_model_fits"] == 9  # 2 x 4 folds + 1 refit
    assert all(t["status"] == "ELIGIBLE" for t in artifact.audit["attempts"])
    assert all(x["disjoint"] for t in artifact.audit["attempts"] for x in t["folds"])
    assert len(artifact.audit["attempts"][0]["folds"]) == 4
    pred = artifact.predict(rec[1]["measurements"])
    assert pred == [rec[1]["metadata"]["label"]]


def test_satellite_and_bearing_share_search_core_but_have_different_output_classes():
    sat = task("satellite_eps_joint_demo.json")
    bearing = task("industrial_bearing_classification_demo.json")
    a = run_design(sat, synthetic_records(sat))
    b = run_design(bearing, synthetic_records(bearing))
    assert a.plan["eligible_task_nodes"] != b.plan["eligible_task_nodes"]
    assert len(a.plan["ontology"]["labels"]) == 5
    assert len(b.plan["ontology"]["labels"]) == 3
    assert a.trial.feature == b.trial.feature == "temporal"
    assert a.audit["final_evaluation_performed"] is False
    assert b.audit["final_evaluation_performed"] is False


def test_one_neural_tcn_and_one_linear_use_same_train_heldout_protocol():
    spec = task("industrial_bearing_classification_demo.json",
                compare=("linear", "tcn"))
    rec = synthetic_records(spec)
    result = run_design(spec, rec, acceptance=DesignAcceptance(
        accuracy_min=0, worst_environment_class_recall_min=0,
        healthy_false_alarm_max=1))
    trials = result.audit["attempts"]
    assert [t["trial"]["learner"] for t in trials] == ["linear", "torch_tcn"]
    assert all(len(t["folds"]) == 4 for t in trials)
    assert trials[1]["status"] == "ELIGIBLE"
    assert all(f["disjoint"] for t in trials for f in t["folds"])
    assert result.plan["qualified"] is False


def test_strict_unattainable_constraints_reject_without_final_refit(monkeypatch):
    import phm_design.industrial.design_engine as engine

    class WrongAlways:
        classes_ = np.array([0, 1, 2], dtype=int)

        def predict_proba(self, x):
            pred = np.zeros((len(x), 3))
            pred[:, 0] = 1.0
            return pred

    monkeypatch.setattr(engine, "_fitted", lambda *args: WrongAlways())
    spec = task("industrial_bearing_classification_demo.json",
                compare=("linear",))
    with pytest.raises(IndustrialDesignRejected) as captured:
        run_design(spec, synthetic_records(spec))
    audit = captured.value.audit
    assert audit["decision"] == "NO_ELIGIBLE_DESIGN"
    assert audit["model_delivered"] is False
    assert audit["final_evaluation_performed"] is False
    assert len(audit["attempts"]) == 1
    assert audit["attempts"][0]["status"] == "REJECTED_BY_POLICY"
    assert audit["attempts"][0]["acceptance"]["failed_gates"]


def test_fraudulent_final_role_and_extra_truth_channel_are_rejected_pretraining(monkeypatch):
    import phm_design.industrial.design_engine as engine
    model_fits = []
    monkeypatch.setattr(engine, "_fitted", lambda *args: model_fits.append(args))
    spec = task("industrial_bearing_classification_demo.json",
                compare=("linear",))
    rec = synthetic_records(spec)
    rec[0]["metadata"]["role"] = "FINAL_OOD"
    with pytest.raises(TaskSpecRejected, match="DESIGN_WORKER_PROTECTED_ROLE"):
        run_design(spec, rec)
    assert not model_fits
    rec = synthetic_records(spec)
    rec[0]["measurements"][0]["fault_label"] = 1
    with pytest.raises(TaskSpecRejected, match="UNDECLARED_OR_PROTECTED_OBSERVATION"):
        run_design(spec, rec)
    assert not model_fits


def test_physical_group_cross_environment_label_support_and_causal_are_blocked():
    spec = task("industrial_bearing_classification_demo.json",
                compare=("linear",))
    rec = synthetic_records(spec)
    rec[1]["metadata"]["bearing_instance_id"] = rec[0]["metadata"]["bearing_instance_id"]
    rec[1]["metadata"]["speed_regime"] = "alternate"
    with pytest.raises(TaskSpecRejected, match="PHYSICAL_GROUP_CROSSES_ENVIRONMENTS"):
        run_design(spec, rec)
    spec = task("industrial_bearing_classification_demo.json",
                compare=("linear",))
    rec = synthetic_records(spec)
    rec = [r for r in rec if r["metadata"]["label"] != "OUTER_DAMAGE"]
    with pytest.raises(TaskSpecRejected, match="UNSUPPORTED_TRAINING_CLASS_OR_GROUP_SUPPORT"):
        run_design(spec, rec)
    spec["task"]["decision_mode"] = "causal_window"
    with pytest.raises(TaskSpecRejected, match="CAUSAL_R81_EXECUTION_NOT_IMPLEMENTED"):
        run_design(spec, [] if False else synthetic_records(spec))


def test_artifact_export_has_content_hash_and_explicit_no_qualification(tmp_path):
    spec = task("industrial_bearing_classification_demo.json",
                compare=("linear",))
    a = run_design(spec, synthetic_records(spec))
    output = a.export(tmp_path / "model")
    assert output["qualified"] is False
    info = json.loads((tmp_path / "model" / "audit.json").read_text())
    assert info["independent_qualification"] is False
    assert info["trained"] is True
    assert hashlib.sha256((tmp_path / "model" / "model.joblib").read_bytes()).hexdigest() == info["model_sha256"]
    with pytest.raises(ValueError, match="ARTIFACT_DIRECTORY_MUST_BE_EMPTY"):
        a.export(tmp_path / "model")


def test_invalid_acceptance_and_insufficient_trial_budget_fail():
    with pytest.raises(ValueError, match="INVALID_DESIGN_ACCEPTANCE"):
        DesignAcceptance(healthy_false_alarm_max=-.1)
    spec = task("industrial_bearing_classification_demo.json", compare=("linear",))
    spec["design"]["budget"]["max_fits"] = 1
    with pytest.raises(TaskSpecRejected, match="SEARCH_BUDGET_FITS_BELOW_TRIALS"):
        run_design(spec, synthetic_records(spec))


def test_no_r71_r78_or_final_targets_in_any_design_record():
    spec = task("industrial_bearing_classification_demo.json", compare=("linear",))
    result = run_design(spec, synthetic_records(spec))
    assert result.audit["target_feedback_used"] is False
    assert result.audit["independent_qualification"] is False
    assert result.audit["final_evaluation_performed"] is False
    assert result.audit["search_budget"]["max_cpu_seconds"] > 0
