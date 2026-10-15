"""R81c hard wall-worker termination and in-process fail-closed CPU checks."""
import copy
import time

import pytest

from phm_design.industrial.design_engine import (
    DesignAcceptance, IndustrialDesignRejected, run_design,
)
from phm_design.industrial.resource_guard import (
    GuardedDesignTimeout, run_design_guarded,
)
from phm_design.industrial.taskspec import TaskSpecRejected, compile_task_spec
from test_r81a_unified_design_engine import task, synthetic_records


SCREEN = DesignAcceptance(accuracy_min=0,
                          worst_environment_class_recall_min=0,
                          healthy_false_alarm_max=1)


def spec(max_wall=30, *, cpu=60):
    s = task("industrial_bearing_classification_demo.json", compare=("linear",))
    s["design"]["budget"]["max_wall_seconds"] = max_wall
    s["design"]["budget"]["max_cpu_seconds"] = cpu
    return s


def test_opt_in_hard_deadline_rejects_direct_inprocess_runner():
    s = spec()
    plan = compile_task_spec(s)
    assert plan["budget"]["max_wall_seconds"] == 30
    with pytest.raises(TaskSpecRejected, match="HARD_WALL_BUDGET_REQUIRES_GUARDED_RUNNER"):
        run_design(s, synthetic_records(s, environment_count=3),
                   acceptance=SCREEN)


@pytest.mark.parametrize("bad", [0, -1, 1.5, True, "5"])
def test_invalid_deadline_is_rejected_before_launch(bad):
    s = spec(max_wall=bad)
    with pytest.raises(TaskSpecRejected):
        compile_task_spec(s)


def test_guard_requires_predeclared_deadline():
    s = spec()
    del s["design"]["budget"]["max_wall_seconds"]
    with pytest.raises(TaskSpecRejected, match="HARD_WALL_BUDGET_NOT_DECLARED"):
        run_design_guarded(s, synthetic_records(s, environment_count=3),
                           acceptance=SCREEN)


def test_process_guard_real_fit_and_trusted_artifact_replay(tmp_path):
    from phm_design.industrial.artifact_io import load_trusted_artifact

    s = spec(max_wall=60)
    records = synthetic_records(s, environment_count=3)
    model = run_design_guarded(s, records, acceptance=SCREEN)
    assert model.audit["decision"] == "TRAIN_ONLY_SELECTION_REFITTED"
    assert model.audit["model_delivered"] is True
    assert model.audit["budget_enforcement"]["wall_deadline"] == (
        "PROCESS_ISOLATED_PARENT_TERMINATION")
    assert model.audit["budget_enforcement"]["preprocessing_charged"] is True
    assert model.audit["independent_qualification"] is False
    path = tmp_path / "guard"
    model.export(path)
    restored = load_trusted_artifact(path, trust_pickle=True)
    assert restored.predict_decisions(records[0]["measurements"]) == (
        model.predict_decisions(records[0]["measurements"]))


def test_wall_timeout_terminates_worker_without_delivering_model():
    s = spec(max_wall=1)
    records = synthetic_records(s, environment_count=3)
    with pytest.raises(GuardedDesignTimeout) as captured:
        run_design_guarded(s, records, acceptance=SCREEN)
    assert captured.value.audit["decision"] == "HARD_WALL_TIMEOUT"
    assert captured.value.audit["model_delivered"] is False
    assert captured.value.audit["independent_qualification"] is False


def test_cpu_overrun_after_fit_refuses_delivery(monkeypatch):
    import phm_design.industrial.design_engine as engine

    s = spec(cpu=1)
    del s["design"]["budget"]["max_wall_seconds"]

    def cpu_hungry_fit(*args, **kwargs):
        start = time.process_time()
        while time.process_time() - start < 1.1:
            pass
        raise RuntimeError("TRIAL_CPU_BUDGET_EXCEEDED")

    monkeypatch.setattr(engine, "_fitted", cpu_hungry_fit)
    with pytest.raises(IndustrialDesignRejected) as captured:
        run_design(s, synthetic_records(s, environment_count=3),
                   acceptance=SCREEN)
    audit = captured.value.audit
    assert audit["model_delivered"] is False
    assert audit["decision"] == "BUDGET_EXHAUSTED_BEFORE_SELECTION"
    assert audit["fold_model_fits"] == 1


def test_train_fingerprint_binds_record_shapes_and_boundaries():
    import numpy as np
    from phm_design.industrial.design_engine import _observations_digest
    one = [np.array([[1., 2.], [3., 4.]], dtype=float)]
    two = [np.array([[1., 2.]], dtype=float),
           np.array([[3., 4.]], dtype=float)]
    assert _observations_digest(one) != _observations_digest(two)
    assert _observations_digest(one) == _observations_digest([
        np.asfortranarray(one[0])])


def test_full_audit_and_runtime_tampering_are_rejected_before_pickle(tmp_path):
    import json
    from phm_design.industrial.artifact_io import load_trusted_artifact

    s = spec(max_wall=60)
    model = run_design_guarded(s, synthetic_records(s, environment_count=3),
                               acceptance=SCREEN)
    folder = tmp_path / "audit"
    model.export(folder)
    audit_path = folder / "audit.json"
    original = json.loads(audit_path.read_text())

    changed = copy.deepcopy(original)
    changed["trial_selection_audit"]["model_delivered"] = False
    audit_path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="ARTIFACT_FULL_AUDIT_HASH_MISMATCH"):
        load_trusted_artifact(folder, trust_pickle=True)

    changed = copy.deepcopy(original)
    changed["runtime_fingerprint"]["packages"]["scikit-learn"] = "0.0.0"
    audit_path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="ARTIFACT_RUNTIME_DEPENDENCY_MISMATCH"):
        load_trusted_artifact(folder, trust_pickle=True)

    audit_path.write_text(json.dumps(original))
    restored = load_trusted_artifact(folder, trust_pickle=True)
    assert restored.audit["training_observations_hash_contract"] == (
        "LENGTH_PREFIXED_FLOAT64_LE_V1")
