"""R83b spawn worker and hard deadline: synthetic-only research contract."""
import json
import time

import pytest

from phm_design.industrial.design_engine import (
    DesignAcceptance, IndustrialDesignRejected,
)
from phm_design.industrial.resource_guard import (
    GuardedDesignTimeout, _invoke_worker,
)
from phm_design.industrial.self_organizing import EvolutionPolicy
from phm_design.industrial.self_organizing_guard import (
    run_self_organizing_design_guarded,
)
from phm_design.industrial.taskspec import TaskSpecRejected
from test_r81a_unified_design_engine import task, synthetic_records


def _spec():
    spec = task("industrial_bearing_classification_demo.json", compare=("tcn",))
    spec["design"]["candidate_max"] = 2
    spec["design"]["budget"]["max_fits"] = 12
    spec["design"]["budget"]["max_cpu_seconds"] = 120
    spec["design"]["budget"]["max_wall_seconds"] = 90
    return spec


def _blocked_worker(conn):
    try:
        time.sleep(10)
    finally:
        conn.close()


def test_process_guard_enforces_deadline_with_blocked_worker():
    with pytest.raises(GuardedDesignTimeout) as ex:
        _invoke_worker(_blocked_worker, tuple(), 1)
    assert ex.value.audit["decision"] == "HARD_WALL_TIMEOUT"
    assert ex.value.audit["model_delivered"] is False


def test_real_cpu_torch_training_returns_picklable_model_and_can_replay(tmp_path):
    spec = _spec()
    records = synthetic_records(spec, environment_count=3)
    artifact = run_self_organizing_design_guarded(
        spec, records, policy=EvolutionPolicy(max_candidates=2, epochs=1),
        acceptance=DesignAcceptance(
            accuracy_min=0, worst_environment_class_recall_min=0,
            healthy_false_alarm_max=1),
    )
    assert artifact.audit["wall_guard"] == "PARENT_SPAWN_TERMINATION"
    assert artifact.audit["charged_fits"] == 7  # 2x3 heldout fits + final refit
    assert artifact.audit["independent_qualification"] is False
    before = artifact.predict(records[0]["measurements"])
    artifact.export(tmp_path / "guarded_model")
    from phm_design.industrial.self_organizing import SelfOrganizingArtifact
    restored = SelfOrganizingArtifact.load(tmp_path / "guarded_model")
    assert restored.predict(records[0]["measurements"])["label"] == before["label"]


def test_worker_refusal_preserves_source_train_audit_not_partial_model():
    spec = _spec()
    spec["design"]["budget"]["max_fits"] = 3  # valid TaskSpec, insufficient for 3 folds + final
    with pytest.raises(IndustrialDesignRejected) as err:
        run_self_organizing_design_guarded(
            spec, synthetic_records(spec, environment_count=3),
            policy=EvolutionPolicy(max_candidates=2, epochs=1))
    assert err.value.audit["decision"] == "SEARCH_FIT_BUDGET_EXHAUSTED"
    assert err.value.audit["model_delivered"] is False
    assert err.value.audit["source_data_role"] == "TRAIN_ONLY"


def test_unguarded_entrance_and_missing_required_wall_contract_reject():
    spec = _spec()
    del spec["design"]["budget"]["max_wall_seconds"]
    with pytest.raises(TaskSpecRejected, match="HARD_WALL_BUDGET_REQUIRED"):
        run_self_organizing_design_guarded(
            spec, synthetic_records(spec, environment_count=3))
    spec = _spec()
    spec["design"]["candidate_max"] = 1
    with pytest.raises(TaskSpecRejected, match="TRIAL_BUDGET_EXCEEDED"):
        run_self_organizing_design_guarded(
            spec, synthetic_records(spec, environment_count=3))
