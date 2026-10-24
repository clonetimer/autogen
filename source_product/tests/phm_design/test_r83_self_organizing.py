"""R83 self-organizing neural design contract (synthetic functional tests only)."""
import copy
import json

import numpy as np
import pytest

from phm_design.industrial.design_engine import DesignAcceptance, IndustrialDesignRejected
from phm_design.industrial.taskspec import TaskSpecRejected
from phm_design.industrial.self_organizing import (
    EvolutionPolicy, NetworkGenome, SelfOrganizingArtifact,
    propose_genome, run_self_organizing_design,
)
from test_r81a_unified_design_engine import task, synthetic_records


OPEN_SCREEN = DesignAcceptance(
    accuracy_min=0.0, worst_environment_class_recall_min=0.0,
    healthy_false_alarm_max=1.0)


def case_spec():
    spec = task("industrial_bearing_classification_demo.json", compare=("tcn",))
    spec["design"]["budget"]["max_fits"] = 16
    spec["design"]["budget"]["max_cpu_seconds"] = 120
    return spec


def test_genome_strict_grammar_mutates_width_depth_kernel_pool_and_residual():
    g = NetworkGenome()
    neighbors = g.neighbors()
    assert len(neighbors) == len({child.digest for child in neighbors})
    assert any(child.width > g.width for child in neighbors)
    assert any(child.depth > g.depth for child in neighbors)
    assert any(child.kernel != g.kernel for child in neighbors)
    assert any(child.pooling != g.pooling for child in neighbors)
    assert any(child.residual != g.residual for child in neighbors)
    assert any(child.width < 32 for child in NetworkGenome(width=32).neighbors())
    assert any(child.depth < 3 for child in NetworkGenome(depth=3).neighbors())
    for kwargs in ({"width": 10}, {"depth": 5}, {"kernel": 7},
                   {"pooling": "arbitrary"}, {"residual": 1}):
        with pytest.raises(ValueError, match="UNREGISTERED_NETWORK_TOPOLOGY"):
            NetworkGenome(**kwargs)
    with pytest.raises(ValueError, match="INVALID_EVOLUTION_POLICY"):
        EvolutionPolicy(max_candidates=100)


def test_search_changes_third_proposal_when_training_feedback_changes():
    seed_a = NetworkGenome()
    seed_b = NetworkGenome(16, 2, 3, "mean_max", True)
    def propose(loss_a, loss_b):
        history = [
            {"genome": seed_a.__dict__, "genome_sha256": seed_a.digest,
             "selection_loss": loss_a},
            {"genome": seed_b.__dict__, "genome_sha256": seed_b.digest,
             "selection_loss": loss_b},
        ]
        child, parent = propose_genome(
            history, {seed_a.digest, seed_b.digest})
        assert child.digest not in (seed_a.digest, seed_b.digest)
        assert parent in (seed_a.digest, seed_b.digest)
        return parent, child
    parent_a, child_a = propose(.1, .9)
    parent_b, child_b = propose(.9, .1)
    assert parent_a == seed_a.digest
    assert parent_b == seed_b.digest
    assert child_a.digest != child_b.digest


def test_actual_torch_training_group_oof_growth_and_safe_replay(tmp_path):
    spec = case_spec()
    records = synthetic_records(spec, environment_count=3)
    result = run_self_organizing_design(
        spec, records, policy=EvolutionPolicy(max_candidates=3, epochs=2),
        acceptance=OPEN_SCREEN)
    audit = result.audit
    assert audit["source_data_role"] == "TRAIN_ONLY"
    assert audit["source_environment_count"] == 3
    assert audit["feedback_is_independent_test"] is False
    assert audit["target_feedback_used"] is False
    assert audit["independent_qualification"] is False
    assert audit["charged_fits"] == 10
    assert len(audit["attempts"]) == 3
    assert len({row["genome_sha256"] for row in audit["attempts"]}) == 3
    assert audit["attempts"][2]["parent_genome_sha256"] in {
        audit["attempts"][0]["genome_sha256"],
        audit["attempts"][1]["genome_sha256"]}
    assert audit["attempts"][2]["proposal"] == "TRAIN_OOF_FEEDBACK_MUTATION"
    assert all(row["feedback_role"] == "SOURCE_TRAIN_OOF_ONLY"
               for row in audit["attempts"])
    original = result.predict(records[0]["measurements"])
    assert original["label"] in spec["task"]["labels"]
    assert original["independent_qualification"] is False
    assert 0 <= original["confidence_uncalibrated"] <= 1
    output = result.export(tmp_path / "neural")
    assert output["independent_qualification"] is False
    restored = SelfOrganizingArtifact.load(tmp_path / "neural")
    replay = restored.predict(records[0]["measurements"])
    assert original["label"] == replay["label"]
    assert original["confidence_uncalibrated"] == pytest.approx(
        replay["confidence_uncalibrated"], abs=1e-6)
    with pytest.raises(FileExistsError):
        result.export(tmp_path / "neural")
    with open(tmp_path / "neural" / "weights.npz", "ab") as f:
        f.write(b"tamper")
    with pytest.raises(ValueError, match="INTEGRITY"):
        SelfOrganizingArtifact.load(tmp_path / "neural")


def test_protected_roles_truth_columns_and_duplicate_physical_group_fail_closed():
    spec = case_spec()
    rows = synthetic_records(spec, environment_count=3)
    protected = copy.deepcopy(rows)
    protected[0]["metadata"]["role"] = "FINAL_OOD"
    with pytest.raises(TaskSpecRejected, match="DESIGN_WORKER_PROTECTED_ROLE"):
        run_self_organizing_design(spec, protected)
    truth = copy.deepcopy(rows)
    truth[0]["measurements"][0]["fault_label"] = 2
    with pytest.raises(TaskSpecRejected, match="UNDECLARED_OR_PROTECTED_OBSERVATION"):
        run_self_organizing_design(spec, truth)
    duplicate = copy.deepcopy(rows)
    duplicate[1]["measurements"] = copy.deepcopy(duplicate[0]["measurements"])
    with pytest.raises(TaskSpecRejected, match="DUPLICATE_TRAJECTORY"):
        run_self_organizing_design(spec, duplicate)
    cross = copy.deepcopy(rows)
    cross[1]["metadata"][spec["data"]["group_keys"][0]] = (
        cross[0]["metadata"][spec["data"]["group_keys"][0]])
    cross[1]["metadata"][spec["data"]["environment_axes"][0]] = "another_env"
    with pytest.raises(TaskSpecRejected, match="PHYSICAL_GROUP_CROSSES_ENVIRONMENTS"):
        run_self_organizing_design(spec, cross)


def test_insufficient_budget_refuses_before_training():
    spec = case_spec()
    spec["design"]["budget"]["max_fits"] = 3
    with pytest.raises(IndustrialDesignRejected) as error:
        run_self_organizing_design(
            spec, synthetic_records(spec, environment_count=3),
            policy=EvolutionPolicy(max_candidates=3, epochs=1),
            acceptance=OPEN_SCREEN)
    assert error.value.audit["decision"] == "SEARCH_FIT_BUDGET_EXHAUSTED"
    assert error.value.audit["model_delivered"] is False


def test_strict_screen_refuses_without_final_refit(monkeypatch):
    from phm_design.industrial import self_organizing as evolving
    orig = evolving.EvolvingTemporalEstimator.predict_proba

    def always_normal(self, x):
        result = np.zeros((len(x), len(self.classes_)))
        result[:, 0] = 1
        return result

    monkeypatch.setattr(evolving.EvolvingTemporalEstimator,
                        "predict_proba", always_normal)
    spec = case_spec()
    with pytest.raises(IndustrialDesignRejected) as error:
        run_self_organizing_design(
            spec, synthetic_records(spec, environment_count=3),
            policy=EvolutionPolicy(max_candidates=2, epochs=1))
    assert error.value.audit["decision"] == "NO_ELIGIBLE_SELF_ORGANIZING_DESIGN"
    assert error.value.audit["model_delivered"] is False
    assert error.value.audit["charged_fits"] == 6
    assert len(error.value.audit["attempts"]) == 2
    monkeypatch.setattr(evolving.EvolvingTemporalEstimator, "predict_proba", orig)


def test_causal_mode_wall_guard_and_unobservable_class_are_refused():
    spec = case_spec()
    spec["task"]["decision_mode"] = "causal_window"
    with pytest.raises(TaskSpecRejected, match="CAUSAL_SELF_ORGANIZING_NOT_IMPLEMENTED"):
        run_self_organizing_design(spec, synthetic_records(spec, environment_count=3))
    spec = case_spec()
    spec["design"]["budget"]["max_wall_seconds"] = 60
    with pytest.raises(TaskSpecRejected, match="SELF_ORGANIZING_WALL_GUARD_NOT_IMPLEMENTED"):
        run_self_organizing_design(spec, synthetic_records(spec, environment_count=3))
    spec = case_spec()
    rows = synthetic_records(spec, environment_count=3)
    bad = [row for row in rows if not (
        row["metadata"][spec["data"]["environment_axes"][0]] == "regime_2" and
        row["metadata"]["label"] == spec["task"]["labels"][-1])]
    with pytest.raises(TaskSpecRejected):
        run_self_organizing_design(spec, bad)
