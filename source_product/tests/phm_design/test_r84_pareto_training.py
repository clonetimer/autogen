"""R84 actual Torch training smoke on SYNTHETIC industrial bearing records.

This is not an independent blind trial or physical sensor evidence.
"""
from __future__ import annotations

from phm_design.industrial.design_engine import DesignAcceptance
from phm_design.industrial.self_organizing import (
    EvolutionPolicy, run_self_organizing_design)
from test_r81a_unified_design_engine import synthetic_records
from test_r83_self_organizing import case_spec


def test_real_torch_pareto_and_static_share_train_folds_seeds_and_fit_cap():
    spec = case_spec()
    rows = synthetic_records(spec, environment_count=3)
    policy = EvolutionPolicy(max_candidates=3, epochs=1)
    screen = DesignAcceptance(accuracy_min=0.0,
        worst_environment_class_recall_min=0.0,
        healthy_false_alarm_max=1.0)
    static = run_self_organizing_design(
        spec, rows, policy=policy, acceptance=screen,
        seed_schedule="r83_matched_train",
        proposal_mode="predeclared_static")
    pareto = run_self_organizing_design(
        spec, rows, policy=policy, acceptance=screen,
        seed_schedule="r83_matched_train",
        proposal_mode="feedback_pareto")
    sa, pa = static.audit, pareto.audit
    assert sa["charged_fits"] == pa["charged_fits"] == 10
    assert sa["train_manifest_sha256"] == pa["train_manifest_sha256"]
    assert sa["train_observations_sha256"] == pa["train_observations_sha256"]
    assert sa["seed_schedule"] == pa["seed_schedule"] == "r83_matched_train"
    assert sa["split"] == pa["split"] == "LEAVE_ONE_ENTIRE_SOURCE_ENVIRONMENT_OUT"
    assert len(sa["attempts"]) == len(pa["attempts"]) == 3
    assert [a["genome_sha256"] for a in sa["attempts"][:2]] == [
        a["genome_sha256"] for a in pa["attempts"][:2]]
    assert pa["attempts"][2]["proposal"] == "TRAIN_OOF_PARETO_MUTATION"
    assert pa["attempts"][2]["parent_genome_sha256"] in {
        a["genome_sha256"] for a in pa["attempts"][:2]}
    assert pa["pareto_proposal_method"].startswith("TRAIN_OOF_")
    assert pa["source_data_role"] == "TRAIN_ONLY"
    assert pa["feedback_is_independent_test"] is False
    assert pa["independent_qualification"] is False
    assert sa["independent_qualification"] is False
    assert pareto.predict(rows[0]["measurements"])["independent_qualification"] is False
