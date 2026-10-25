"""R99 design explanation and independent measured-evidence protocol regression.

The source design tests use SYNTHETIC vibration/tachometer fixtures.
Independent-steward tests use explicitly GENERATED raw binary files that are
honestly not industrial measurements, testing only mandatory SHA, grouping,
pairing and no-false-qualification gates. No superiority experiment is
claimed to have been run on actual Paderborn/CWRU independent bearings.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
import hashlib
import json

import pytest

from phm_design.industrial_contracts.design_review import (
    DesignReviewRejected, explain_source_design)
from phm_design.industrial_contracts.independent_review import (
    IndependentReviewRejected, freeze_comparison,
    evaluate_sealed_measured_comparison)
from phm_design.industrial_contracts.physical_graph_design import (
    propose_physical_graph_study, evaluate_physical_graph_study)
from test_r94_torch_like_bridge import SpyTorchLike
from test_r98_physical_graph_design import _cards, _signals


def _source_report():
    equipment, source, trial = _cards(speed=True)
    frozen = propose_physical_graph_study(equipment, source, trial)
    result = evaluate_physical_graph_study(
        frozen, equipment, source, trial, _signals(source),
        provider=SpyTorchLike())
    return frozen, result


def test_r99_explains_each_physical_dag_from_registered_observables():
    frozen, source = _source_report()
    explanation = explain_source_design(frozen, source)
    assert explanation["status"] == "DESCRIPTIVE_SOURCE_EXPLANATION_ONLY"
    assert explanation["evidence_scope"] == "SOURCE_ONLY_NOT_INDEPENDENT"
    assert len(explanation["explanation_sha256"]) == 64
    assert len(explanation["why_generated"]) == len(frozen["graphs"]) == 4
    assert "tachometer" in " ".join(
        explanation["why_generated"][2]["physical_prerequisite_reasons"])
    assert "recording_level" in " ".join(
        explanation["why_generated"][3]["physical_prerequisite_reasons"])
    assert len(explanation["failure_audit"]) == sum(
        len(g["pipeline_plan"]["pipeline_designs"]) for g in frozen["graphs"])
    assert explanation["partial_cost_pareto"]
    assert set(explanation["partial_cost_pareto"]) <= {
        x["candidate_key"] for x in explanation["failure_audit"]}
    assert explanation["fair_compute_cost_measured"] is False
    assert explanation["feature_extract_time_measured"] is False
    assert not explanation["algorithm_improvement_independently_proven"]
    assert explanation == explain_source_design(frozen, source)


def test_r99_failure_reason_finds_high_far_then_only_predeclared_revision():
    frozen, report = _source_report()
    selected = report["evaluated_graphs"][1][
        "source_only_pipeline_comparison"]["candidates"][0]
    selected["source_healthy_false_alarm_fraction"] = 1.0
    selected["source_worst_group_accuracy"] = 0.
    explained = explain_source_design(frozen, report)
    key = "graph_02/"+selected["pipeline_id"]
    row = next(x for x in explained["failure_audit"]
               if x["candidate_key"] == key)
    assert "HIGH_HEALTHY_FALSE_ALARM" in row["failures"]
    assert "POOR_SOURCE_WORST_GROUP_ACCURACY" in row["failures"]
    assert row["causal_explanation"] == "not_identified_from_source_oof"
    revised = next(x for x in explained["bounded_revisions"]
                   if x["failed_candidate_key"] == key)
    assert revised["allowed_action"] == "consider_predeclared_lower_complexity_baseline"
    assert revised["candidate_already_in_frozen_study"].startswith("graph_01/")
    assert revised["counts_as_new_independent_evidence"] is False
    assert revised["new_model_architecture_implemented"] is False
    assert revised["threshold_or_alarm_postprocessing_changed"] is False


def test_r99_rejects_fake_external_promotion_unregistered_graph_and_model():
    frozen, report = _source_report()
    fake = deepcopy(report)
    fake["independent_diagnostic_generalization_proven"] = True
    with pytest.raises(DesignReviewRejected,
                       match="R99_EXTERNAL_VALIDATION_CLAIM_FROM_SOURCE_ONLY_REPORT"):
        explain_source_design(frozen, fake)
    fake = deepcopy(report)
    fake["evaluated_graphs"][0]["source_only_pipeline_comparison"][
        "candidates"][0]["algorithm_family"] = "unapproved_neural_network"
    with pytest.raises(DesignReviewRejected,
                       match="R99_MODEL_OR_PREPROCESSOR_UNREGISTERED_IN_REPORT"):
        explain_source_design(frozen, fake)
    fake_plan = deepcopy(frozen)
    fake_plan["graphs"][0]["stages"][1] = "eval_arbitrary_code"
    with pytest.raises(DesignReviewRejected, match="R99_FROZEN_GRAPH_SHA256_MISMATCH"):
        explain_source_design(fake_plan, report)


def test_r99_missing_source_groups_abstain_no_fake_model_explanation():
    equipment, train, trial = _cards(speed=True)
    train["records"] = train["records"][:3]
    frozen = propose_physical_graph_study(equipment, train, trial)
    result = evaluate_physical_graph_study(
        frozen, equipment, train, trial,
        _signals(train), provider=SpyTorchLike())
    explained = explain_source_design(frozen, result)
    assert explained["status"] == "ABSTAIN_SOURCE_UNSUPPORTED"
    assert explained["failure_audit"] == []
    assert explained["bounded_revisions"] == []
    assert explained["algorithm_improvement_independently_proven"] is False


def _binary_measured_protocol(tmp_path):
    """Synthetic bytes masquerade only within test for protocol rejection.

    They are NOT measured data; dataset.origin='measured' is intentionally
    self-declared to prove metadata/byte checks cannot establish authenticity.
    """
    equipment, source, trial = _cards(speed=False)
    manifest = deepcopy(source)
    manifest["origin"] = "measured"  # self assertion, not authenticated
    manifest["license"] = "fixture-local-only-no-publication"
    manifest["dataset_id"] = "test_independent_steward_manifest"
    originals = {}
    labels = trial["labels"]
    all_records = list(manifest["records"])
    base = datetime(2026, 10, 2, tzinfo=timezone.utc)
    for i in range(6):
        label = labels[i % 2]
        start = base+timedelta(hours=i)
        row = deepcopy(all_records[0])
        row.update({
            "record_id": f"val_record_{i}",
            "physical_id": f"validation_physical_bearing_{i}",
            "session_id": f"independent_session_{i}",
            "environment_id": "environment_A",
            "role": "VALIDATION",
            "start_utc": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end_utc": (start+timedelta(minutes=2)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "label": label,
        })
        manifest["records"].append(row)
    for i, row in enumerate(manifest["records"]):
        p = tmp_path / (row["record_id"]+".bin")
        p.write_bytes(bytes([i+1])*1024)
        row["source_sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
        originals[row["record_id"]] = str(p)
    frozen = freeze_comparison(
        "a"*64, "frozen-challenger-sha256", "fixed-baseline-sha256", trial)
    return equipment, manifest, trial, originals, frozen


def test_real_file_sha_and_declared_bearing_group_paired_report_never_qualifies(
        tmp_path):
    equipment, manifest, trial, originals, frozen = _binary_measured_protocol(tmp_path)
    truth = {r["record_id"]: r["label"] for r in manifest["records"]
             if r["role"] == "VALIDATION"}
    challenger = dict(truth)
    baseline = {k: next(x for x in trial["labels"] if x != v)
                for k, v in truth.items()}
    response = evaluate_sealed_measured_comparison(
        frozen, equipment, manifest, trial, originals, {
            frozen["challenger_artifact_id"]: challenger,
            frozen["baseline_artifact_id"]: baseline,
        })
    assert response["raw_original_file_count_hashed"] == len(manifest["records"])
    assert response["declared_heldout_physical_groups"] == 6
    assert all(x == 3 for x in response[
        "declared_heldout_physical_groups_per_class"].values())
    assert response["observed_challenger_minus_baseline_group_macro"] == 1
    assert response["one_sided_group_paired_sign_p"] == 1/64
    assert response["directional_submission_evidence_meets_preregistered_threshold"]
    assert not response["physical_entity_authenticity_independently_certified"]
    assert not response["predictions_bound_to_frozen_executable_artifacts"]
    assert not response["independent_real_device_superiority_proven"]
    assert not response["eligible_for_stable"]


def test_measurement_file_one_byte_mutation_rejected(tmp_path):
    equipment, manifest, trial, originals, frozen = _binary_measured_protocol(tmp_path)
    rid = manifest["records"][0]["record_id"]
    p = Path(originals[rid])
    p.write_bytes(p.read_bytes()[:-1]+b"x")
    with pytest.raises(IndependentReviewRejected,
                       match="R99_ORIGINAL_RAW_BYTES_SHA_MISMATCH"):
        evaluate_sealed_measured_comparison(
            frozen, equipment, manifest, trial, originals, {})


def test_same_device_across_train_and_val_cannot_pass(tmp_path):
    equipment, manifest, trial, originals, frozen = _binary_measured_protocol(tmp_path)
    val = next(x for x in manifest["records"] if x["role"] == "VALIDATION")
    val["physical_id"] = manifest["records"][0]["physical_id"]
    with pytest.raises(ValueError, match="R93_PHYSICAL_DEVICE_CROSSES_DATA_ROLES"):
        evaluate_sealed_measured_comparison(
            frozen, equipment, manifest, trial, originals, {})


def test_real_device_gate_rejects_insufficient_sealed_val_groups(tmp_path):
    equipment, manifest, trial, originals, frozen = _binary_measured_protocol(tmp_path)
    manifest["records"] = [r for r in manifest["records"]
                           if r["role"] == "TRAIN" or
                           r["record_id"] != "val_record_5"]
    del originals["val_record_5"]
    with pytest.raises(IndependentReviewRejected,
                       match="R99_TOO_FEW_REAL_HELDOUT_DEVICES_PER_CLASS"):
        evaluate_sealed_measured_comparison(
            frozen, equipment, manifest, trial, originals, {})


def test_frozen_models_and_predictions_cannot_be_changed_after_seal(tmp_path):
    equipment, manifest, trial, originals, frozen = _binary_measured_protocol(tmp_path)
    altered = deepcopy(frozen)
    altered["challenger_artifact_id"] = "post-hoc-best-model"
    with pytest.raises(IndependentReviewRejected,
                       match="R99_COMPARISON_CHANGED_AFTER_SEALING"):
        evaluate_sealed_measured_comparison(
            altered, equipment, manifest, trial, originals, {})
    frozen = freeze_comparison(
        "a"*64, "the-challenger", "the-baseline", trial)
    with pytest.raises(IndependentReviewRejected,
                       match="R99_EXACT_TWO_PREREGISTERED_PREDICTION_STREAMS_REQUIRED"):
        evaluate_sealed_measured_comparison(
            frozen, equipment, manifest, trial, originals, {})


def test_source_only_fixtures_never_qualify_as_real_measured_evidence(tmp_path):
    equipment, manifest, trial, originals, frozen = _binary_measured_protocol(tmp_path)
    manifest["origin"] = "synthetic"
    with pytest.raises(IndependentReviewRejected,
                       match="R99_EXPERIMENTAL_MEASURED_DATA_ONLY"):
        evaluate_sealed_measured_comparison(
            frozen, equipment, manifest, trial, originals, {})
