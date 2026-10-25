"""R88 frozen no-pickle VALIDATION steward integration.

All trajectories in these tests are *synthetic fixtures*, not actual machine
or native Basilisk measurements. VAL role is deliberately not TEST/FINAL.
"""
from __future__ import annotations

import copy
import hashlib
import shutil

import numpy as np
import pytest

from phm_design.industrial.r86_neuro_fuzzy import FuzzyRulePolicy
from phm_design.industrial.r88_fuzzy_validation_steward import (
    ARMS, freeze_fuzzy_source_models, evaluate_frozen_fuzzy_validation)
from phm_design.industrial.taskspec import TaskSpecRejected
from test_r81a_unified_design_engine import task, synthetic_records


@pytest.fixture(scope="module")
def research_frozen(tmp_path_factory):
    base = tmp_path_factory.mktemp("r88-fuzzy-research")
    spec = task("industrial_bearing_classification_demo.json", compare=("linear",))
    train = synthetic_records(spec, environment_count=3)
    data = synthetic_records(spec, environment_count=2)
    # Artificially separate synthetic physical IDs AND observation bytes.
    # These edits are NOT evidence that external real cohorts are independent.
    for row in data:
        m = row["metadata"]
        for col in spec["data"]["group_keys"]:
            m[col] = "validation_" + m[col]
        for col in spec["data"]["environment_axes"]:
            m[col] = "unseen_" + m[col]
        m["role"] = "VALIDATION"
        for record in row["measurements"]:
            for c in spec["observations"]["channels"]:
                record[c["name"]] += .173
    root = base / "frozen"
    receipt = freeze_fuzzy_source_models(
        spec, train, root, policy=FuzzyRulePolicy(
            max_rules=2, premise_steps=2),
        evidence_origin="SYNTHETIC_TEST_FIXTURE")
    return spec, train, data, root, receipt


def test_actual_frozen_models_replay_all_arms_with_validated_separate_cohort(research_frozen):
    spec, train, val, root, freeze = research_frozen
    result = evaluate_frozen_fuzzy_validation(
        root, val, expected_manifest_sha256=freeze["manifest_sha256"])
    assert len(freeze["source_oof_report"]["arms"]) == 5
    assert len(result["arms"]) == 5
    assert set(result["arms"]) == set(ARMS)
    assert result["validation_role"] == "VALIDATION_ONLY"
    assert result["validation_environment_count"] == 2
    assert result["validation_group_count"] == len(val)
    assert result["frozen_model_hashes_verified"] is True
    assert result["source_artifact_fits_performed"] == 0
    assert result["post_validation_refit_or_selection_performed"] is False
    assert result["blind_test_or_final_performed"] is False
    assert result["independent_generalization_proven"] is False
    assert result["physical_independence_attested_by_external_steward"] is False
    assert result["one_time_validation_enforced"] is False
    assert result["eligible_for_stable"] is False
    assert result["origin_self_reported"] == "SYNTHETIC_TEST_FIXTURE"
    for item in result["arms"].values():
        for value in item.values():
            assert 0 <= value <= 1
    for interval in result["paired_group_intervals"].values():
        assert interval["replicates"] == 499
        assert interval["confirmatory_significance_claim"] is False
        assert len(interval["cluster_bootstrap_95_percentile_interval"]) == 2
    replay = evaluate_frozen_fuzzy_validation(
        root, val, expected_manifest_sha256=freeze["manifest_sha256"])
    assert replay == result


def test_protected_test_or_final_role_refused_by_readonly_reader(research_frozen):
    _, _, val, root, frozen = research_frozen
    for role in ("TRAIN", "TEST", "FINAL", "FINAL_OOD", "CAL"):
        probe = copy.deepcopy(val)
        probe[0]["metadata"]["role"] = role
        with pytest.raises(TaskSpecRejected, match="R88_PROTECTED_ROLE_OR_METADATA"):
            evaluate_frozen_fuzzy_validation(
                root, probe, expected_manifest_sha256=frozen["manifest_sha256"])


def test_group_overlap_source_replay_and_renamed_same_observations_rejected(
        research_frozen):
    spec, train, val, root, frozen = research_frozen
    overlap = copy.deepcopy(val)
    for key in spec["data"]["group_keys"]:
        overlap[0]["metadata"][key] = train[0]["metadata"][key]
    with pytest.raises(TaskSpecRejected, match="R88_VALIDATION_SOURCE_PHYSICAL_GROUP_OVERLAP"):
        evaluate_frozen_fuzzy_validation(
            root, overlap, expected_manifest_sha256=frozen["manifest_sha256"])
    replayed = copy.deepcopy(val)
    replayed[0]["measurements"] = copy.deepcopy(train[0]["measurements"])
    with pytest.raises(TaskSpecRejected, match="R88_VALIDATION_SOURCE_TRAJECTORY_REPLAY"):
        evaluate_frozen_fuzzy_validation(
            root, replayed, expected_manifest_sha256=frozen["manifest_sha256"])


def test_source_refuses_protected_records_before_artifact_directory_created(
        research_frozen, tmp_path):
    spec, train, _, _, _ = research_frozen
    p = tmp_path / "not_created"
    bad = copy.deepcopy(train)
    bad[0]["metadata"]["role"] = "FINAL_OOD"
    with pytest.raises(TaskSpecRejected, match="R88_PROTECTED_ROLE_OR_METADATA"):
        freeze_fuzzy_source_models(
            spec, bad, p, policy=FuzzyRulePolicy(
                max_rules=2, premise_steps=2))
    assert not p.exists()
    bad = copy.deepcopy(train)
    bad[0]["measurements"][0]["fault_label"] = 1
    with pytest.raises(TaskSpecRejected, match="UNDECLARED_OR_PROTECTED_OBSERVATION"):
        freeze_fuzzy_source_models(
            spec, bad, p, policy=FuzzyRulePolicy(
                max_rules=2, premise_steps=2))
    assert not p.exists()


def test_source_rejects_unsafe_pickle_policy_and_reusing_existing_output(
        research_frozen, tmp_path):
    spec, train, _, root, _ = research_frozen
    with pytest.raises(ValueError, match="R88_EXPLICIT_LEARNABLE_PREMISE_POLICY_REQUIRED"):
        freeze_fuzzy_source_models(
            spec, train, tmp_path / "wrong",
            policy=FuzzyRulePolicy(max_rules=2))
    with pytest.raises(TaskSpecRejected, match="R88_FROZEN_OUTPUT_MUST_BE_NEW"):
        freeze_fuzzy_source_models(
            spec, train, root, policy=FuzzyRulePolicy(
                max_rules=2, premise_steps=2))


def test_frozen_checksum_mismatch_prevents_exposure_of_modified_weights(
        research_frozen, tmp_path):
    _, _, val, root, frozen = research_frozen
    bad_hash = "0"*64 if frozen["manifest_sha256"] != "0"*64 else "1"*64
    with pytest.raises(TaskSpecRejected, match="R88_FROZEN_MANIFEST_HASH_MISMATCH"):
        evaluate_frozen_fuzzy_validation(
            root, val, expected_manifest_sha256=bad_hash)
    modified = tmp_path / "copied"
    shutil.copytree(root, modified)
    weights = modified / (ARMS[0] + ".npz")
    weights.write_bytes(weights.read_bytes() + b"poisoned")
    with pytest.raises(TaskSpecRejected, match="R88_FROZEN_MODEL_HASH_MISMATCH"):
        evaluate_frozen_fuzzy_validation(
            modified, val, expected_manifest_sha256=frozen["manifest_sha256"])


def test_relabelled_duplicated_validation_observations_are_blocked(
        research_frozen):
    spec, _, val, root, frozen = research_frozen
    repeated = copy.deepcopy(val)
    repeated[1]["measurements"] = copy.deepcopy(repeated[0]["measurements"])
    with pytest.raises(TaskSpecRejected, match="R88_RELABELED_DUPLICATE_OBSERVATION"):
        evaluate_frozen_fuzzy_validation(
            root, repeated, expected_manifest_sha256=frozen["manifest_sha256"])
