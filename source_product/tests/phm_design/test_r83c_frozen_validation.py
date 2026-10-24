"""R83c reviewer-controlled frozen VALIDATION contracts; synthetic test data only."""
from __future__ import annotations

import copy
import json

import pytest

from phm_design.industrial.design_engine import DesignAcceptance
from phm_design.industrial.r83_matched_budget import run_matched_source_train_study
from phm_design.industrial.r83_frozen_validation import (
    evaluate_r83_frozen_validation, _preflight, BOOTSTRAP_REPLICATES,
)
from phm_design.industrial.taskspec import TaskSpecRejected
from test_r81a_unified_design_engine import task, synthetic_records


ACCEPT_ALL = DesignAcceptance(accuracy_min=0,
    worst_environment_class_recall_min=0, healthy_false_alarm_max=1)


def _spec():
    spec = task("industrial_bearing_classification_demo.json",
                compare=("tcn",))
    spec["design"]["candidate_max"] = 3
    spec["design"]["budget"]["max_fits"] = 10
    spec["design"]["budget"]["max_cpu_seconds"] = 120
    spec["design"]["budget"]["max_wall_seconds"] = 180
    return spec


def _heldout(spec):
    """Independent synthetic groups and changed traces; not Basilisk."""
    val = synthetic_records(spec, environment_count=3)
    for item in val:
        item["metadata"]["role"] = "VALIDATION"
        for key in spec["data"]["group_keys"]:
            item["metadata"][key] = "new_physical_" + item["metadata"][key]
        for r in item["measurements"]:
            for channel in spec["observations"]["channels"]:
                r[channel["name"]] += 0.03125
    return val


def _source_receipt(spec, train):
    from phm_design.industrial.taskspec import TaskSpec
    from phm_design.industrial.r83_matched_budget import _preflight
    import hashlib
    import numpy as np
    from phm_design.industrial.r83_frozen_validation import _fingerprint, _digest_group
    fields = [spec["observations"]["time"]["field"]] + [
        c["name"] for c in spec["observations"]["channels"]]
    groups = sorted({_digest_group("\x1f".join(
        row["metadata"][k] for k in spec["data"]["group_keys"]))
        for row in train})
    traces = sorted({_fingerprint(row["measurements"], fields)
                     for row in train})
    return {
        "schema": "phm.r83.matched-source-train-study.v1",
        "source_task_spec_sha256": TaskSpec.from_dict(spec).sha256,
        "source_data_role": "TRAIN_ONLY",
        "source_train_manifest_sha256": "1"*64,
        "source_train_observations_sha256": "2"*64,
        "independent_qualification": False,
        "source_train_test_or_final_accessed": False,
        "complete_three_arm_comparison": True,
        "fairness_gate": {"source_train_comparison_allowed": True},
        "source_train_physical_group_sha256": groups,
        "source_train_trajectory_sha256": traces,
        "source_trajectory_digest_contract": "RAW_FLOAT64_LE_NO_METADATA_V1",
        "source_group_digest_contract": "UTF8_UNIT_SEPARATOR_V1",
        "arms": {key: {} for key in (
            "fixed_portfolio", "static_topology", "self_organizing")},
    }


def test_validation_rejects_protected_roles_overlap_and_oracle_before_model_load():
    spec = _spec()
    source = synthetic_records(spec, environment_count=3)
    receipt = _source_receipt(spec, source)
    val = _heldout(spec)
    rows, support, groups = _preflight(receipt, spec, val)
    assert len(rows) == len(val)
    assert len(groups) == len(val)
    assert len(support) == 3

    bad = copy.deepcopy(val)
    bad[0]["metadata"]["role"] = "FINAL"
    with pytest.raises(TaskSpecRejected, match="PROTECTED_ROLE_OR_LABEL"):
        _preflight(receipt, spec, bad)

    bad = copy.deepcopy(val)
    bad[0]["metadata"]["role"] = "TRAIN"
    with pytest.raises(TaskSpecRejected, match="PROTECTED_ROLE_OR_LABEL"):
        _preflight(receipt, spec, bad)

    bad = copy.deepcopy(val)
    bad[0]["metadata"][spec["data"]["group_keys"][0]] = (
        source[0]["metadata"][spec["data"]["group_keys"][0]])
    with pytest.raises(TaskSpecRejected, match="OVERLAPS_TRAIN"):
        _preflight(receipt, spec, bad)

    bad = copy.deepcopy(val)
    bad[0]["measurements"] = copy.deepcopy(source[0]["measurements"])
    with pytest.raises(TaskSpecRejected, match="REUSED_FROM_TRAIN"):
        _preflight(receipt, spec, bad)

    bad = copy.deepcopy(val)
    bad[0]["measurements"][0]["ground_truth"] = 1
    with pytest.raises(TaskSpecRejected, match="UNDECLARED_OR_PROTECTED_OBSERVATION"):
        _preflight(receipt, spec, bad)

    bad_receipt = copy.deepcopy(receipt)
    bad_receipt["fairness_gate"]["source_train_comparison_allowed"] = False
    with pytest.raises(TaskSpecRejected, match="FROZEN_COMPLETE_TRAIN_REPORT"):
        _preflight(bad_receipt, spec, val)

    bad = copy.deepcopy(val)
    bad.pop()   # 1 missing in one env/class cell
    with pytest.raises(TaskSpecRejected, match="INSUFFICIENT_INDEPENDENT_CLASS_GROUPS"):
        _preflight(receipt, spec, bad)


def test_real_frozen_models_scored_on_separate_validation_only(tmp_path):
    spec = _spec()
    train = synthetic_records(spec, environment_count=3)
    frozen = tmp_path / "research_models"
    report = run_matched_source_train_study(
        spec, train, acceptance=ACCEPT_ALL, artifacts_root=frozen)
    assert report["complete_three_arm_comparison"] is True
    heldout = _heldout(spec)

    with pytest.raises(ValueError, match="EXPLICIT_FIXED_PICKLE_TRUST_REQUIRED"):
        evaluate_r83_frozen_validation(report, spec, frozen, heldout)

    actual = evaluate_r83_frozen_validation(
        report, spec, frozen, heldout, trusted_fixed_pickle=True)
    assert actual["validation_role"] == "VALIDATION_ONLY_REVIEWER_CONTROLLED"
    assert actual["train_group_overlap"] is False
    assert actual["train_trajectory_overlap"] is False
    assert actual["model_fit_or_selection_performed"] is False
    assert actual["independent_final_test_performed"] is False
    assert actual["independent_qualification"] is False
    assert actual["minimum_groups_per_cell"] == 2
    assert actual["recommend_five_groups_per_cell_met"] is False
    assert set(actual["evaluation_metrics"]) == {
        "fixed_portfolio", "static_topology", "self_organizing"}
    for group in actual["paired_differences"].values():
        assert group["replicates"] == BOOTSTRAP_REPLICATES
        assert group["confirmatory_significance_claim"] is False
        assert len(group["cluster_bootstrap_95_percentile_interval"]) == 2
    # Re-evaluation is deterministic, without any model refits/updates.
    repeat = evaluate_r83_frozen_validation(
        report, spec, frozen, heldout, trusted_fixed_pickle=True)
    assert json.dumps(actual, sort_keys=True) == json.dumps(repeat, sort_keys=True)

    tampered = copy.deepcopy(report)
    tampered["arms"]["static_topology"]["frozen_research_artifact"][
        "manifest_sha256"] = "0"*64
    with pytest.raises(TaskSpecRejected, match="FROZEN_MANIFEST_HASH_MISMATCH"):
        evaluate_r83_frozen_validation(
            tampered, spec, frozen, heldout, trusted_fixed_pickle=True)

    extra = copy.deepcopy(heldout)
    extra[1]["measurements"] = copy.deepcopy(extra[0]["measurements"])
    with pytest.raises(TaskSpecRejected, match="RELABELED_DUPLICATE_TRAJECTORY"):
        evaluate_r83_frozen_validation(
            report, spec, frozen, extra, trusted_fixed_pickle=True)
