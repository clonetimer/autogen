"""R83e: real-source adapter's explicit TEST-free research contract.

Fixtures construct fake source receipts and numeric CSVs; they are NOT
Basilisk execution. A separate Linux Action must test actual native physics.
"""
from __future__ import annotations

import pytest

from phm_design.adapters.sat_rw import prepare
from phm_design.io import read_json
from test_r82_native_campaign_gate import campaign_fixture


def test_opt_in_test_free_native_preparation_preserves_observation_contract(tmp_path):
    candidate = campaign_fixture(
        tmp_path / "synthetic_campaign", splits=("train", "validation"))
    # No TEST split is a deliberate R83d scientific design, not permission to
    # quietly relax the general-purpose adapter for every caller.
    with pytest.raises(ValueError, match="EMPTY_DATASET"):
        prepare(candidate, tmp_path / "strict_rejects_empty_test")
    with pytest.raises(ValueError, match="ALLOW_EMPTY_HOLDOUT_MUST_BE_BOOLEAN"):
        prepare(candidate, tmp_path / "wrong_flag", allow_empty_holdout="yes")
    assert not (tmp_path / "wrong_flag").exists()

    result = prepare(candidate, tmp_path / "research_no_test",
                     allow_empty_holdout=True)
    assert result["status"] == "PREPARED"
    assert result["channel_count"] == 6
    assert result["fit_receives_holdout"] is False
    assert result["counts"]["train"]["cases"] == 2
    assert result["counts"]["validation"]["cases"] == 2
    assert "test" not in result["counts"]
    assert result["native_train_candidate_gate"]["train_pairs"] == 1

    dev = read_json(tmp_path / "research_no_test/development/dataset.json")
    holdout = read_json(tmp_path / "research_no_test/holdout/dataset.json")
    assert {item["split"] for item in dev["samples"]} == {
        "train", "validation"}
    assert holdout["samples"] == []
    assert dev["contract"]["evidence_type"] == "simulation"
    assert dev["provenance"]["source_kind"] == "native_basilisk"
    assert len(dev["contract"]["channels"]) == 6
    assert all(channel["role"] == "deployment_input"
               for channel in dev["contract"]["channels"])


def test_test_free_flag_does_not_hide_bad_train_physics_or_missing_pairs(tmp_path):
    candidate = campaign_fixture(tmp_path / "source", splits=("train",))
    native_report = candidate / "cases/train-rw-fault/astrograph/dataset_contract.json"
    contract = read_json(native_report)
    contract["quality"]["physics_validation_passed"] = False
    from phm_design.io import write_json
    write_json(native_report, contract)
    with pytest.raises(ValueError, match="R82_NATIVE_PAIR_PHYSICS_DIAGNOSTIC_REVIEW_REQUIRED"):
        prepare(candidate, tmp_path / "reject_bad_physics",
                allow_empty_holdout=True)
    assert not (tmp_path / "reject_bad_physics").exists()
