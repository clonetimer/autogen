"""R82 native-fidelity candidate gate tests with FAKE CSV/source receipts.

The constructed fixtures are NOT Basilisk outputs and cannot support physical
fidelity or independent OOD claims. The tests verify only rejection contracts.
"""
import copy
import csv

import pytest

from phm_design.io import read_json, write_json
from phm_design.industrial.native_campaign_gate import (
    review_native_rw_train_campaign, require_native_rw_candidate,
)
from phm_design.adapters.sat_rw import prepare


def campaign_fixture(root, *, splits=("train", "validation", "test")):
    fields = ["time_s", *(f"adcs.rw.speed_rad_s_{i}" for i in range(3)),
              *(f"adcs.rw.command_torque_nm_{i}" for i in range(3))]
    for index, split in enumerate(splits):
        for role in ("fault", "nominal"):
            case_id = f"{split}-rw-{role}"
            folder = root / "cases" / case_id
            (folder / "astrograph").mkdir(parents=True)
            seed = 8001 + index
            fidelity = {
                "fidelity_level": "A_ENGINEERING_BASILISK_NATIVE",
                "formal_training_eligible": True,
                "bsk_version_accepted": True,
                "instantiated_module_count": 2,
                "runtime_truth_status": "instantiated_connected_recorded",
            }
            quality = {
                "formal_training_eligible": True,
                "physics_validation_passed": True,
                "formal_training_qualified": True,
                "physics_candidate_ready": True,
                "diagnostic_validation_passed": True,
                "basilisk_diagnostic_candidate": True,
                "training_ready": False,
            }
            spec = {
                "capability_id": "subsystem.adcs_unified_native.v1",
                "simulation": {
                    "backend": "basilisk", "sample_s": 1., "seed": seed},
                "parameters": {"wheel_config": "orthogonal_3",
                               "rate": float(100 + index)},
            }
            c = {
                "dataset_id": case_id, "split": split, "status": "complete",
                "pairing": {"pair_id": f"{split}-rw", "case_role": role},
                "ground_truth": {
                    "class_id": ("ADCS_RW_JAM" if role == "fault" else "NORMAL"),
                    "pair_id": f"{split}-rw"},
                "simulation": {
                    "capability_id": "subsystem.adcs_unified_native.v1",
                    "backend": "basilisk", "bsk_version": "2.11.0+satfix1",
                    "seed": seed, "sample_s": 1., "fidelity": fidelity},
                "quality": quality,
                "telemetry": {"source_trace_file": "trace.csv",
                              "row_count": 20},
                "simulation_report_file": "astrograph/simulation_report.json",
            }
            trace = folder / "trace.csv"
            with trace.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for t in range(20):
                    row = {name: float(100 * index + 7 * int(role == "fault") +
                                   channel * .01 + t * .1)
                           for channel, name in enumerate(fields[1:])}
                    writer.writerow({"time_s": float(t), **row})
            write_json(folder / "task_spec.json", spec)
            write_json(folder / "astrograph/dataset_contract.json", c)
            write_json(folder / "astrograph/simulation_report.json", {
                "fidelity": fidelity,
                "runtime_summary": {
                    "runtime_truth_status": "instantiated_connected_recorded"},
                "runtime_evidence": {
                    "runtime_manifest": {
                        "instantiated_modules": ["rwEffector", "rwMotorTorque"]}},
            })
    return root


def _contract(root, *, role="fault", split="train"):
    return root / "cases" / f"{split}-rw-{role}" / (
        "astrograph/dataset_contract.json")


def test_upstream_normal_to_downstream_nominal_is_label_only_mapping(tmp_path):
    campaign = campaign_fixture(tmp_path / "campaign", splits=("train",))
    contract = read_json(_contract(campaign, role="nominal"))
    assert contract["ground_truth"]["class_id"] == "NORMAL"
    assert review_native_rw_train_campaign(campaign)["train_pairs"] == 1
    from phm_design.adapters.sat_rw import CLASSES
    assert "NOMINAL" in CLASSES and "NORMAL" not in CLASSES


def test_complete_candidate_pair_checks_train_only_and_prepares_six_channels(tmp_path):
    campaign = campaign_fixture(tmp_path / "campaign")
    audit = review_native_rw_train_campaign(campaign)
    assert audit["train_pairs"] == 1
    assert audit["train_cases"] == 2
    assert audit["physical_execution_independently_attested"] is False
    assert audit["validation_test_trace_accessed"] is False
    prepared = tmp_path / "prepared"
    result = prepare(campaign, prepared)
    assert result["native_train_candidate_gate"]["train_pairs"] == 1
    assert (prepared / "development/dataset.json").exists()
    assert (prepared / "holdout/dataset.json").exists()
    assert result["channel_count"] == 6
    assert result["fit_receives_holdout"] is False


@pytest.mark.parametrize("section,key,code", [
    ("simulation", "backend", "R82_NATIVE_RUNTIME_CANDIDATE_EVIDENCE_REQUIRED"),
    ("fidelity", "formal_training_eligible",
     "R82_NATIVE_RUNTIME_CANDIDATE_EVIDENCE_REQUIRED"),
    ("fidelity", "runtime_truth_status",
     "R82_NATIVE_RUNTIME_CANDIDATE_EVIDENCE_REQUIRED"),
    ("quality", "physics_validation_passed",
     "R82_NATIVE_PAIR_PHYSICS_DIAGNOSTIC_REVIEW_REQUIRED"),
    ("quality", "basilisk_diagnostic_candidate",
     "R82_NATIVE_PAIR_PHYSICS_DIAGNOSTIC_REVIEW_REQUIRED"),
])
def test_native_claims_without_actual_candidate_signals_refused(tmp_path, section, key, code):
    campaign = campaign_fixture(tmp_path / "campaign", splits=("train",))
    cp = _contract(campaign)
    c = read_json(cp)
    if section == "fidelity":
        c["simulation"]["fidelity"][key] = False if key != "runtime_truth_status" else None
    elif section == "quality":
        c["quality"][key] = False
    else:
        c["simulation"][key] = "python"
    write_json(cp, c)
    with pytest.raises(ValueError, match=code):
        review_native_rw_train_campaign(campaign)
    with pytest.raises(ValueError, match=code):
        prepare(campaign, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_missing_nominal_pair_refused(tmp_path):
    campaign = campaign_fixture(tmp_path / "campaign", splits=("train",))
    (_contract(campaign, role="nominal")).unlink()
    with pytest.raises(ValueError, match="R82_NATIVE_RW_PAIR_INCOMPLETE"):
        review_native_rw_train_campaign(campaign)


def test_runtime_report_missing_modules_refused(tmp_path):
    campaign = campaign_fixture(tmp_path / "campaign", splits=("train",))
    report_path = _contract(campaign).parent / "simulation_report.json"
    report = read_json(report_path)
    report["runtime_evidence"]["runtime_manifest"]["instantiated_modules"] = []
    write_json(report_path, report)
    with pytest.raises(ValueError, match="R82_NATIVE_RUNTIME_REPORT_INCONSISTENT"):
        review_native_rw_train_campaign(campaign)


def test_nominal_fault_parameters_must_match_for_same_pair(tmp_path):
    campaign = campaign_fixture(tmp_path / "campaign", splits=("train",))
    path = _contract(campaign, role="nominal").parent.parent / "task_spec.json"
    doc = read_json(path)
    doc["parameters"]["rate"] += 1
    write_json(path, doc)
    with pytest.raises(ValueError, match="R82_RW_PAIR_PARAMETER_SEED_MISMATCH"):
        review_native_rw_train_campaign(campaign)


def test_native_train_gate_does_not_open_test_or_validation_files(tmp_path):
    campaign = campaign_fixture(tmp_path / "campaign")
    for split in ("validation", "test"):
        cp = _contract(campaign, split=split)
        c = read_json(cp)
        c["simulation"]["backend"] = "python"  # only metadata; not read by gate
        write_json(cp, c)
        (_contract(campaign, split=split).parent / "simulation_report.json").unlink()
    audit = review_native_rw_train_campaign(campaign)
    assert audit["train_pairs"] == 1
    assert audit["validation_test_trace_accessed"] is False
    # Full preparation, unlike TRAIN gate, must refuse invalid non-TRAIN cases.
    with pytest.raises(ValueError, match="R82_NATIVE_RUNTIME_CANDIDATE_EVIDENCE_REQUIRED"):
        prepare(campaign, tmp_path / "full_prepared")


def test_identical_telemetry_bytes_across_split_are_rejected(tmp_path):
    """Preserve existing Dataset CROSS_SPLIT_LEAKAGE gate; no weakening."""
    from shutil import copyfile
    campaign = campaign_fixture(tmp_path / "campaign")
    from_file = _contract(campaign, role="nominal", split="train").parent.parent / "trace.csv"
    to_file = _contract(campaign, role="fault", split="validation").parent.parent / "trace.csv"
    copyfile(from_file, to_file)
    with pytest.raises(ValueError, match="CROSS_SPLIT_LEAKAGE"):
        prepare(campaign, tmp_path / "prepared")
