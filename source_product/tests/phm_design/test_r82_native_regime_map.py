"""Prespecified native RW study regimes must never be inferred from labels."""
from pathlib import Path

import pytest

from phm_design.io import read_json, write_json
from phm_design.industrial.r82_native_regime_map import (
    _regime, create_prespecified_native_regime_map,
)

HERE = Path(__file__).resolve().parents[2]
TASK = HERE / "examples/industrial/satellite_rw_native_offline_train.json"


def fixture(tmp_path):
    campaign = tmp_path / "campaign"
    origin_records, samples = [], []
    for n, inclination in enumerate((25., 47., 67.)):
        pair = f"case-{n}"
        folder = campaign / "cases" / pair
        folder.mkdir(parents=True)
        write_json(folder / "task_spec.json",
                   {"parameters": {"inclination_deg": inclination}})
        for i in range(2):
            ident = f"train-{n}-{i}"
            # Deliberately no fault labels: source mapping must need none.
            samples.append({"id": ident, "split": "train",
                            "group_id": f"physical-{n}"})
            origin_records.append({"id": ident, "source_case": pair})
    report = tmp_path / "prepared.json"
    manifest = tmp_path / "train.json"
    write_json(report, {"source_records": origin_records})
    write_json(manifest, {"samples": samples})
    return campaign, report, manifest


def test_predeclared_source_inclination_groups_without_label_access(tmp_path):
    campaign, report, manifest = fixture(tmp_path)
    output = tmp_path / "regime_map.json"
    audit = create_prespecified_native_regime_map(
        source_campaign=campaign, preparation_report=report,
        train_manifest=manifest, task_spec=TASK, output=output,
        research_only=True)
    assert audit["count_by_regime"] == {
        "inclination_low": 1, "inclination_medium": 1,
        "inclination_high": 1}
    assert audit["train_samples"] == 6
    assert audit["train_physical_groups"] == 3
    assert audit["source_labels_not_used_in_mapping"] is True
    assert audit["operating_regime_expert_reviewed"] is False
    assigned = read_json(output)["records"]
    assert assigned["train-0-0"] == assigned["train-0-1"]
    assert len({item["operating_regime"] for item in assigned.values()}) == 3
    assert read_json(output.with_suffix(".research_audit.json")) == audit


def test_mapping_requires_explicit_nonqualified_research_mode(tmp_path):
    campaign, report, manifest = fixture(tmp_path)
    with pytest.raises(ValueError, match="R82_INCLINATION_REGIME_EXPERT_REVIEW_NOT_PROVIDED"):
        create_prespecified_native_regime_map(
            source_campaign=campaign, preparation_report=report,
            train_manifest=manifest, task_spec=TASK,
            output=tmp_path / "regime.json")


def test_fault_truth_column_never_needed_by_regime_map(tmp_path):
    campaign, report, manifest = fixture(tmp_path)
    data = read_json(manifest)
    data["samples"][0]["split"] = "test"
    write_json(manifest, data)
    with pytest.raises(ValueError, match="R82_RESEARCH_REGIME_TRAIN_ONLY"):
        create_prespecified_native_regime_map(
            source_campaign=campaign, preparation_report=report,
            train_manifest=manifest, task_spec=TASK,
            output=tmp_path / "regime.json", research_only=True)


def test_invalid_physical_inclinations_refused():
    for invalid in (-1., 92., "45", None):
        with pytest.raises(ValueError, match="R82_NATIVE_INCLINATION_OUTSIDE_PHYSICAL_RANGE"):
            _regime(invalid)
    assert _regime(38.) == "inclination_medium"
    assert _regime(56.) == "inclination_high"
