"""R83d preregistration fixtures — synthetic metadata/CSV, NOT Basilisk."""
import copy
import csv
from collections import Counter
from pathlib import Path

import pytest

from phm_design.io import read_json, sha_file, write_json
from phm_design.industrial.r83d_native_protocol import (
    SCHEMA, COUNTS, FAULTS, CLASSES, REGIMES, freeze_balanced_campaign,
    materialize_balanced_development,
)
from phm_design.industrial.taskspec import TaskSpecRejected


def _factory(_):
    cases = []
    seed = 83_000
    for split, count in COUNTS.items():
        for fault in FAULTS:
            for index in range(count):
                pair = f"{split}-{fault.lower()}-{index:03d}"
                common = {
                    "astrograph_pair_id": pair,
                    "astrograph_split": split,
                    "astrograph_fault_id": fault,
                }
                parameters = {
                    "inclination_deg": 44.4,
                    "simulation_seed": seed,
                    "controller_k": 3.0 + index * .01,
                }
                for role in ("fault", "nominal"):
                    cases.append({
                        "case_id": pair + "-" + role,
                        "overrides": {
                            "metadata": {**common, "astrograph_case_role": role},
                            "simulation.seed": seed,
                            "parameters": copy.deepcopy(parameters),
                        },
                    })
                seed += 1
    return {"campaign": {"cases": cases}, "metadata": {}}


def _config():
    return {
        "counts": dict(COUNTS), "fault_ids": list(FAULTS),
        "duration_s": 60.0, "sample_s": 1.0, "solver_step_s": 0.2,
        "capability_id": "subsystem.adcs_unified_native.v1",
        "simulation_backend": "basilisk",
        "allow_test_proxy": False,
    }


def test_freeze_equal_strata_preserves_all_native_nominal_fault_twins():
    spec, receipt = freeze_balanced_campaign(_config(), campaign_builder=_factory)
    assert receipt["planned_cases"] == 126
    assert receipt["planned_pairs"] == 63
    assert receipt["physics_executed"] is False
    assert len(receipt["frozen_plan_canonical_sha256"]) == 64
    assert spec["metadata"]["r83d_preregistration"]["train_or_validation_scores_used"] is False
    for split in ("train", "validation"):
        for fault in FAULTS:
            for regime in REGIMES:
                want = COUNTS[split] // 3
                assert receipt["fault_pairs_by_split_regime"][
                    "|".join((split, fault, regime))] == want
    # Same protocol, same builder -> same preregistered cases and incline.
    again, audit = freeze_balanced_campaign(_config(), campaign_builder=_factory)
    assert spec == again
    assert receipt == audit
    from phm_design.io import digest
    assert receipt["frozen_plan_canonical_sha256"] == digest(spec)

    wrong = _config()
    wrong["allow_test_proxy"] = True
    with pytest.raises(TaskSpecRejected, match="BASILISK_CONTRACT_REQUIRED"):
        freeze_balanced_campaign(wrong, campaign_builder=_factory)
    def broken(config):
        result = _factory(config)
        result["campaign"]["cases"][0]["overrides"]["simulation.seed"] = 999
        return result
    with pytest.raises(TaskSpecRejected, match="PHYSICAL_TWIN_DRIFT"):
        freeze_balanced_campaign(_config(), campaign_builder=broken)


def _data(tmp_path):
    """Synthetic CSV fixture. Names resemble native inputs but no physics runs."""
    from phm_design.adapters.sat_rw import CHANNELS
    spec, _ = freeze_balanced_campaign(_config(), campaign_builder=_factory)
    spec_path = tmp_path / "frozen_campaign.json"
    write_json(spec_path, spec)
    root = tmp_path / "development"
    samples, source_records = [], []
    fieldnames = ["time_s", *(c["name"] for c in CHANNELS)]
    cases = spec["campaign"]["cases"]
    for index, case in enumerate(cases):
        m = case["overrides"]["metadata"]
        pair = m["astrograph_pair_id"]
        role = m["astrograph_case_role"]
        label = "NOMINAL" if role == "nominal" else m["astrograph_fault_id"]
        sample_id = f"synthetic_{index:04d}"
        relative = f"observations/{sample_id}.csv"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            for t in range(18):
                writer.writerow({"time_s": float(t), **{
                    c["name"]: index * 0.01 + t * .0002 + j * .01
                    for j, c in enumerate(CHANNELS)}})
        split = m["astrograph_split"]
        samples.append({
            "id": sample_id,
            "group_id": "reviewed_lineage_" + pair,
            "lineage_id": "reviewed_lineage_" + pair,
            "split": split, "label": label,
            "path": relative,
            "sha256": sha_file(path),
        })
        source_records.append({"id": sample_id,
                               "source_case": case["case_id"]})
    dataset = {
        "schema": "phm.sequence-dataset.v1",
        "dataset_id": "r83d_synthetic_contract_only",
        "contract": {
            "device_family": "satellite.adcs.reaction_wheel.fixed_three_wheels",
            "task": "fixed_length_case_classification",
            "channels": CHANNELS,
            "classes": list(CLASSES),
            "sample_s": 1.0,
            "evidence_type": "software_synthetic",
            "scope": "TEST FIXTURE NOT ACTUAL BASILISK",
        },
        "samples": samples,
        "provenance": {"source_kind": "synthetic_fixture_only"},
    }
    write_json(root / "dataset.json", dataset)
    preparation_report = tmp_path / "preparation.json"
    write_json(preparation_report, {"source_records": source_records})
    return spec_path, root / "dataset.json", preparation_report


def test_select_balanced_cohort_without_model_outcome_or_fault_twins_loss(tmp_path):
    plan, dataset, preparation = _data(tmp_path)
    dest = tmp_path / "balanced"
    audit = materialize_balanced_development(
        frozen_plan=plan, prepared_manifest=dataset,
        preparation_report=preparation, destination=dest)
    assert audit["train_selected"] == 60   # 3 regimes x 4 classes x 5 groups
    assert audit["validation_selected"] == 24  # 3 x 4 x 2
    assert audit["source_outcome_or_telemetry_guided_selection"] is False
    assert audit["physics_execution_attested"] is False
    subset = read_json(dest / "dataset.json")
    assert len(subset["samples"]) == 84
    assert subset["provenance"]["telemetry_used_for_nominal_selection"] is False
    assert set(s["split"] for s in subset["samples"]) == {"train", "validation"}
    groups = {}
    for sample in subset["samples"]:
        assert (dest / sample["path"]).exists()
        assigned = read_json(dest / "review" / (
            sample["split"] + "_physical_map.json"))["records"][sample["id"]]
        key = (sample["split"], assigned["operating_regime"], sample["label"])
        groups.setdefault(key, set()).add(sample["group_id"])
        assert assigned["rw_run_lineage_id"] == sample["group_id"]
    assert all(len(groups[(split, regime, label)]) == desired for
               split, desired in (("train", 5), ("validation", 2))
               for regime in REGIMES for label in CLASSES)
    # Healthy example subsampling must ALSO be balanced across the 3 native
    # fault-template sources, not just against the fault class counts.
    origin = {row["id"]: row["source_case"]
              for row in read_json(preparation)["source_records"]}
    mechanism_counts = Counter()
    for sample in subset["samples"]:
        if sample["label"] != "NOMINAL":
            continue
        source_case = origin[sample["id"]]
        regime = read_json(dest / "review" / (
            sample["split"] + "_physical_map.json"))["records"][
                sample["id"]]["operating_regime"]
        mechanism = next(f for f in FAULTS if source_case.startswith(
            sample["split"] + "-" + f.lower() + "-"))
        mechanism_counts[(sample["split"], regime, mechanism)] += 1
    for split, expected in (("train", 5), ("validation", 2)):
        for phase, regime in enumerate(REGIMES):
            base, remainder = divmod(expected, len(FAULTS))
            for index, fault in enumerate(FAULTS):
                quota = base + (1 if (index - phase) % len(FAULTS) < remainder else 0)
                assert mechanism_counts[(split, regime, fault)] == quota
    with pytest.raises(FileExistsError):
        materialize_balanced_development(
            frozen_plan=plan, prepared_manifest=dataset,
            preparation_report=preparation, destination=dest)


def test_protected_role_or_corrupted_selected_sample_refused(tmp_path):
    plan, dataset, preparation = _data(tmp_path)
    raw = read_json(dataset)
    raw["samples"][0]["split"] = "test"
    write_json(dataset, raw)
    with pytest.raises(ValueError, match="DATA_ROLE_VIOLATION"):
        materialize_balanced_development(
            frozen_plan=plan, prepared_manifest=dataset,
            preparation_report=preparation, destination=tmp_path / "no_test")
    assert not (tmp_path / "no_test").exists()
    raw["samples"][0]["split"] = "train"
    write_json(dataset, raw)
    # All faulty and a quota of healthy traces are selected. Pick a fault.
    fault = next(s for s in raw["samples"] if s["label"] != "NOMINAL")
    with (dataset.parent / fault["path"]).open("a", encoding="utf-8") as file:
        file.write("tampered without updating SHA\n")
    with pytest.raises(TaskSpecRejected, match="SOURCE_TRACE_TAMPERED"):
        materialize_balanced_development(
            frozen_plan=plan, prepared_manifest=dataset,
            preparation_report=preparation, destination=tmp_path / "no_tamper")
    assert not (tmp_path / "no_tamper").exists()



def test_two_nominal_fault_pairs_cannot_forge_one_physical_group(tmp_path):
    plan, manifest, prep = _data(tmp_path)
    raw = read_json(manifest)
    sample_a = raw["samples"][0]
    sample_b = next(x for x in raw["samples"]
                    if x["group_id"].endswith("-003")
                    and x["split"] == sample_a["split"])
    # Cases 000 and 003 were preregistered in the *same* inclination bin.
    # Same physical lineage renamed onto a different pair must never raise
    # independent per-class support; rejection happens before reading traces.
    sample_b["group_id"] = sample_a["group_id"]
    write_json(manifest, raw)
    with pytest.raises(TaskSpecRejected, match="DISTINCT_PAIRS_REUSE_PHYSICAL_GROUP"):
        materialize_balanced_development(
            frozen_plan=plan, prepared_manifest=manifest,
            preparation_report=prep, destination=tmp_path / "no_forged_groups")
    assert not (tmp_path / "no_forged_groups").exists()


def test_nominal_fault_twins_must_both_exist_in_source_before_subsampling(tmp_path):
    plan, manifest, prep = _data(tmp_path)
    raw = read_json(manifest)
    normal = next(x for x in raw["samples"] if x["label"] == "NOMINAL")
    raw["samples"].remove(normal)
    source = read_json(prep)
    source["source_records"] = [x for x in source["source_records"]
                                if x["id"] != normal["id"]]
    write_json(prep, source)
    write_json(manifest, raw)
    with pytest.raises(TaskSpecRejected, match="NOMINAL_FAULT_PAIR_INCOMPLETE"):
        materialize_balanced_development(
            frozen_plan=plan, prepared_manifest=manifest,
            preparation_report=prep, destination=tmp_path / "no_incomplete")
