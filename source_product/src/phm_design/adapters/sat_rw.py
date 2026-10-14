"""Basilisk campaign -> fixed physical-index wheel telemetry. No oracle remapping."""
from __future__ import annotations
import csv
from pathlib import Path
from ..data import Dataset, validate_contract
from ..io import digest, read_json, safe_child, sha_file, write_json

CLASSES = ["NOMINAL", "ADCS_RW_FRICTION_INCREASE", "ADCS_RW_JAM", "ADCS_RW_TORQUE_AUTHORITY_LOSS"]
CHANNELS = ([{"name": f"adcs.rw.speed_rad_s_{i}", "unit": "rad/s", "role": "deployment_input",
              "provenance": "Basilisk wheel-speed recorder; ideal simulated measurement; hardware sensor error unvalidated"} for i in range(3)] +
            [{"name": f"adcs.rw.command_torque_nm_{i}", "unit": "N*m", "role": "deployment_input",
              "provenance": "Basilisk controller motor-command recorder; NOT actual applied torque"} for i in range(3)])


def prepare(campaign: str | Path, output: str | Path) -> dict:
    campaign, output = Path(campaign), Path(output)
    if output.exists():
        raise FileExistsError("OUTPUT_EXISTS")
    contracts = sorted((campaign / "cases").glob("*/astrograph/dataset_contract.json"))
    if not contracts:
        raise ValueError("NO_CASE_CONTRACTS")
    records, source_records, sample_s = [], [], None
    output.mkdir(parents=True)
    for cp in contracts:
        c = read_json(cp)
        root = cp.parent.parent
        simulation = c["simulation"]
        if c.get("status") != "complete" or simulation.get("backend") != "basilisk" or simulation.get("capability_id") != "subsystem.adcs_unified_native.v1":
            raise ValueError("NATIVE_COMPLETE_ADCS_REQUIRED")
        if sample_s is not None and float(simulation["sample_s"]) != sample_s:
            raise ValueError("MIXED_SAMPLE_PERIODS")
        sample_s = float(simulation["sample_s"])
        role, split = c["pairing"]["case_role"], c["split"]
        if role not in {"nominal", "fault"} or split not in {"train", "validation", "test"}:
            raise ValueError("CASE_ROLE_OR_SPLIT_INVALID")
        label = "NOMINAL" if role == "nominal" else c["ground_truth"]["class_id"]
        if label not in CLASSES:
            raise ValueError("FAULT_OUTSIDE_ONTOLOGY")
        spec = read_json(root / "task_spec.json")
        # Same parameters+seed define a lineage even if someone changes Pair IDs.
        lineage = digest({"parameters": spec["parameters"], "seed": simulation["seed"]})
        group_id = lineage
        source = safe_child(root, c["telemetry"]["source_trace_file"])
        if source == cp.parent / "telemetry.csv":
            raise ValueError("ORACLE_PROJECTED_TELEMETRY_FORBIDDEN")
        item_id = digest({"lineage": lineage, "role": role, "label": label})[:24]
        destination = output / ("holdout" if split == "test" else "development") / "observations" / f"{item_id}.csv"
        destination.parent.mkdir(parents=True, exist_ok=True)
        fields = ["time_s", *[x["name"] for x in CHANNELS]]
        with source.open(newline="", encoding="utf-8") as src, destination.open("w", newline="", encoding="utf-8") as dst:
            reader, writer = csv.DictReader(src), csv.DictWriter(dst, fieldnames=fields)
            if not set(fields).issubset(reader.fieldnames or []):
                raise ValueError("RAW_FIXED_WHEEL_CHANNELS_MISSING")
            writer.writeheader()
            for row in reader:
                # No use of ground-truth wheel index, onset, or nominal twin here.
                writer.writerow({name: row[name] for name in fields})
        record = {"id": item_id, "group_id": group_id, "lineage_id": lineage, "split": split, "label": label,
                  "path": f"observations/{item_id}.csv", "sha256": sha_file(destination)}
        records.append(record)
        source_records.append({"id": item_id, "source_case": c["dataset_id"], "source_trace_sha256": sha_file(source),
                               "contract_sha256": sha_file(cp), "quality": c.get("quality", {})})
    contract = {"device_family": "satellite.adcs.reaction_wheel.fixed_three_wheels",
                "task": "fixed_length_case_classification", "channels": CHANNELS, "classes": CLASSES,
                "sample_s": sample_s, "evidence_type": "simulation",
                "scope": "Three fixed orthogonal wheel channels, one injected known fault per case, ideal simulated sensing; no flight, RUL, unknown-fault or online-alarm claim"}
    validate_contract(contract)
    for name, splits in (("development", {"train", "validation"}), ("holdout", {"test"})):
        write_json(output / name / "dataset.json", {"schema": "phm.sequence-dataset.v1", "dataset_id": f"sat-rw-v02-{name}",
                   "contract": contract, "samples": [r for r in records if r["split"] in splits],
                   "provenance": {"adapter": "sat_rw.fixed_index.v1", "role": "trusted_data_preparation",
                                  "source_kind": "native_basilisk", "source_campaign_hash": digest(source_records),
                                  "real_device_evidence": False}})
    dev = Dataset(output / "development/dataset.json", allowed_splits={"train", "validation"})
    test = Dataset(output / "holdout/dataset.json", allowed_splits={"test"})
    from ..data import reject_overlap
    reject_overlap(dev.samples, test.samples)
    dev.verify_files(); test.verify_files()
    summary = {"status": "PREPARED", "counts": {**dev.counts(), **test.counts()},
               "channel_count": len(CHANNELS), "oracle_wheel_remapping": False,
               "actual_torque_input": False, "source_records": source_records,
               "fit_receives_holdout": False, "scope": contract["scope"]}
    write_json(output / "preparation_report.json", summary)
    write_json(output / "units.json", {c["name"]: c["unit"] for c in CHANNELS})
    return summary
