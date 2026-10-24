"""R83d preregistered, balanced Basilisk RW source campaign.

Planning is metadata-only; actual physics MUST run via SatAgentSimulation's
native fault-dataset factory. No proxy sensor simulation and no TEST/FINAL.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
from pathlib import Path
from shutil import copyfile

from ..data import Dataset
from ..io import read_json, safe_child, sha_file, write_json
from .r82_native_regime_map import _regime
from .taskspec import TaskSpecRejected

SCHEMA = "phm.r83d.native-balanced-protocol.v1"
COUNTS = {"train": 15, "validation": 6, "test": 0}
CENTERS = (29.0, 47.0, 65.0)
REGIMES = ("inclination_low", "inclination_medium", "inclination_high")
FAULTS = (
    "ADCS_RW_FRICTION_INCREASE",
    "ADCS_RW_JAM",
    "ADCS_RW_TORQUE_AUTHORITY_LOSS",
)
CLASSES = ("NOMINAL", *FAULTS)


def _reject(reason):
    raise TaskSpecRejected(reason)


def _source_identity(config):
    if (config.get("counts") != COUNTS or
            config.get("fault_ids") != list(FAULTS) or
            config.get("simulation_backend") != "basilisk" or
            config.get("allow_test_proxy") is not False or
            config.get("capability_id") != "subsystem.adcs_unified_native.v1" or
            (config.get("duration_s"), config.get("sample_s"),
             config.get("solver_step_s")) != (60.0, 1.0, 0.2)):
        _reject("R83D_PREDECLARED_BASILISK_CONTRACT_REQUIRED")


def freeze_balanced_campaign(config, *, campaign_builder):
    """Set inclination strata BEFORE all simulations and model outcomes.

    The upstream builder produces the actual SatAgentSimulation paired task
    specification. Pair seed/parameters other than inclination remain intact.
    """
    _source_identity(config)
    campaign = campaign_builder(config)
    cases = campaign["campaign"]["cases"]
    expected = 2 * len(FAULTS) * sum(COUNTS.values())
    if len(cases) != expected:
        _reject("R83D_FACTORY_CASE_COUNT_CHANGED")
    pairs, seeds = defaultdict(dict), {}
    per_cell = Counter()
    for entry in cases:
        override = entry["overrides"]
        m = override["metadata"]
        pair = m["astrograph_pair_id"]
        split, fault, role = (m["astrograph_split"],
                              m["astrograph_fault_id"],
                              m["astrograph_case_role"])
        if (split not in COUNTS or fault not in FAULTS or
                role not in ("nominal", "fault") or
                role in pairs[pair] or
                not pair.startswith(split + "-" + fault.lower() + "-")):
            _reject("R83D_INVALID_NATIVE_PAIR_METADATA")
        try:
            index = int(pair.rsplit("-", 1)[-1])
        except (ValueError, TypeError) as exc:
            raise TaskSpecRejected("R83D_INVALID_SPLIT_PAIR_INDEX") from exc
        if not 0 <= index < COUNTS[split]:
            _reject("R83D_PAIR_INDEX_OUTSIDE_FROZEN_SPLIT")
        override["parameters"]["inclination_deg"] = CENTERS[index % 3]
        pairs[pair][role] = override
        if role == "fault":
            seed = override["simulation.seed"]
            if seed in seeds:
                _reject("R83D_DUPLICATE_PAIR_SEED")
            seeds[seed] = pair
            per_cell[(split, fault, REGIMES[index % 3])] += 1
    if len(pairs) != expected // 2:
        _reject("R83D_MISSING_NATIVE_PAIRS")
    for pair, roles in pairs.items():
        if set(roles) != {"nominal", "fault"}:
            _reject("R83D_UNPAIRED_NATIVE_SOURCE")
        normal, fault = roles["nominal"], roles["fault"]
        if (normal["parameters"] != fault["parameters"] or
                normal["simulation.seed"] != fault["simulation.seed"]):
            _reject("R83D_PHYSICAL_TWIN_DRIFT")
    for split, count in COUNTS.items():
        for fault in FAULTS:
            for regime in REGIMES:
                if per_cell[(split, fault, regime)] != count // 3:
                    _reject("R83D_REGIME_FAULT_GROUP_IMBALANCE")
    campaign["metadata"]["r83d_preregistration"] = {
        "schema": SCHEMA,
        "centers_deg": list(CENTERS),
        "split_pairs_per_fault": dict(COUNTS),
        "regime_is_split_index_mod_three": True,
        "train_or_validation_scores_used": False,
        "physical_execution_attested": False,
        "independent_qualification": False,
    }
    return campaign, {
        "schema": SCHEMA,
        "planned_cases": len(cases),
        "planned_pairs": len(pairs),
        "fault_pairs_by_split_regime": {
            "|".join(key): count for key, count in sorted(per_cell.items())},
        "physics_executed": False,
        "qualification": False,
    }


def _healthy_subset(pairs, *, split, regime, wanted):
    """Nominal selection based on pair IDs only, NEVER telemetry/results."""
    order = sorted(pairs, key=lambda pair: hashlib.sha256(
        ("R83D_HEALTHY_V1|" + split + "|" + regime + "|" + pair)
        .encode("utf-8")).hexdigest())
    if len(order) < wanted:
        _reject("R83D_INSUFFICIENT_PAIRED_HEALTHY")
    return set(order[:wanted])


def materialize_balanced_development(
    *, frozen_plan, prepared_manifest, preparation_report, destination,
):
    """Keep every fault and a preregistered, equal number of nominal twins.

    Trusted preparation may copy selected TRAIN/VAL traces. Model TRAIN workers
    use only TRAIN staging; reviewer validators use only VAL staging. The
    original full source campaign remains unchanged for physical audit.
    """
    plan = read_json(frozen_plan)
    if plan.get("metadata", {}).get("r83d_preregistration", {}).get("schema") != SCHEMA:
        _reject("R83D_FROZEN_CAMPAIGN_RECEIPT_REQUIRED")
    source = Dataset(prepared_manifest, allowed_splits={"train", "validation"})
    report = read_json(preparation_report)
    from_id = {entry["id"]: entry["source_case"] for entry in report["source_records"]}
    if set(from_id) != {s["id"] for s in source.samples}:
        _reject("R83D_PREPARATION_LINEAGE_INCOMPLETE")
    cases = {case["case_id"]: case for case in plan["campaign"]["cases"]}
    rows, groups, nominal, fault_counts = [], {}, defaultdict(set), Counter()
    for sample in source.samples:
        case = cases.get(from_id[sample["id"]])
        if case is None:
            _reject("R83D_CASE_NOT_IN_PREREGISTERED_PLAN")
        meta = case["overrides"]["metadata"]
        split = meta["astrograph_split"]
        fault, role = meta["astrograph_fault_id"], meta["astrograph_case_role"]
        pair = meta["astrograph_pair_id"]
        regime = _regime(case["overrides"]["parameters"]["inclination_deg"])
        label = "NOMINAL" if role == "nominal" else fault
        if sample["split"] != split or sample["label"] != label:
            _reject("R83D_PREPARED_ROLE_OR_LABEL_MISMATCH")
        if sample["group_id"] in groups and groups[sample["group_id"]] != (split, regime):
            _reject("R83D_GROUP_CROSSES_SOURCE_BOUNDARIES")
        groups[sample["group_id"]] = (split, regime)
        if role == "nominal":
            nominal[(split, regime)].add(pair)
        else:
            fault_counts[(split, regime, label)] += 1
        rows.append((sample, split, role, regime, pair))
    chosen_normal = {}
    for split, count in COUNTS.items():
        if count == 0:
            continue
        for regime in REGIMES:
            needed = count // 3
            if len(nominal[(split, regime)]) != needed * len(FAULTS):
                _reject("R83D_NATIVE_NOMINAL_PAIR_COVERAGE_DRIFT")
            for fault in FAULTS:
                if fault_counts[(split, regime, fault)] != needed:
                    _reject("R83D_NATIVE_FAULT_PAIR_COVERAGE_DRIFT")
            chosen_normal[(split, regime)] = _healthy_subset(
                nominal[(split, regime)], split=split,
                regime=regime, wanted=needed)
    selected = [(sample, split, regime) for sample, split, role, regime, pair in rows
                if role == "fault" or pair in chosen_normal[(split, regime)]]
    counts = Counter((split, regime, sample["label"])
                     for sample, split, regime in selected)
    for split, count in COUNTS.items():
        if count:
            for regime in REGIMES:
                for label in CLASSES:
                    if counts[(split, regime, label)] != count // 3:
                        _reject("R83D_SELECTED_DATA_NOT_EQUALLY_BALANCED")
    dest = Path(destination)
    if dest.exists():
        raise FileExistsError("R83D_BALANCED_DESTINATION_EXISTS")
    # Only selected TRAIN/VAL bytes are opened; no TEST/FINAL in source Dataset.
    for sample, _, _ in selected:
        path = safe_child(source.root, sample["path"])
        if not path.is_file() or sha_file(path) != sample["sha256"]:
            _reject("R83D_SELECTED_SOURCE_TRACE_TAMPERED")
    dest.mkdir(parents=True)
    train_map, val_map = {}, {}
    for sample, split, regime in selected:
        original, target = (safe_child(source.root, sample["path"]),
                            safe_child(dest, sample["path"]))
        target.parent.mkdir(parents=True, exist_ok=True)
        copyfile(original, target)
        if sha_file(target) != sample["sha256"]:
            _reject("R83D_STAGED_TRACE_CHANGED")
        row = {"rw_run_lineage_id": sample["group_id"],
               "operating_regime": regime}
        (train_map if split == "train" else val_map)[sample["id"]] = row
    subset = dict(source.raw)
    subset["samples"] = [dict(sample) for sample in source.samples
                         if sample["id"] in {
                             chosen[0]["id"] for chosen in selected}]
    subset["dataset_id"] = source.raw["dataset_id"] + "-r83d-balanced"
    subset["provenance"] = {
        **source.raw["provenance"],
        "r83d_balance_protocol": SCHEMA,
        "frozen_plan_sha256": sha_file(Path(frozen_plan)),
        "prepared_source_sha256": sha_file(Path(prepared_manifest)),
        "telemetry_used_for_nominal_selection": False,
        "final_test_opened": False,
        "independent_qualification": False,
    }
    write_json(dest / "dataset.json", subset)
    review = dest / "review"
    for label, entries in (("train", train_map), ("validation", val_map)):
        write_json(review / (label + "_physical_map.json"), {
            "schema": "phm.industrial.source-physical-map.v1",
            "records": entries,
        })
    audit = {
        "schema": SCHEMA,
        "class_regime_support": {
            "|".join(key): count for key, count in sorted(counts.items())},
        "train_selected": len(train_map),
        "validation_selected": len(val_map),
        "source_outcome_or_telemetry_guided_selection": False,
        "original_native_nominal_fault_pairs_preserved": True,
        "physics_execution_attested": False,
        "independent_qualification": False,
    }
    write_json(review / "balance_audit.json", audit)
    return audit


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("plan", "balance"), required=True)
    parser.add_argument("--factory-config", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--prepared-manifest", type=Path)
    parser.add_argument("--preparation-report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arg = parser.parse_args(argv)
    if arg.mode == "plan":
        if arg.factory_config is None:
            parser.error("--factory-config required")
        from sat_sim.fault_dataset_factory import (
            FaultDatasetFactoryConfig, SplitCounts, build_fault_dataset_campaign)
        config = dict(read_json(arg.factory_config))
        config.update(campaign_id="r83d_balanced_native_rw",
                      counts=dict(COUNTS), fault_ids=list(FAULTS),
                      simulation_backend="basilisk", allow_test_proxy=False,
                      duration_s=60.0, sample_s=1.0, solver_step_s=0.2)
        factory = FaultDatasetFactoryConfig(
            counts=SplitCounts(**config.pop("counts")),
            fault_ids=tuple(config.pop("fault_ids")), **config)
        campaign, audit = freeze_balanced_campaign(
            factory.to_dict(),
            campaign_builder=lambda _: build_fault_dataset_campaign(factory))
        if arg.output.exists():
            raise FileExistsError("R83D_PLAN_ALREADY_EXISTS")
        write_json(arg.output, campaign)
        write_json(arg.output.with_suffix(".audit.json"), audit)
    else:
        if any(x is None for x in (
                arg.plan, arg.prepared_manifest, arg.preparation_report)):
            parser.error("--plan --prepared-manifest --preparation-report required")
        materialize_balanced_development(
            frozen_plan=arg.plan, prepared_manifest=arg.prepared_manifest,
            preparation_report=arg.preparation_report, destination=arg.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
