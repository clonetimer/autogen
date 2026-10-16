"""Pre-registered *research-only* operating-regime mapping for native RW data.

Split thresholds apply to a real source simulation parameter, inclination_deg.
No fault label, source split, outcome metric, or telemetry values determine an
environment. This creates a provisional research grouping, NOT expert physics
review or independent OOD qualification.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from ..io import read_json, write_json, sha_file
from .taskspec import compile_task_spec

# Frozen BEFORE source performance or label inspection.
INCLINATION_BINS_DEG = (38.0, 56.0)


def _regime(inclination):
    if type(inclination) not in (float, int) or not 0.0 <= inclination <= 90.0:
        raise ValueError("R82_NATIVE_INCLINATION_OUTSIDE_PHYSICAL_RANGE")
    return ("inclination_low" if inclination < INCLINATION_BINS_DEG[0]
            else "inclination_medium" if inclination < INCLINATION_BINS_DEG[1]
            else "inclination_high")


def create_prespecified_native_regime_map(*, source_campaign, preparation_report,
                                         train_manifest, task_spec, output,
                                         research_only=False):
    """Create an explicit source TRAIN-only mapping, never a FINAL report.

    Requires research_only=True to acknowledge that inclination bands have
    NOT received independent expert approval. No possibility of claiming
    cross-equipment validation from three parameter bands of one simulator.
    """
    if research_only is not True:
        raise ValueError("R82_INCLINATION_REGIME_EXPERT_REVIEW_NOT_PROVIDED")
    plan = compile_task_spec(read_json(task_spec))
    if (plan["train_data_contract"]["group_keys"] != ["rw_run_lineage_id"] or
            plan["train_data_contract"]["environment_axes"] != ["operating_regime"]):
        raise ValueError("R82_NATIVE_REGIME_TASK_CONTRACT_UNSUPPORTED")
    report = read_json(preparation_report)
    manifest = read_json(train_manifest)
    if any(x.get("split") != "train" for x in manifest["samples"]):
        raise ValueError("R82_RESEARCH_REGIME_TRAIN_ONLY")
    origins = {r["id"]: r["source_case"] for r in report["source_records"]}
    ids = [sample["id"] for sample in manifest["samples"]]
    if len(ids) != len(set(ids)) or not all(key in origins for key in ids):
        raise ValueError("R82_REGIME_SOURCE_CASE_MAPPING_INCOMPLETE")
    result, groups, source_specs = {}, defaultdict(set), {}
    for sample in manifest["samples"]:
        case_id = origins[sample["id"]]
        source_file = (Path(source_campaign) / "cases" / case_id /
                       "task_spec.json")
        if not source_file.is_file():
            raise FileNotFoundError("R82_NATIVE_REGIME_SOURCE_TASK_NOT_FOUND")
        doc = read_json(source_file)
        inclination = doc["parameters"].get("inclination_deg")
        regime = _regime(inclination)
        group = sample["group_id"]
        groups[group].add(regime)
        result[sample["id"]] = {
            "rw_run_lineage_id": group,
            "operating_regime": regime,
        }
        source_specs[case_id] = sha_file(source_file)
    if any(len(regimes) != 1 for regimes in groups.values()):
        raise ValueError("R82_NATIVE_PHYSICAL_GROUP_REGIME_DRIFT")
    output = Path(output)
    if output.exists():
        raise FileExistsError("R82_NATIVE_REGIME_MAP_OUTPUT_EXISTS")
    write_json(output, {
        "schema": "phm.industrial.source-physical-map.v1",
        "records": result,
    })
    audit = {
        "schema": "phm.industrial.r82-prespecified-regime-research.v1",
        "research_only": True,
        "physical_source_independently_attested": False,
        "operating_regime_expert_reviewed": False,
        "selection_scores_not_used_in_mapping": True,
        "source_labels_not_used_in_mapping": True,
        "boundaries_deg": list(INCLINATION_BINS_DEG),
        "count_by_regime": {
            name: sum(1 for val in groups.values() if name in val)
            for name in ("inclination_low", "inclination_medium", "inclination_high")
        },
        "train_samples": len(ids),
        "train_physical_groups": len(groups),
        "source_task_sha256": source_specs,
    }
    write_json(output.with_suffix(".research_audit.json"), audit)
    return audit
