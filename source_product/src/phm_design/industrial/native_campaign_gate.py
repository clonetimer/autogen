"""Trusted-preparation gate for actually produced SatAgentSimulation RW cases.

Inspects source TRAIN contract, matching task spec, source trace and runtime
report. Other splits are never opened by this gate. It checks candidate
evidence consistency, not the truth of the physical simulator or independent
expert qualification. This module does not fit, select or evaluate models.
"""
from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

from ..io import read_json, safe_child, sha_file, digest

_RW_CLASSES = frozenset({
    "ADCS_RW_FRICTION_INCREASE", "ADCS_RW_JAM",
    "ADCS_RW_TORQUE_AUTHORITY_LOSS",
})
_NATIVE_FIDELITY = "A_ENGINEERING_BASILISK_NATIVE"
_NATIVE_CAPABILITY = "subsystem.adcs_unified_native.v1"
_RAW_RW_FIELDS = {
    *(f"adcs.rw.speed_rad_s_{i}" for i in range(3)),
    *(f"adcs.rw.command_torque_nm_{i}" for i in range(3)),
}
_ACCEPTED_BSK = frozenset({"2.11.0", "2.11.0+satfix1"})


def require_native_rw_candidate(contract):
    """Reject incomplete or proxy Basilisk data before *any* preparation."""
    s = contract.get("simulation") or {}
    fidelity = s.get("fidelity") or {}
    q = contract.get("quality") or {}
    if (contract.get("status") != "complete" or
            s.get("backend") != "basilisk" or
            s.get("capability_id") != _NATIVE_CAPABILITY or
            s.get("bsk_version") not in _ACCEPTED_BSK or
            fidelity.get("fidelity_level") != _NATIVE_FIDELITY or
            fidelity.get("formal_training_eligible") is not True or
            fidelity.get("bsk_version_accepted") is not True or
            fidelity.get("runtime_truth_status") is None or
            not str(fidelity.get("runtime_truth_status")).startswith(
                "instantiated_connected_recorded") or
            not isinstance(fidelity.get("instantiated_module_count"), int) or
            fidelity["instantiated_module_count"] <= 0):
        raise ValueError("R82_NATIVE_RUNTIME_CANDIDATE_EVIDENCE_REQUIRED")
    if (q.get("formal_training_eligible") is not True or
            q.get("physics_validation_passed") is not True or
            q.get("formal_training_qualified") is not True or
            q.get("physics_candidate_ready") is not True or
            q.get("diagnostic_validation_passed") is not True or
            q.get("basilisk_diagnostic_candidate") is not True):
        raise ValueError("R82_NATIVE_PAIR_PHYSICS_DIAGNOSTIC_REVIEW_REQUIRED")
    # Formal training_ready is intentionally not required here: the producer
    # keeps it false while independent expert and FINAL reviews remain pending.


def review_native_rw_train_campaign(campaign):
    """Return a source TRAIN evidence report, not a Basilisk execution receipt.

    Examine only case-contracts declared TRAIN. Do not open VALIDATION/TEST
    task specs, reports or CSVs and do not supply this report to selection
    as an independent score.
    """
    campaign = Path(campaign)
    files = sorted((campaign / "cases").glob("*/astrograph/dataset_contract.json"))
    if not files:
        raise ValueError("R82_NO_RW_CASE_CONTRACTS")
    pairs = defaultdict(dict)
    version = None
    checks = []
    for cp in files:
        c = read_json(cp)
        if c.get("split") != "train":
            continue
        root = cp.parent.parent
        require_native_rw_candidate(c)
        pairing = c.get("pairing") or {}
        role = pairing.get("case_role")
        pair_id = pairing.get("pair_id")
        label = c.get("ground_truth", {}).get("class_id")
        if (not isinstance(pair_id, str) or not pair_id or
                role not in ("fault", "nominal") or
                label not in _RW_CLASSES | {"NORMAL"} or
                (role == "nominal") != (label == "NORMAL") or
                role in pairs[pair_id] or
                c.get("ground_truth", {}).get("pair_id") != pair_id):
            raise ValueError("R82_RW_TRAIN_PAIR_CLASS_MISMATCH")
        spec_path = root / "task_spec.json"
        spec = read_json(spec_path)
        simulation = spec.get("simulation") or {}
        declared = c["simulation"]
        if (spec.get("capability_id") != _NATIVE_CAPABILITY or
                simulation.get("backend") != "basilisk" or
                simulation.get("seed") != declared.get("seed") or
                simulation.get("sample_s") != declared.get("sample_s")):
            raise ValueError("R82_RW_SOURCE_TASK_CONTRACT_MISMATCH")
        case_version = declared["bsk_version"]
        if version is not None and case_version != version:
            raise ValueError("R82_MIXED_BASILISK_SOURCE_VERSIONS")
        version = case_version
        # Runtime report must contain the instantiated Basilisk evidence,
        # not merely a file named 'dataset_contract.json'.
        report_path = safe_child(
            root, c.get("simulation_report_file", "astrograph/simulation_report.json"))
        report = read_json(report_path)
        manifest = (report.get("runtime_evidence") or {}).get("runtime_manifest") or {}
        instantiated = manifest.get("instantiated_modules")
        summary = report.get("runtime_summary") or {}
        if (report.get("fidelity") != declared.get("fidelity") or
                not isinstance(instantiated, list) or not instantiated or
                not str(summary.get("runtime_truth_status", "")).startswith(
                    "instantiated_connected_recorded")):
            raise ValueError("R82_NATIVE_RUNTIME_REPORT_INCONSISTENT")
        # SHA-based lineage is shared by paired nominal/fault cases, and must
        # not be derived from the fault label or target wheel index.
        source_lineage = digest({
            "parameters": spec.get("parameters"),
            "seed": simulation.get("seed"),
        })
        path = safe_child(root, c["telemetry"]["source_trace_file"])
        if path == root / "astrograph" / "telemetry.csv" or not path.is_file():
            raise ValueError("R82_RAW_NATIVE_TRACE_REQUIRED")
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if not _RAW_RW_FIELDS.issubset(reader.fieldnames or []):
                raise ValueError("R82_RAW_RW_OBSERVATION_FIELDS_MISSING")
            count = 0
            previous = None
            for row in reader:
                try:
                    t = float(row["time_s"])
                    for name in _RAW_RW_FIELDS:
                        if not math.isfinite(float(row[name])):
                            raise ValueError("R82_NONFINITE_NATIVE_RW_TELEMETRY")
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError("R82_INVALID_NATIVE_RW_TELEMETRY") from exc
                if previous is not None and (
                        t <= previous or not math.isclose(
                            t - previous, float(declared["sample_s"]),
                            rel_tol=1e-6, abs_tol=1e-7)):
                    raise ValueError("R82_NATIVE_SAMPLE_CADENCE_MISMATCH")
                previous = t
                count += 1
        if count < 16 or count != c["telemetry"]["row_count"]:
            raise ValueError("R82_NATIVE_TRACE_LENGTH_MISMATCH")
        pairs[pair_id][role] = (label, source_lineage, simulation["seed"])
        checks.append({
            "case_id": c["dataset_id"], "case_contract_sha256": sha_file(cp),
            "source_trace_sha256": sha_file(path),
            "simulation_report_sha256": sha_file(report_path),
            "task_spec_sha256": sha_file(spec_path),
            "source_lineage_sha256": source_lineage,
        })
    if not checks:
        raise ValueError("R82_NO_NATIVE_RW_TRAIN_CASES")
    for pair_id, pair in pairs.items():
        if set(pair) != {"fault", "nominal"}:
            raise ValueError("R82_NATIVE_RW_PAIR_INCOMPLETE")
        fault = pair["fault"]
        nominal = pair["nominal"]
        if fault[0] not in _RW_CLASSES or nominal[0] != "NORMAL" or (
                fault[1], fault[2]) != (nominal[1], nominal[2]):
            raise ValueError("R82_RW_PAIR_PARAMETER_SEED_MISMATCH")
    return {
        "schema": "phm.industrial.r82-native-campaign-gate.v1",
        "role": "TRUSTED_PREPARATION_TRAIN_ONLY",
        "native_basilisk_candidate_source_checked": True,
        "train_pairs": len(pairs),
        "train_cases": len(checks),
        "source_bsk_version_claimed": version,
        "train_case_evidence": checks,
        "validation_test_trace_accessed": False,
        "physical_execution_independently_attested": False,
        "model_trained": False,
        "independent_qualification": False,
    }


def main(argv=None):
    """CLI is a trusted preparation-side review, not an AutoDesign worker."""
    import argparse
    from ..io import write_json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.report.exists():
        raise FileExistsError("R82_NATIVE_CAMPAIGN_REVIEW_EXISTS")
    reviewed = review_native_rw_train_campaign(args.campaign)
    write_json(args.report, reviewed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
