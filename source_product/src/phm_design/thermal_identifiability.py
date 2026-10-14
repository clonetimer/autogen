"""R64 offline thermal signature identifiability audit (never a diagnostic inference path).

Requires five *paired, labelled, already disclosed* Basilisk trajectories per world,
including simulator-only onset truth. The baseline is an oracle control not available
at deployment; no outputs of this module can qualify a model or be used as live features.
No model is fitted or tuned on the historical evaluation worlds.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "phm.r64.thermal-identifiability.v1"
EXPECTED_HEADER = (
    "time_s", "illumination", "heat_a_command_w", "heat_b_command_w",
    "temperature_a_k", "temperature_b_k",
)
CLASSES = (
    "NOMINAL", "THERMAL_A_EMISSIVITY_LOSS", "THERMAL_B_EMISSIVITY_LOSS",
    "THERMAL_A_EXTRA_HEAT", "THERMAL_B_EXTRA_HEAT",
)
# These describe R62's *simulated* measurement noise (Gaussian sigma=0.1 K).
# They are not empirically calibrated sensor uncertainty bounds.
SIMULATED_TEMPERATURE_NOISE_SIGMA_K = 0.1


class AuditContractError(ValueError):
    """Rejects incomplete, inconsistent, or altered historical audit evidence."""


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_observations(path: Path) -> np.ndarray:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            fields = tuple(next(reader))
        except StopIteration as exc:
            raise AuditContractError("EMPTY_OBSERVATION_FILE") from exc
        if fields != EXPECTED_HEADER:
            raise AuditContractError("DEPLOYMENT_FIELDS_NOT_EXACT")
        try:
            rows = [list(map(float, row)) for row in reader]
        except (TypeError, ValueError) as exc:
            raise AuditContractError("NON_NUMERIC_OBSERVATION") from exc
    a = np.asarray(rows, dtype=float)
    if a.ndim != 2 or a.shape[1] != 6 or len(a) < 12 or not np.isfinite(a).all():
        raise AuditContractError("INVALID_OBSERVATION_SHAPE_OR_VALUES")
    if not (np.diff(a[:, 0]) > 0).all() or not np.allclose(np.diff(a[:, 0]), 20.0, atol=1e-9, rtol=0):
        raise AuditContractError("UNEXPECTED_SAMPLING_CADENCE")
    return a


def _read_case(world: Path, label: int) -> tuple[np.ndarray, np.ndarray, dict]:
    obs_path = world / f"L{label}_DEPLOYMENT.csv"
    meta_path = world / f"L{label}_CASE.json"
    truth_path = world / f"L{label}_SIMULATOR_ONLY.json"
    metadata = _read_json(meta_path)
    truth_data = _read_json(truth_path)
    if (metadata.get("offline_label") != label
            or metadata.get("domain") != "Thermal_localization"
            or metadata.get("world_id") != world.name
            or metadata.get("role") != "new_development_evaluation"
            or truth_data.get("native_audit") is not True):
        raise AuditContractError("WORLD_METADATA_MISMATCH")
    if hashlib.sha256(obs_path.read_bytes()).hexdigest() != metadata.get("observation_sha256"):
        raise AuditContractError("OBSERVATION_SHA256_MISMATCH")
    a = _read_observations(obs_path)
    b = np.asarray(truth_data.get("truth"), dtype=float)
    if b.ndim != 2 or b.shape[0] != len(a) or b.shape[1] < 2 or not np.isfinite(b).all():
        raise AuditContractError("TRUTH_ALIGNMENT_MISMATCH")
    if not np.array_equal(a[:, 0], b[:, 0]):
        raise AuditContractError("TRUTH_TIME_ALIGNMENT_MISMATCH")
    if metadata.get("rows") != len(a):
        raise AuditContractError("ROW_COUNT_MISMATCH")
    if not np.isin(b[:, 1], [0, 1]).all():
        raise AuditContractError("INVALID_FAULT_ACTIVE_INDICATOR")
    return a, b, metadata


def _correlation(a: np.ndarray, b: np.ndarray) -> float | None:
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _contrast(x: np.ndarray, y: np.ndarray) -> dict:
    na = float(np.linalg.norm(x))
    nb = float(np.linalg.norm(y))
    if na < 1e-12 or nb < 1e-12:
        return {"pearson": None, "cosine": None,
                "best_scale_y_over_x": None, "scale_fit_error_over_y_norm": None}
    coeff = float(np.dot(x, y) / np.dot(x, x))
    return {
        "pearson": _correlation(x, y),
        "cosine": float(np.dot(x, y) / (na * nb)),
        "best_scale_y_over_x": coeff,
        "scale_fit_error_over_y_norm": float(np.linalg.norm(y - coeff*x) / nb),
    }


def audit_world(world_dir: str | Path) -> dict[str, Any]:
    world = Path(world_dir)
    if not world.is_dir():
        raise AuditContractError("WORLD_DIRECTORY_MISSING")
    pairs = [_read_case(world, k) for k in range(5)]
    data = [p[0] for p in pairs]
    truth = [p[1] for p in pairs]
    if not np.array_equal(truth[0][:, 1], np.zeros(len(data[0]))):
        raise AuditContractError("NOMINAL_HAS_ACTIVE_FAULT")
    if not all(np.array_equal(data[0][:, 0], a[:, 0]) for a in data[1:]):
        raise AuditContractError("PAIR_TIME_GRID_MISMATCH")
    fault_rows = np.flatnonzero(truth[1][:, 1] == 1)
    if not len(fault_rows) or fault_rows[0] < 2 or len(fault_rows) == len(data[0]):
        raise AuditContractError("INVALID_ONSET")
    onset = int(fault_rows[0]); expected_active = np.arange(len(data[0])) >= onset
    for i in range(1, 5):
        if not np.array_equal(truth[i][:, 1], expected_active.astype(float)):
            raise AuditContractError("PAIR_ONSET_NOT_EQUAL_OR_NONCONTIGUOUS")
        if not np.array_equal(data[0][:onset], data[i][:onset]):
            raise AuditContractError("PREFAULT_PAIR_MISMATCH")
        if not np.array_equal(data[0][:, 1:4], data[i][:, 1:4]):
            raise AuditContractError("FAULT_CHANGED_BOUNDARY_OR_COMMAND")
    mechanisms = []
    for label in range(1, 5):
        # The index of the faulty component is determined offline by the audit
        # label. It must never be inferred from or attached to deployment input.
        active_component_col = 4 if label in (1, 3) else 5
        unaffected_component_col = 5 if active_component_col == 4 else 4
        delta = data[label][onset:, active_component_col] - data[0][onset:, active_component_col]
        unintended = data[label][onset:, unaffected_component_col] - data[0][onset:, unaffected_component_col]
        mechanisms.append({
            "label": CLASSES[label],
            "post_onset_temperature_delta_rms_k": float(np.sqrt(np.mean(delta**2))),
            "post_onset_temperature_delta_max_abs_k": float(np.max(np.abs(delta))),
            "unaffected_component_max_abs_change_k": float(np.max(np.abs(unintended))),
            "rms_pair_difference_over_independent_noise_sigma":
                float(np.sqrt(np.mean(delta**2))/(np.sqrt(2)*SIMULATED_TEMPERATURE_NOISE_SIGMA_K)),
        })
    similarity = {}
    excitation = {}
    for component, col, e, h, cmd in (("a", 4, 1, 3, 2), ("b", 5, 2, 4, 3)):
        emiss = data[e][onset:, col] - data[0][onset:, col]
        extra = data[h][onset:, col] - data[0][onset:, col]
        similarity[component] = _contrast(emiss, extra)
        T = data[0][onset:, col]
        basis = np.column_stack((np.ones(len(T)), (T/300.)**4))
        # 1 and T^4 are the local linearized effect shapes for additional heat
        # and altered radiative loss, respectively. Condition number is a
        # descriptive proxy, not a formal nonlinear observability proof.
        excitation[component] = {
            "nominal_temperature_span_k": float(np.ptp(T)),
            "heat_command_levels_observed": int(len(np.unique(data[0][:, cmd]))),
            "constant_vs_t4_design_condition_number": float(np.linalg.cond(basis)),
        }
    return {
        "world_id": world.name, "labelled_fault_pair_count": 4,
        "post_onset_rows": len(data[0]) - onset,
        "onset_time_s_simulator_only": float(data[0][onset, 0]),
        "prefault_byte_equivalent_numeric_rows": True,
        "illumination_transitions": int(np.count_nonzero(np.diff(data[0][:, 1]))),
        "mechanism_deltas": mechanisms,
        "within_component_signature_similarity": similarity,
        "nominal_excitation": excitation,
        "qualified": False,
    }


def audit_worlds(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    worlds = sorted(p for p in root.glob("r62-independent-thermal-*") if p.is_dir())
    if not worlds:
        raise AuditContractError("NO_R62_THERMAL_WORLDS")
    reports = [audit_world(p) for p in worlds]
    all_pearsons = [w["within_component_signature_similarity"][k]["pearson"]
                    for w in reports for k in ("a", "b")]
    correlations = [v for v in all_pearsons if v is not None]
    non_target = [m["unaffected_component_max_abs_change_k"]
                  for w in reports for m in w["mechanism_deltas"]]
    return {
        "schema": SCHEMA,
        "evidence_role": "DISCLOSED_R62_EVALUATION_POST_HOC_DIAGNOSTIC_ONLY",
        "training_or_model_selection_performed": False,
        "new_worlds_generated": 0,
        "claim_scope": "low_fidelity_two_parallel_Basilisk_SensorThermal_components",
        "oracle_nominal_and_simulator_fault_onset_used": True,
        "simulated_independent_noise_sigma_k_for_reference": SIMULATED_TEMPERATURE_NOISE_SIGMA_K,
        "world_count": len(reports),
        "fault_pairs_examined": 4*len(reports),
        "within_component_mechanism_pairs": 2*len(reports),
        "min_pearson": min(correlations) if correlations else None,
        "median_pearson": float(np.median(correlations)) if correlations else None,
        "max_unaffected_component_change_k": max(non_target),
        "worlds": reports,
        "interpretation": [
            "High pairwise curve correlation indicates shape overlap, not mathematical non-identifiability.",
            "A no-fault counterfactual and onset truth were used for offline audit only; neither is available to the diagnostic inference interface.",
            "Exact zero cross-component response is a limitation of uncoupled benchmark physics, not evidence of full spacecraft behavior.",
            "The noise reference comes from the simulator code; paired same-seed noise cancels, so it does not establish operational detection risk.",
            "Disclosed R62 worlds cannot be reused for a fresh blind qualification.",
        ],
        "diagnostic_qualification": "NOT_EVALUATED",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="R64 read-only retrospective Thermal audit; not inference")
    parser.add_argument("--native-root", type=Path, required=True,
                        help="Directory containing disclosed r62-independent-thermal-* worlds")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("OUTPUT_ALREADY_EXISTS: evidence is immutable")
    result = audit_worlds(args.native_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(f"R64 audited {result['world_count']} disclosed worlds, no new qualification")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
