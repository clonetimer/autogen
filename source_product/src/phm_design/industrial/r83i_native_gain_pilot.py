"""R83i: prospective engineering pilot using ONLY verified Basilisk FSW gain knobs.

NOT a time-varying torque-profile interface. Not a diagnostic benchmark.
Same predeclared TRAIN pair, seed, fault/severity and physical initial state
under two static gain settings; native SatAgentSimulation executes dynamics.
Increased PD gain does not guarantee excitation and safety; verify from
actual native telemetry *after* execution.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import argparse
import math

from ..io import digest, read_json, write_json
from .r83d_native_protocol import FAULTS
from .taskspec import TaskSpecRejected

SCHEMA = "phm.r83i.predeclared-native-static-gain-pilot.v1"
SOURCE_PLAN = "c54bfcd154322a620c7d8058ce8a40db6e155602593c5ff5da8bdba43ef3b714"
SOURCE_PRODUCER_SHA = "1f709beeae97baec65db9ca13ae373d88728f87d"
ARMS = ("reference", "gain125")
INDEXES = (0, 1, 2)
GAIN = {"reference": 1., "gain125": 1.25}
PERMITTED = frozenset(("controller_k", "controller_p"))


def _reject(x):
    raise TaskSpecRejected(x)


def _check_original(plan):
    if (digest(plan) != SOURCE_PLAN or
            plan.get("campaign", {}).get("base_spec", {}).get("capability_id") !=
            "subsystem.adcs_unified_native.v1" or
            plan["campaign"]["base_spec"]["simulation"].get("backend") != "basilisk" or
            len(plan["campaign"]["cases"]) != 126 or
            plan.get("metadata", {}).get("r83d_preregistration", {}).get("schema") !=
            "phm.r83d.native-balanced-protocol.v1"):
        _reject("R83I_REQUIRES_EXACT_ORIGINAL_R83F_NATIVE_PLAN")


def _registered_gain_paths(registry):
    """Pass the real upstream 2.11.0 registry contract to this checker."""
    contract = registry("subsystem.adcs_unified_native.v1")
    if contract is None:
        _reject("R83I_NATIVE_GAIN_REGISTRY_UNAVAILABLE")
    paths = set(contract.path_map)
    required = {"parameters.values.controller_k",
                "parameters.values.controller_p",
                "parameters.values.rw_max_torque_nm"}
    if not required.issubset(paths):
        _reject("R83I_NATIVE_GAIN_REGISTRY_NOT_VERIFIED")
    # No verified time-varying command profile: never claim one is supported.
    if any("profile" in path or "schedule" in path for path in paths):
        _reject("R83I_UNREVIEWED_NATIVE_TIME_PROFILE_SURFACE")
    return sorted(required)


def preregister_static_gain_pilot(plan, *, native_gain_registry):
    """36 actual physical tasks = 3 faults x 3 inclinations x 2 twins x 2 arms."""
    _check_original(plan)
    supported = _registered_gain_paths(native_gain_registry)
    pairs = defaultdict(dict)
    for case in plan["campaign"]["cases"]:
        m = case["overrides"]["metadata"]
        if m.get("astrograph_split") != "train":
            continue
        pair = m["astrograph_pair_id"]
        if m["astrograph_fault_id"] not in FAULTS or \
                m["astrograph_case_role"] not in ("fault", "nominal"):
            _reject("R83I_UNEXPECTED_ORIGINAL_SOURCE_CASE")
        index = int(pair.rsplit("-", 1)[-1])
        if index in INDEXES:
            role = m["astrograph_case_role"]
            if role in pairs[pair]:
                _reject("R83I_DUPLICATE_ORIGINAL_TWIN")
            pairs[pair][role] = case
    if len(pairs) != len(FAULTS) * len(INDEXES):
        _reject("R83I_ORIGINAL_PHYSICAL_SOURCE_COVERAGE_CHANGED")
    result = deepcopy(plan)
    output = []
    for pair in sorted(pairs):
        templates = pairs[pair]
        if set(templates) != {"fault", "nominal"}:
            _reject("R83I_ORIGINAL_PHYSICAL_PAIR_INCOMPLETE")
        a = templates["fault"]["overrides"]
        b = templates["nominal"]["overrides"]
        if (a["parameters"] != b["parameters"] or
                a["simulation.seed"] != b["simulation.seed"]):
            _reject("R83I_ORIGINAL_PAIR_NOT_MATCHED")
        baseline = a["parameters"]
        for arm in ARMS:
            suffix = f"-r83i-{arm}"
            new_pair = pair + suffix
            modified = deepcopy(baseline)
            for key in PERMITTED:
                if (type(modified.get(key)) not in (float, int) or
                        not math.isfinite(modified[key]) or modified[key] <= 0):
                    _reject("R83I_INVALID_ORIGINAL_CONTROLLER_GAIN")
                modified[key] = round(modified[key] * GAIN[arm], 8)
            # Hardware rating, current cap, gyro noise, initial condition,
            # seed, severity, injection wheel and fault onset do NOT change.
            if modified["rw_max_torque_nm"] != baseline["rw_max_torque_nm"]:
                _reject("R83I_HARDWARE_TORQUE_LIMIT_CHANGED")
            for role in ("fault", "nominal"):
                child = deepcopy(templates[role])
                meta = child["overrides"]["metadata"]
                ident = child["case_id"]
                if not ident.endswith("-" + role):
                    _reject("R83I_UNEXPECTED_SOURCE_CASE_ID")
                child["case_id"] = ident[:-len(role)] + suffix.lstrip("-") + "-" + role
                child["overrides"]["parameters"] = deepcopy(modified)
                meta["astrograph_pair_id"] = new_pair
                meta["astrograph_fault_case_id"] = (
                    templates["fault"]["case_id"][:-len("fault")] +
                    suffix.lstrip("-") + "-fault")
                meta["astrograph_nominal_case_id"] = (
                    templates["nominal"]["case_id"][:-len("nominal")] +
                    suffix.lstrip("-") + "-nominal")
                meta["astrograph_independent_run_id"] = new_pair
                meta["experiment_id"] = f"r83i:{new_pair}:{role}"
                meta["r83i_original_source_pair_id"] = pair
                meta["r83i_static_gain_arm"] = arm
                meta["r83i_controller_multiplier"] = GAIN[arm]
                meta["r83i_no_time_varying_command"] = True
                output.append(child)
    if len(output) != 36 or len({c["case_id"] for c in output}) != 36:
        _reject("R83I_INCORRECT_PAIRED_SOURCE_CAPACITY")
    result["task_id"] = "r83i_static_controller_excitability_native_pilot"
    result["campaign"]["cases"] = output
    result["campaign"]["count"] = len(output)
    result["campaign"]["max_cases"] = len(output)
    result["metadata"]["r83i"] = {
        "schema": SCHEMA,
        "source_canonical_sha256": SOURCE_PLAN,
        "pinned_native_source_commit": SOURCE_PRODUCER_SHA,
        "verified_native_controller_paths": supported,
        "source_role": "TRAIN_PHYSICS_QA_ONLY",
        "fault_mechanisms": list(FAULTS),
        "original_train_pair_indexes": list(INDEXES),
        "original_inclinations_deg": [29., 47., 65.],
        "independent_original_pairs": 9,
        "physics_case_count": 36,
        "static_controller_gain_multipliers": GAIN,
        "hardware_torque_limit_held_constant": True,
        "fault_event_and_seed_held_constant": True,
        "no_verified_time_varying_command_support": True,
        "claim_policy": "EXPLORATORY_ACTUATOR_PHYSICS_ONLY_NO_ML_OR_VAL",
        "real_physics_executed": False,
        "independent_qualification": False,
    }
    return result, {"schema": SCHEMA, "frozen_plan_canonical_sha256": digest(result),
                    "original_plan_sha256": SOURCE_PLAN,
                    "source_cases": len(output),
                    "source_base_pairs": len(pairs),
                    "actual_basilisk_executed": False}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--r83f-frozen-plan", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a=p.parse_args(argv)
    if a.output.exists():
        raise FileExistsError("R83I_PREREGISTRATION_EXISTS")
    from sat_sim.experiments.native_controller_registry import native_controller_contract
    plan, receipt = preregister_static_gain_pilot(
        read_json(a.r83f_frozen_plan),native_gain_registry=native_controller_contract)
    write_json(a.output,plan)
    write_json(a.output.with_suffix(".audit.json"),receipt)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
