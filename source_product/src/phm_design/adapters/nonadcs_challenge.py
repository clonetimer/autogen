"""Recovered challenge.v2 research configuration, without changing legacy RNG.

The settings are sensitivity assumptions, not hardware specifications. Four
factors can be varied independently using the same world IDs: fault strength,
onset, normal plant/measurement nuisance and boundary cycles. Shared worlds
across test cohorts are paired observations, not additional independent units.
"""
from __future__ import annotations
import math
import random
from . import sat_nonadcs as native
from ..io import digest

SCHEMA = "phm.native-nonadcs.challenge.v2"
RANGES = {
    "fault_fraction_by_role": (0.01, 0.60),
    "onset_fraction_by_role": (0.10, 0.90),
    "nuisance_scale_by_role": (0.0, 3.0),
    "boundary_cycles_by_role": (1.0, 6.0),
}


def check_challenge(c: dict) -> None:
    if not isinstance(c, dict) or set(c) != set(RANGES):
        raise ValueError("CHALLENGE_CONFIG_SCHEMA")
    for name, (lower, upper) in RANGES.items():
        if not isinstance(c[name], dict) or set(c[name]) != set(native.ROLES):
            raise ValueError("CHALLENGE_ROLE_RANGES_REQUIRED:" + name)
        for role, pair in c[name].items():
            if (not isinstance(pair, list) or len(pair) != 2 or
                any(type(v) not in (int, float) or not math.isfinite(v) for v in pair) or
                not lower <= pair[0] <= pair[1] <= upper):
                raise ValueError("INVALID_CHALLENGE_RANGE:" + name + ":" + role)


def check_config(c: dict) -> None:
    if not isinstance(c, dict) or set(c) != {"schema", "seed", "groups_per_role", "domains", "challenge"} or c["schema"] != SCHEMA:
        raise ValueError("NONADCS_CONFIG_SCHEMA")
    # Reuse legacy common-field checks without weakening the v1 schema.
    base = {k: v for k, v in c.items() if k != "challenge"}
    base["schema"] = native.PROTOCOL
    native.check_config(base)
    check_challenge(c["challenge"])


def parameters(world: str, domain: str, role: str, c: dict) -> dict:
    check_challenge(c)
    if role not in native.ROLES or domain not in native.FAMILIES:
        raise ValueError("CHALLENGE_DOMAIN_OR_ROLE")
    rng = random.Random(int(digest([world, domain])[:16], 16))
    spec = native.FAMILIES[domain]
    def draw(key):
        return rng.uniform(*c[key][role])
    onset = draw("onset_fraction_by_role")
    cycles = draw("boundary_cycles_by_role")
    fault = draw("fault_fraction_by_role")
    nuisance = draw("nuisance_scale_by_role")
    p = {
        "phase": rng.uniform(0, 2 * math.pi),
        "boundary_period_s": spec["duration_s"] / cycles,
        "onset_s": spec["sample_s"] * round(spec["duration_s"] * onset / spec["sample_s"]),
        "remaining_fraction": 1 - fault, "fault_fraction": fault,
        "panel_area_m2": rng.uniform(.09, .14), "panel_efficiency": rng.uniform(.20, .28),
        "load_w": rng.uniform(6., 12.),
        "thermal_area_m2": rng.uniform(.015, .025), "absorptivity": rng.uniform(.4, .6),
        "emissivity": rng.uniform(.7, .9), "mass_kg": rng.uniform(.3, .6),
        "initial_temperature_k": rng.uniform(300., 315.), "heat_w": rng.uniform(2., 7.),
        "generation_bps": rng.uniform(8000., 15000.), "service_bps": rng.uniform(45000., 65000.),
        "measurement_seed": int(digest([world, domain, "measurement"])[:16], 16),
        "challenge_mode": True, "nuisance_scale": nuisance,
    }
    p.update({
        "extra_load_w": p["load_w"] * fault, "extra_heat_w": 8. * fault,
        "excess_generation_multiplier": 1. + fault,
        "solar_gain": 1 + rng.uniform(-.04, .04) * nuisance,
        "load_actuator_gain": 1 + rng.uniform(-.05, .05) * nuisance,
        "panel_meter_gain": 1 + rng.uniform(-.02, .02) * nuisance,
        "panel_meter_bias_w": rng.uniform(-.35, .35) * nuisance,
        "bus_meter_gain": 1 + rng.uniform(-.02, .02) * nuisance,
        "bus_meter_bias_w": rng.uniform(-.35, .35) * nuisance,
        "power_noise_std_w": .1 + .12 * nuisance,
        "thermal_absorptivity_gain": 1 + rng.uniform(-.04, .04) * nuisance,
        "thermal_emissivity_gain": 1 + rng.uniform(-.03, .03) * nuisance,
        "heat_actuator_gain": 1 + rng.uniform(-.05, .05) * nuisance,
        "temperature_bias_k": rng.uniform(-.45, .45) * nuisance,
        "temperature_noise_std_k": .1 + .12 * nuisance,
        "generation_gain": 1 + rng.uniform(-.05, .05) * nuisance,
        "service_gain": 1 + rng.uniform(-.05, .05) * nuisance,
        "queue_meter_gain": 1 + rng.uniform(-.005, .005) * nuisance,
        "queue_bias_bits": rng.uniform(-5000., 5000.) * nuisance,
    })
    return p


def plan_campaign(c: dict) -> list[dict]:
    check_config(c)
    result = []
    for role in native.ROLES:
        for i in range(c["groups_per_role"][role]):
            world = digest([SCHEMA, c["seed"], role, i])
            for domain in c["domains"]:
                result.append({"world_id": world, "role": role, "index_in_role": i,
                    "domain": domain, "parameters": parameters(world, domain, role, c["challenge"])})
    return result


def measurement_assumptions() -> dict:
    return {
        "source": "engineering_sensitivity_assumptions_not_hardware_specifications",
        "power_white_noise_std_w": "0.1 + 0.12 * nuisance_scale",
        "temperature_white_noise_std_k": "0.1 + 0.12 * nuisance_scale",
        "normal_plant_gains": "hidden, sampled per world/domain; see generation_plan",
        "meter_gains_and_biases": "hidden, shared across paired fault variants",
        "queue_counter": "affine virtual meter, clamped to [0, design capacity] (dev6 explicit saturation)",
        "nuisance_not_in_normal_passport": True,
        "fault_effects": "native parameters/input messages during simulation, not post-hoc fault arrays",
    }
