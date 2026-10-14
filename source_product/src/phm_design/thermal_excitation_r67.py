"""R67 predeclared equal-energy crossed-phase heater COMMANDS for native Basilisk.

Never reads fault truth, state, simulated temperatures or diagnosis decisions.
Two crossed arms have identical command-level multisets and integrated energy
in *each* half of the 14,400 s horizon; a third constant-power control has
identical mean energy in each half. These are research-only open-loop inputs.
"""
from __future__ import annotations
import math

CROSS_AB = 'r67_phase_a0_b3_isoenergy'
CROSS_BA = 'r67_phase_a3_b0_isoenergy'
CONSTANT = 'r67_constant_isoenergy'
PROFILES = (CROSS_AB, CROSS_BA, CONSTANT)
SEGMENT_S = 2400
LEVELS = (0.50, 1.50, 1.00, 1.50, 0.50, 1.00)
DURATION_S = 14400
ONSET_S = 7200


def commanded_heat(base_w: float, time_s: int, component_index: int, profile: str) -> float:
    if (isinstance(base_w, bool) or not isinstance(base_w, (int, float))
            or not math.isfinite(base_w) or base_w <= 0
            or type(time_s) is not int or time_s < 0 or time_s >= DURATION_S or time_s % 20
            or type(component_index) is not int or component_index not in (0, 1)
            or profile not in PROFILES):
        raise ValueError('R67_INVALID_COMMAND')
    if profile == CONSTANT:
        factor = 1.0
    else:
        shift = (0 if component_index == 0 else 3) if profile == CROSS_AB else (3 if component_index == 0 else 0)
        factor = LEVELS[(time_s // SEGMENT_S + shift) % len(LEVELS)]
    return float(base_w) * factor


def schedule_energy_audit() -> dict:
    """Nominal commanded energy using actual twenty-second simulation intervals."""
    out = {}
    for profile in PROFILES:
        out[profile] = {}
        for component in (0, 1):
            half = []
            for start in (0, ONSET_S):
                vals = [commanded_heat(1., t, component, profile) for t in range(start, start+ONSET_S, 20)]
                half.append(20. * sum(vals))
            out[profile][str(component)] = half
            if any(abs(h - ONSET_S) > 1e-6 for h in half):
                raise ValueError('R67_NON_EQUAL_COMMAND_ENERGY')
    return out
