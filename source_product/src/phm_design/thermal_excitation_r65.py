"""R65 *command* schedule for native SensorThermal development experiments.

This module changes a commanded heater input only. It does NOT approximate,
advance, or replace the thermal state: the physical dynamics remain the
Basilisk 2.11.0 ``SensorThermal`` implementation.
"""
from __future__ import annotations

import math

SCHEMA = "phm.thermal-excitation.r65.v1"
BASELINE = "legacy_two_level"
EXPERIMENT = "multilevel_step_r65"
LEVELS = (0.55, 1.50, 0.75, 1.75, 0.65, 1.25, 0.90, 1.60)
SEGMENT_S = 220


def commanded_heat(base_w: float, time_s: int, component_index: int, profile: str) -> float:
    """Output only a prescribed deployment-visible heat *command* in W.

    Independent of fault label, onset, hidden state and future measurements.
    """
    if (profile not in (BASELINE, EXPERIMENT)
            or type(component_index) is not int or component_index not in (0, 1)
            or type(time_s) is not int or time_s < 0 or time_s % 20
            or not isinstance(base_w, (int, float)) or not math.isfinite(base_w)
            or base_w <= 0):
        raise ValueError("R65_INVALID_COMMAND_PROFILE_OR_INPUT")
    if profile == BASELINE:
        return float(base_w)*(1+.25*float(math.sin(time_s/(115+15*component_index))>0))
    index = (time_s // SEGMENT_S + 3*component_index) % len(LEVELS)
    return float(base_w)*LEVELS[index]
