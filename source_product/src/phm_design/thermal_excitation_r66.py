"""Frozen R66 slow-heating *command* profile; native Basilisk integrates dynamics.

No fault truth, onset, measurements, or simulator state enters this function.
This is a development-only research excitation, not a deployed controller.
"""
from __future__ import annotations
import math

SLOW = 'slow_thermal_plateaus_r66'
SEGMENT_S = 2400
LEVELS = (0.50, 1.70, 0.55, 1.80, 0.60, 1.65)


def commanded_heat(base_w: float, time_s: int, component_index: int) -> float:
    if (not isinstance(base_w, (int, float)) or isinstance(base_w, bool)
            or not math.isfinite(base_w) or base_w <= 0
            or type(time_s) is not int or time_s < 0 or time_s % 20
            or type(component_index) is not int or component_index not in (0,1)):
        raise ValueError('R66_INVALID_COMMAND')
    return float(base_w) * LEVELS[(time_s // SEGMENT_S + 3*component_index) % len(LEVELS)]
