"""Observation-derived availability vectors for R81c industrial late fusion.

Every atom uses its own declared physical sensor signal and a *deployment
observed* binary validity channel. A 0-mask is never computed from labels,
source roles, injected faults, or the physical-environment split metadata.
Finite placeholders under an unavailable flag are not treated as observations.
Current contract is offline trajectory-wide availability, not mid-window loss.
"""
from __future__ import annotations

import numpy as np

from ..open_design import ObservationSpec, Program, generic_registry
from .feature_program import ALLOWED_ATOMS


def atom_dimensions(atom):
    if atom not in ALLOWED_ATOMS:
        raise ValueError("UNREGISTERED_AVAILABILITY_ATOM")
    # A synthetic, controlled one-channel vector determines registered output
    # dimension; never inspect invalid or unavailable real telemetry for sizing.
    demo = np.column_stack((np.arange(32, dtype=float),
                            np.zeros(32, dtype=float)))
    spec = ObservationSpec("availability_dimension", ("time_s", "signal"))
    return int(len(generic_registry().transform(
        Program((atom,), "linear"), demo, spec)))


def execute_available(array, bindings):
    """Concatenate zero-coded absent base inputs plus a binary availability mask.

    Each binding: (atom, sensor_column_index, validity_column_index), indices
    from a prevalidated TaskSpec field order (time index 0).
    """
    a = np.asarray(array, dtype=float)
    if a.ndim != 2 or a.shape[0] < 16 or a.shape[1] < 3 or not np.isfinite(a).all():
        raise ValueError("INVALID_AVAILABLE_SENSOR_TRAJECTORY")
    if np.any(np.diff(a[:, 0]) <= 0):
        raise ValueError("NONMONOTONE_AVAILABLE_SENSOR_TIME")
    if not 2 <= len(bindings) <= 3:
        raise ValueError("AVAILABILITY_REQUIRES_TWO_OR_THREE_BRANCHES")
    registry = generic_registry()
    parts, mask = [], []
    spec = ObservationSpec("availability_sensor", ("time_s", "signal"))
    for atom, sensor, validity in bindings:
        if (atom not in ALLOWED_ATOMS or type(sensor) is not int or
                type(validity) is not int or not 0 < sensor < a.shape[1] or
                not 0 < validity < a.shape[1] or sensor == validity):
            raise ValueError("INVALID_AVAILABLE_SENSOR_BINDING")
        flags = a[:, validity]
        if not np.isin(flags, (0., 1.)).all():
            raise ValueError("NONBINARY_AVAILABLE_SENSOR_FLAG")
        if not np.all(flags == flags[0]):
            raise ValueError("CHANGING_AVAILABILITY_IN_TRAJECTORY_UNSUPPORTED")
        active = bool(flags[0])
        mask.append(float(active))
        if active:
            observed = a[:, [0, sensor]]
            v = registry.transform(Program((atom,), "linear"), observed, spec)
            if len(v) != atom_dimensions(atom):
                raise ValueError("AVAILABILITY_ATOM_DIMENSION_DRIFT")
            parts.append(v)
        else:
            parts.append(np.zeros(atom_dimensions(atom), dtype=float))
    return np.concatenate([*parts, np.asarray(mask, dtype=float)])
