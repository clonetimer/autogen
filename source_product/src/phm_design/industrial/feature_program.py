"""Safe, executable R81c bridge to the existing generic feature-program grammar.

This bridge intentionally supports finite, explicit *concatenation* of 1--3
existing generic atoms. Availability-aware classifier composition needs
separately grouped inner OOF training and is NOT silently approximated here.
No imports, callbacks, truth columns, or environment metadata can be specified.
"""
from __future__ import annotations

import hashlib
import json

# PLAN compilation must remain stdlib-only; R80 G0 runs without NumPy.
# This allowlist mirrors the existing generic_registry. Runtime verifies it.
ALLOWED_ATOMS = frozenset(("temporal_bins", "random_kernels",
                           "difference_kernels"))
PREFIX = "program:concat:"


def encode_program(atoms) -> str:
    """Canonical identifier for a predeclared, order-insensitive atom set."""
    if (not isinstance(atoms, (tuple, list)) or
            not 1 <= len(atoms) <= 3 or
            any(type(atom) is not str for atom in atoms) or
            len(set(atoms)) != len(atoms) or
            not set(atoms) <= ALLOWED_ATOMS):
        raise ValueError("UNREGISTERED_OR_DUPLICATE_INDUSTRIAL_ATOMS")
    return PREFIX + "+".join(sorted(atoms))


def decode_program(identifier: str) -> tuple[str, ...]:
    if not isinstance(identifier, str) or not identifier.startswith(PREFIX):
        raise ValueError("INVALID_INDUSTRIAL_FEATURE_PROGRAM")
    atoms = tuple(identifier[len(PREFIX):].split("+"))
    if identifier != encode_program(atoms):
        raise ValueError("NONCANONICAL_INDUSTRIAL_FEATURE_PROGRAM")
    return atoms


def program_digest(identifier: str) -> str:
    atoms = decode_program(identifier)
    payload = {"schema": "phm.industrial.feature-program.v1",
               "atoms": list(atoms), "composition": "concat",
               "registry": "phm_design.open_design.generic_registry"}
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def execute_program(array, identifier: str):
    """Materialize only deployment-visible numeric observations via old Program.

    Synthetic *field identifiers* preserve matrix order; actual deployment
    field names and roles are validated by industrial.validate_observations.
    Existing generic FeatureRegistry supplies the atom implementations.
    """
    import numpy as np
    from ..open_design import ObservationSpec, Program, generic_registry

    if set(generic_registry().names) != ALLOWED_ATOMS:
        raise ValueError("INDUSTRIAL_FEATURE_REGISTRY_DRIFT")
    atoms = decode_program(identifier)
    a = np.asarray(array, dtype=float)
    if a.ndim != 2 or a.shape[1] < 2:
        raise ValueError("INVALID_FEATURE_PROGRAM_MATRIX")
    fields = ("time_s", *(f"sensor_{i}" for i in range(a.shape[1] - 1)))
    spec = ObservationSpec("industrial_deployment_observations", fields,
                           time_field="time_s", min_rows=4)
    program = Program(atoms, learner="linear", composition="concat")
    return np.asarray(generic_registry().transform(program, a, spec),
                      dtype=np.float64)
