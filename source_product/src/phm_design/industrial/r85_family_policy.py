"""R85 source-TRAIN-only feedback-driven choice across registered algorithm families.

Pure candidate scheduler; no labels, raw measurements, protected roles or
model weights are accepted. The R81 executor trains every chosen candidate.
This is NOT evidence of across-domain generalization.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

SCHEMA = "phm.r85.family-feedback.v1"
FAMILIES = ("linear", "tcn", "tree", "gru", "cnn1d", "transformer")
NEXT_NEIGHBORS = {
    "linear": ("tree", "gru", "cnn1d", "transformer"),
    "tree": ("linear", "gru", "tcn", "cnn1d", "transformer"),
    "tcn": ("gru", "cnn1d", "transformer", "tree"),
    "gru": ("transformer", "cnn1d", "tree", "tcn"),
    "cnn1d": ("tcn", "gru", "transformer", "tree"),
    "transformer": ("gru", "tcn", "cnn1d", "tree"),
}
KEYS = ("accuracy", "worst_environment_class_recall", "healthy_false_alarm")


def _fail(code):
    raise ValueError("R85_" + code)


def validate_pool(grid: Sequence[Mapping]) -> dict[str, int]:
    if not isinstance(grid, list) or len(grid) < 4 or len(grid) > 6:
        _fail("CANDIDATE_POOL_MUST_HAVE_FOUR_TO_SIX_FAMILIES")
    positions = {}
    for i, candidate in enumerate(grid):
        if not isinstance(candidate, Mapping):
            _fail("INVALID_CANDIDATE_MAPPING")
        family = candidate.get("backbone")
        if (family not in FAMILIES or family in positions
                or candidate.get("objective") != "ERM"
                or candidate.get("topology") != "direct"
                or candidate.get("representation") != (
                    "temporal_bins" if family in ("linear", "tree")
                    else "learned_sequence")
                or set(candidate) != {"backbone", "objective", "topology", "representation"}):
            _fail("UNSUPPORTED_OR_DUPLICATE_FAMILY_CANDIDATE")
        positions[family] = i
    if not {"linear", "tree", "tcn", "gru"}.issubset(positions):
        _fail("FOUR_REFERENCE_FAMILIES_REQUIRED")
    return positions


def _metric_loss(metrics):
    if metrics is None:
        return None
    if not isinstance(metrics, Mapping) or not set(KEYS).issubset(metrics):
        _fail("INVALID_SOURCE_FEEDBACK")
    values = []
    for key in KEYS:
        number = metrics[key]
        if (type(number) not in (int, float) or
                not math.isfinite(number) or not 0 <= number <= 1):
            _fail("INVALID_SOURCE_METRIC")
        values.append(float(number))
    return .45*(1-values[0]) + .40*(1-values[1]) + .15*values[2]


def choose_next(grid: list[Mapping], history: Sequence[Mapping], *,
                mode: str = "feedback") -> tuple[int, dict]:
    """Deterministic proposal: first two anchors, then source-feedback neighbor.

    History is a sanitized TRAIN-OOF summary, NOT a held-out/test score.
    If no valid fit exists, fallback follows fixed order and audit says so.
    """
    positions = validate_pool(grid)
    if mode not in ("feedback", "predeclared_static"):
        _fail("INVALID_PROPOSAL_MODE")
    if not isinstance(history, (list, tuple)):
        _fail("INVALID_FEEDBACK_HISTORY")
    seen = set()
    successful = []
    for entry in history:
        if (not isinstance(entry, Mapping)
                or set(entry) != {"candidate_index", "role", "metrics"}
                or entry.get("role") != "SOURCE_TRAIN_OOF_ONLY"):
            _fail("UNTRUSTED_OR_PROTECTED_FEEDBACK")
        idx = entry["candidate_index"]
        if type(idx) is not int or not 0 <= idx < len(grid) or idx in seen:
            _fail("DUPLICATE_OR_INVALID_CANDIDATE_INDEX")
        seen.add(idx)
        loss = _metric_loss(entry["metrics"])
        if loss is not None:
            successful.append((loss, FAMILIES.index(grid[idx]["backbone"]), idx))
    if len(seen) == len(grid):
        _fail("EXHAUSTED_FAMILY_POOL")
    anchors = ("linear", "tcn")
    for name in anchors:
        idx = positions[name]
        if idx not in seen:
            return idx, {"schema": SCHEMA, "strategy": "PREDECLARED_ANCHOR",
                         "parent_index": None, "feedback_source": "SOURCE_TRAIN_OOF_ONLY"}
    ordered = [positions[x] for x in FAMILIES if x in positions]
    if mode == "predeclared_static":
        selected = next(x for x in ordered if x not in seen)
        return selected, {"schema": SCHEMA, "strategy": "STATIC_FAMILY_ORDER",
                          "parent_index": None, "feedback_source": "SOURCE_TRAIN_OOF_ONLY"}
    for _, _, parent_idx in sorted(successful):
        parent_family = grid[parent_idx]["backbone"]
        for neighbor in NEXT_NEIGHBORS[parent_family]:
            if neighbor in positions and positions[neighbor] not in seen:
                return positions[neighbor], {
                    "schema": SCHEMA, "strategy": "TRAIN_OOF_FAMILY_FEEDBACK",
                    "parent_index": parent_idx,
                    "feedback_source": "SOURCE_TRAIN_OOF_ONLY"}
    fallback = next(x for x in ordered if x not in seen)
    return fallback, {"schema": SCHEMA,
                      "strategy": "FAILED_FITS_PREDECLARED_FALLBACK",
                      "parent_index": None, "feedback_source": "SOURCE_TRAIN_OOF_ONLY"}
