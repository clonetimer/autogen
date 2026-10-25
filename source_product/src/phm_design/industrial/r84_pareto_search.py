"""R84 bounded Pareto-informed *TRAIN-OOF* topology proposal.

Pure-Python proposal only: no sensor data, model fitting, VAL/TEST/FINAL,
training labels, or independent generalization evidence. Used optionally by
the R83 self-organizing trainer; its legacy and static modes are unchanged.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import math

FIELDS = ("width", "depth", "kernel", "pooling", "residual")
METRICS = ("accuracy", "worst_environment_class_recall", "healthy_false_alarm")
METHOD = "TRAIN_OOF_PARETO_RISK_COMPLEXITY_DIVERSITY_V1"


def _require(ok, reason):
    if not ok:
        raise ValueError("R84_" + reason)


def _hash(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, ensure_ascii=False, allow_nan=False,
        separators=(",", ":")).encode("utf-8")).hexdigest()


def _topology(obj):
    _require(isinstance(obj, Mapping) and set(obj) == set(FIELDS),
             "INVALID_GENOME_FIELDS")
    _require(type(obj["width"]) is int and obj["width"] in (8, 16, 32)
             and type(obj["depth"]) is int and obj["depth"] in (1, 2, 3)
             and type(obj["kernel"]) is int and obj["kernel"] in (3, 5)
             and obj["pooling"] in ("mean", "mean_max")
             and type(obj["residual"]) is bool, "UNREGISTERED_GENOME")
    return {k: obj[k] for k in FIELDS}


def _risk(m):
    return .45 * (1 - m["accuracy"]) + .40 * (
        1 - m["worst_environment_class_recall"]) + .15 * m["healthy_false_alarm"]


def _parse(history, visited):
    _require(isinstance(history, (list, tuple)) and len(history) >= 2,
             "INSUFFICIENT_SOURCE_FEEDBACK")
    _require(isinstance(visited, set) and all(
        isinstance(v, str) for v in visited), "INVALID_VISITED_SET")
    parsed = []
    for item in history:
        _require(isinstance(item, Mapping) and
                 item.get("feedback_role") == "SOURCE_TRAIN_OOF_ONLY",
                 "PROTECTED_OR_UNKNOWN_FEEDBACK")
        genome = _topology(item.get("genome"))
        digest = _hash(genome)
        _require(item.get("genome_sha256") == digest,
                 "GENOME_HASH_MISMATCH")
        m = item.get("metrics")
        _require(isinstance(m, Mapping) and set(m) == set(METRICS),
                 "INVALID_TRAIN_FEEDBACK_METRICS")
        for key in METRICS:
            val = m[key]
            _require(type(val) in (int, float) and math.isfinite(val)
                     and 0 <= val <= 1, "INVALID_TRAIN_FEEDBACK_VALUE")
        loss = item.get("selection_loss")
        _require(type(loss) in (float, int) and math.isfinite(loss)
                 and abs(loss - _risk(m)) < 1e-8, "INCONSISTENT_TRAIN_OOF_LOSS")
        params = item.get("parameter_count")
        _require(type(params) is int and params > 0,
                 "UNKNOWN_PARAMETER_COUNT")
        parsed.append({"genome": genome, "digest": digest,
                       "metrics": m, "params": params, "loss": loss})
    _require(len({x["digest"] for x in parsed}) == len(parsed),
             "DUPLICATED_FEEDBACK_GENOME")
    _require(visited == {x["digest"] for x in parsed},
             "VISITED_HISTORY_MISMATCH")
    return parsed


def _objectives(item):
    m = item["metrics"]
    return (1-m["accuracy"], 1-m["worst_environment_class_recall"],
            m["healthy_false_alarm"], item["params"])


def _dominates(left, right):
    a, b = _objectives(left), _objectives(right)
    return all(x <= y for x, y in zip(a, b)) and any(
        x < y for x, y in zip(a, b))


def _complexity_proxy(genome):
    # A structural tie-break ONLY, not a Torch parameter/FLOP measurement.
    return (genome["depth"] * genome["kernel"] * genome["width"] ** 2
            + genome["width"] * (2 if genome["pooling"] == "mean_max" else 1))


def _distance(a, b):
    # Symmetric bounded structural distance, not learned metric similarity.
    return (abs(a["width"] - b["width"]) / 24
            + abs(a["depth"] - b["depth"]) / 2
            + (a["kernel"] != b["kernel"]) * 1.0
            + (a["pooling"] != b["pooling"]) * 1.0
            + (a["residual"] != b["residual"]) * 1.0)


def propose_pareto(history: Sequence[Mapping], visited: set[str],
                   neighbors: Callable[[dict], Sequence[Mapping]]
                   ) -> tuple[dict, str]:
    """Produce ONE novel genotype from verified TRAIN-only multiobjective feedback.

    Parent: nondominated on three source-OOF rates and observed parameter count;
    rank by declared source risk plus modest relative parameter penalty.
    Child: maximum structural novelty relative to all tried candidates,
    then minimum analytical complexity proxy, then stable digest.
    Nothing here estimates a child's future accuracy.
    """
    tried = _parse(history, visited)
    front = [candidate for candidate in tried if not any(
        _dominates(other, candidate) for other in tried
        if other["digest"] != candidate["digest"])]
    _require(bool(front), "EMPTY_PARETO_FRONT")
    smallest = min(row["params"] for row in tried)
    parents = sorted(front, key=lambda row: (
        row["loss"] + .03 * math.log2(row["params"] / smallest),
        row["loss"], row["params"], row["digest"]))
    for parent in parents:
        raw_neighbors = neighbors(dict(parent["genome"]))
        _require(isinstance(raw_neighbors, (tuple, list)),
                 "INVALID_NEIGHBOR_GENERATOR")
        candidates = {}
        for child in raw_neighbors:
            genome = _topology(child)
            digest = _hash(genome)
            if digest not in visited:
                candidates[digest] = genome
        if not candidates:
            continue
        choice = min(candidates.items(), key=lambda pair: (
            -min(_distance(pair[1], x["genome"]) for x in tried),
            _complexity_proxy(pair[1]), pair[0]))
        return choice[1], parent["digest"]
    raise ValueError("R84_EXHAUSTED_PARETO_NEIGHBORHOODS")
