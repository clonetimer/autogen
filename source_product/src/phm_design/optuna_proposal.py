"""Optional Optuna-backed topology proposal backend.

This module deliberately does *not* replace PHM Design's typed DAG compiler,
resource gates, trainer, selection rules or evidence ledger.  Optuna only
selects a genome in the already-reviewed grammar.  Every proposal is lowered
and audited by the same native code path as native feedback/random search.

Engineering-only in dev27.  Formal promotion requires a fresh-seed study.
"""
from __future__ import annotations

import itertools
import math
from collections import defaultdict
from typing import Any

from .graph import analyze_graph, graph_identity
from .models import Shape
from .synthesis import (
    apply_initial_tail_pool_coverage,
    graph_budget_reason,
    lower_genome,
    propose as native_propose,
    validate_request,
)


def _optuna():
    try:
        import optuna
    except Exception as exc:  # pragma: no cover - dependency absence is explicit
        raise RuntimeError("OPTUNA_OPTIONAL_DEPENDENCY_REQUIRED") from exc
    return optuna


def _pool_choices(grammar: dict) -> list[str]:
    out = ["statistics", "mean", "last"]
    out.extend(f"tail_mean_{int(w)}" for w in grammar.get("tail_pool_windows", []))
    return out


def _perm_from_code(n: int, code: int) -> list[int]:
    items = list(range(n))
    code %= math.factorial(n)
    out: list[int] = []
    for k in range(n, 0, -1):
        f = math.factorial(k - 1)
        idx, code = divmod(code, f)
        out.append(items.pop(idx))
    return out


def _perm_code(order: list[int]) -> int:
    items = list(range(len(order)))
    code = 0
    for i, value in enumerate(order):
        idx = items.index(value)
        code += idx * math.factorial(len(order) - i - 1)
        items.pop(idx)
    return code


def suggest_genome(trial: Any, grammar: dict) -> dict:
    """Suggest one full genome in a fixed, conditional Optuna search space."""
    max_branches = int(grammar["max_branches"])
    max_depth = int(grammar["max_depth"])
    allow_passthrough = bool(grammar.get("allow_passthrough_branches", False))
    layer_ops = list(grammar.get("layer_ops", ["conv", "linear"]))
    pools = _pool_choices(grammar)
    branches = []
    branch_count = trial.suggest_int("branch_count", 2, max_branches)
    for bi in range(branch_count):
        source = trial.suggest_categorical(f"b{bi}_source", ["components", "global"])
        difference = False if bi == 0 else trial.suggest_categorical(f"b{bi}_difference", [False, True])
        min_depth = 0 if allow_passthrough else 1
        depth = trial.suggest_int(f"b{bi}_depth", min_depth, max_depth)
        layers = []
        for li in range(depth):
            op = trial.suggest_categorical(f"b{bi}_l{li}_op", layer_ops)
            if source == "components" and op in {"grid_res3d", "grid_st_res"}:
                raise ValueError("OPTUNA_GRID_COMPONENT_INCOMPATIBLE")
            width = trial.suggest_categorical(f"b{bi}_l{li}_width", list(grammar["widths"]))
            layer = {
                "op": op,
                "width": int(width),
                "activation": trial.suggest_categorical(f"b{bi}_l{li}_activation", ["relu", "silu", "tanh"]),
            }
            if op in {"conv", "res_tcn", "grid_res3d", "grid_st_res"}:
                layer["kernel"] = int(trial.suggest_categorical(f"b{bi}_l{li}_kernel", list(grammar["kernels"])))
                layer["dilation"] = int(trial.suggest_categorical(f"b{bi}_l{li}_dilation", list(grammar["dilations"])))
            elif op == "self_attention":
                heads = int(trial.suggest_categorical(f"b{bi}_l{li}_heads", list(grammar["attention_heads"])))
                if int(width) % heads:
                    raise ValueError("OPTUNA_ATTENTION_WIDTH_HEAD_INCOMPATIBLE")
                layer["heads"] = heads
            layers.append(layer)
        pool = trial.suggest_categorical(f"b{bi}_pool", pools)
        skip = bool(depth and trial.suggest_categorical(f"b{bi}_skip", [False, True]))
        share = bool(
            depth
            and source == "components"
            and grammar["allow_sharing"]
            and trial.suggest_categorical(f"b{bi}_share", [False, True])
        )
        branches.append({
            "source": source,
            "difference": bool(difference),
            "layers": layers,
            "pool": pool,
            "skip": skip,
            "share": share,
        })

    fusion = trial.suggest_categorical("fusion", ["concat", "add", "gated_add"])
    fusion_width = int(trial.suggest_categorical("fusion_width", list(grammar["widths"])))
    order_code = trial.suggest_int("fusion_order_code", 0, math.factorial(max_branches) - 1)
    fusion_order = _perm_from_code(branch_count, order_code)
    head_count = trial.suggest_int("head_count", 0, 2)
    head_widths = [int(trial.suggest_categorical(f"head_width_{i}", list(grammar["widths"]))) for i in range(head_count)]
    return {
        "branches": branches,
        "fusion": fusion,
        "fusion_width": fusion_width,
        "fusion_order": fusion_order,
        "head_widths": head_widths,
    }


def genome_params(genes: dict, grammar: dict) -> dict:
    """Flatten one native genome to parameters consumed by ``suggest_genome``."""
    params: dict[str, Any] = {
        "branch_count": len(genes["branches"]),
        "fusion": genes["fusion"],
        "fusion_width": int(genes["fusion_width"]),
        "fusion_order_code": _perm_code(list(genes["fusion_order"])),
        "head_count": len(genes["head_widths"]),
    }
    for i, w in enumerate(genes["head_widths"]):
        params[f"head_width_{i}"] = int(w)
    for bi, branch in enumerate(genes["branches"]):
        params[f"b{bi}_source"] = branch["source"]
        if bi:
            params[f"b{bi}_difference"] = bool(branch["difference"])
        params[f"b{bi}_depth"] = len(branch["layers"])
        params[f"b{bi}_pool"] = branch["pool"]
        if branch["layers"]:
            params[f"b{bi}_skip"] = bool(branch["skip"])
            if branch["source"] == "components" and grammar["allow_sharing"]:
                params[f"b{bi}_share"] = bool(branch["share"])
        for li, layer in enumerate(branch["layers"]):
            params[f"b{bi}_l{li}_op"] = layer["op"]
            params[f"b{bi}_l{li}_width"] = int(layer["width"])
            params[f"b{bi}_l{li}_activation"] = layer["activation"]
            if layer["op"] in {"conv", "res_tcn", "grid_res3d", "grid_st_res"}:
                params[f"b{bi}_l{li}_kernel"] = int(layer["kernel"])
                params[f"b{bi}_l{li}_dilation"] = int(layer["dilation"])
            elif layer["op"] == "self_attention":
                params[f"b{bi}_l{li}_heads"] = int(layer["heads"])
    return params


def topology_history(archive: list[dict]) -> list[dict]:
    """Collapse raw/bounded/unbounded route records into one topology objective."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in archive:
        if record.get("kind") == "generated" and record.get("status") == "COMPLETED" and "topology_sha256" in record:
            grouped[record["topology_sha256"]].append(record)
    history = []
    for topology, records in grouped.items():
        best = max(records, key=lambda r: float(r["mean_validation_macro_f1"]))
        history.append({
            "topology_sha256": topology,
            "genes": best["genes"],
            "objective": float(best["mean_validation_macro_f1"]),
            "best_route": best.get("mode"),
            "source_id": best.get("id"),
        })
    return sorted(history, key=lambda x: x["topology_sha256"])


def _study_from_archive(request: dict, archive: list[dict], generation: int):
    optuna = _optuna()
    ev = request["evolution"]
    grammar = request["grammar"]
    history = topology_history(archive)
    sampler = optuna.samplers.TPESampler(
        seed=int(ev["seed"]) + generation * 1_000_003 + 27,
        n_startup_trials=max(1, min(3, len(history))),
        multivariate=False,
    )
    study = optuna.create_study(direction="maximize", sampler=sampler)
    for item in history:
        fixed = genome_params(item["genes"], grammar)
        study.enqueue_trial(fixed, skip_if_exists=True)

        def objective(trial, value=item["objective"]):
            suggest_genome(trial, grammar)
            return value

        study.optimize(objective, n_trials=1, show_progress_bar=False)
    return study, history


def propose_optuna_tpe(request: dict, contract: dict, generation: int, archive: list[dict], seen: set[str]) -> dict:
    """Matched proposal backend using Optuna TPE after native generation-0.

    Generation-0 is byte-for-byte the native legal random startup so comparisons
    begin from the same initial topology set.  Later generations reconstruct an
    in-memory TPE study from completed topology-level development evidence.
    """
    if generation == 0:
        result = native_propose(request, contract, generation, archive, seen)
        result = dict(result)
        result["feedback_role"] = "none_optuna_shared_startup"
        return result

    optuna = _optuna()
    groups = validate_request(request, contract)
    shape = Shape(request["window"], len(contract["channels"]), len(contract["classes"]))
    ev, grammar = request["evolution"], request["grammar"]
    count = int(ev["offspring_per_generation"])
    study, history = _study_from_archive(request, archive, generation)
    if not history:
        raise ValueError("NO_ELIGIBLE_OPTUNA_HISTORY")
    accepted, rejected = [], []
    known = set(seen)
    for attempt in range(int(ev["max_proposal_attempts"])):
        trial = study.ask()
        genes = None
        try:
            genes = suggest_genome(trial, grammar)
            graph = lower_genome(genes, groups, shape)
            audit = analyze_graph(shape, graph)
            graph_hash = graph_identity(graph)
            if graph_hash in known:
                raise ValueError("DUPLICATE_GRAPH")
            reason = graph_budget_reason(audit, request["budget"], len(request["seeds"]))
            if reason:
                raise ValueError(reason)
            known.add(graph_hash)
            accepted.append({
                "spec": {"family": "composed_dag", "parameters": graph},
                "genes": genes,
                "graph_sha256": graph_hash,
                "topology_sha256": graph_identity(graph, topology_only=True),
                "parents": [],
                "edit": "optuna_tpe_full_genome",
                "proposal_attempt": attempt,
                "optuna_trial_number": trial.number,
                "resources": {k: v for k, v in audit.items() if k not in {"nodes", "modules"}},
            })
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)
        except ValueError as exc:
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            rejected.append({
                "attempt": attempt,
                "parents": [],
                "edit": "optuna_tpe_full_genome",
                "genes": genes,
                "reason": str(exc),
                "optuna_trial_number": trial.number,
            })
        if len(accepted) == count:
            break
    if len(accepted) != count:
        raise ValueError(f"OPTUNA_PROPOSAL_BUDGET_EXHAUSTED: {len(accepted)}/{count}; last_rejections={rejected[-3:]}")
    return {
        "schema": "phm.graph-proposals.v1",
        "generation": generation,
        "feedback_role": "development_validation_optuna_tpe",
        "parent_pool": [
            {"topology_sha256": h["topology_sha256"], "objective": h["objective"], "best_route": h["best_route"]}
            for h in history
        ],
        "accepted": accepted,
        "rejected": rejected,
        "holdout_read": False,
        "external_optimizer": {"library": "optuna", "version": optuna.__version__, "sampler": "TPESampler"},
    }
