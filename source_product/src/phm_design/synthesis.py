"""Contract-conditioned graph synthesis with validation-guided structural evolution.

The grammar is reviewed code. It generates topology, not an unrestricted new
mathematical operator. Every offspring records its parent(s) and edit. Neither
proposal nor fit accepts a holdout path. This is workflow, not OS isolation.
"""
from __future__ import annotations
import copy
from pathlib import Path
import random
import shutil
from typing import Callable
import numpy as np
import torch
from .data import Dataset
from .engine import (fit, config_check, environment, implementation_hash)
from .graph import (SCHEMA, analyze_graph, graph_identity, graph_to_dot, compositional_registry)
from .io import digest, read_json, write_json, seal, verify, run_lock, sha_file
from .models import Shape, integer
from .resource_planner import plan_fixed_controls


def validate_request(request: dict, contract: dict) -> list[list[int]]:
    required = {"schema", "window", "groups", "structure_reference", "grammar", "evolution", "controls", "seeds", "training", "budget", "selection", "abstention"}
    if set(request) != required or request["schema"] != "phm.compositional-design.v1":
        raise ValueError("DESIGN_REQUEST_SCHEMA")
    if not isinstance(request["structure_reference"], str) or not request["structure_reference"].strip():
        raise ValueError("MISSING_STRUCTURAL_PRIOR_REFERENCE")
    shape = Shape(request["window"], len(contract["channels"]), len(contract["classes"]))
    names = [c["name"] for c in contract["channels"]]
    units = [c["unit"] for c in contract["channels"]]
    groups, ids = [], set()
    if not isinstance(request["groups"], list) or not 1 <= len(request["groups"]) <= 8:
        raise ValueError("GROUP_COUNT")
    for group in request["groups"]:
        if set(group) != {"id", "channels"} or not isinstance(group["id"], str) or not group["id"] or group["id"] in ids:
            raise ValueError("GROUP_SCHEMA")
        ids.add(group["id"])
        if not isinstance(group["channels"], list) or not group["channels"] or any(n not in names for n in group["channels"]):
            raise ValueError("GROUP_CHANNEL_NOT_IN_INPUT_CONTRACT")
        groups.append([names.index(n) for n in group["channels"]])
    flat = [i for group in groups for i in group]
    if sorted(flat) != list(range(shape.channels)):
        raise ValueError("GROUPS_MUST_PARTITION_INPUT_CHANNELS")
    grammar = request["grammar"]
    base_grammar={"widths","max_depth","max_branches","kernels","dilations","allow_sharing"}
    optional_grammar={"layer_ops","attention_heads","tail_pool_windows","allow_passthrough_branches","initial_tail_pool_coverage"}
    keys=set(grammar)
    if not base_grammar.issubset(keys) or keys-(base_grammar|optional_grammar):
        raise ValueError("GRAMMAR_SCHEMA")
    if ("layer_ops" in grammar)!=("attention_heads" in grammar):
        raise ValueError("GRAMMAR_SCHEMA")
    for field, hi in (("widths", 128), ("kernels", 9), ("dilations", 8)):
        if not isinstance(grammar[field], list) or not grammar[field] or len(set(grammar[field])) != len(grammar[field]):
            raise ValueError("GRAMMAR_CHOICES")
        for v in grammar[field]:
            integer(v, field, 2 if field == "widths" else 1, hi)
    if any(k % 2 != 1 for k in grammar["kernels"]):
        raise ValueError("ODD_KERNELS_REQUIRED")
    integer(grammar["max_depth"], "max_depth", 1, 4)
    integer(grammar["max_branches"], "max_branches", 2, 4)
    if type(grammar["allow_sharing"]) is not bool:
        raise ValueError("ALLOW_SHARING_MUST_BE_BOOLEAN")
    if "layer_ops" in grammar:
        allowed={"linear","conv","res_tcn","grid_res3d","grid_st_res","self_attention"}
        if not isinstance(grammar["layer_ops"],list) or not grammar["layer_ops"] or len(set(grammar["layer_ops"]))!=len(grammar["layer_ops"]) or any(x not in allowed for x in grammar["layer_ops"]):
            raise ValueError("GRAMMAR_LAYER_OPS")
        if not isinstance(grammar["attention_heads"],list) or not grammar["attention_heads"] or len(set(grammar["attention_heads"]))!=len(grammar["attention_heads"]):
            raise ValueError("GRAMMAR_ATTENTION_HEADS")
        for h in grammar["attention_heads"]:
            integer(h,"attention_heads",1,16)
        if "self_attention" in grammar["layer_ops"] and not any(w%h==0 for w in grammar["widths"] for h in grammar["attention_heads"]):
            raise ValueError("NO_COMPATIBLE_ATTENTION_WIDTH_HEADS")
    if "tail_pool_windows" in grammar:
        windows=grammar["tail_pool_windows"]
        if not isinstance(windows,list) or not windows or len(set(windows))!=len(windows):
            raise ValueError("GRAMMAR_TAIL_POOL_WINDOWS")
        for w in windows:
            integer(w,"tail_pool_windows",1,shape.window)
    if "allow_passthrough_branches" in grammar and type(grammar["allow_passthrough_branches"]) is not bool:
        raise ValueError("ALLOW_PASSTHROUGH_BRANCHES_MUST_BE_BOOLEAN")
    if "initial_tail_pool_coverage" in grammar:
        if type(grammar["initial_tail_pool_coverage"]) is not bool:
            raise ValueError("INITIAL_TAIL_POOL_COVERAGE_MUST_BE_BOOLEAN")
        if grammar["initial_tail_pool_coverage"] and not grammar.get("tail_pool_windows"):
            raise ValueError("INITIAL_TAIL_POOL_COVERAGE_REQUIRES_WINDOWS")
    if grammar["allow_sharing"]:
        signatures = [tuple(units[i] for i in group) for group in groups]
        if any(s != signatures[0] for s in signatures):
            raise ValueError("SHARING_REQUIRES_MATCHED_GROUP_UNIT_ROLES")
    ev = request["evolution"]
    if set(ev) != {"seed", "initial_graphs", "generations", "offspring_per_generation", "elite_parents", "max_proposal_attempts", "max_total_trials"}:
        raise ValueError("EVOLUTION_SCHEMA")
    integer(ev["seed"], "proposal_seed", 0, 2**32-1)
    for k in ("initial_graphs", "offspring_per_generation", "elite_parents"):
        integer(ev[k], k, 1, 64)
    integer(ev["generations"], "generations", 1, 10)
    integer(ev["max_proposal_attempts"], "max_proposal_attempts", 1, 10_000)
    integer(ev["max_total_trials"], "max_total_trials", 1, 10_000)
    if not isinstance(request["controls"], list) or not request["controls"] or any(c.get("family") == "composed_dag" for c in request["controls"]):
        raise ValueError("FIXED_NEURAL_CONTROLS_REQUIRED")
    control_plans=plan_fixed_controls(shape,request["controls"],request["budget"],len(request["seeds"]))
    config_check(training_config(request,[p["effective_spec"] for p in control_plans]), shape.channels, shape.classes, compositional_registry())
    planned = (ev["initial_graphs"] + (ev["generations"] - 1) * ev["offspring_per_generation"] + len(request["controls"])) * len(request["seeds"])
    if planned > ev["max_total_trials"]:
        raise ValueError("GLOBAL_DESIGN_TRIAL_BUDGET_EXCEEDED")
    return groups


def training_config(request: dict, specs: list[dict]) -> dict:
    return {"schema": "phm.search.v1", "window": request["window"], "candidates": specs,
            **{k: copy.deepcopy(request[k]) for k in ("seeds", "training", "budget", "selection", "abstention")}}


def fixed_control_resource_plan(request:dict,contract:dict)->list[dict]:
    shape=Shape(request["window"],len(contract["channels"]),len(contract["classes"]))
    return plan_fixed_controls(shape,request["controls"],request["budget"],len(request["seeds"]))


class GraphBuilder:
    def __init__(self):
        self.nodes, self.shares = [], {}

    def add(self, op: str, inputs: list[str], args: dict | None = None, share: str = "") -> str:
        if share and share not in self.shares:
            self.shares[share] = f"s{len(self.shares)}"
        name = f"n{len(self.nodes)}"
        self.nodes.append({"id": name, "op": op, "inputs": inputs, "args": args or {}, "share": self.shares.get(share, "")})
        return name


def lower_genome(genes: dict, groups: list[list[int]], shape: Shape) -> dict:
    """Lower a variable branch/operation genome to an explicit DAG program."""
    b = GraphBuilder()
    vectors = []
    for branch_i, branch in enumerate(genes["branches"]):
        sources = [b.add("select", ["input"], {"indices": group}) for group in groups] if branch["source"] == "components" else ["input"]
        locals_ = []
        for source in sources:
            current = b.add("difference", [source]) if branch["difference"] else source
            skip_source = current
            share_prefix = f"b{branch_i}" if branch["share"] and branch["source"] == "components" else ""
            for layer_i, layer in enumerate(branch["layers"]):
                args = {"width": layer["width"]}
                if layer["op"] in {"conv","res_tcn","grid_res3d","grid_st_res"}:
                    args.update(kernel=layer["kernel"], dilation=layer["dilation"])
                elif layer["op"] == "self_attention":
                    args.update(heads=layer["heads"])
                current = b.add(layer["op"], [current], args, f"{share_prefix}_l{layer_i}" if share_prefix else "")
                current = b.add(layer["activation"], [current])
            if branch["skip"]:
                projected = b.add("linear", [skip_source], {"width": branch["layers"][-1]["width"]}, f"{share_prefix}_skip" if share_prefix else "")
                current = b.add("add", [current, projected])
            pool=branch["pool"]
            if isinstance(pool,str) and pool.startswith("tail_mean_"):
                locals_.append(b.add("tail_mean", [current], {"window": int(pool.rsplit("_",1)[1])}))
            else:
                locals_.append(b.add(pool, [current]))
        vector = locals_[0] if len(locals_) == 1 else b.add("concat", locals_)
        vectors.append(vector)
    if genes["fusion"] != "concat":
        vectors = [b.add("linear", [v], {"width": genes["fusion_width"]}) for v in vectors]
    ordered = [vectors[i] for i in genes["fusion_order"]]
    current = ordered[0]
    for v in ordered[1:]:
        current = b.add(genes["fusion"], [current, v])
    for width in genes["head_widths"]:
        current = b.add("linear", [current], {"width": width})
        current = b.add("silu", [current])
    current = b.add("linear", [current], {"width": shape.classes})
    graph = {"schema": SCHEMA, "nodes": b.nodes, "output": current}
    analyze_graph(shape, graph)
    return graph


def random_layer(rng: random.Random, grammar: dict, *, source: str | None = None) -> dict:
    ops=grammar.get("layer_ops",["conv","linear"])
    # Small component branches do not have enough channels to form a useful
    # 2-D channel grid. The grid primitive is therefore proposed only globally.
    if source == "components":
        ops=[op for op in ops if op not in {"grid_res3d","grid_st_res"}] or ["linear"]
    op=rng.choice(ops)
    width=rng.choice(grammar["widths"])
    layer={"op":op,"width":width,"activation":rng.choice(["relu","silu","tanh"])}
    if op in {"conv","res_tcn","grid_res3d","grid_st_res"}:
        layer.update(kernel=rng.choice(grammar["kernels"]),dilation=rng.choice(grammar["dilations"]))
    elif op=="self_attention":
        compatible=[h for h in grammar["attention_heads"] if width%h==0]
        if not compatible:
            widths=[w for w in grammar["widths"] if any(w%h==0 for h in grammar["attention_heads"])]
            width=rng.choice(widths);layer["width"]=width
            compatible=[h for h in grammar["attention_heads"] if width%h==0]
        layer["heads"]=rng.choice(compatible)
    return layer


def _pool_choices(grammar:dict)->list[str]:
    choices=["statistics","mean","last"]
    choices.extend(f"tail_mean_{int(w)}" for w in grammar.get("tail_pool_windows",[]))
    return choices


def random_branch(rng: random.Random, grammar: dict) -> dict:
    source=rng.choice(["components", "global"])
    # Preserve the exact legacy RNG consumption order when passthrough is off:
    # source -> difference -> depth/layers -> pool -> skip -> sharing.
    difference=rng.choice([False, True])
    min_depth=0 if grammar.get("allow_passthrough_branches",False) else 1
    layers=[random_layer(rng, grammar, source=source) for _ in range(rng.randint(min_depth, grammar["max_depth"]))]
    pool=rng.choice(_pool_choices(grammar))
    skip=bool(layers) and rng.choice([False, True])
    share=bool(layers) and grammar["allow_sharing"] and rng.choice([False, True])
    return {"source": source, "difference": difference, "layers": layers,
            "pool": pool, "skip": skip, "share": share}


def random_genome(rng: random.Random, grammar: dict) -> dict:
    branches = [random_branch(rng, grammar) for _ in range(rng.randint(2, grammar["max_branches"]))]
    # At least one level branch: differences alone cannot preserve DC offsets.
    branches[0]["difference"] = False
    order = list(range(len(branches))); rng.shuffle(order)
    return {"branches": branches, "fusion": rng.choice(["concat", "add", "gated_add"]),
            "fusion_width": rng.choice(grammar["widths"]), "fusion_order": order,
            "head_widths": [rng.choice(grammar["widths"]) for _ in range(rng.randint(0, 2))]}


def apply_initial_tail_pool_coverage(genes:dict,grammar:dict,accepted_index:int)->dict:
    """Stratify declared tail windows across generation-0 proposals.

    This is opt-in and consumes no extra RNG, so legacy seeded policies replay
    exactly.  The first non-differenced branch is used because it preserves level
    information; matched random-control generation applies the identical rule.
    """
    if not grammar.get("initial_tail_pool_coverage",False):
        return genes
    windows=grammar["tail_pool_windows"]
    target=f"tail_mean_{int(windows[accepted_index%len(windows)])}"
    idx=next(i for i,b in enumerate(genes["branches"]) if not b["difference"])
    genes["branches"][idx]["pool"]=target
    return genes


MUTATIONS = ("add_branch", "replace_subgraph", "change_fusion", "insert_operator", "toggle_skip", "toggle_sharing", "change_pool", "toggle_difference", "change_scope", "remove_branch", "change_activation", "reorder_fusion")


def _grid_layer(branch:dict)->bool:
    return any(layer.get("op") in {"grid_res3d","grid_st_res"} for layer in branch.get("layers",[]))


def _has_level_information(branches:list[dict])->bool:
    """A legal genome must retain at least one non-differenced branch.

    Pure-difference graphs discard all DC/level information and are intentionally
    excluded by the design grammar.  Mutation feasibility therefore has to be
    checked *before* an edit is sampled, not after a whole search is aborted.
    """
    return any(not b["difference"] for b in branches)


def _safe_replace_choices(branches:list[dict],donor_branches:list[dict])->list[tuple[int,dict]]:
    choices=[]
    for i,b in enumerate(branches):
        for candidate in donor_branches:
            if candidate==b:
                continue
            trial=list(branches);trial[i]=candidate
            if _has_level_information(trial):
                choices.append((i,candidate))
    return choices


def _safe_remove_indices(branches:list[dict])->list[int]:
    return [i for i in range(len(branches)) if _has_level_information(branches[:i]+branches[i+1:])]


def _applicable_mutations(parent:dict,donor:dict,grammar:dict)->list[str]:
    branches=parent["branches"]
    ops=[]
    if len(branches)<grammar["max_branches"]:ops.append("add_branch")
    if _safe_replace_choices(branches,donor["branches"]):ops.append("replace_subgraph")
    ops.append("change_fusion")
    if any(len(b["layers"])<grammar["max_depth"] for b in branches):ops.append("insert_operator")
    if any(b["layers"] for b in branches):ops.append("toggle_skip")
    if grammar["allow_sharing"]:ops.append("toggle_sharing")
    ops.append("change_pool")
    ops.append("toggle_difference")
    if any(b["source"]=="components" or not _grid_layer(b) for b in branches):ops.append("change_scope")
    if len(branches)>2 and _safe_remove_indices(branches):ops.append("remove_branch")
    if any(b["layers"] for b in branches):ops.append("change_activation")
    # Reordering concat or sequential gated-add changes the computation. Plain
    # addition is commutative/associative for design purposes and is excluded.
    if parent["fusion"] in {"concat","gated_add"} and len(branches)>1:ops.append("reorder_fusion")
    return ops


def mutate_genome(parent: dict, donor: dict, rng: random.Random, grammar: dict, attempt: int) -> tuple[dict, str]:
    """Apply exactly one *effective* structural mutation.

    Older code selected an operation before checking whether it was applicable
    and then shuffled ``fusion_order`` unconditionally.  A saturated
    ``add_branch`` therefore became a mislabeled fusion reorder; with ``add``
    fusion this was semantically equivalent and wasted feedback trials.  The
    revised mutation engine chooses only applicable operations and never adds a
    second hidden mutation.
    """
    genes=copy.deepcopy(parent)
    branches=genes["branches"]
    applicable=_applicable_mutations(parent,donor,grammar)
    if not applicable:raise ValueError("NO_APPLICABLE_STRUCTURAL_MUTATION")
    # Preserve deterministic attempt scheduling while skipping impossible ops.
    preferred=MUTATIONS[attempt%len(MUTATIONS)]
    operation=preferred if preferred in applicable else applicable[attempt%len(applicable)]

    if operation=="add_branch":
        old_order=list(genes["fusion_order"]);new_idx=len(branches)
        branches.append(random_branch(rng,grammar))
        genes["fusion_order"]=old_order+[new_idx]
    elif operation=="replace_subgraph":
        choices=_safe_replace_choices(branches,donor["branches"])
        i,candidate=rng.choice(choices);branches[i]=copy.deepcopy(candidate)
    elif operation=="change_fusion":
        genes["fusion"]=rng.choice([x for x in ("concat","add","gated_add") if x!=genes["fusion"]])
    elif operation=="insert_operator":
        choices=[b for b in branches if len(b["layers"])<grammar["max_depth"]]
        branch=rng.choice(choices)
        branch["layers"].insert(rng.randrange(len(branch["layers"])+1),random_layer(rng,grammar,source=branch["source"]))
    elif operation=="toggle_skip":
        branch=rng.choice([b for b in branches if b["layers"]]);branch["skip"]=not branch["skip"]
    elif operation=="toggle_sharing":
        branch=rng.choice(branches);branch["source"]="components";branch["share"]=not branch["share"]
    elif operation=="change_pool":
        branch=rng.choice(branches);branch["pool"]=rng.choice([x for x in _pool_choices(grammar) if x!=branch["pool"]])
    elif operation=="toggle_difference":
        # Avoid immediately undoing the edit via the at-least-one-level guard.
        false_idx=[i for i,b in enumerate(branches) if not b["difference"]]
        if len(false_idx)==1 and all(branches[i]["difference"] for i in range(len(branches)) if i!=false_idx[0]):
            choices=[b for b in branches if b["difference"]]
        else:choices=branches
        branch=rng.choice(choices);branch["difference"]=not branch["difference"]
    elif operation=="change_scope":
        choices=[b for b in branches if b["source"]=="components" or not _grid_layer(b)]
        branch=rng.choice(choices);branch["source"]="global" if branch["source"]=="components" else "components"
        if branch["source"]!="components":branch["share"]=False
    elif operation=="remove_branch":
        idx=rng.choice(_safe_remove_indices(branches));branches.pop(idx)
        genes["fusion_order"]=[j-(j>idx) for j in genes["fusion_order"] if j!=idx]
    elif operation=="reorder_fusion":
        old=list(genes["fusion_order"]);new=list(old)
        while new==old:rng.shuffle(new)
        genes["fusion_order"]=new
    else:
        branch=rng.choice([b for b in branches if b["layers"]]);layer=rng.choice(branch["layers"])
        layer["activation"]=rng.choice([a for a in ("relu","silu","tanh") if a!=layer["activation"]])

    if all(b["difference"] for b in branches):
        raise ValueError("MUTATION_REMOVED_ALL_LEVEL_INFORMATION")
    if genes==parent:raise ValueError("NO_OP_STRUCTURAL_MUTATION")
    return genes,operation


def ranking(record: dict) -> tuple:
    return (-record["mean_validation_macro_f1"], record["seed_std_macro_f1"], record["description"]["parameter_count"], record["id"])


def graph_budget_reason(audit: dict, budget: dict, seed_count: int) -> str | None:
    """Static generated-graph resource gate; runtime timeout remains a backstop.

    The optional MAC ceiling is explicit policy, not a learned threshold. It is
    applied only to generated DAG proposals because their graph audit exposes an
    implementation-independent operation estimate. Fixed controls remain direct
    competitors and are still guarded by the empirical per-trial time budget.
    """
    if audit["parameter_count"] > budget["max_parameters_per_model"] or audit["parameter_count"] * seed_count > budget["max_ensemble_parameters"]:
        return "PARAMETER_BUDGET_EXCEEDED"
    cap = budget.get("max_multiply_accumulate_per_case")
    if cap is not None and audit["multiply_accumulate_estimate_per_case"] > cap:
        return "MAC_BUDGET_EXCEEDED"
    return None


def propose(request: dict, contract: dict, generation: int, archive: list[dict], seen: set[str]) -> dict:
    groups = validate_request(request, contract)
    shape = Shape(request["window"], len(contract["channels"]), len(contract["classes"]))
    ev, grammar = request["evolution"], request["grammar"]
    rng = random.Random(ev["seed"] + generation * 1_000_003)
    count = ev["initial_graphs"] if generation == 0 else ev["offspring_per_generation"]
    parents = sorted([a for a in archive if a["kind"] == "generated" and a["status"] == "COMPLETED"], key=ranking)[:ev["elite_parents"]]
    if generation > 0 and not parents:
        raise ValueError("NO_ELIGIBLE_COMPOSED_PARENT")
    accepted, rejected = [], []
    known = set(seen)
    for attempt in range(ev["max_proposal_attempts"]):
        parent_ids = []
        genes,edit=None,"grammar_synthesis" if generation==0 else "mutation_rejected"
        try:
            if generation == 0:
                genes = random_genome(rng, grammar)
                apply_initial_tail_pool_coverage(genes,grammar,len(accepted))
            else:
                parent, donor = rng.choice(parents), rng.choice(parents)
                parent_ids = [parent["id"]]
                genes, edit = mutate_genome(parent["genes"], donor["genes"], rng, grammar, attempt)
                if edit == "replace_subgraph":
                    parent_ids.append(donor["id"])
            graph = lower_genome(genes, groups, shape)
            audit = analyze_graph(shape, graph)
            graph_hash = graph_identity(graph)
            if graph_hash in known:
                raise ValueError("DUPLICATE_GRAPH")
            reason = graph_budget_reason(audit, request["budget"], len(request["seeds"]))
            if reason:
                raise ValueError(reason)
            # Parents must actually differ in exact and topological identity.
            if generation > 0 and graph_identity(graph, topology_only=True) == parent["topology_sha256"]:
                raise ValueError("NO_STRUCTURAL_EDIT")
            known.add(graph_hash)
            accepted.append({"spec": {"family": "composed_dag", "parameters": graph}, "genes": genes,
                             "graph_sha256": graph_hash, "topology_sha256": graph_identity(graph, topology_only=True),
                             "parents": parent_ids, "edit": edit, "proposal_attempt": attempt,
                             "resources": {k: v for k, v in audit.items() if k not in {"nodes", "modules"}}})
        except ValueError as exc:
            rejected.append({"attempt": attempt, "parents": parent_ids, "edit": edit, "genes": genes, "reason": str(exc)})
        if len(accepted) == count:
            break
    if len(accepted) != count:
        raise ValueError(f"GRAMMAR_PROPOSAL_BUDGET_EXHAUSTED: {len(accepted)}/{count}; last_rejections={rejected[-3:]}")
    return {"schema": "phm.graph-proposals.v1", "generation": generation,
            "feedback_role": "development_validation_only" if generation else "none",
            "parent_pool": [{"id": p["id"], "mean_validation_macro_f1": p["mean_validation_macro_f1"]} for p in parents],
            "accepted": accepted, "rejected": rejected, "holdout_read": False}



def propose_random_control(request: dict, contract: dict, generation: int, archive: list[dict], seen: set[str]) -> dict:
    """Matched-budget legal-graph control that never uses validation feedback.

    Generation 0 intentionally matches ``propose`` exactly. Later generations
    draw fresh legal genomes with the same count/budget instead of selecting
    parents or applying validation-guided mutations.
    """
    groups = validate_request(request, contract)
    shape = Shape(request["window"], len(contract["channels"]), len(contract["classes"]))
    ev, grammar = request["evolution"], request["grammar"]
    rng = random.Random(ev["seed"] + generation * 1_000_003)
    count = ev["initial_graphs"] if generation == 0 else ev["offspring_per_generation"]
    accepted, rejected = [], []
    known = set(seen)
    for attempt in range(ev["max_proposal_attempts"]):
        genes, edit = random_genome(rng, grammar), "grammar_synthesis"
        if generation==0:
            apply_initial_tail_pool_coverage(genes,grammar,len(accepted))
        try:
            graph = lower_genome(genes, groups, shape)
            audit = analyze_graph(shape, graph)
            graph_hash = graph_identity(graph)
            if graph_hash in known:
                raise ValueError("DUPLICATE_GRAPH")
            reason = graph_budget_reason(audit, request["budget"], len(request["seeds"]))
            if reason:
                raise ValueError(reason)
            known.add(graph_hash)
            accepted.append({"spec": {"family": "composed_dag", "parameters": graph}, "genes": genes,
                             "graph_sha256": graph_hash, "topology_sha256": graph_identity(graph, topology_only=True),
                             "parents": [], "edit": edit, "proposal_attempt": attempt,
                             "resources": {k: v for k, v in audit.items() if k not in {"nodes", "modules"}}})
        except ValueError as exc:
            rejected.append({"attempt": attempt, "parents": [], "edit": edit, "genes": genes, "reason": str(exc)})
        if len(accepted) == count:
            break
    if len(accepted) != count:
        raise ValueError(f"GRAMMAR_PROPOSAL_BUDGET_EXHAUSTED: {len(accepted)}/{count}; last_rejections={rejected[-3:]}")
    return {"schema": "phm.graph-proposals.v1", "generation": generation,
            "feedback_role": "none_random_control", "parent_pool": [],
            "accepted": accepted, "rejected": rejected, "holdout_read": False}

def export_selection(output: Path, selection: dict, fingerprint: str, destination: Path) -> None:
    """Export one selected candidate using the existing frozen-ensemble format."""
    run = output / f"generation_{selection['generation']:02d}" / "run"
    verify(run)
    destination.mkdir(parents=True, exist_ok=False)
    plan = read_json(run / "plan.json")
    config = plan["config"]
    development = read_json(run / "development_snapshot.json")
    weights = []
    for seed in config["seeds"]:
        old = run / "trials" / selection["id"] / f"seed_{seed}" / "model.pt"
        new = destination / "weights" / f"seed_{seed}.pt"
        new.parent.mkdir(exist_ok=True)
        shutil.copyfile(old, new)
        weights.append(str(new.relative_to(destination)))
    for filename in ("baseline.json", "preprocessing.json", "development_snapshot.json", "observation_hashes.json", "data_counts.json"):
        shutil.copyfile(run / filename, destination / filename)
    bundle = {"schema": "phm.neural-ensemble.v1", "spec": selection["spec"],
              "architecture_description": selection["description"],
              "shape": {"window": config["window"], "channels": len(development["contract"]["channels"]), "classes": len(development["contract"]["classes"])},
              "contract": development["contract"], "scaler": read_json(run / "preprocessing.json"),
              "weights": weights, "seeds": config["seeds"], "max_parameters": config["budget"]["max_parameters_per_model"],
              "abstention": config["abstention"], "score_semantics": "uncalibrated_known_class_softmax",
              "fingerprint": digest({"design": fingerprint, "selection": selection["id"]}),
              "torch_version": str(torch.__version__), "training_roles": ["train"], "validation_used_for_early_stopping": True,
              "refit_on_validation": False, "claim_status": "RESEARCH_ONLY", "source_design_fingerprint": fingerprint}
    write_json(destination / "model_bundle.json", bundle)
    write_json(destination / "selection.json", selection)
    seal(destination, [str(p.relative_to(destination)) for p in destination.rglob('*') if p.is_file()])


def design(dataset_path: str | Path, request_path: str | Path, output: str | Path, *, resume: bool = False,
           on_trial_complete: Callable[[dict], None] | None = None) -> dict:
    data = Dataset(dataset_path, allowed_splits={"train", "validation"})
    data.verify_files()
    request = read_json(request_path)
    groups = validate_request(request, data.contract)
    fingerprint = digest({"request": request, "dataset": data.hash, "code": implementation_hash(), "environment": environment()})
    output = Path(output)
    with run_lock(output):
        if (output / "design_plan.json").exists():
            plan = read_json(output / "design_plan.json")
            if plan["fingerprint"] != fingerprint:
                raise ValueError("DESIGN_RESUME_CONTRACT_OR_CODE_CHANGED")
            if not resume:
                raise FileExistsError("DESIGN_EXISTS_USE_RESUME")
            if (output / "manifest.json").exists():
                verify(output)
                return read_json(output / "design_report.json")
        else:
            write_json(output / "design_plan.json", {"fingerprint": fingerprint, "request": request, "contract": data.contract,
                       "dataset_hash": data.hash, "groups": groups, "environment": environment(), "holdout_access": False,
                       "meaning_of_design": "grammar-generated operator topology and validation-guided structural mutations; not new operator discovery"})
        control_plans=fixed_control_resource_plan(request,data.contract)
        write_json(output / "fixed_control_resource_plan.json", {"schema":"phm.fixed-control-resource-plans.v1","plans":control_plans})
        effective_controls=[p["effective_spec"] for p in control_plans]
        control_plan_by_spec={digest(p["effective_spec"]):p for p in control_plans}
        archive = []
        seen = set()
        for generation in range(request["evolution"]["generations"]):
            folder = output / f"generation_{generation:02d}"
            proposals = propose(request, data.contract, generation, archive, seen)
            specs = [p["spec"] for p in proposals["accepted"]]
            if generation == 0:
                specs += effective_controls
            cfg = training_config(request, specs)
            commit = {"proposals": digest(proposals), "config": digest(cfg)}
            if (folder / "commit.json").exists():
                if read_json(folder / "commit.json") != commit or digest(read_json(folder / "proposals.json")) != commit["proposals"] or digest(read_json(folder / "search.json")) != commit["config"]:
                    raise ValueError("GENERATION_PLAN_CHANGED")
            else:
                write_json(folder / "proposals.json", proposals)
                write_json(folder / "search.json", cfg)
                write_json(folder / "commit.json", commit)
            report = fit(dataset_path, folder / "search.json", folder / "run", resume=resume, on_trial_complete=on_trial_complete)
            by_spec = {digest(p["spec"]): p for p in proposals["accepted"]}
            for candidate in report["candidate_results"]:
                proposal = by_spec.get(digest(candidate["spec"]))
                record = {k: candidate[k] for k in ("id", "spec", "description", "status")}
                record.update(generation=generation, kind="generated" if proposal else "fixed_control")
                if not proposal:
                    record["resource_plan"]=control_plan_by_spec.get(digest(candidate["spec"]))
                if candidate["status"] == "COMPLETED":
                    record.update({k: candidate[k] for k in ("mean_validation_macro_f1", "seed_std_macro_f1", "ensemble_validation")})
                if proposal:
                    record.update({k: proposal[k] for k in ("genes", "graph_sha256", "topology_sha256", "parents", "edit", "resources")})
                    seen.add(proposal["graph_sha256"])
                archive.append(record)
            write_json(output / "lineage.json", archive)
        generated = sorted([a for a in archive if a["kind"] == "generated" and a["status"] == "COMPLETED"], key=ranking)
        controls = sorted([a for a in archive if a["kind"] == "fixed_control" and a["status"] == "COMPLETED"], key=ranking)
        winner, control = (generated[0] if generated else None), (controls[0] if controls else None)
        for name, selected in (("selected_composed", winner), ("selected_control", control)):
            dest = output / name
            if dest.exists():
                # Only an interrupted, unsealed export may be reconstructed.
                if (dest / "manifest.json").exists():
                    verify(dest)
                    if read_json(dest / "selection.json") != selected:
                        raise ValueError("FROZEN_SELECTION_CHANGED")
                    continue
                shutil.rmtree(dest)
            if selected:
                export_selection(output, selected, fingerprint, dest)
        if winner:
            write_json(output / "best_architecture.json", winner["spec"])
            (output / "best_architecture.dot").write_text(graph_to_dot(winner["spec"]["parameters"]), encoding="utf-8")
        report = {"schema": "phm.compositional-design-report.v1", "status": "COMPLETED" if winner else "NO_FEASIBLE_DESIGNED_CANDIDATE",
                  "fingerprint": fingerprint, "counts": data.counts(), "candidate_count": len(archive),
                  "generated_graphs": sum(a["kind"] == "generated" for a in archive),
                  "distinct_topologies": len({a["topology_sha256"] for a in archive if a["kind"] == "generated"}),
                  "best_composed": winner, "best_fixed_neural_control": control,
                  "validation_mean_macro_f1_difference": winner["mean_validation_macro_f1"] - control["mean_validation_macro_f1"] if winner and control else None,
                  "lineage": "lineage.json", "test_used": False, "physical_law_discovery": False,
                  "novel_scientific_architecture_claim": False, "claim_status": "RESEARCH_ONLY",
                  "simulation_performance_qualified": False, "production_release_qualified": False,
                  "limits": ["finite reviewed grammar", "small adaptive development validation", "fixed case classification only", "no OS isolation", "same-environment replay only", "trained from scratch, no morphism weight inheritance"]}
        write_json(output / "design_report.json", report)
        files = [str(p.relative_to(output)) for p in output.rglob('*') if p.is_file() and p.name not in {".lock", "manifest.json", "evaluation_ledger.json"} and not p.name.endswith('.tmp')]
        seal(output, files)
        return report
