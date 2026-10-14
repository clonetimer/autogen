from __future__ import annotations
import copy
import random


def _branch(*, difference: bool, width: int = 8):
    return {
        "difference": difference,
        "layers": [{"activation": "relu", "op": "linear", "width": width}],
        "pool": "mean",
        "share": False,
        "skip": False,
        "source": "global",
    }


def _genome():
    return {
        "branches": [_branch(difference=False), _branch(difference=True), _branch(difference=True, width=16)],
        "fusion": "concat",
        "fusion_order": [0, 1, 2],
        "fusion_width": 8,
        "head_widths": [8],
    }


def _grammar():
    return {
        "widths": [8, 16], "max_depth": 2, "max_branches": 3,
        "kernels": [3], "dilations": [1], "allow_sharing": False,
        "layer_ops": ["linear", "conv", "res_tcn", "self_attention"],
        "attention_heads": [1, 2], "tail_pool_windows": [4],
        "allow_passthrough_branches": True,
    }


def test_remove_branch_never_removes_only_level_information_branch():
    from phm_design.synthesis import mutate_genome
    parent = _genome()
    child, op = mutate_genome(parent, parent, random.Random(9), _grammar(), 9)
    assert op == "remove_branch"
    assert any(not b["difference"] for b in child["branches"])


def test_replace_subgraph_never_replaces_only_level_branch_with_difference_only_donor():
    from phm_design.synthesis import mutate_genome
    parent = _genome()
    donor = copy.deepcopy(parent)
    for b in donor["branches"]:
        b["difference"] = True
    child, op = mutate_genome(parent, donor, random.Random(1), _grammar(), 1)
    assert op == "replace_subgraph"
    assert any(not b["difference"] for b in child["branches"])


def test_propose_rejects_one_mutation_exception_and_continues(monkeypatch, tmp_path):
    import phm_design.synthesis as synthesis
    from phm_design.data import Dataset
    from tests.phm_design.test_compositional_design import request
    from tests.phm_design.test_design_system import fixture_dataset

    dev, _ = fixture_dataset(tmp_path)
    contract = Dataset(dev, allowed_splits={"train", "validation"}).contract
    cfg = request()
    first = synthesis.propose(cfg, contract, 0, [], set())["accepted"][0]
    archive = [{
        **first,
        "id": "parent0",
        "kind": "generated",
        "status": "COMPLETED",
        "mean_validation_macro_f1": 0.7,
        "seed_std_macro_f1": 0.01,
        "description": {"parameter_count": first["resources"]["parameter_count"]},
    }]
    real = synthesis.mutate_genome
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("SYNTHETIC_INVALID_MUTATION")
        return real(*args, **kwargs)

    monkeypatch.setattr(synthesis, "mutate_genome", flaky)
    out = synthesis.propose(cfg, contract, 1, archive, {first["graph_sha256"]})
    assert len(out["accepted"]) == cfg["evolution"]["offspring_per_generation"]
    assert any(r["reason"] == "SYNTHETIC_INVALID_MUTATION" for r in out["rejected"])
