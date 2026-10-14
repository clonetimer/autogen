import random
import torch
from phm_design.models import Shape
from phm_design.graph import analyze_graph,build_graph
from phm_design.synthesis import validate_request,random_genome,lower_genome,graph_budget_reason
from phm_design.family_design import make_request
from phm_design.io import read_json


def test_grid_res3d_compiles_preserves_time_and_matches_parameter_audit():
    shape=Shape(128,19,5)
    graph={"schema":"phm.typed-dag.v1","nodes":[
      {"id":"n0","op":"grid_res3d","inputs":["input"],"args":{"width":24,"kernel":3,"dilation":1},"share":""},
      {"id":"n1","op":"mean","inputs":["n0"],"args":{},"share":""},
      {"id":"n2","op":"linear","inputs":["n1"],"args":{"width":5},"share":""}],"output":"n2"}
    audit=analyze_graph(shape,graph)
    model=build_graph(shape,graph)
    assert sum(p.numel() for p in model.parameters())==audit["parameter_count"]
    assert audit["parameter_count"]<50_000
    with torch.no_grad():
        y=model(torch.randn(2,128,19))
    assert y.shape==(2,5)


def test_grid_res3d_rejects_too_small_channel_grid():
    shape=Shape(32,2,3)
    graph={"schema":"phm.typed-dag.v1","nodes":[
      {"id":"n0","op":"grid_res3d","inputs":["input"],"args":{"width":16,"kernel":3,"dilation":1},"share":""},
      {"id":"n1","op":"mean","inputs":["n0"],"args":{},"share":""},
      {"id":"n2","op":"linear","inputs":["n1"],"args":{"width":3},"share":""}],"output":"n2"}
    try:
        analyze_graph(shape,graph)
    except ValueError as exc:
        assert "GRID_RES3D" in str(exc)
    else:
        raise AssertionError("small channel grid unexpectedly accepted")


def test_grid_policy_validates_and_generator_uses_grid_only_on_global_branches():
    from phm_design.adapters.sat_multicomponent import contract,binding
    policy=read_json("configs/phm/reference_design_grid_dev13.json")["base_design"]
    request=make_request(binding("thermal"),policy)
    groups=validate_request(request,contract("thermal"))
    shape=Shape(request["window"],len(contract("thermal")["channels"]),len(contract("thermal")["classes"]))
    found=0
    for seed in range(120):
        genes=random_genome(random.Random(seed),request["grammar"])
        for branch in genes["branches"]:
            if any(layer["op"]=="grid_res3d" for layer in branch["layers"]):
                found+=1
                assert branch["source"]=="global"
        graph=lower_genome(genes,groups,shape)
        analyze_graph(shape,graph)
    assert found>0


def test_generated_graph_mac_budget_gate_is_explicit_and_parameter_gate_still_applies():
    audit={"parameter_count":37333,"multiply_accumulate_estimate_per_case":89518184}
    budget={"max_parameters_per_model":50000,"max_ensemble_parameters":150000,"max_trials":40,
            "max_trial_seconds":90,"max_multiply_accumulate_per_case":60000000}
    assert graph_budget_reason(audit,budget,3)=="MAC_BUDGET_EXCEEDED"
    audit["multiply_accumulate_estimate_per_case"]=53000000
    assert graph_budget_reason(audit,budget,3) is None
    audit["parameter_count"]=60000
    assert graph_budget_reason(audit,budget,3)=="PARAMETER_BUDGET_EXCEEDED"
