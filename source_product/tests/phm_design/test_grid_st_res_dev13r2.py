import random
import torch
from phm_design.models import Shape
from phm_design.graph import analyze_graph,build_graph
from phm_design.synthesis import validate_request,random_genome,lower_genome
from phm_design.family_design import make_request
from phm_design.adapters.sat_multicomponent import contract,binding


def _policy():
    return {
      "schema":"phm.family-design-policy.v1",
      "grammar":{"widths":[8,16,24],"max_depth":2,"max_branches":3,
                 "kernels":[3,5],"dilations":[1,2],"allow_sharing":False,
                 "layer_ops":["linear","conv","res_tcn","grid_st_res","self_attention"],
                 "attention_heads":[1,2,4]},
      "evolution":{"seed":1,"initial_graphs":3,"generations":2,"offspring_per_generation":1,
                   "elite_parents":1,"max_proposal_attempts":300,"max_total_trials":24},
      "controls":[{"family":"mlp","parameters":{"widths":[32],"representation":"statistics"}}],
      "seeds":[11,23,37],
      "training":{"epochs":2,"batch_size":24,"learning_rate":.003,"weight_decay":.001,"patience":2},
      "budget":{"max_parameters_per_model":50000,"max_ensemble_parameters":150000,"max_trials":40,
                "max_trial_seconds":90,"max_multiply_accumulate_per_case":60000000},
      "selection":"mean_seed_validation_macro_f1","abstention":{"min_uncalibrated_score":.6}}


def test_grid_st_res_compiles_preserves_time_and_matches_parameter_audit():
    shape=Shape(128,19,5)
    graph={"schema":"phm.typed-dag.v1","nodes":[
      {"id":"n0","op":"grid_st_res","inputs":["input"],"args":{"width":24,"kernel":5,"dilation":2},"share":""},
      {"id":"n1","op":"mean","inputs":["n0"],"args":{},"share":""},
      {"id":"n2","op":"linear","inputs":["n1"],"args":{"width":5},"share":""}],"output":"n2"}
    audit=analyze_graph(shape,graph); model=build_graph(shape,graph)
    assert sum(p.numel() for p in model.parameters())==audit["parameter_count"]
    assert audit["parameter_count"] < 5000
    assert audit["multiply_accumulate_estimate_per_case"] < 8_000_000
    y=model(torch.randn(2,128,19))
    assert y.shape==(2,5)
    y.square().mean().backward()


def test_grid_st_res_rejects_too_small_channel_grid():
    shape=Shape(32,2,3)
    graph={"schema":"phm.typed-dag.v1","nodes":[
      {"id":"n0","op":"grid_st_res","inputs":["input"],"args":{"width":16,"kernel":3,"dilation":1},"share":""},
      {"id":"n1","op":"mean","inputs":["n0"],"args":{},"share":""},
      {"id":"n2","op":"linear","inputs":["n1"],"args":{"width":3},"share":""}],"output":"n2"}
    try: analyze_graph(shape,graph)
    except ValueError as exc: assert "GRID_ST_RES" in str(exc)
    else: raise AssertionError("small channel grid unexpectedly accepted")


def test_grid_st_res_generator_uses_grid_only_on_global_branches():
    request=make_request(binding("thermal"),_policy())
    groups=validate_request(request,contract("thermal"))
    shape=Shape(request["window"],len(contract("thermal")["channels"]),len(contract("thermal")["classes"]))
    found=0
    for seed in range(120):
        genes=random_genome(random.Random(seed),request["grammar"])
        for branch in genes["branches"]:
            if any(layer["op"]=="grid_st_res" for layer in branch["layers"]):
                found+=1; assert branch["source"]=="global"
        analyze_graph(shape,lower_genome(genes,groups,shape))
    assert found>0
