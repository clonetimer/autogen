
import torch
from phm_design.models import Shape
from phm_design.graph import analyze_graph,build_graph
from phm_design.synthesis import validate_request,random_genome,lower_genome
from phm_design.family_design import make_request
from phm_design.io import read_json


def test_res_tcn_and_attention_primitives_compile_and_preserve_sequence():
    shape=Shape(32,6,5)
    graph={"schema":"phm.typed-dag.v1","nodes":[
      {"id":"n0","op":"res_tcn","inputs":["input"],"args":{"width":32,"kernel":3,"dilation":2},"share":""},
      {"id":"n1","op":"self_attention","inputs":["n0"],"args":{"width":32,"heads":4},"share":""},
      {"id":"n2","op":"mean","inputs":["n1"],"args":{},"share":""},
      {"id":"n3","op":"linear","inputs":["n2"],"args":{"width":5},"share":""}],
      "output":"n3"}
    audit=analyze_graph(shape,graph)
    assert audit["parameter_count"]>0
    model=build_graph(shape,graph)
    y=model(torch.randn(3,32,6))
    assert y.shape==(3,5)


def test_advanced_policy_validates_on_multicomponent_contract():
    from phm_design.adapters.sat_multicomponent import contract,binding
    policy=read_json("configs/phm/advanced_spatiotemporal_dev12.json")["base_design"]
    request=make_request(binding("thermal"),policy)
    groups=validate_request(request,contract("thermal"))
    assert groups


def test_advanced_random_genome_can_lower_many_seeded_samples():
    import random
    from phm_design.adapters.sat_multicomponent import contract,binding
    policy=read_json("configs/phm/advanced_spatiotemporal_dev12.json")["base_design"]
    request=make_request(binding("thermal"),policy)
    groups=validate_request(request,contract("thermal"))
    shape=Shape(request["window"],len(contract("thermal")["channels"]),len(contract("thermal")["classes"]))
    seen=set()
    found=set()
    for seed in range(30):
        genes=random_genome(random.Random(seed),request["grammar"])
        graph=lower_genome(genes,groups,shape)
        analyze_graph(shape,graph)
        found|={layer["op"] for b in genes["branches"] for layer in b["layers"]}
    assert {"res_tcn","self_attention"} <= found
