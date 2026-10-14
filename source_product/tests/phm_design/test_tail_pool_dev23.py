import torch
from phm_design.graph import GraphNetwork, analyze_graph
from phm_design.models import Shape
from phm_design.synthesis import validate_request, random_genome, lower_genome, mutate_genome
import random


def _contract(c=25):
    return {"channels":[{"name":f"c{i}","unit":"1"} for i in range(c)],"classes":["a","b"]}

def _request(tails=None,passthrough=False):
    grammar={"widths":[8,16,24],"max_depth":2,"max_branches":3,"kernels":[3,5],"dilations":[1,2],"allow_sharing":False,
             "layer_ops":["linear","conv","res_tcn","self_attention"],"attention_heads":[1,2,4]}
    if tails is not None: grammar["tail_pool_windows"]=tails
    if passthrough: grammar["allow_passthrough_branches"]=True
    return {"schema":"phm.compositional-design.v1","window":128,
      "groups":[{"id":"all","channels":[f"c{i}" for i in range(25)]}],"structure_reference":"test",
      "grammar":grammar,"evolution":{"seed":1,"initial_graphs":1,"generations":1,"offspring_per_generation":1,"elite_parents":1,"max_proposal_attempts":20,"max_total_trials":20},
      "controls":[{"family":"mlp","parameters":{"widths":[8],"representation":"statistics"}}],"seeds":[11,23],
      "training":{"epochs":1,"batch_size":4,"learning_rate":0.001,"weight_decay":0.0,"patience":1},
      "budget":{"max_parameters_per_model":50000,"max_ensemble_parameters":100000,"max_trials":20,"max_trial_seconds":90,"max_multiply_accumulate_per_case":60000000},
      "selection":"mean_seed_validation_macro_f1","abstention":{"min_uncalibrated_score":0.6}}

def test_tail_mean_graph_operator_is_typed_and_causal_tail_pool():
    graph={"schema":"phm.typed-dag.v1","nodes":[
      {"id":"n0","op":"tail_mean","inputs":["input"],"args":{"window":16},"share":""},
      {"id":"n1","op":"linear","inputs":["n0"],"args":{"width":2},"share":""}],"output":"n1"}
    audit=analyze_graph(Shape(128,25,2),graph)
    assert audit["parameter_count"]==52
    net=GraphNetwork(Shape(128,25,2),graph)
    x=torch.zeros(1,128,25);x[:,112:]=2.0
    with torch.no_grad():
        pooled=net.operators["node_n0"](x)
    assert torch.allclose(pooled,torch.full((1,25),2.0))

def test_tail_pool_is_opt_in_and_validated():
    r=_request([8,16,32]); groups=validate_request(r,_contract())
    assert groups==[list(range(25))]
    bad=_request([16,129])
    try: validate_request(bad,_contract())
    except ValueError: pass
    else: raise AssertionError("window beyond sequence must fail")

def test_old_grammar_remains_without_tail_pool():
    r=_request(None); validate_request(r,_contract())
    g=random_genome(random.Random(5),r["grammar"]); assert all(not b["pool"].startswith("tail_mean_") for b in g["branches"])

def test_tail_enabled_genomes_lower_tail_mean_when_selected():
    r=_request([8,16,32]); groups=validate_request(r,_contract()); grammar=r["grammar"]
    found=False
    for seed in range(200):
        g=random_genome(random.Random(seed),grammar)
        if any(b["pool"].startswith("tail_mean_") for b in g["branches"]):
            graph=lower_genome(g,groups,Shape(128,25,2)); found=True
            assert any(n["op"]=="tail_mean" for n in graph["nodes"]); break
    assert found


def test_passthrough_branch_is_opt_in_and_can_pool_raw_tail():
    r=_request([8,16,32],passthrough=True); groups=validate_request(r,_contract()); grammar=r["grammar"]
    found=None
    for seed in range(1000):
        g=random_genome(random.Random(seed),grammar)
        for b in g["branches"]:
            if b["source"]=="global" and not b["layers"] and b["pool"].startswith("tail_mean_") and not b["difference"]:
                found=g;break
        if found is not None:break
    assert found is not None
    graph=lower_genome(found,groups,Shape(128,25,2))
    assert any(n["op"]=="tail_mean" and n["inputs"]==["input"] for n in graph["nodes"])

def test_passthrough_disabled_keeps_at_least_one_layer_per_branch():
    grammar=_request([8,16,32],passthrough=False)["grammar"]
    for seed in range(50):
        assert all(b["layers"] for b in random_genome(random.Random(seed),grammar)["branches"])


def test_initial_tail_pool_coverage_stratifies_level_branches_without_rng():
    import random
    from phm_design.synthesis import random_genome, apply_initial_tail_pool_coverage
    grammar={"widths":[8,16],"max_depth":2,"max_branches":3,"kernels":[3],"dilations":[1],"allow_sharing":False,
             "layer_ops":["linear","conv","res_tcn","self_attention"],"attention_heads":[1,2],
             "tail_pool_windows":[8,16,32],"allow_passthrough_branches":True,"initial_tail_pool_coverage":True}
    rng=random.Random(170003)
    pools=[]
    for i in range(3):
        g=random_genome(rng,grammar);apply_initial_tail_pool_coverage(g,grammar,i)
        level=next(b for b in g["branches"] if not b["difference"])
        pools.append(level["pool"])
    assert pools==["tail_mean_8","tail_mean_16","tail_mean_32"]


def test_generation0_feedback_and_random_control_match_with_tail_coverage(tmp_path):
    from phm_design.data import Dataset
    from phm_design.synthesis import propose, propose_random_control
    from tests.phm_design.test_design_system import fixture_dataset
    from tests.phm_design.test_compositional_design import request
    dev,_=fixture_dataset(tmp_path)
    contract=Dataset(dev,allowed_splits={"train","validation"}).contract
    req=request();req["grammar"].update(tail_pool_windows=[2,4],initial_tail_pool_coverage=True)
    a=propose(req,contract,0,[],set());b=propose_random_control(req,contract,0,[],set())
    assert [x["graph_sha256"] for x in a["accepted"]]==[x["graph_sha256"] for x in b["accepted"]]
    for i,x in enumerate(a["accepted"]):
        level=next(z for z in x["genes"]["branches"] if not z["difference"])
        assert level["pool"]==f"tail_mean_{[2,4][i%2]}"
