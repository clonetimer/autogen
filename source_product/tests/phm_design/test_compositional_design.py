from __future__ import annotations
import copy
import json
from pathlib import Path
import random
import numpy as np
import pytest
import torch
from phm_design.data import Dataset
from phm_design.engine import load_ensemble, ensemble_scores, evaluate, deterministic_cpu
from phm_design.graph import (GraphNetwork, analyze_graph, graph_identity, compositional_registry, SCHEMA)
from phm_design.synthesis import (GraphBuilder, design, propose, lower_genome, random_genome, validate_request)
from phm_design.models import Shape
from phm_design.io import read_json, write_json, sha_file, verify
from tests.phm_design.test_design_system import fixture_dataset, config


def request(channels=2):
    old = config()
    return {"schema": "phm.compositional-design.v1", "window": 8,
            "groups": [{"id": f"part{i}", "channels": [f"sensor.x{i}"]} for i in range(channels)],
            "structure_reference": "software fixture only, no physics claim",
            "grammar": {"widths": [4, 8], "max_depth": 2, "max_branches": 3, "kernels": [3, 5], "dilations": [1, 2], "allow_sharing": True},
            "evolution": {"seed": 722, "initial_graphs": 3, "generations": 2, "offspring_per_generation": 2, "elite_parents": 2, "max_proposal_attempts": 200, "max_total_trials": 20},
            "controls": old["candidates"][:1], "seeds": [3, 5],
            "training": {**old["training"], "epochs": 3}, "budget": {**old["budget"], "max_parameters_per_model": 40000, "max_ensemble_parameters": 80000, "max_trials": 20},
            "selection": old["selection"], "abstention": old["abstention"]}


def shared_graph(shared=True):
    b = GraphBuilder()
    a = b.add("select", ["input"], {"indices": [0]})
    c = b.add("select", ["input"], {"indices": [1]})
    a = b.add("conv", [a], {"width": 4, "kernel": 3, "dilation": 2}, "group" if shared else "")
    c = b.add("conv", [c], {"width": 4, "kernel": 3, "dilation": 2}, "group" if shared else "")
    a = b.add("tanh", [a]); c = b.add("silu", [c])
    v = b.add("gated_add", [a, c])
    v = b.add("statistics", [v])
    v = b.add("linear", [v], {"width": 3})
    return {"schema": SCHEMA, "nodes": b.nodes, "output": v}


@pytest.mark.parametrize("seed", range(12))
def test_generated_graphs_compile_backward_and_exact_counts(seed):
    cfg = request(); groups = [[0], [1]]; shape = Shape(8, 2, 3)
    genes = random_genome(random.Random(seed), cfg["grammar"])
    graph = lower_genome(genes, groups, shape)
    with deterministic_cpu():
        registry = compositional_registry()
        model = registry.build({"family": "composed_dag", "parameters": graph}, shape, seed, 40000)
        y = model(torch.randn(4, 8, 2))
        assert y.shape == (4, 3)
        torch.nn.functional.cross_entropy(y, torch.tensor([0, 1, 2, 0])).backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        assert sum(p.numel() for p in model.parameters()) == analyze_graph(shape, graph)["parameter_count"]


def test_sharing_is_real_not_copied_parameters():
    s = Shape(8, 2, 3)
    tied = GraphNetwork(s, shared_graph())
    untied = GraphNetwork(s, shared_graph(False))
    assert sum(p.numel() for p in untied.parameters()) - sum(p.numel() for p in tied.parameters()) == 16
    assert len([n for n, _ in tied.named_parameters() if 'shared_s0.conv.weight' in n]) == 1
    a = analyze_graph(s, shared_graph())
    assert a['shared_parameter_groups'] == 1
    assert a['multiply_accumulate_estimate_per_case'] > 0


def test_graph_initialization_restores_rng_and_replays():
    r = compositional_registry(); s = Shape(8,2,3); spec = {"family":"composed_dag", "parameters":shared_graph()}
    with deterministic_cpu():
        before = torch.get_rng_state().clone()
        a = r.build(spec,s,123,10000)
        assert torch.equal(before,torch.get_rng_state())
        b = r.build(spec,s,123,10000)
        assert all(torch.equal(x,y) for x,y in zip(a.parameters(),b.parameters()))


@pytest.mark.parametrize("issue", ["cycle", "dead", "extra", "unknown", "share_shape", "vector_conv", "output", "duplicate_ref", "invalid_selection", "bad_share", "bool_width"])
def test_graph_invalid_structure_fails_closed(issue):
    g=shared_graph()
    if issue == "cycle": g["nodes"][0]["inputs"]=[g["output"]]
    elif issue == "dead": g["nodes"].append({"id":"n99","op":"mean","inputs":["input"],"args":{},"share":""})
    elif issue == "extra": g["nodes"][0]["code"]="print('not executed')"
    elif issue == "unknown": g["nodes"][0]["op"]="exec"
    elif issue == "share_shape": g["nodes"][1]["args"]["indices"]=[0,1]
    elif issue == "vector_conv": g["nodes"][-1].update(op="conv",args={"width":3,"kernel":3,"dilation":1})
    elif issue == "output": g["output"]="input"
    elif issue == "duplicate_ref": g["nodes"][6]["inputs"]=["n4","n4"]
    elif issue == "invalid_selection": g["nodes"][0]["args"]["indices"]=[999]
    elif issue == "bad_share": g["nodes"][0]["share"]="s0"
    else: g["nodes"][-1]["args"]["width"]=True
    with pytest.raises((ValueError,TypeError)):
        analyze_graph(Shape(8,2,3),g)


def test_equal_parameter_count_does_not_mean_same_topology():
    a=shared_graph();b=copy.deepcopy(a);b['nodes'][4]['op']='relu'
    assert analyze_graph(Shape(8,2,3),a)['parameter_count']==analyze_graph(Shape(8,2,3),b)['parameter_count']
    assert graph_identity(a,topology_only=True)!=graph_identity(b,topology_only=True)


def test_causal_convolution_has_no_future_dependence():
    from phm_design.graph import CausalConvolution
    model=CausalConvolution(2,{'width':4,'kernel':3,'dilation':2})
    x=torch.randn(2,12,2);y=x.clone();y[:,7:]+=100
    torch.testing.assert_close(model(x)[:,:7],model(y)[:,:7],rtol=0,atol=0)


def test_input_shape_and_budget_blocked_before_construction():
    r=compositional_registry();s=Shape(8,2,3);spec={'family':'composed_dag','parameters':shared_graph()}
    with pytest.raises(ValueError,match='PARAMETER_BUDGET'):
        r.build(spec,s,1,1)
    m=r.build(spec,s,1,10000)
    with pytest.raises(ValueError,match='INPUT_SHAPE'):
        m(torch.zeros(2,7,2))


@pytest.mark.parametrize('channels',[1,2,4])
def test_generic_groups_and_classes_no_satellite_hardcoding(tmp_path,channels):
    dev,_=fixture_dataset(tmp_path,channels=channels,classes=2)
    contract=Dataset(dev,allowed_splits={'train','validation'}).contract
    cfg=request(channels);p=propose(cfg,contract,0,[],set())
    for q in p['accepted']:
        with deterministic_cpu():
            m=compositional_registry().build(q['spec'],Shape(8,channels,2),3,40000)
            assert m(torch.zeros(1,8,channels)).shape==(1,2)


def test_group_unit_roles_and_global_budget_enforced(tmp_path):
    dev,_=fixture_dataset(tmp_path);contract=read_json(dev)['contract'];cfg=request()
    contract['channels'][1]['unit']='N*m'
    with pytest.raises(ValueError,match='UNIT_ROLES'):
        validate_request(cfg,contract)
    cfg['grammar']['allow_sharing']=False
    validate_request(cfg,contract)
    cfg['evolution']['max_total_trials']=1
    with pytest.raises(ValueError,match='GLOBAL_DESIGN'):
        validate_request(cfg,contract)


@pytest.fixture(scope='module')
def completed_design(tmp_path_factory):
    root=tmp_path_factory.mktemp('composition')
    dev,test=fixture_dataset(root/'data');cfg=root/'request.json';write_json(cfg,request())
    out=root/'design'
    report=design(dev,cfg,out)
    return root,dev,test,cfg,out,report


def test_design_evolution_has_parents_structural_edits_and_separate_control(completed_design):
    _,_,_,_,out,report=completed_design
    assert report['status']=='COMPLETED'
    assert report['generated_graphs']==5
    assert report['best_composed']['kind']=='generated'
    assert report['best_fixed_neural_control']['kind']=='fixed_control'
    lineage=read_json(out/'lineage.json');byid={a['id']:a for a in lineage}
    children=[a for a in lineage if a['generation']==1]
    assert len(children)==2
    for child in children:
        assert child['parents'] and child['edit']!='grammar_synthesis'
        assert child['topology_sha256']!=byid[child['parents'][0]]['topology_sha256']
    assert not report['test_used']
    assert not report['novel_scientific_architecture_claim']


def test_recovery_evaluation_and_same_output_no_labels(completed_design):
    root,_,test,_,out,_=completed_design
    with deterministic_cpu():
        b,m=load_ensemble(out/'selected_composed')
        _,again=load_ensemble(out/'selected_composed')
        x=np.random.default_rng(88).normal(size=(3,8,2)).astype('float32')
        np.testing.assert_array_equal(ensemble_scores(m,x),ensemble_scores(again,x))
    report=evaluate(out/'selected_composed',test,root/'evaluation')
    assert report['neural']['cases']==6
    assert report['test_used_for_selection'] is False


def test_resume_reuses_same_weights_and_rejects_changed_contract(completed_design,tmp_path):
    _,dev,_,cfg,out,report=completed_design
    before={str(p): (p.stat().st_mtime_ns,sha_file(p)) for p in out.rglob('model.pt')}
    assert design(dev,cfg,out,resume=True)==report
    after={str(p): (p.stat().st_mtime_ns,sha_file(p)) for p in out.rglob('model.pt')}
    assert before==after
    new=read_json(cfg);new['evolution']['seed']+=1;p=tmp_path/'changed.json';write_json(p,new)
    with pytest.raises(ValueError,match='DESIGN_RESUME'):
        design(dev,p,out,resume=True)


def test_interrupted_design_resumes_completed_trial(tmp_path):
    dev,_=fixture_dataset(tmp_path/'data');cfg=tmp_path/'request.json';write_json(cfg,request());out=tmp_path/'design'
    def interrupt(record): raise KeyboardInterrupt('after committed trial')
    with pytest.raises(KeyboardInterrupt):
        design(dev,cfg,out,on_trial_complete=interrupt)
    weights=list(out.glob('generation_00/run/trials/*/*/model.pt'));assert len(weights)==1
    before=(weights[0].stat().st_mtime_ns,sha_file(weights[0]))
    assert design(dev,cfg,out,resume=True)['status']=='COMPLETED'
    assert before==(weights[0].stat().st_mtime_ns,sha_file(weights[0]))


def test_holdout_manifest_not_accepted_as_design_input(tmp_path):
    _,test=fixture_dataset(tmp_path/'data');cfg=tmp_path/'request.json';write_json(cfg,request())
    with pytest.raises(ValueError,match='DATA_ROLE'):
        design(test,cfg,tmp_path/'bad')


def test_partial_plan_tamper_rejected(tmp_path):
    dev,_=fixture_dataset(tmp_path/'data');cfg=tmp_path/'r.json';write_json(cfg,request());out=tmp_path/'d'
    def interrupt(record): raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):design(dev,cfg,out,on_trial_complete=interrupt)
    pp=out/'generation_00/proposals.json';p=read_json(pp);p['accepted'][0]['edit']='tampered';write_json(pp,p)
    with pytest.raises(ValueError,match='GENERATION_PLAN_CHANGED'):design(dev,cfg,out,resume=True)


def test_structural_ablations_are_retrained_without_replacing_champion(completed_design):
    from phm_design.ablation import ablate
    root,dev,_,_,out,report=completed_design
    frozen=sha_file(out/'selected_composed/model_bundle.json')
    result=ablate(dev,out,root/'ablations')
    assert not result['test_read'] and not result['champion_changed']
    assert not result['parameter_matched']
    assert sha_file(out/'selected_composed/model_bundle.json')==frozen
    assert all(r['status']=='COMPLETED' for r in result['results'])
