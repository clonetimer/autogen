from phm_design.models import Shape
from phm_design.resource_planner import (
    fixed_control_mac_per_case, plan_fixed_control, plan_fixed_controls,
)

BUDGET={
    'max_parameters_per_model':50000,
    'max_ensemble_parameters':150000,
    'max_trials':40,
    'max_trial_seconds':90,
    'max_multiply_accumulate_per_case':60000000,
}


def test_grid3d_contract_aware_width_19_vs_25_channels():
    preferred={'family':'grid3d_res','parameters':{'width':12}}
    p19=plan_fixed_control(Shape(128,19,5),preferred,BUDGET,3)
    p25=plan_fixed_control(Shape(128,25,5),preferred,BUDGET,3)
    assert p19['effective_spec']['parameters']['width']==11
    assert p25['effective_spec']['parameters']['width']==9
    assert p19['effective_resources']['multiply_accumulate_estimate_per_case']<=60000000
    assert p25['effective_resources']['multiply_accumulate_estimate_per_case']<=60000000
    assert fixed_control_mac_per_case(Shape(128,25,5),preferred)>60000000


def test_multiscale_shrinks_only_when_channel_count_pushes_parameter_budget():
    preferred={'family':'multiscale_conv_transformer','parameters':{'d_model':48}}
    p19=plan_fixed_control(Shape(128,19,5),preferred,BUDGET,3)
    p25=plan_fixed_control(Shape(128,25,5),preferred,BUDGET,3)
    assert p19['effective_spec']==preferred
    assert p19['adapted'] is False
    assert p25['effective_spec']['parameters']['d_model']==44
    assert p25['effective_resources']['description']['parameter_count']<=50000
    assert p25['adapted'] is True
    assert p25['preferred_spec']==preferred


def test_transformer_preferred_spec_is_unchanged_when_feasible():
    preferred={'family':'transformer_ts','parameters':{'d_model':40,'heads':4,'layers':2}}
    plan=plan_fixed_control(Shape(128,25,5),preferred,BUDGET,3)
    assert plan['effective_spec']==preferred
    assert plan['attempt_count']==1
    assert plan['effective_resources']['feasible'] is True


def test_planner_returns_one_effective_control_per_policy_control():
    controls=[
        {'family':'mlp','parameters':{'widths':[32],'representation':'statistics'}},
        {'family':'grid3d_res','parameters':{'width':12}},
        {'family':'transformer_ts','parameters':{'d_model':40,'heads':4,'layers':2}},
        {'family':'multiscale_conv_transformer','parameters':{'d_model':48}},
    ]
    plans=plan_fixed_controls(Shape(128,25,5),controls,BUDGET,3)
    assert len(plans)==len(controls)
    assert [p['effective_spec']['family'] for p in plans]==[x['family'] for x in controls]
    assert all(p['effective_resources']['feasible'] for p in plans)


def test_grid3d_mac_is_monotone_in_width_and_channel_grid():
    s19=Shape(128,19,5);s25=Shape(128,25,5)
    a=fixed_control_mac_per_case(s19,{'family':'grid3d_res','parameters':{'width':9}})
    b=fixed_control_mac_per_case(s19,{'family':'grid3d_res','parameters':{'width':10}})
    c=fixed_control_mac_per_case(s25,{'family':'grid3d_res','parameters':{'width':10}})
    assert a<b<c


def _mutation_fixture():
    import json
    parent={"branches":[
        {"difference":False,"layers":[{"activation":"silu","heads":1,"op":"self_attention","width":8}],"pool":"mean","share":False,"skip":False,"source":"global"},
        {"difference":False,"layers":[{"activation":"relu","op":"linear","width":24},{"activation":"silu","dilation":1,"kernel":5,"op":"res_tcn","width":24}],"pool":"mean","share":False,"skip":True,"source":"global"},
        {"difference":False,"layers":[{"activation":"silu","dilation":2,"kernel":3,"op":"conv","width":8},{"activation":"tanh","heads":2,"op":"self_attention","width":24}],"pool":"last","share":False,"skip":False,"source":"global"}],
        "fusion":"add","fusion_order":[1,2,0],"fusion_width":16,"head_widths":[8,16]}
    donor={"branches":[
        {"difference":False,"layers":[{"activation":"tanh","dilation":2,"kernel":3,"op":"res_tcn","width":8}],"pool":"last","share":False,"skip":False,"source":"components"},
        {"difference":False,"layers":[{"activation":"relu","op":"linear","width":16}],"pool":"statistics","share":False,"skip":True,"source":"global"}],
        "fusion":"gated_add","fusion_order":[0,1],"fusion_width":16,"head_widths":[8]}
    grammar={"widths":[8,16,24],"max_depth":2,"max_branches":3,"kernels":[3,5],"dilations":[1,2],"allow_sharing":False,
             "layer_ops":["linear","conv","res_tcn","self_attention"],"attention_heads":[1,2,4]}
    return parent,donor,grammar


def test_saturated_add_branch_is_not_mislabeled_noop_or_hidden_reorder():
    import random
    from phm_design.synthesis import mutate_genome, lower_genome
    from phm_design.graph import graph_identity
    parent,donor,grammar=_mutation_fixture()
    child,op=mutate_genome(parent,donor,random.Random(17),grammar,0)
    assert op!="add_branch"
    assert child!=parent
    assert child["fusion_order"]==parent["fusion_order"]
    groups=[[0,1,2],[3,4],[5,6]]
    shape=Shape(128,7,5)
    assert graph_identity(lower_genome(child,groups,shape))!=graph_identity(lower_genome(parent,groups,shape))


def test_effective_mutations_never_return_parent_genome():
    import random
    from phm_design.synthesis import mutate_genome
    parent,donor,grammar=_mutation_fixture()
    for attempt in range(12):
        child,op=mutate_genome(parent,donor,random.Random(100+attempt),grammar,attempt)
        assert child!=parent, op
        if op!="reorder_fusion":
            # Structural mutations no longer smuggle in an unrelated reorder.
            assert child["fusion_order"]==parent["fusion_order"] or len(child["branches"])!=len(parent["branches"])
