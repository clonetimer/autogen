from phm_design.models import Shape
from phm_design.resource_planner import (_project_runtime_seconds, plan_fixed_control_runtime,
                                         plan_fixed_controls_runtime)

BUDGET={'max_parameters_per_model':50000,'max_ensemble_parameters':150000,'max_trials':40,
        'max_trial_seconds':90,'max_multiply_accumulate_per_case':60000000}
TRAIN={'epochs':60,'batch_size':24,'learning_rate':0.003,'weight_decay':0.001,'patience':60}
SHAPE=Shape(128,25,5)

def fake_measure(shape,spec,training,budget,validation_cases):
    fam=spec['family']; p=spec['parameters']
    if fam=='grid3d_res':
        # width 9 is predicted over the 85% safety envelope; width 8 is safe.
        t={9:0.265,8:0.155}.get(p['width'],0.12)
        v={9:0.21,8:0.125}.get(p['width'],0.10)
    else:
        t,v=0.01,0.01
    return {'schema':'fake','seed':0,'warmup_steps':0,'measured_steps':1,
            'train_step_seconds':[t],'validation_seconds':[v],
            'conservative_train_step_seconds':t,'conservative_validation_seconds':v}

def test_projection_matches_trial_shape():
    assert _project_runtime_seconds(0.2,0.1,TRAIN,120)==66.0

def test_runtime_planner_shrinks_grid3d_after_static_feasibility():
    preferred={'family':'grid3d_res','parameters':{'width':12}}
    p=plan_fixed_control_runtime(SHAPE,preferred,BUDGET,3,TRAIN,120,80,measure_fn=fake_measure)
    assert p['effective_spec']=={'family':'grid3d_res','parameters':{'width':8}}
    assert p['schema']=='phm.fixed-control-resource-plan.v2'
    assert any(a.get('runtime_preflight',{}).get('runtime_feasible') is False for a in p['attempts'] if a.get('runtime_preflight'))

def test_runtime_planner_preserves_fast_control():
    preferred={'family':'transformer_ts','parameters':{'d_model':40,'heads':4,'layers':2}}
    p=plan_fixed_control_runtime(SHAPE,preferred,BUDGET,3,TRAIN,120,80,measure_fn=fake_measure)
    assert p['effective_spec']==preferred and not p['adapted']

def test_runtime_plans_remain_distinct():
    controls=[{'family':'mlp','parameters':{'representation':'statistics','widths':[32]}},
              {'family':'grid3d_res','parameters':{'width':12}},
              {'family':'transformer_ts','parameters':{'d_model':40,'heads':4,'layers':2}},
              {'family':'multiscale_conv_transformer','parameters':{'d_model':48}}]
    plans=plan_fixed_controls_runtime(SHAPE,controls,BUDGET,3,TRAIN,120,80,measure_fn=fake_measure)
    assert len(plans)==4
    assert plans[1]['effective_spec']['parameters']['width']==8
