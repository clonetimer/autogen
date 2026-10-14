import copy, random
from phm_design.robust_model import subgraph, mutate_spec
from phm_design.robust_engine import validate_request


def grammar():
    return {'widths':[8,16],'max_depth':2,'kernels':[3,5],'dilations':[1,2],
            'pools':['mean','last','statistics','tail_mean'],'tail_pool_windows':[8,16,24,32,48]}


def contract():
    units=['rad/s','rad/s','rad/s','N*m','N*m','N*m','rad/s','rad/s','rad/s']
    return {'device_family':'test.adcs','task':'fixed_length_fault_mechanism_and_component_localization',
            'channels':[{'name':f'c{i}','unit':units[i],'role':'deployment_input','provenance':'test'} for i in range(9)],
            'classes':['N','F'],'sample_s':0.1,'evidence_type':'simulation','scope':'test'}


def request():
    # Minimal complete request matching robust_engine.validate_request; profiles are simple valid no-op stresses.
    p={'id':'clean','noise_std':[0.0]*9,'bias':[0.0]*9,'delay_steps':[0]*9,'drop_probability':[0.0]*9,
       'gap_start':0,'gap_length':0,'drop_channels':[],'diagnostic_inertia_multiplier':1.0,'role':'selection'}
    return {'schema':'phm.robust-design.v1','window':48,'proposal_seed':1,'initial_candidates':1,'mutations':0,'max_proposals':10,
      'grammar':grammar(),'seeds':[1,2],
      'training':{'epochs':1,'batch_size':2,'learning_rate':0.001,'weight_decay':0.0,'checkpoint_every':1},
      'budget':{'max_parameters':100000,'max_ensemble_parameters':200000,'max_trials':2,'max_trial_seconds':10,'max_neural_macs':2000000},
      'quality_policy':{'min_raw_coverage':0.5,'min_physical_coverage':0.4,'max_branch_final_age_s':1.0},
      'mechanism':{'schema':'phm.robust-momentum.v1','groups':[[0,3,6],[1,4,7],[2,5,8]],'inertia_bounds_kg_m2':[[0.05,0.06]]*3,
                   'nominal_limits_nm':[0.2]*3,'source':{'reference':'x','sha256':'0'*64},'assumptions':['x'],'max_age_s':1.0,
                   'sync_tolerance_s':0.02,'max_interval_s':0.2,'inertia_scope':'hypothetical_diagnostic_parameter_sensitivity_not_hardware_tolerance'},
      'train_profiles':[copy.deepcopy(p)],'validation_profiles':[copy.deepcopy(p)],'evaluation_profiles':[copy.deepcopy(p)],
      'train_perturbation_seed':11,'validation_perturbation_seed':12,'evaluation_perturbation_seed':13,'threshold':0.6,
      'evidence_scope':'hypothetical_measurement_stress_not_hardware_specification'}


def test_horizon_aware_subgraph_can_generate_all_tail_windows():
    seen=set()
    for seed in range(400):
        _, gene=subgraph(random.Random(seed),9,8,grammar())
        if gene['pool'].startswith('tail_mean_'):
            seen.add(int(gene['pool'].split('_')[-1]))
    assert seen == {8,16,24,32,48}


def test_tail_gene_lowers_to_typed_tail_mean_operator():
    for seed in range(100):
        graph,gene=subgraph(random.Random(seed),9,8,grammar())
        if gene['pool'].startswith('tail_mean_'):
            w=int(gene['pool'].split('_')[-1])
            nodes=[n for n in graph['nodes'] if n['op']=='tail_mean']
            assert len(nodes)==1 and nodes[0]['args']=={'window':w}
            return
    raise AssertionError('no tail_mean generated')


def test_robust_request_accepts_horizon_grammar():
    validate_request(request(), contract())


def test_robust_request_rejects_tail_window_larger_than_model_window():
    r=request(); r['grammar']['tail_pool_windows']=[8,49]
    try: validate_request(r, contract())
    except ValueError as e: assert str(e) in {'tail_pool_window','tail_pool_window_MAXIMUM'} or 'tail_pool_window' in str(e)
    else: raise AssertionError('expected invalid tail window')


def test_legacy_robust_grammar_remains_valid():
    r=request(); r['grammar']={k:v for k,v in r['grammar'].items() if k not in {'pools','tail_pool_windows'}}
    validate_request(r, contract())
