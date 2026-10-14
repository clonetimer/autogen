import copy
from phm_design.optuna_proposal import genome_params, propose_optuna_tpe, suggest_genome, topology_history
from phm_design.synthesis import propose
from tests.phm_design.test_compositional_design import request


def _contract(channels=9, classes=5):
    return {
        "channels": [{"name": f"x{i}", "unit": "1"} for i in range(channels)],
        "classes": [f"c{i}" for i in range(classes)],
    }


def _req():
    r=request(channels=9)
    r['groups']=[{'id':'a','channels':[f'x{i}' for i in range(3)]},
                 {'id':'b','channels':[f'x{i}' for i in range(3,6)]},
                 {'id':'g','channels':[f'x{i}' for i in range(6,9)]}]
    r['grammar']['allow_sharing']=False
    r['grammar']['layer_ops']=['linear','conv','res_tcn','self_attention']
    r['grammar']['attention_heads']=[1,2]
    r['grammar']['tail_pool_windows']=[4,8]
    r['grammar']['allow_passthrough_branches']=True
    r['evolution'].update(initial_graphs=3,generations=2,offspring_per_generation=2,elite_parents=2,max_total_trials=60)
    r['budget']['max_parameters_per_model']=100000
    r['budget']['max_ensemble_parameters']=300000
    r['budget']['max_multiply_accumulate_per_case']=100000000
    return r


def test_generation_zero_matches_native_graphs():
    r=_req();c=_contract();a=propose(r,c,0,[],set());b=propose_optuna_tpe(r,c,0,[],set())
    assert [x['graph_sha256'] for x in a['accepted']]==[x['graph_sha256'] for x in b['accepted']]
    assert b['feedback_role']=='none_optuna_shared_startup'


def test_genome_roundtrip_through_queued_trial():
    import optuna
    r=_req();c=_contract();g=propose(r,c,0,[],set())['accepted'][0]['genes']
    study=optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=1,n_startup_trials=1))
    study.enqueue_trial(genome_params(g,r['grammar']))
    seen={}
    def objective(trial):
        seen['g']=suggest_genome(trial,r['grammar']);return .5
    study.optimize(objective,n_trials=1)
    assert seen['g']==g


def test_topology_history_uses_best_route():
    genes={'branches':[]}
    a={'kind':'generated','status':'COMPLETED','topology_sha256':'t','genes':genes,'mode':'bounded','id':'a','mean_validation_macro_f1':.7}
    b={**a,'mode':'raw','id':'b','mean_validation_macro_f1':.9}
    h=topology_history([a,b])
    assert len(h)==1 and h[0]['objective']==.9 and h[0]['best_route']=='raw'


def test_generation_one_produces_legal_unique_budgeted_graphs():
    r=_req();c=_contract();g0=propose_optuna_tpe(r,c,0,[],set())
    archive=[]
    for i,p in enumerate(g0['accepted']):
        for mode,score in [('raw',.70+i*.05),('bounded',.65+i*.05),('unbounded',.68+i*.05)]:
            archive.append({
                'id':f'{i}-{mode}','kind':'generated','status':'COMPLETED','mode':mode,
                'mean_validation_macro_f1':score,'seed_std_macro_f1':0.01,
                'description':{'parameter_count':p['resources']['parameter_count']},
                'genes':copy.deepcopy(p['genes']),'graph_sha256':p['graph_sha256'],
                'topology_sha256':p['topology_sha256'],
            })
    g1=propose_optuna_tpe(r,c,1,archive,{p['graph_sha256'] for p in g0['accepted']})
    assert len(g1['accepted'])==2 and not g1['holdout_read']
    assert len({p['graph_sha256'] for p in g1['accepted']})==2
    assert all(p['resources']['parameter_count']<=r['budget']['max_parameters_per_model'] for p in g1['accepted'])
    assert g1['external_optimizer']['library']=='optuna'
