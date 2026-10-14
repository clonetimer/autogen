import copy
from phm_design.synthesis import propose, propose_random_control
from phm_design.optuna_proposal import propose_optuna_tpe
from tests.phm_design.test_compositional_design import request


def _contract(channels=9, classes=5):
    return {"channels":[{"name":f"x{i}","unit":"1"} for i in range(channels)],"classes":[f"c{i}" for i in range(classes)]}


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
    r['evolution'].update(initial_graphs=3,generations=5,offspring_per_generation=1,elite_parents=2,
                          max_proposal_attempts=500,max_total_trials=100)
    r['budget']['max_parameters_per_model']=100000
    r['budget']['max_ensemble_parameters']=300000
    r['budget']['max_multiply_accumulate_per_case']=100000000
    return r


def _rec(p, score, generation, idx):
    return {
        'id':f'g{generation}-{idx}','kind':'generated','status':'COMPLETED','mode':'raw',
        'mean_validation_macro_f1':score,'seed_std_macro_f1':0.01,
        'description':{'parameter_count':p['resources']['parameter_count']},
        'genes':copy.deepcopy(p['genes']),'graph_sha256':p['graph_sha256'],
        'topology_sha256':p['topology_sha256'],'spec':copy.deepcopy(p['spec']),
    }


def _exercise(proposer, *, optuna=False):
    r=_req(); c=_contract(); seen=set(); archive=[]; parent_archive=[]; topologies=[]
    for generation in range(r['evolution']['generations']):
        history=archive if optuna else parent_archive
        out=proposer(r,c,generation,history,seen)
        expected=3 if generation==0 else 1
        assert len(out['accepted'])==expected
        for idx,p in enumerate(out['accepted']):
            assert p['graph_sha256'] not in seen
            seen.add(p['graph_sha256']); topologies.append(p['topology_sha256'])
            rec=_rec(p,.70+.03*generation+.001*idx,generation,idx)
            archive.append(rec); parent_archive.append(rec)
    assert len(topologies)==7
    assert len(set(topologies))==7
    return topologies


def test_native_feedback_five_generations_unique():
    assert len(_exercise(propose))==7


def test_random_control_five_generations_unique():
    assert len(_exercise(propose_random_control))==7


def test_optuna_tpe_five_generations_unique():
    assert len(_exercise(propose_optuna_tpe,optuna=True))==7
