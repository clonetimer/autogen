"""Matched-budget mechanism-conditioned graph construction.

Topology candidates are generated without labels, then independently selected
in two arms. Both arms see the SAME nine-channel raw contract. The added branch
receives a normalized physical innovation OR normalized observed deltas. Equal
trial/epoch/parameter/MAC accounting is not an exact wall-time or power match.
Existing v0.3 validation-guided evolutionary `design` remains available.
"""
from __future__ import annotations
import copy
import random
from pathlib import Path
import numpy as np
from .data import Dataset,fit_scaler,reject_numeric_overlap
from .engine import fit,environment,implementation_hash,config_check
from .graph import SCHEMA,analyze_graph,graph_identity,graph_to_dot
from .io import digest,read_json,write_json,run_lock,seal,verify
from .models import Shape,integer
from .physics import analyze_program,fit_feature_scaler,analyze_mechanism
from .registry import default_registry
from .synthesis import GraphBuilder,random_layer


def generate_graph(rng,request,shape,groups,features):
    """Synthesize operator branches using component and equation dependency priors."""
    grammar=request['grammar'];b=GraphBuilder();vectors=[];genes=[]
    sources=['components','mechanisms']
    if rng.choice([True,False]):sources.append('global')
    for number,source in enumerate(sources):
        shared=(source=='components' and rng.choice([True,False]))
        layers=[random_layer(rng,grammar) for _ in range(rng.randint(1,grammar['max_depth']))]
        pool=rng.choice(['mean','last','statistics']);skip=rng.choice([True,False])
        selections=groups if source=='components' else ([features] if source=='mechanisms' else [list(range(shape.channels-len(features)))])
        branch=[]
        for indices in selections:
            cur=b.add('select',['input'],{'indices':indices});start=cur
            for i,layer in enumerate(layers):
                args={k:v for k,v in layer.items() if k not in {'op','activation'}}
                cur=b.add(layer['op'],[cur],args,f'b{number}_layer{i}' if shared else '')
                cur=b.add(layer['activation'],[cur])
            if skip:
                projected=b.add('linear',[start],{'width':layers[-1]['width']},f'b{number}_skip' if shared else '')
                cur=b.add('add',[cur,projected])
            branch.append(b.add(pool,[cur]))
        vectors.append(branch[0] if len(branch)==1 else b.add('concat',branch))
        genes.append({'source':source,'input_indices':selections,'layers':layers,'pool':pool,'skip':skip,'share':shared})
    fusion=rng.choice(['concat','add','gated_add']);width=rng.choice(grammar['widths'])
    if fusion!='concat':vectors=[b.add('linear',[v],{'width':width}) for v in vectors]
    rng.shuffle(vectors);cur=vectors[0]
    for v in vectors[1:]:cur=b.add(fusion,[cur,v])
    if rng.choice([True,False]):
        cur=b.add('linear',[cur],{'width':width});cur=b.add('silu',[cur])
    cur=b.add('linear',[cur],{'width':shape.classes})
    graph={'schema':SCHEMA,'nodes':b.nodes,'output':cur}
    return graph,{'branches':genes,'fusion':fusion,'fusion_width':width}


def build_matched_plan(data:Dataset,program:dict,request:dict)->dict:
    if set(request)!={'schema','window','groups','control_channels','candidate_pairs','proposal_seed','max_proposal_attempts','grammar','seeds','training','budget','max_neural_macs_per_case','selection','abstention'} or request['schema']!='phm.mechanism-design.v1':
        raise ValueError('MECHANISM_DESIGN_REQUEST_SCHEMA')
    audit=analyze_program(program,data.contract)
    names=[c['name'] for c in data.contract['channels']]
    groups=request['groups']
    if not isinstance(groups,list) or not 1<=len(groups)<=8 or any(not isinstance(g,list) or not g or any(n not in names for n in g) for g in groups):
        raise ValueError('MECHANISM_GROUP_CONTRACT')
    indices=[[names.index(n) for n in g] for g in groups]
    if sorted(i for g in indices for i in g)!=list(range(len(names))):raise ValueError('GROUPS_MUST_PARTITION_INPUT_CHANNELS')
    signatures=[tuple(data.contract['channels'][i]['unit'] for i in g) for g in indices]
    if any(s!=signatures[0] for s in signatures):raise ValueError('SHARED_BRANCH_UNIT_ROLES_MUST_MATCH')
    cc=request['control_channels']
    if not isinstance(cc,list) or len(cc)!=audit['output_count'] or any(n not in names for n in cc) or len(set(cc))!=len(cc):
        raise ValueError('CONTROL_CHANNEL_CONTRACT')
    ci=[names.index(n) for n in cc]
    integer(request['candidate_pairs'],'candidate_pairs',1,32)
    integer(request['proposal_seed'],'proposal_seed',0,2**32-1)
    integer(request['max_proposal_attempts'],'max_proposal_attempts',1,10000)
    integer(request['max_neural_macs_per_case'],'max_neural_macs_per_case',1,100000000)
    grammar=request['grammar']
    if set(grammar)!={'widths','max_depth','kernels','dilations'}:raise ValueError('MECHANISM_GRAMMAR_SCHEMA')
    for k,lo,hi in [('widths',2,128),('kernels',1,9),('dilations',1,8)]:
        if not isinstance(grammar[k],list) or not grammar[k] or len(set(grammar[k]))!=len(grammar[k]):raise ValueError('GRAMMAR_CHOICES')
        for v in grammar[k]:integer(v,k,lo,hi)
    if any(k%2==0 for k in grammar['kernels']):raise ValueError('ODD_KERNELS_REQUIRED')
    integer(grammar['max_depth'],'max_depth',1,3)
    # Fixed epochs (including patience) keep gradient steps equal, irrespective
    # of one arm's validation loss. Best checkpoint still uses development data.
    if request['training']['patience']!=request['training']['epochs']:
        raise ValueError('MATCHED_BUDGET_REQUIRES_FIXED_EPOCHS')
    shape=Shape(request['window'],len(names),len(data.contract['classes']))
    train=data.load('train',shape.window);val=data.load('validation',shape.window);reject_numeric_overlap(train,val)
    scaler=fit_scaler(train.x)
    modes=('physical','observed_delta')
    fs={m:fit_feature_scaler(program,data.contract,train.x,m,ci) for m in modes}
    n=audit['output_count'];aug=Shape(shape.window,shape.channels+n,shape.classes)
    rng=random.Random(request['proposal_seed']);seen=set();pairs=[];rejected=[]
    for attempt in range(request['max_proposal_attempts']):
        try:
            graph,genes=generate_graph(rng,request,aug,indices,list(range(shape.channels,shape.channels+n)))
            ga=analyze_graph(aug,graph);identity=graph_identity(graph)
            if identity in seen:raise ValueError('DUPLICATE_GRAPH')
            if ga['parameter_count']>request['budget']['max_parameters_per_model'] or ga['parameter_count']*len(request['seeds'])>request['budget']['max_ensemble_parameters']:
                raise ValueError('PARAMETER_BUDGET_EXCEEDED')
            if ga['multiply_accumulate_estimate_per_case']>request['max_neural_macs_per_case']:
                raise ValueError('NEURAL_MAC_BUDGET_EXCEEDED')
            specs={m:{'family':'mechanism_dag','parameters':{'program':copy.deepcopy(program),'input_contract':copy.deepcopy(data.contract),
                   'input_scaler':scaler,'feature_scaler':fs[m],'feature_mode':m,'control_indices':ci,'graph':graph}} for m in modes}
            pairs.append({'pair_id':identity,'genes':genes,'graph':graph,'specs':specs,'resources':{k:ga[k] for k in ('parameter_count','multiply_accumulate_estimate_per_case','node_count','shared_parameter_groups')},'proposal_attempt':attempt})
            seen.add(identity)
        except ValueError as exc:rejected.append({'attempt':attempt,'reason':str(exc)})
        if len(pairs)==request['candidate_pairs']:break
    if len(pairs)!=request['candidate_pairs']:raise ValueError('MATCHED_PROPOSAL_BUDGET_EXHAUSTED')
    configs={}
    for m in modes:
        cfg={k:copy.deepcopy(request[k]) for k in ('window','seeds','training','budget','selection','abstention')}
        cfg.update(schema='phm.search.v1',candidates=[p['specs'][m] for p in pairs])
        config_check(cfg,shape.channels,shape.classes,default_registry());configs[m]=cfg
    return {'schema':'phm.matched-mechanism-plan.v1','request':request,'dataset_hash':data.hash,'program_audit':audit,
            'pairs':pairs,'rejected':rejected,'configs':configs,'counts':data.counts(),
            'source_role':'train_for_scalers; validation_for_selection; no test path accepted',
            'neural_parameters_and_macs_matched_per_pair':True,'wall_time_or_energy_matched':False,
            'source_physical_laws_invented':False,'architecture_generation':'operator DAG synthesis from component and mechanism dependency groups',
            'proposal_uses_labels':False,'prospective_hardware_qualification':False}


def mechanism_design(dataset_path,program_path,request_path,output,*,resume=False,on_trial_complete=None):
    data=Dataset(dataset_path,allowed_splits={'train','validation'});data.verify_files()
    program=read_json(program_path);request=read_json(request_path)
    plan=build_matched_plan(data,program,request)
    fingerprint=digest({'plan':plan,'code':implementation_hash(),'environment':environment()})
    output=Path(output)
    with run_lock(output):
        if (output/'plan.json').exists():
            if read_json(output/'plan.json')['fingerprint']!=fingerprint:raise ValueError('MECHANISM_RESUME_CHANGED')
            if not resume:raise FileExistsError('RUN_EXISTS_USE_RESUME')
            if (output/'manifest.json').exists():verify(output);return read_json(output/'report.json')
        else:write_json(output/'plan.json',{**plan,'fingerprint':fingerprint,'environment':environment()})
        reports={}
        for mode,cfg in plan['configs'].items():
            write_json(output/mode/'search.json',cfg)
            reports[mode]=fit(dataset_path,output/mode/'search.json',output/mode/'run',resume=resume,on_trial_complete=on_trial_complete)
        table=[]
        for pair in plan['pairs']:
            row={'pair_id':pair['pair_id'],'resources':pair['resources']}
            for mode,report in reports.items():
                candidate=next(c for c in report['candidate_results'] if digest(c['spec'])==digest(pair['specs'][mode]))
                row[mode]={'id':candidate['id'],'status':candidate['status'],'mean_validation_macro_f1':candidate.get('mean_validation_macro_f1'),
                           'epochs_executed':[t.get('epochs_executed') for t in candidate['trials']]}
            table.append(row)
        complete=all(row[m]['status']=='COMPLETED' for row in table for m in reports)
        matched=complete and all(row['physical']['epochs_executed']==row['observed_delta']['epochs_executed'] for row in table)
        report={'schema':'phm.mechanism-design-report.v1','status':'COMPLETED' if complete else 'INCOMPLETE_TRIALS',
                'fingerprint':fingerprint,'candidate_pairs':len(table),'training_trials':sum(len(x['candidate_results'])*len(request['seeds']) for x in reports.values()),
                'paired_validation_results':table,'same_completed_epoch_budget':matched,'test_used':False,
                'best_ids':{m:r['best_neural_candidate'] for m,r in reports.items()},'claim_status':'RESEARCH_ONLY',
                'physical_equations_discovered':False,'real_device_qualified':False,'production_release_qualified':False,
                'confounds':['one generated candidate pool; no independent repeated architecture-search campaigns',
                             'neural MACs and parameters matched per pair, not exact CPU time, memory or energy',
                             'normal model innovation includes sensor and integration error',
                             'chosen model complexity may differ between arms; paired rows control topology exactly']}
        for mode,r in reports.items():
            winner=next((c for c in r['candidate_results'] if c['id']==r['best_neural_candidate']),None)
            if winner:
                write_json(output/f'{mode}_architecture.json',winner['spec'])
                (output/f'{mode}_architecture.dot').write_text(graph_to_dot(winner['spec']['parameters']['graph']))
        write_json(output/'report.json',report)
        seal(output,[str(p.relative_to(output)) for p in output.rglob('*') if p.is_file() and p.name not in {'.lock','manifest.json','evaluation_ledger.json'}])
        return report
