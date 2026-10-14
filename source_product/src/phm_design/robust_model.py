"""Composition of generated typed subgraphs with quality-carrying branches.

Old graph/physics implementations remain unchanged for 0.3/0.4 bundle recovery.
Only assess(Observation) is used by the robust execution and deployment path.
A zero logits row when all branches are unavailable is NEVER a normal decision.
"""
from __future__ import annotations
import copy
import random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .data import fit_scaler
from .graph import GraphNetwork,analyze_graph,graph_identity
from .models import Shape,integer
from .synthesis import GraphBuilder,random_layer
from .io import digest,sha_file
from .robust_ops import Observation,MomentumBank,QualityFusion,validate_mechanism,number


def code_binding():
    base=Path(__file__).parent
    return {n:sha_file(base/n) for n in ('robust_ops.py','robust_model.py','graph.py')}


def subgraph(rng,channels,width,grammar):
    b=GraphBuilder();cur='input';layers=[]
    for _ in range(rng.randint(1,grammar['max_depth'])):
        layer=random_layer(rng,grammar);layers.append(layer)
        args={k:v for k,v in layer.items() if k not in {'op','activation'}}
        cur=b.add(layer['op'],[cur],args);cur=b.add(layer['activation'],[cur])
    skip=rng.choice([False,True])
    if skip:
        projected=b.add('linear',['input'],{'width':layers[-1]['width']});cur=b.add('add',[cur,projected])
    pools=grammar.get('pools',['mean','last','statistics'])
    pool=rng.choice(pools);pool_gene=pool
    if pool=='tail_mean':
        tail_window=rng.choice(grammar['tail_pool_windows'])
        cur=b.add('tail_mean',[cur],{'window':tail_window});pool_gene=f'tail_mean_{tail_window}'
    else:
        cur=b.add(pool,[cur])
    cur=b.add('linear',[cur],{'width':width})
    return {'schema':'phm.typed-dag.v1','nodes':b.nodes,'output':cur},{'layers':layers,'skip':skip,'pool':pool_gene,'embedding':width}


def generate_spec(rng,contract,mechanism,scalers,window,grammar,quality):
    width=rng.choice(grammar['widths']);raw,rg=subgraph(rng,9,width,grammar)
    phys,pg=subgraph(rng,5,width,grammar)
    use_global=rng.choice([False,True]);glob,gg=subgraph(rng,len(contract['channels'])*3,width,grammar)
    return {'schema':'phm.robust-composition.v1','contract':copy.deepcopy(contract),'mechanism':copy.deepcopy(mechanism),
        'scalers':copy.deepcopy(scalers),'window':window,'embedding':width,'share_components':rng.choice([True,False]),
        'fusion':rng.choice(['quality_gate','masked_mean','masked_concat']),
        'graphs':{'raw':raw,'physical':phys,**({'global':glob} if use_global else {})},
        'quality_policy':copy.deepcopy(quality)}, {'operation':'synthesize_subgraphs','raw':rg,'physical':pg,'global':gg if use_global else None}


def mutate_spec(parent,rng,grammar):
    child=copy.deepcopy(parent);ops=['change_fusion','toggle_sharing','replace_raw','replace_physical','toggle_global'];op=rng.choice(ops)
    width=child['embedding'];genes={'operation':op}
    if op=='change_fusion':child['fusion']=rng.choice([x for x in ('quality_gate','masked_mean','masked_concat') if x!=child['fusion']])
    elif op=='toggle_sharing':child['share_components']=not child['share_components']
    elif op in ('replace_raw','replace_physical'):
        name=op.split('_')[1];graph,gene=subgraph(rng,9 if name=='raw' else 5,width,grammar);child['graphs'][name]=graph;genes['replacement']=gene
    else:
        if 'global' in child['graphs']:del child['graphs']['global']
        else:child['graphs']['global'],genes['replacement']=subgraph(rng,len(child['contract']['channels'])*3,width,grammar)
    return child,genes


def analyze_robust(spec):
    required={'schema','contract','mechanism','scalers','window','embedding','share_components','fusion','graphs','quality_policy'}
    if set(spec)!=required or spec['schema']!='phm.robust-composition.v1':raise ValueError('ROBUST_SPEC_SCHEMA')
    validate_mechanism(spec['contract'],spec['mechanism'])
    c=len(spec['contract']['channels']);w=len(spec['mechanism']['groups']);t=spec['window'];d=spec['embedding'];k=len(spec['contract']['classes'])
    integer(t,'window',2,4096);integer(d,'embedding',2,128)
    if type(spec['share_components']) is not bool:raise ValueError('SHARING_BOOLEAN')
    if spec['fusion'] not in {'quality_gate','masked_mean','masked_concat'}:raise ValueError('UNKNOWN_QUALITY_FUSION')
    if set(spec['graphs']) not in ({'raw','physical'},{'raw','physical','global'}):raise ValueError('REQUIRED_RAW_AND_PHYSICAL_BRANCHES')
    q=spec['quality_policy']
    if set(q)!={'min_raw_coverage','min_physical_coverage','max_branch_final_age_s'}:raise ValueError('QUALITY_POLICY_SCHEMA')
    number(q['min_raw_coverage'],'raw_coverage',1e-9,1);number(q['min_physical_coverage'],'physical_coverage',1e-9,1);number(q['max_branch_final_age_s'],'final_age',0)
    if set(spec['scalers'])!={'raw','physical'}:raise ValueError('ROBUST_SCALER_KEYS')
    for name,size in [('raw',c),('physical',w*3)]:
        s=spec['scalers'][name]
        if set(s)!={'mean','scale','fit_role'} or s['fit_role']!='train_only' or len(s['mean'])!=size or len(s['scale'])!=size:raise ValueError('TRAIN_ONLY_SCALER_REQUIRED')
        for v in s['mean']:number(v,'scaler_mean')
        for v in s['scale']:number(v,'scaler_scale',1e-12)
    params=macs=nodes=0;audits={}
    for name,g in spec['graphs'].items():
        channels={'raw':9,'physical':5,'global':c*3}[name];a=analyze_graph(Shape(t,channels,d),g);audits[name]=a
        repeats=w if name!='global' else 1
        params+=a['parameter_count']*(1 if spec['share_components'] or name=='global' else repeats)
        macs+=a['multiply_accumulate_estimate_per_case']*repeats;nodes+=a['node_count']*repeats
    branches=2*w+('global' in spec['graphs'])
    params+=(3 if spec['fusion']=='quality_gate' else 0)+((branches*d if spec['fusion']=='masked_concat' else d)*k+k)
    macs+=(branches*2 if spec['fusion']=='quality_gate' else 0)+(branches*d if spec['fusion']=='masked_concat' else d)*k
    return {'parameter_count':params,'neural_mac_estimate':macs,'executed_subgraph_nodes':nodes,'branches':branches,
        'component_weight_sharing':spec['share_components'],'subgraphs':{n:{'node_count':a['node_count'],'parameter_count':a['parameter_count'],'topology':graph_identity(spec['graphs'][n],topology_only=True)} for n,a in audits.items()},
        'source_sha256':code_binding(),'physical_uncertainty_calibrated':False,'physical_zero_parameter_hypotheses':3*w}


class RobustGraph(nn.Module):
    def __init__(self,spec):
        super().__init__();audit=analyze_robust(spec);self.spec=copy.deepcopy(spec);self.bank=MomentumBank(spec['contract'],spec['mechanism'])
        self.groups=spec['mechanism']['groups'];self.w=len(self.groups);self.c=len(spec['contract']['channels']);self.t=spec['window'];d=spec['embedding'];self.k=len(spec['contract']['classes'])
        for name in ('raw','physical'):
            for field in ('mean','scale'):
                self.register_buffer(f'{name}_{field}',torch.tensor(spec['scalers'][name][field],dtype=torch.float64))
        self.encoders=nn.ModuleDict()
        self.branch_keys=[]
        for mode in ('raw','physical'):
            for i in range(self.w):
                key=mode if spec['share_components'] else f'{mode}_{i}'
                if key not in self.encoders:self.encoders[key]=GraphNetwork(Shape(self.t,9 if mode=='raw' else 5,d),spec['graphs'][mode])
                self.branch_keys.append(key)
        if 'global' in spec['graphs']:
            self.encoders['global']=GraphNetwork(Shape(self.t,self.c*3,d),spec['graphs']['global']);self.branch_keys.append('global')
        self.gate=QualityFusion() if spec['fusion']=='quality_gate' else None
        self.head=nn.Linear(d*len(self.branch_keys) if spec['fusion']=='masked_concat' else d,self.k)
        if sum(p.numel() for p in self.parameters())!=audit['parameter_count']:raise ValueError('ROBUST_PARAMETER_ESTIMATE_MISMATCH')

    def assess(self,obs:Observation,*,inertia_multiplier=1.0):
        if obs.values.shape[1:]!=(self.t,self.c):raise ValueError('ROBUST_INPUT_WINDOW_OR_CHANNELS')
        clean,good,age=obs.quality(self.spec['mechanism']['max_age_s'])
        intervals=obs.time_s[:,1:].double()-obs.time_s[:,:-1].double()
        expected=torch.full_like(intervals,float(self.spec['contract']['sample_s']))
        if not torch.allclose(intervals,expected,rtol=1e-6,atol=1e-7):
            raise ValueError('PROCESSING_GRID_CHANGED_REQUIRES_REQUALIFICATION')
        normalized=(clean.double()-self.raw_mean.double())/self.raw_scale.double()
        normalized=torch.where(good,normalized,torch.zeros_like(normalized)).float()
        age_den=max(self.spec['mechanism']['max_age_s'],1.0)
        bank=self.bank(obs,inertia_multiplier=inertia_multiplier);valid=bank['computable'];res=bank['residual']
        res=(res-self.physical_mean.double().reshape(self.w,3))/self.physical_scale.double().reshape(self.w,3)
        res=torch.where(valid.unsqueeze(-1),res,torch.zeros_like(res)).float()
        embeddings=[];available=[];qualities=[];policy=self.spec['quality_policy']
        def add(key,sequence,mask,tick_age,physical):
            coverage=mask.float().mean(1)
            # Final age measures time since latest usable observation, NOT a
            # mean over old data which could hide a long outage at window end.
            effective=obs.time_s.double()-tick_age.double()
            last=effective.masked_fill(~mask,float('-inf')).amax(1)
            final_age=obs.time_s[:,-1].double()-last
            finite=torch.isfinite(final_age)
            safe_age=torch.where(finite,final_age,torch.full_like(final_age,1e6)).float()
            ok=(coverage>=(policy['min_physical_coverage'] if physical else policy['min_raw_coverage'])) & finite & (final_age<=policy['max_branch_final_age_s'])
            embeddings.append(self.encoders[key](sequence))
            available.append(ok);qualities.append(torch.stack([coverage,safe_age/age_den],-1))
        for i,g in enumerate(self.groups):
            seq=torch.cat([normalized[...,g],good[...,g].float(),(age[...,g]/age_den).float()],-1)
            # Raw branch can ignore a missing gyro if the wheel/command remain;
            # explicit channel masks accompany zero-filled numerical values.
            tick=good[...,g].float().mean(-1)>=2/3
            add(self.branch_keys[i],seq,tick,age[...,g].amax(-1),False)
        for i in range(self.w):
            seq=torch.cat([res[:,:,i],valid[:,:,i:i+1].float(),(bank['age_s'][:,:,i:i+1]/age_den).float()],-1)
            add(self.branch_keys[self.w+i],seq,valid[:,:,i],bank['age_s'][:,:,i],True)
        if 'global' in self.encoders:
            seq=torch.cat([normalized,good.float(),(age/age_den).float()],-1)
            add('global',seq,good.float().mean(-1)>=2/3,age.amax(-1),False)
        e=torch.stack(embeddings,1);a=torch.stack(available,1);q=torch.stack(qualities,1)
        if self.gate is not None:embedding,computable,weights=self.gate(e,a,q)
        else:
            computable=a.any(1);weights=a.float()/a.sum(1,keepdim=True).clamp(min=1)
            masked=torch.where(a.unsqueeze(-1),e,torch.zeros_like(e))
            embedding=masked.flatten(1) if self.spec['fusion']=='masked_concat' else (masked*weights.unsqueeze(-1)).sum(1)
        logits=self.head(embedding)
        logits=torch.where(computable.unsqueeze(-1),logits,torch.zeros_like(logits))
        if not torch.isfinite(logits).all():raise ValueError('ROBUST_NONFINITE_LOGITS')
        return {'logits':logits,'computable':computable,'branch_available':a,'branch_weights':weights,'branch_quality':q}


def build_model(spec,seed,max_parameters):
    audit=analyze_robust(spec)
    if audit['parameter_count']>max_parameters:raise ValueError('PARAMETER_BUDGET_EXCEEDED')
    integer(seed,'seed',0,2**32-1)
    with torch.random.fork_rng(devices=[]),torch.device('cpu'):
        torch.set_rng_state(torch.Generator(device='cpu').manual_seed(seed).get_state())
        model=RobustGraph(spec)
    return model


def fit_robust_scalers(raw,packets,contract,mechanism):
    raw_scaler=fit_scaler(raw);bank=MomentumBank(contract,mechanism)
    features=[];masks=[]
    with torch.inference_mode():
        for obs in packets:
            r=bank(obs);features.append(r['residual'].numpy());masks.append(r['computable'].numpy())
    x=np.concatenate(features);valid=np.concatenate(masks);means=[];scales=[]
    for i in range(len(mechanism['groups'])):
        a=x[:,:,i,:][valid[:,:,i]]
        if len(a)<2:raise ValueError('INSUFFICIENT_TRAIN_MECHANISM_OBSERVATIONS')
        means.extend(a.mean(0).tolist());scales.extend(np.maximum(a.std(0),1e-8).tolist())
    return {'raw':raw_scaler,'physical':{'mean':means,'scale':scales,'fit_role':'train_only'}}
