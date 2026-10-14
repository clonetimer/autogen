"""Dimension-checked, non-executable physical signal programs.

Only declared deployment channels and sourced design constants are admitted.
SI units are checked before evaluation; radians are dimensionless. A dimension
check is NOT model validation, observability, causal identification or approval.
"""
from __future__ import annotations
import copy
import math
import re
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .data import validate_contract, fit_scaler, transform
from .graph import analyze_graph, GraphNetwork
from .io import digest, sha_file
from .models import Shape

SCHEMA = 'phm.physical-program.v1'
# kg, m, s. Deliberately closed unit vocabulary: no guessing or silent conversion.
UNITS = {'1':(0,0,0), 'rad':(0,0,0), 's':(0,0,1), 'm':(0,1,0),
         'kg':(1,0,0), 'rad/s':(0,0,-1), 'rad/s^2':(0,0,-2),
         'N*m':(1,2,-2), 'N*m*s':(1,2,-1), 'kg*m^2':(1,2,0),
         'kg*m^2/s':(1,2,-1), 'N*m*s/rad':(1,2,-1),
         'W':(1,2,-3), 'J':(1,2,-2), 'K':(0,0,0)}
# Temperature is intentionally NOT supported by the mechanical compiler.
UNITS.pop('K')
UNARY = {'delta', 'rate', 'trapezoid_integral', 'sign'}
BINARY = {'add','sub','mul','symmetric_clip'}


def _keys(value, keys, reason):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(reason)


def _number(value, name):
    if type(value) not in (int,float) or not math.isfinite(value):
        raise ValueError(f'NONFINITE_OR_INVALID_NUMBER: {name}')
    return float(value)


def _source(value):
    _keys(value, ('reference','sha256'), 'SOURCE_SCHEMA')
    if not isinstance(value['reference'],str) or not value['reference'].strip() or not isinstance(value['sha256'],str) or not re.fullmatch('[a-f0-9]{64}',value['sha256']):
        raise ValueError('SOURCE_REFERENCE_AND_HASH_REQUIRED')


def unit_dimension(unit):
    if not isinstance(unit,str) or unit not in UNITS:
        raise ValueError(f'UNSUPPORTED_EXPLICIT_UNIT: {unit}')
    return UNITS[unit]


def analyze_program(program: dict, contract: dict) -> dict:
    validate_contract(contract)
    _keys(program, ('schema','inputs','sample_s','constants','nodes','outputs','assumptions','source','evidence_scope'), 'PHYSICAL_PROGRAM_SCHEMA')
    if program['schema'] != SCHEMA or program['evidence_scope'] not in {'simulation_research','analytic_fixture'}:
        raise ValueError('PHYSICAL_PROGRAM_SCOPE')
    if program['inputs'] != contract['channels'] or program['sample_s'] != contract['sample_s']:
        raise ValueError('PHYSICAL_INPUT_CONTRACT_MISMATCH')
    dt = _number(program['sample_s'],'sample_s')
    if dt <= 0: raise ValueError('POSITIVE_SAMPLE_PERIOD_REQUIRED')
    if not isinstance(program['assumptions'],list) or not program['assumptions'] or any(not isinstance(a,str) or not a.strip() for a in program['assumptions']):
        raise ValueError('EXPLICIT_ASSUMPTIONS_REQUIRED')
    _source(program['source'])
    dims = {f'x{i}':unit_dimension(c['unit']) for i,c in enumerate(program['inputs'])}
    dependencies = {name:[] for name in dims}
    constants = {}
    if not isinstance(program['constants'],list) or not 1 <= len(program['constants']) <= 64:
        raise ValueError('CONSTANT_BUDGET')
    for c in program['constants']:
        _keys(c, ('id','value','unit','role','source'), 'CONSTANT_SCHEMA')
        if not isinstance(c['id'],str) or not re.fullmatch('k[0-9]+',c['id']) or c['id'] in dims or c['role'] != 'design_constant':
            raise ValueError('INVALID_DESIGN_CONSTANT')
        _source(c['source']); constants[c['id']] = _number(c['value'],c['id'])
        dims[c['id']] = unit_dimension(c['unit']); dependencies[c['id']] = []
    if not isinstance(program['nodes'],list) or not 1 <= len(program['nodes']) <= 128:
        raise ValueError('PHYSICAL_NODE_BUDGET')
    records = []
    for node in program['nodes']:
        _keys(node, ('id','op','inputs'), 'PHYSICAL_NODE_SCHEMA')
        name, op, refs = node['id'],node['op'],node['inputs']
        if not isinstance(name,str) or not re.fullmatch('p[0-9]+',name) or name in dims:
            raise ValueError('PHYSICAL_NODE_ID')
        if not isinstance(op,str) or op not in UNARY | BINARY:
            raise ValueError('UNREGISTERED_PHYSICAL_OPERATOR')
        if not isinstance(refs,list) or len(refs) != (1 if op in UNARY else 2) or any(not isinstance(r,str) or r not in dims for r in refs):
            raise ValueError('PHYSICAL_ARITY_OR_FORWARD_REFERENCE')
        a = dims[refs[0]]
        if op in {'add','sub','symmetric_clip'}:
            if a != dims[refs[1]]: raise ValueError('PHYSICAL_UNIT_MISMATCH')
            if op == 'symmetric_clip' and (refs[1] not in constants or constants[refs[1]] <= 0):
                raise ValueError('CLIP_REQUIRES_POSITIVE_DESIGN_LIMIT')
            out = a
        elif op == 'mul': out = tuple(x+y for x,y in zip(a,dims[refs[1]]))
        elif op == 'rate': out = (a[0],a[1],a[2]-1)
        elif op == 'trapezoid_integral': out = (a[0],a[1],a[2]+1)
        elif op == 'sign': out = (0,0,0)
        else: out = a
        if any(abs(v)>8 for v in out): raise ValueError('DIMENSION_EXPONENT_BUDGET')
        dims[name]=out; dependencies[name]=refs
        records.append({**node,'dimension_kg_m_s':list(out)})
    outputs = program['outputs']
    if not isinstance(outputs,list) or not 1 <= len(outputs) <= 32:
        raise ValueError('PHYSICAL_OUTPUT_BUDGET')
    names=set(); visited=set(); stack=[]
    for o in outputs:
        _keys(o,('name','node','unit','component'), 'PHYSICAL_OUTPUT_SCHEMA')
        if not isinstance(o['name'],str) or not re.fullmatch('[a-z][a-z0-9_]*',o['name']) or o['name'] in names or not isinstance(o['component'],str) or not o['component']:
            raise ValueError('PHYSICAL_OUTPUT_NAME')
        names.add(o['name'])
        if o['node'] not in dims or dims[o['node']] != unit_dimension(o['unit']):
            raise ValueError('PHYSICAL_OUTPUT_DIMENSION_MISMATCH')
        stack.append(o['node'])
    while stack:
        key=stack.pop()
        if key in visited: continue
        visited.add(key);stack.extend(dependencies[key])
    if any(k not in visited for k in dims if not k.startswith('x')):
        raise ValueError('UNUSED_PHYSICAL_NODE_OR_CONSTANT')
    return {'schema':'phm.physical-audit.v1','program_sha256':digest(program),'node_count':len(records),
            'output_count':len(outputs),'nodes':records,'outputs':copy.deepcopy(outputs),
            'input_dependencies':[program['inputs'][i]['name'] for i in range(len(program['inputs'])) if f'x{i}' in visited],
            'dimension_check':'PASS','physical_validity':'NOT_PROVEN',
            'external_source_content_verified_by_compiler':False,
            'limitations':copy.deepcopy(program['assumptions'])}


class PhysicalProgram(nn.Module):
    """Causal operations on uniformly sampled *raw SI* observation sequences."""
    def __init__(self, program: dict, contract: dict):
        super().__init__(); self.audit=analyze_program(program,contract)
        self.program=copy.deepcopy(program); self.channel_count=len(program['inputs'])

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        if raw.ndim!=3 or raw.shape[-1]!=self.channel_count or raw.shape[1]<2:
            raise ValueError('PHYSICAL_INPUT_SHAPE')
        if not torch.isfinite(raw).all(): raise ValueError('NONFINITE_PHYSICAL_INPUT')
        x=raw.to(torch.float64); dt=self.program['sample_s']
        values={f'x{i}':x[...,i:i+1] for i in range(self.channel_count)}
        for c in self.program['constants']: values[c['id']]=torch.full_like(x[...,:1],c['value'])
        for n in self.program['nodes']:
            op=n['op']; xs=[values[r] for r in n['inputs']]; a=xs[0]
            if op=='add': out=a+xs[1]
            elif op=='sub': out=a-xs[1]
            elif op=='mul': out=a*xs[1]
            elif op=='symmetric_clip': out=torch.maximum(-xs[1],torch.minimum(xs[1],a))
            elif op=='sign': out=a.sign()
            elif op=='delta': out=a-a[:,:1]
            elif op=='rate': out=torch.cat((torch.zeros_like(a[:,:1]),(a[:,1:]-a[:,:-1])/dt),1)
            else: out=torch.cat((torch.zeros_like(a[:,:1]),torch.cumsum((a[:,1:]+a[:,:-1])*(dt/2),dim=1)),1)
            values[n['id']]=out
        result=torch.cat([values[o['node']] for o in self.program['outputs']],-1)
        if not torch.isfinite(result).all(): raise ValueError('NONFINITE_PHYSICAL_OUTPUT')
        return result


def _scaler_check(scaler: dict, channels: int):
    if not isinstance(scaler,dict) or set(scaler)!={'mean','scale','fit_role'} or scaler['fit_role']!='train_only':
        raise ValueError('TRAIN_ONLY_SCALER_REQUIRED')
    for key in ('mean','scale'):
        if not isinstance(scaler[key],list) or len(scaler[key])!=channels:
            raise ValueError('SCALER_DIMENSION_MISMATCH')
        for v in scaler[key]: _number(v,key)
    if any(v<=0 for v in scaler['scale']): raise ValueError('SCALER_POSITIVE_SCALE_REQUIRED')


def analyze_mechanism(shape: Shape, p: dict) -> dict:
    _keys(p, ('program','input_contract','input_scaler','feature_scaler','feature_mode','control_indices','graph'), 'MECHANISM_GRAPH_SCHEMA')
    contract=p['input_contract']; audit=analyze_program(p['program'],contract)
    if len(contract['channels'])!=shape.channels or len(contract['classes'])!=shape.classes:
        raise ValueError('MECHANISM_SHAPE_CONTRACT')
    count=audit['output_count']; _scaler_check(p['input_scaler'],shape.channels); _scaler_check(p['feature_scaler'],count)
    if p['feature_mode'] not in {'physical','observed_delta'}: raise ValueError('UNKNOWN_MECHANISM_MODE')
    idx=p['control_indices']
    if not isinstance(idx,list) or len(idx)!=count or any(type(i) is not int or not 0<=i<shape.channels for i in idx):
        raise ValueError('CONTROL_FEATURE_INDICES')
    network=analyze_graph(Shape(shape.window,shape.channels+count,shape.classes),p['graph'])
    return {'physical_program':audit,'network':network,'parameter_count':network['parameter_count'],
            'feature_mode':p['feature_mode'],'physical_program_evaluated_in_both_modes':True,
            'neural_mac_estimate':network['multiply_accumulate_estimate_per_case'],
            'physical_signal_operations':audit['node_count']*shape.window,
            'exact_hardware_compute_matched':False}


class MechanismGraph(nn.Module):
    def __init__(self, shape: Shape, p: dict):
        super().__init__(); audit=analyze_mechanism(shape,p)
        self.program=PhysicalProgram(p['program'],p['input_contract']);self.mode=p['feature_mode'];self.indices=list(p['control_indices'])
        for name,values in (('raw_mean',p['input_scaler']['mean']),('raw_scale',p['input_scaler']['scale']),('feature_mean',p['feature_scaler']['mean']),('feature_scale',p['feature_scaler']['scale'])):
            # Registry.build converts the model to float32; forward promotes the
            # constants explicitly. No hidden fitted state lives outside bundle.
            self.register_buffer(name,torch.tensor(values,dtype=torch.float64))
        self.network=GraphNetwork(Shape(shape.window,shape.channels+audit['physical_program']['output_count'],shape.classes),p['graph'])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw=x.double()*self.raw_scale.double()+self.raw_mean.double()
        physical=self.program(raw)
        control=raw[...,self.indices]-raw[:,:1,self.indices]
        features=physical if self.mode=='physical' else control
        normalized=((features-self.feature_mean.double())/self.feature_scale.double()).float()
        return self.network(torch.cat((x,normalized),-1))


def mechanism_count(shape: Shape, p: dict) -> int:
    return analyze_mechanism(shape,p)['parameter_count']


def build_mechanism(shape: Shape,p:dict)->nn.Module:
    return MechanismGraph(shape,p)


def fit_feature_scaler(program: dict, contract: dict, raw: np.ndarray, mode: str, indices: list[int]) -> dict:
    if mode=='physical':
        with torch.inference_mode(): features=PhysicalProgram(program,contract)(torch.from_numpy(raw)).numpy()
    elif mode=='observed_delta': features=raw[...,indices].astype(np.float64)-raw[:,:1,indices].astype(np.float64)
    else: raise ValueError('UNKNOWN_MECHANISM_MODE')
    return fit_scaler(features)
