"""Deterministic measurement perturbations, NOT new physics simulations.

Derived cases retain the base case identity. Hypothetical values are explicitly
NOT sensor specifications. A delayed channel keeps its source timestamp; early
samples without history become missing, rather than being filled from the future.
"""
from __future__ import annotations
import re
import numpy as np
import torch
from .io import digest
from .models import integer
from .robust_ops import Observation,number

PROFILE_KEYS={'id','noise_std','bias','delay_steps','drop_probability','gap_start','gap_length',
              'drop_channels','diagnostic_inertia_multiplier','role'}


def check_profile(profile,channels,window):
    if set(profile)!=PROFILE_KEYS or not isinstance(profile['id'],str) or not re.fullmatch('[a-z][a-z0-9_]*',profile['id']):raise ValueError('STRESS_PROFILE_SCHEMA')
    if profile['role'] not in {'selection','safety_only','evaluation_only'}:raise ValueError('STRESS_ROLE')
    for name in ('noise_std','bias','delay_steps','drop_probability'):
        if not isinstance(profile[name],list) or len(profile[name])!=channels:raise ValueError('STRESS_CHANNEL_COUNT')
    for v in profile['noise_std']:number(v,'noise_std',0)
    for v in profile['bias']:number(v,'bias')
    for v in profile['drop_probability']:number(v,'drop_probability',0,1)
    for v in profile['delay_steps']:integer(v,'delay_steps',0,window-1)
    integer(profile['gap_start'],'gap_start',0,window)
    integer(profile['gap_length'],'gap_length',0,window)
    if profile['gap_start']+profile['gap_length']>window:raise ValueError('GAP_OUTSIDE_WINDOW')
    if not isinstance(profile['drop_channels'],list) or len(set(profile['drop_channels']))!=len(profile['drop_channels']):raise ValueError('DROP_CHANNELS')
    for i in profile['drop_channels']:integer(i,'drop_channel',0,channels-1)
    number(profile['diagnostic_inertia_multiplier'],'inertia_multiplier',0.1,10)


def clean_profile(channels,id='clean',role='selection'):
    return {'id':id,'noise_std':[0.]*channels,'bias':[0.]*channels,'delay_steps':[0]*channels,
            'drop_probability':[0.]*channels,'gap_start':0,'gap_length':0,'drop_channels':[],
            'diagnostic_inertia_multiplier':1.0,'role':role}


def perturb(raw:np.ndarray,samples:list[dict],sample_s:float,profile:dict,seed:int)->Observation:
    """Fixed measurement wrapper; does not receive labels or fault onset."""
    if raw.ndim!=3 or len(samples)!=len(raw) or not np.isfinite(raw).all():raise ValueError('BASE_OBSERVATIONS_INVALID')
    integer(seed,'perturbation_seed',0,2**32-1);number(sample_s,'sample_s',1e-12)
    n,t,c=raw.shape;check_profile(profile,c,t)
    x=raw.astype(np.float64).copy();mask=np.ones(raw.shape,dtype=bool)
    time=np.broadcast_to(np.arange(t,dtype=np.float64)*sample_s,(n,t)).copy()
    stamps=np.broadcast_to(time[...,None],raw.shape).copy()
    for b,sample in enumerate(samples):
        # Only base identity is used. Ordering or label changes cannot affect
        # perturbations; sample identity is NOT passed into the model.
        key=int(digest({'seed':seed,'profile':profile,'base_id':sample['id']})[:16],16)
        rng=np.random.default_rng(key)
        x[b]+=rng.normal(size=(t,c))*np.asarray(profile['noise_std'])+np.asarray(profile['bias'])
        for j,d in enumerate(profile['delay_steps']):
            if d:
                x[b,d:,j]=x[b,:-d,j].copy();stamps[b,d:,j]=time[b,:-d]
                mask[b,:d,j]=False;x[b,:d,j]=0;stamps[b,:d,j]=np.nan
        mask[b]&=rng.uniform(size=(t,c))>=np.asarray(profile['drop_probability'])
    a=profile['gap_start'];length=profile['gap_length']
    mask[:,a:a+length,:]=False
    mask[:,:,profile['drop_channels']]=False
    # Deliberately represent missing measurements as NaNs so accidental legacy
    # inference fails rather than accepting zero-filled input silently.
    x[~mask]=np.nan;stamps[~mask]=np.nan
    with np.errstate(over='ignore'):x=x.astype(np.float32)
    if np.any(mask & ~np.isfinite(x)):raise ValueError('PERTURBATION_OVERFLOW')
    return Observation(torch.from_numpy(x),torch.from_numpy(mask),torch.from_numpy(stamps),torch.from_numpy(time))


def packet_hash(packet:Observation)->str:
    return digest({'values':np.nan_to_num(packet.values.numpy(),nan=0.).tolist(),
                   'observed':packet.observed.numpy().tolist(),
                   'source_s':np.nan_to_num(packet.source_s.numpy(),nan=0.).tolist(),
                   'time_s':packet.time_s.numpy().tolist()})


def packet_from_json(data,contract,window):
    required={'schema','channel_names','units','values','observed','source_s','time_s'}
    if set(data)!=required or data['schema']!='phm.quality-observation.v1':raise ValueError('PREDICTION_PACKET_SCHEMA')
    if data['channel_names']!=[c['name'] for c in contract['channels']] or data['units']!=[c['unit'] for c in contract['channels']]:raise ValueError('PREDICTION_INPUT_CONTRACT_CHANGED')
    # JSON boolean masks, not integers/strings coerced to true.
    m=np.asarray(data['observed'])
    if m.dtype!=np.bool_:raise ValueError('BOOLEAN_OBSERVATION_MASK_REQUIRED')
    raw=np.asarray(data['values'],dtype=np.float64)
    with np.errstate(over='ignore'):x=raw.astype(np.float32)
    if np.any(np.isfinite(raw) & ~np.isfinite(x)):raise ValueError('INPUT_FLOAT32_OVERFLOW')
    o=Observation(torch.tensor(x),torch.tensor(m),torch.tensor(np.asarray(data['source_s'],dtype=np.float64)),torch.tensor(np.asarray(data['time_s'],dtype=np.float64)))
    o.validate()
    if o.values.shape[1:]!=(window,len(contract['channels'])):raise ValueError('PREDICTION_WINDOW_CHANGED')
    return o


def packet_to_json(obs,contract):
    def serial(t):
        a=t.numpy().astype(object);a[~np.isfinite(t.numpy())]=None;return a.tolist()
    return {'schema':'phm.quality-observation.v1','channel_names':[c['name'] for c in contract['channels']],
            'units':[c['unit'] for c in contract['channels']],'values':serial(obs.values),
            'observed':obs.observed.tolist(),'source_s':serial(obs.source_s),'time_s':serial(obs.time_s)}
