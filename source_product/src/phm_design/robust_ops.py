"""Quality-carrying observations and fixed-axis momentum hypotheses.

No forward filling from future measurements; missing intervals reset cumulative
innovations. A numeric zero at an invalid position is a placeholder, NEVER proof
of nominal operation. Inertia hypotheses are engineering sensitivity assumptions,
not a calibrated posterior. CPU float32 model / float64 physical calculations.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch
from torch import nn
from .data import validate_contract
from .models import integer


def number(x, name, lo=None, hi=None):
    if type(x) not in (int, float) or not math.isfinite(x):
        raise ValueError(f'{name}: FINITE_NUMBER_REQUIRED')
    if (lo is not None and x < lo) or (hi is not None and x > hi):
        raise ValueError(f'{name}: OUT_OF_RANGE')
    return float(x)


@dataclass
class Observation:
    values: torch.Tensor       # [B,T,C], physical units, invalid entries may be NaN
    observed: torch.Tensor     # [B,T,C], bool
    source_s: torch.Tensor     # [B,T,C], actual measurement timestamp
    time_s: torch.Tensor       # [B,T], processing timestamp

    def select(self, indices):
        return Observation(*(getattr(self,k)[indices] for k in ('values','observed','source_s','time_s')))

    def validate(self):
        x=self.values
        if x.ndim!=3 or min(x.shape)<1 or x.shape[1]<2 or x.dtype not in (torch.float32,torch.float64):
            raise ValueError('OBSERVATION_SHAPE_OR_DTYPE')
        if self.observed.dtype!=torch.bool or self.observed.shape!=x.shape or self.source_s.shape!=x.shape or self.time_s.shape!=x.shape[:2]:
            raise ValueError('OBSERVATION_METADATA_SHAPE')
        if any(t.device!=x.device for t in (self.observed,self.source_s,self.time_s)):
            raise ValueError('OBSERVATION_DEVICE_MISMATCH')
        if any(t.dtype not in (torch.float32,torch.float64) for t in (self.source_s,self.time_s)):
            raise ValueError('TIMESTAMP_DTYPE')
        if not torch.isfinite(self.time_s).all() or not (self.time_s[:,1:]>self.time_s[:,:-1]).all():
            raise ValueError('PROCESSING_TIME_NOT_FINITE_OR_ORDERED')
        finite=self.observed & torch.isfinite(x) & torch.isfinite(self.source_s)
        # A malformed timestamp is not silently turned into an old observation.
        if (finite & (self.source_s>self.time_s.unsqueeze(-1)+1e-9)).any():
            raise ValueError('FUTURE_MEASUREMENT')
        # Repeated stamps (sample-and-hold) are allowed, backwards stamps are not.
        previous=torch.full_like(self.source_s[:,0],float('-inf'))
        for t in range(x.shape[1]):
            if (finite[:,t] & (self.source_s[:,t]<previous-1e-9)).any():
                raise ValueError('SOURCE_TIME_REGRESSION')
            previous=torch.where(finite[:,t],self.source_s[:,t],previous)
        return finite

    def quality(self,max_age_s):
        number(max_age_s,'max_age_s',0)
        finite=self.validate()
        raw_age=self.time_s.unsqueeze(-1).double()-self.source_s.double()
        good=finite & (raw_age>=-1e-9) & (raw_age<=max_age_s)
        age=torch.where(good,raw_age.clamp(min=0),torch.zeros_like(raw_age))
        clean=torch.where(good,self.values,torch.zeros_like(self.values))
        return clean,good,age


def validate_mechanism(contract, spec):
    validate_contract(contract)
    required={'schema','groups','inertia_bounds_kg_m2','nominal_limits_nm','source','assumptions',
              'max_age_s','sync_tolerance_s','max_interval_s','inertia_scope'}
    if set(spec)!=required or spec['schema']!='phm.robust-momentum.v1':
        raise ValueError('ROBUST_MECHANISM_SCHEMA')
    if spec['inertia_scope']!='hypothetical_diagnostic_parameter_sensitivity_not_hardware_tolerance':
        raise ValueError('INERTIA_EVIDENCE_SCOPE')
    src=spec['source']
    if not isinstance(src,dict) or set(src)!={'reference','sha256'} or not src['reference'] or not isinstance(src['sha256'],str) or len(src['sha256'])!=64 or any(c not in '0123456789abcdef' for c in src['sha256']):
        raise ValueError('MECHANISM_SOURCE_REQUIRED')
    if not isinstance(spec['assumptions'],list) or not spec['assumptions'] or not all(isinstance(x,str) and x for x in spec['assumptions']):
        raise ValueError('MECHANISM_ASSUMPTIONS_REQUIRED')
    groups=spec['groups'];channels=contract['channels']
    if not isinstance(groups,list) or not 1<=len(groups)<=8:raise ValueError('WHEEL_GROUPS_REQUIRED')
    for g in groups:
        if not isinstance(g,list) or len(g)!=3 or len(set(g))!=3:raise ValueError('WHEEL_GROUP_TRIPLE')
        for i in g:integer(i,'channel',0,len(channels)-1)
        if [channels[i]['unit'] for i in g]!=['rad/s','N*m','rad/s']:raise ValueError('MECHANISM_UNITS')
    # Disjoint groups prevent a hidden label-selected wheel or accidental reuse.
    if sorted(i for g in groups for i in g)!=list(range(len(channels))):
        raise ValueError('WHEEL_GROUPS_MUST_PARTITION_CONTRACT')
    if len(spec['inertia_bounds_kg_m2'])!=len(groups) or len(spec['nominal_limits_nm'])!=len(groups):
        raise ValueError('WHEEL_PARAMETER_COUNTS')
    for b in spec['inertia_bounds_kg_m2']:
        if not isinstance(b,list) or len(b)!=2:raise ValueError('INERTIA_BOUNDS')
        lo=number(b[0],'inertia',1e-12);hi=number(b[1],'inertia',lo)
    for v in spec['nominal_limits_nm']:number(v,'torque_limit',1e-12)
    number(spec['max_age_s'],'max_age',0,3600)
    number(spec['sync_tolerance_s'],'synchrony',0,spec['max_age_s']+1)
    number(spec['max_interval_s'],'max_interval',1e-9,3600)


class MomentumBank(nn.Module):
    """Three inertia hypotheses/wheel; integrate ONLY contiguous valid intervals.

    Gyro channels MUST be pre-projected onto each fixed wheel axis. This class
    does not perform arbitrary-axis coordinate transforms or infer friction.
    """
    def __init__(self,contract,spec):
        super().__init__();validate_mechanism(contract,spec);self.spec=spec
        self.register_buffer('indices',torch.tensor(spec['groups'],dtype=torch.long))
        self.register_buffer('inertias',torch.tensor([[a,a+(b-a)/2,b] for a,b in spec['inertia_bounds_kg_m2']],dtype=torch.float64))
        self.register_buffer('limits',torch.tensor(spec['nominal_limits_nm'],dtype=torch.float64))

    def forward(self,obs:Observation,*,inertia_multiplier=1.0):
        number(inertia_multiplier,'inertia_multiplier',1e-9,10)
        clean,good,age=obs.quality(self.spec['max_age_s'])
        v=clean.double()[:,:,self.indices];valid=good[:,:,self.indices].all(-1)
        stamps=torch.where(good,obs.source_s.double(),obs.time_s.double().unsqueeze(-1))[:,:,self.indices]
        valid=valid & ((stamps.amax(-1)-stamps.amin(-1))<=self.spec['sync_tolerance_s'])
        source=stamps.mean(-1);dt=source[:,1:]-source[:,:-1]
        pairs=valid[:,1:] & valid[:,:-1] & (dt>0) & (dt<=self.spec['max_interval_s'])
        dt=torch.where(pairs,dt,torch.zeros_like(dt))
        rate=v[...,0]+v[...,2];dr=torch.where(pairs,rate[:,1:]-rate[:,:-1],torch.zeros_like(dt))
        u=torch.maximum(-self.limits.double(),torch.minimum(v[...,1],self.limits.double()))
        impulse=(0.5*u[:,1:]+0.5*u[:,:-1])*dt
        intervals=dr.unsqueeze(-1)*self.inertias.double()*inertia_multiplier-impulse.unsqueeze(-1)
        intervals=torch.where(pairs.unsqueeze(-1),intervals,torch.zeros_like(intervals))
        if not torch.isfinite(intervals).all():raise ValueError('PHYSICAL_OVERFLOW')
        state=torch.zeros_like(intervals[:,0]);sequence=[state]
        for t in range(intervals.shape[1]):
            state=torch.where(pairs[:,t].unsqueeze(-1),state+intervals[:,t],torch.zeros_like(state))
            sequence.append(state)
        residual=torch.stack(sequence,1)
        computable=torch.cat([torch.zeros_like(pairs[:,:1]),pairs],1)
        max_age=age[:,:,self.indices].amax(-1)
        return {'residual':residual,'computable':computable,'age_s':max_age,
                'interval_residual':torch.cat([torch.zeros_like(intervals[:,:1]),intervals],1)}


class QualityFusion(nn.Module):
    """Quality-only gate. No residual magnitude is treated as unreliability."""
    def __init__(self):
        super().__init__();self.score=nn.Linear(2,1)
    def forward(self,embeddings,available,quality):
        if embeddings.ndim!=3 or quality.shape!=(*embeddings.shape[:2],2) or available.shape!=embeddings.shape[:2] or available.dtype!=torch.bool:
            raise ValueError('FUSION_SHAPE')
        if quality.dtype!=embeddings.dtype or quality.device!=embeddings.device or available.device!=embeddings.device:raise ValueError('FUSION_DTYPE_OR_DEVICE')
        ok=available & torch.isfinite(embeddings).all(-1) & torch.isfinite(quality).all(-1)
        bad=ok & ((quality[...,0]<0)|(quality[...,0]>1)|(quality[...,1]<0))
        if bad.any():raise ValueError('INVALID_BRANCH_QUALITY')
        ok=ok & (quality[...,0]>0)
        e=torch.where(ok.unsqueeze(-1),embeddings,torch.zeros_like(embeddings))
        q=torch.where(ok.unsqueeze(-1),quality,torch.zeros_like(quality))
        scores=self.score(q).squeeze(-1)
        if not torch.isfinite(scores).all():raise ValueError('FUSION_NUMERICAL_ERROR')
        any_ok=ok.any(-1);scores=scores.masked_fill(~ok,float('-inf'))
        scores=torch.where(any_ok.unsqueeze(-1),scores,torch.zeros_like(scores))
        w=scores.softmax(-1)*ok.to(scores.dtype)
        return (e*w.unsqueeze(-1)).sum(1),any_ok,w
