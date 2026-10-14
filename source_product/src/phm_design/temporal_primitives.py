"""Literature-inspired streaming Algorithm-DAG primitives (research only).

Design lineage: model-based analytical redundancy (Frank, Automatica 1990)
combined with EWMA and two-sided Page CUSUM change accumulation. This is an
independent, fully causal, explicit-UNKNOWN implementation. It DOES NOT claim
numerical equivalence to any named paper's full experimental system.

Candidate selection must use nominal train/validation only; returned candidate
is a development route, not a calibrated fault-risk policy.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Sequence
import numpy as np

METHODS = ('rolling_mean', 'ewma', 'two_sided_cusum')

@dataclass(frozen=True)
class SequentialCandidate:
    method: str
    parameter: float
    drift: float = 0.0

    def __post_init__(self):
        if self.method not in METHODS or not math.isfinite(self.parameter) or self.parameter <= 0:
            raise ValueError('INVALID_SEQUENTIAL_METHOD_OR_PARAMETER')
        if not math.isfinite(self.drift) or self.drift < 0:
            raise ValueError('INVALID_DRIFT')
        if self.method == 'ewma' and self.parameter > 1.0:
            raise ValueError('INVALID_EWMA_ALPHA')
        if self.method == 'rolling_mean' and self.parameter != int(self.parameter):
            raise ValueError('INVALID_ROLLING_WINDOW')


def sequential_scores(residuals: Sequence[float], observable: Sequence[bool],
                      candidate: SequentialCandidate) -> np.ndarray:
    """Compute per-sample causal change statistic; reject invalid visible data.

    An unobservable instant resets the detector's state (avoids integrating
    unknowable gaps). No values from future instants or ground truth are used.
    Every returned score at t depends only on inputs at <=t.
    """
    r=np.asarray(residuals,dtype=float)
    mask=np.asarray(observable,dtype=bool)
    if r.ndim!=1 or mask.ndim!=1 or r.shape!=mask.shape or not len(r):
        raise ValueError('INVALID_SERIES_SHAPE')
    if not np.isfinite(r[mask]).all():
        raise ValueError('NONFINITE_OBSERVABLE_RESIDUAL')
    out=np.zeros(len(r),float)
    gplus=gminus=ewma=0.0
    window=[]
    for idx,(v,valid) in enumerate(zip(r,mask)):
        if not valid:
            gplus=gminus=ewma=0.0
            window=[]
            continue
        v=float(v)
        if candidate.method=='two_sided_cusum':
            gplus=max(0.0,gplus+v-candidate.drift)
            gminus=max(0.0,gminus-v-candidate.drift)
            out[idx]=max(gplus,gminus)
        elif candidate.method=='ewma':
            ewma=candidate.parameter*v+(1.-candidate.parameter)*ewma
            out[idx]=abs(ewma)
        else:
            window.append(v)
            if len(window)>int(candidate.parameter):window.pop(0)
            out[idx]=abs(sum(window)/len(window))
    return out


@dataclass(frozen=True)
class NominalPolicy:
    candidate: SequentialCandidate
    threshold: float
    minimum_group_coverage: float
    train_group_count: int
    validation_group_count: int
    status: str = 'DEVELOPMENT_NOMINAL_ENVELOPE_NOT_RISK_CALIBRATED'


def nominal_envelope(candidate: SequentialCandidate,
                     train: Sequence[tuple[np.ndarray,np.ndarray]],
                     validation: Sequence[tuple[np.ndarray,np.ndarray]],
                     *, min_coverage: float=.65, buffer_fraction:float=.10) -> NominalPolicy:
    """Nominal-only threshold from independent train/validation groups.

    A deliberately conservative max-envelope policy, NOT a binomial false alarm
    probability certificate. Missing groups or low coverage fail closed.
    """
    if len(train)<3 or len(validation)<3 or not(0<min_coverage<=1) or buffer_fraction<0:
        raise ValueError('INSUFFICIENT_NOMINAL_GROUPS_OR_COVERAGE_POLICY')
    maxima=[]
    for groups in (train,validation):
        for residual,mask in groups:
            m=np.asarray(mask,dtype=bool)
            if m.ndim!=1 or len(m)<8 or m.mean()<min_coverage:
                raise ValueError('FAIL_CLOSED_NOMINAL_GROUP_COVERAGE')
            stats=sequential_scores(residual,m,candidate)
            maxima.append(float(stats[m].max()))
    threshold=max(1e-8,(1.+buffer_fraction)*max(maxima))
    return NominalPolicy(candidate,threshold,min_coverage,len(train),len(validation))


def sequential_alarm(residuals:Sequence[float], observable:Sequence[bool],
                     policy:NominalPolicy, *, consecutive:int=3):
    if consecutive<1 or not np.isfinite(policy.threshold) or policy.threshold<=0:
        raise ValueError('INVALID_ALARM_POLICY')
    mask=np.asarray(observable,dtype=bool)
    scores=sequential_scores(residuals,mask,policy.candidate)
    alarm=np.zeros(len(scores),dtype=bool)
    status=[];run=0
    for j,(ok,s) in enumerate(zip(mask,scores)):
        if not ok:
            run=0
            status.append('UNKNOWN_UNOBSERVABLE')
        else:
            run=run+1 if s>policy.threshold else 0
            alarm[j]=run>=consecutive
            status.append('FAULT_ALERT' if alarm[j] else 'NO_ALERT_NOT_PROVEN_HEALTHY')
    return scores,alarm,tuple(status)


def available_candidates(scale:float) -> tuple[SequentialCandidate,...]:
    if not math.isfinite(scale) or scale<=0:raise ValueError('INVALID_SCALE')
    return tuple([SequentialCandidate('rolling_mean',float(w)) for w in (3,5,9)]+[
        SequentialCandidate('ewma',alpha) for alpha in (.05,.15,.30)]+[
        SequentialCandidate('two_sided_cusum',1.0,scale*m) for m in (.5,1.0,1.5)])
