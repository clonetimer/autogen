"""Research Temporal Algorithm-DAG: residual -> sequence monitor -> event gate.

High-level inference primitive for PHM AutoDesign 0.7.0.dev5; it does not
change existing Algorithm-DAG case classification or 0.6.0 Stable. A frozen
candidate can execute with deployable signed residuals and observability flags
supplied by a trusted mechanism program; it never accepts failure truth.

Literature-inspired components: analytical redundancy residual, Page CUSUM,
and EWMA. A model-based residual source must be selected and trained separately.
This module does not assert any fielded statistical false-alarm guarantee.
"""
from __future__ import annotations
import hashlib
import json
import math
from typing import Sequence
import numpy as np
from .temporal_primitives import SequentialCandidate, sequential_scores

SCHEMA = 'phm.algorithm-dag.temporal.v1'


def fingerprint(payload:dict) -> str:
    return hashlib.sha256(json.dumps(payload,sort_keys=True,
       separators=(',',':'),allow_nan=False).encode('utf-8')).hexdigest()


def make_temporal_candidate(*, method:str, parameter:float,
                            threshold:float, consecutive:int=3, drift:float=0.) -> dict:
    alg=SequentialCandidate(method,parameter,drift)
    if not math.isfinite(threshold) or threshold<=0 or consecutive<1:
        raise ValueError('INVALID_TEMPORAL_ALARM_POLICY')
    candidate={
      'schema':SCHEMA,'grammar':{'algorithm':'analytic_residual_sequence_alarm',
                                'monitor':alg.method,'parameter':alg.parameter,'drift':alg.drift,
                                'threshold':float(threshold),'consecutive':int(consecutive)},
      'nodes':{
        'residual':{'op':'input_signed_analytic_residual'},
        'observable':{'op':'input_observability_mask'},
        'statistic':{'op':alg.method,'inputs':['residual','observable'],
                     'params':{'parameter':alg.parameter,'drift':alg.drift}},
        'event':{'op':'threshold_and_consecutive','inputs':['statistic','observable'],
                 'params':{'threshold':float(threshold),'consecutive':int(consecutive)}}
      },
      'outputs':{'event':'event','score':'statistic'},
      'truth_inputs_used_at_runtime':False,
      'calibration_status':'NOT_ASSIGNED_BY_DAG_RUNTIME'
    }
    candidate['fingerprint']=fingerprint(candidate)
    return candidate


def execute_temporal_candidate(candidate:dict,
                               signed_residuals:Sequence[float],
                               observable:Sequence[bool]) -> dict:
    if candidate.get('schema')!=SCHEMA:raise ValueError('TEMPORAL_DAG_SCHEMA')
    clone={k:v for k,v in candidate.items() if k!='fingerprint'}
    if candidate.get('fingerprint')!=fingerprint(clone):
        raise ValueError('TEMPORAL_DAG_FINGERPRINT_DRIFT')
    g=candidate['grammar']
    c=SequentialCandidate(g['monitor'],float(g['parameter']),float(g['drift']))
    threshold=float(g['threshold']);n=int(g['consecutive'])
    if threshold<=0 or not math.isfinite(threshold) or n<1:
        raise ValueError('TEMPORAL_DAG_INVALID_ALARM_PARAMS')
    score=sequential_scores(signed_residuals,observable,c)
    mask=np.asarray(observable,dtype=bool)
    flags=np.zeros(len(score),dtype=bool)
    states=[];streak=0
    for i,(s,valid) in enumerate(zip(score,mask)):
        if not valid:
            streak=0;states.append('UNKNOWN_UNOBSERVABLE')
        else:
            streak=streak+1 if s>threshold else 0
            flags[i]=streak>=n
            states.append('FAULT_ALERT' if flags[i] else 'NO_ALERT_NOT_PROVEN_HEALTHY')
    return {'scores':score,'alert':flags,'status':tuple(states),
            'candidate_fingerprint':candidate['fingerprint']}
