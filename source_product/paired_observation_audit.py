"""Read-only, paired-data information audit over frozen R16 *train* worlds.

Fault labels and normal counterfactual traces are used only in this offline
research audit. They NEVER enter model inference. A pairwise response distance
is *optimistic* and cannot establish classifier performance or observability
under unknown-world parameters.
"""
from __future__ import annotations
import csv
import json
import math
from pathlib import Path
import numpy as np
from phm_design.design_readiness import sha_file

ROOT = Path('/mnt/data/PHM_AUTODESIGN_THERMAL_R16_20261008/new_training')
SENSORS=('equipment_temperature_k','radiator_temperature_k')


def load_case(case_id, path_root=ROOT):
    trace=path_root/'traces'/f'{case_id}.csv'
    audit_path=path_root/'audit'/f'{case_id}.json'
    audit=json.loads(audit_path.read_text(encoding='utf-8'))
    if audit['case_id']!=case_id or sha_file(trace)!=audit['trace_sha256']:
        raise ValueError('R17_TRACE_AUDIT_BINDING_FAILED')
    if audit['plan_sha256']!=sha_file(path_root/'FROZEN_PLAN.json'):
        raise ValueError('R17_PLAN_AUDIT_BINDING_FAILED')
    with trace.open(newline='', encoding='utf-8') as f:
        rows=list(csv.DictReader(f))
    if not rows:raise ValueError('R17_EMPTY_TRACE')
    return np.array([[float(x[k]) for k in ('time_s',)+SENSORS] for x in rows]), audit


def _geometry(x, y):
    a=x.ravel();b=y.ravel()
    aa=float(np.linalg.norm(a));bb=float(np.linalg.norm(b))
    if aa<1e-9 or bb<1e-9:
        return {'cosine':None,'relative_separation':None,'fault_signal_rms':[aa/math.sqrt(len(a)),bb/math.sqrt(len(b))]}
    return {
      'cosine':float(np.dot(a,b)/aa/bb),
      'relative_separation':float(np.linalg.norm(a-b)/max(aa,bb)),
      'fault_signal_rms':[float(np.sqrt(np.mean(a*a))),float(np.sqrt(np.mean(b*b)))],
    }


def analyze_pairs(plan_path=ROOT/'FROZEN_PLAN.json', source_root=ROOT):
    plan=json.loads(plan_path.read_text(encoding='utf-8'))
    if plan.get('role')!='NEW_DEV_SUPERVISED_TRAIN_ONLY' or not plan.get('no_blind') or not plan.get('no_calibration'):
        raise ValueError('R17_PROTECTED_DATA_IN_PLAN')
    assert plan['simulations']==54 and plan['world_groups']==18
    worlds={}
    for spec in plan['cases']:
        if spec['role']!='design_train':raise ValueError('R17_NON_TRAIN_CASE')
        worlds.setdefault(spec['world_group'],{})[spec['label']]=spec['case_id']
    if len(worlds)!=18:raise ValueError('R17_INCOMPLETE_WORLDS')
    out=[]
    for group,cases in sorted(worlds.items()):
        if set(cases)!={0,1,2}:raise ValueError('R17_MISSING_PAIR_CLASS')
        arr={k:load_case(cases[k],source_root) for k in (0,1,2)}
        if {tuple(e[0][:,0]) for e in arr.values()}.__len__()!=1:
            raise ValueError('R17_TIME_GRID_MISMATCH')
        for vals in arr.values():
            if vals[1]['world_group']!=group:raise ValueError('R17_WORLD_GROUP_MISMATCH')
        t=arr[0][0][:,0]
        pref=t<600
        # Same input and random noise until fault onset: should be exact paired sensor telemetry.
        if any(not np.array_equal(arr[k][0][pref,1:],arr[0][0][pref,1:]) for k in (1,2)):
            raise ValueError('R17_PRE_ONSET_PAIR_MISMATCH')
        # 24min decision upper bound is 600+1440 = 2040s.
        eligible=(t>600)&(t<=2040)
        if np.count_nonzero(eligible)<48:raise ValueError('R17_SHORT_INFORMATION_WINDOW')
        # Causal maximum 12/24/48 samples leading up to 2040s.
        deltas={k:arr[k][0][eligible,1:]-arr[0][0][eligible,1:] for k in (1,2)}
        observations={}
        for n in (12,24,48):
            single=_geometry(deltas[1][-n:,0:1],deltas[2][-n:,0:1])
            two=_geometry(deltas[1][-n:,:],deltas[2][-n:,:])
            observations[str(n)]={'single_sensor':single,'two_sensor':two}
        out.append({'world_group':group,'case_ids':[cases[i] for i in range(3)],'geometry_by_window':observations})
    summary={}
    for n in ('12','24','48'):
        summary[n]={}
        for mode in ('single_sensor','two_sensor'):
            vals=[x['geometry_by_window'][n][mode] for x in out]
            sel=[x['relative_separation'] for x in vals if x['relative_separation'] is not None]
            cos=[x['cosine'] for x in vals if x['cosine'] is not None]
            summary[n][mode]={
                'worlds_with_nonzero_both_responses':len(sel),
                'median_relative_fault_response_separation':float(np.median(sel)) if sel else None,
                'minimum_relative_fault_response_separation':float(min(sel)) if sel else None,
                'median_fault_response_cosine':float(np.median(cos)) if cos else None,
                'max_fault_response_cosine':float(max(cos)) if cos else None,
            }
    return {
        'schema':'phm.r17.paired-information.development-audit.v1',
        'source_plan_sha256':sha_file(plan_path),
        'input_role':'TRAIN_ONLY_PAIR_RESPONSE_GEOMETRY',
        'input_world_groups':len(worlds),
        'no_new_simulations':True,
        'no_model_fit':True,
        'no_test_or_holdout':True,
        'labels_and_counterfactuals_used_only_for_offline_research':True,
        'not_a_deployment_observability_proof':True,
        'decision_time_s':2040,
        'summary':summary,'world_audit':out,
    }


if __name__=='__main__':
    dest=Path(__file__).resolve().parent/'evidence'/'THERMAL_R16_PAIRED_INFORMATION_AUDIT.json'
    with dest.open('x',encoding='utf-8') as f:
        json.dump(analyze_pairs(),f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
    print('R17_INFORMATION_AUDIT_SAVED',dest)
