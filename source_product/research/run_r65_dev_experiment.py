"""R65 predeclared native Thermal excitation pilot. NO BLIND/QUALIFICATION.

Usage: PYTHONPATH=src python research/run_r65_dev_experiment.py --output NEW_DIR
Output is immutable (refuses any pre-existing directory). Model training and
access to R60/R62 blinded evidence are explicitly outside this script.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata as md
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from phm_design.adapters.sat_multicomponent import FAMILIES,parameters,run_native
from phm_design.thermal_excitation_r65 import BASELINE,EXPERIMENT,LEVELS,SEGMENT_S

HEADER=("time_s","illumination","heat_a_command_w","heat_b_command_w","temperature_a_k","temperature_b_k")
PROTOCOL_FILE=Path(__file__).with_name("R65_DEV_PROTOCOL.json")


def _sha(data: bytes)->str: return hashlib.sha256(data).hexdigest()
def _dump(path:Path,data:dict)->str:
    by=(json.dumps(data,indent=2,sort_keys=True,allow_nan=False)+"\n").encode()
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(by)
    return _sha(by)


def _csv(path:Path,rows:list)->str:
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",newline="",encoding="utf-8") as f:
        w=csv.writer(f);w.writerow(HEADER);w.writerows(rows)
    return _sha(path.read_bytes())


def validate_protocol(d:dict)->None:
    if d.get('schema')!='phm.r65.excitation-dev-protocol.v1' or d.get('research_phase')!='DEVELOPMENT_PILOT_ONLY' or d.get('open_blind_worlds') is not False:
        raise ValueError('R65_NOT_DEV_ONLY')
    assert d['arms']=={'A':BASELINE,'B':EXPERIMENT}
    e=d['simulation']['excitation']
    if (e['segment_s']!=SEGMENT_S or tuple(e['multipliers'])!=LEVELS or e['component_b_phase_shift_segments']!=3):
        raise ValueError('R65_EXCITATION_PROTOCOL_CHANGED')
    if (d['evaluation']['no_model_training'] is not True or d['evaluation']['no_qualification_or_holdout'] is not True
            or d['evaluation']['deployment_observations_only']!=list(HEADER)
            or len(d['world_ids'])!=4 or len(set(d['world_ids']))!=4):
        raise ValueError('R65_EVALUATION_PROTOCOL_CHANGED')


def assess_world(ref:list[dict], p:dict, arm:str,world:str)->dict:
    baseline=np.asarray(ref[0]['observations'],dtype=float)
    when=np.asarray(ref[1]['truth'],dtype=float)[:,1].astype(bool)
    if when.sum()<12 or not np.array_equal(when,np.asarray(ref[4]['truth'])[:,1].astype(bool)):
        raise AssertionError('INVALID_FAULT_ACTIVITY')
    first=int(np.flatnonzero(when)[0])
    if int(baseline[first,0]) != p['onset_s']+20:
        raise AssertionError('INCORRECT_NATIVE_ONSET_ALIGNMENT')
    comps=[]
    for component,idx,e,h,command in [('a',4,1,3,2),('b',5,2,4,3)]:
        x=np.asarray(ref[e]['observations'],dtype=float)
        y=np.asarray(ref[h]['observations'],dtype=float)
        if not np.array_equal(x[:first],baseline[:first]) or not np.array_equal(y[:first],baseline[:first]):
            raise AssertionError('PRE_FAULT_PAIR_MISMATCH')
        if not np.array_equal(x[:,1:4],baseline[:,1:4]) or not np.array_equal(y[:,1:4],baseline[:,1:4]):
            raise AssertionError('FAULT_CHANGED_KNOWN_BOUNDARY')
        other=5 if idx==4 else 4
        if not np.array_equal(x[:,other],baseline[:,other]) or not np.array_equal(y[:,other],baseline[:,other]):
            raise AssertionError('UNCOUPLED_COMPONENT_CHANGED')
        a=x[first:,idx]-baseline[first:,idx]
        b=y[first:,idx]-baseline[first:,idx]
        if min(np.linalg.norm(a),np.linalg.norm(b))<1e-8:raise AssertionError('NO_FAULT_EFFECT')
        scale=float(np.dot(a,b)/np.dot(a,a))
        T=baseline[first:,idx]
        d=np.column_stack([np.ones(len(T)),(T/300)**4])
        comps.append({
          'component':component,
          'prefault_command_levels':int(len(np.unique(baseline[:first,command]))),
          'prefault_nominal_temperature_span_k':float(np.ptp(baseline[:first,idx])),
          'postfault_nominal_temperature_span_k':float(np.ptp(T)),
          'postfault_mechanism_pearson':float(np.corrcoef(a,b)[0,1]),
          'postfault_scaled_residual_fraction':float(np.linalg.norm(b-scale*a)/np.linalg.norm(b)),
          'postfault_constant_vs_t4_condition_number':float(np.linalg.cond(d)),
          'postfault_nonfault_component_max_change_k':float(max(np.max(np.abs(x[:,other]-baseline[:,other])),np.max(np.abs(y[:,other]-baseline[:,other])))),
        })
    return {'world_id':world,'arm':arm,'onset_s_simulator_only':p['onset_s'],
      'research_oracle_used':True,'components':comps}


def main()->int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if args.output.exists():ap.error('OUTPUT_ALREADY_EXISTS')
    raw=PROTOCOL_FILE.read_bytes();plan=json.loads(raw);validate_protocol(plan)
    if md.version('bsk').split('+')[0]!='2.11.0':raise ValueError('BASILISK_VERSION_NOT_2110')
    args.output.mkdir(parents=True)
    (args.output/'PROTOCOL_FROZEN_COPY.json').write_bytes(raw)
    base_hash=_sha(raw)
    try:
        commit=subprocess.check_output(['git','-C',str(PROTOCOL_FILE.parent.parent),'rev-parse','HEAD'],stderr=subprocess.DEVNULL,text=True).strip()
    except Exception: commit=None
    file_manifest={"PROTOCOL_FROZEN_COPY.json":base_hash}
    world_reports=[]
    for world in plan['world_ids']:
        p=parameters(world,'thermal')
        wr={'world_id':world,'arms':[]}
        for arm,profile in plan['arms'].items():
            paired=[]
            # Re-using the same noise realization is allowed ONLY for oracle
            # numerical physics auditing; never use these CSVs as a dataset.
            for label in range(5):
                s=run_native('thermal',p,label,thermal_excitation=profile)
                if not all(a==b==int(i*20*1e9) for i,(a,b) in enumerate(s['output_times_ns'])):
                    raise AssertionError('NATIVE_TIMESTAMP_MISMATCH')
                paired.append(s)
                path=Path('research_oracle_pairs')/arm/world/f'L{label}.csv'
                file_manifest[str(path)]=_csv(args.output/path,s['observations'])
            metrics=assess_world(paired,p,arm,world)
            wr['arms'].append(metrics)
            # Independent measurement noise by arm, world and class. The
            # physical simulation is identical; no oracle features leak in CSV.
            for label in range(5):
                independent=dict(p)
                independent['measurement_seed']=int(_sha(f'{world}:{arm}:{label}:independent_noise'.encode())[:16],16)
                s=run_native('thermal',independent,label,thermal_excitation=profile)
                path=Path('deployment_observations_dev_only')/arm/world/f'L{label}.csv'
                file_manifest[str(path)]=_csv(args.output/path,s['observations'])
        world_reports.append(wr)
        print(f'completed_dev_world={world}',flush=True)
    all_a=[c for r in world_reports for a in r['arms'] if a['arm']=='A' for c in a['components']]
    all_b=[c for r in world_reports for a in r['arms'] if a['arm']=='B' for c in a['components']]
    def aggregate(all_c):
        return {key:{'median':float(np.median([c[key] for c in all_c])),
                     'min':float(np.min([c[key] for c in all_c])),
                     'max':float(np.max([c[key] for c in all_c]))}
           for key in ['prefault_command_levels','prefault_nominal_temperature_span_k',
           'postfault_nominal_temperature_span_k','postfault_mechanism_pearson',
           'postfault_scaled_residual_fraction','postfault_constant_vs_t4_condition_number']}
    report={
      'schema':'phm.r65.native-dev-excitation-pilot.v1',
      'status':'DEVELOPMENT_PILOT_ONLY_NO_DIAGNOSTIC_QUALIFICATION',
      'bsk_version':md.version('bsk'), 'python':sys.version.split()[0],
      'protocol_sha256':base_hash,'source_commit_sha':commit,
      'worlds':world_reports,'paired_native_trajectories':40,
      'independent_noise_native_trajectories':40,
      'development_worlds_per_arm':4,'blind_worlds_opened':0,
      'model_training_performed':False,
      'archival_warning':'oracle paired same-noise traces and known fault times must never be passed to inference or model selection',
      'aggregate':{'A':aggregate(all_a),'B':aggregate(all_b)},
      'diagnostic_qualification':'NOT_EVALUATED'}
    file_manifest['R65_DEV_RESULTS.json']=_dump(args.output/'R65_DEV_RESULTS.json',report)
    file_manifest['SIMULATOR_PARAMETERS_ONLY.json']=_dump(args.output/'SIMULATOR_PARAMETERS_ONLY.json',
        {'parameters':{w:parameters(w,'thermal') for w in plan['world_ids']},
         'role':'SIMULATOR_ORACLE_ONLY_NOT_DEPLOYMENT'})
    _dump(args.output/'MANIFEST_SHA256.json',{'schema':'phm.r65.dev-evidence-manifest.v1', 'files':file_manifest})
    print(json.dumps({'status':report['status'],'aggregate':report['aggregate']},indent=2),flush=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
