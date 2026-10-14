"""R66 frozen two-arm native Basilisk development pilot. Never opens blind worlds.

Run from source checkout: PYTHONPATH=src python research/run_r66_dev_experiment.py --output /NEW/DIR
"""
from __future__ import annotations
import argparse, csv, hashlib, importlib.metadata as md, json, sys
from pathlib import Path
import numpy as np
from phm_design.adapters.sat_multicomponent import parameters, run_native
from phm_design.thermal_excitation_r65 import BASELINE
from phm_design.thermal_excitation_r66 import SLOW, LEVELS, SEGMENT_S
try:
    from research.run_r65_dev_experiment import assess_world, HEADER
except ModuleNotFoundError:
    from run_r65_dev_experiment import assess_world, HEADER

PROTOCOL_FILE = Path(__file__).with_name('R66_DEV_PROTOCOL.json')


def sha(blob:bytes)->str:
    return hashlib.sha256(blob).hexdigest()


def write_json(path:Path,obj)->str:
    path.parent.mkdir(parents=True,exist_ok=True)
    blob=(json.dumps(obj,sort_keys=True,indent=2,allow_nan=False,ensure_ascii=False)+'\n').encode('utf-8')
    path.write_bytes(blob)
    return sha(blob)


def write_csv(path:Path, rows)->str:
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x', newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(HEADER);w.writerows(rows)
    return sha(path.read_bytes())


def validate_protocol(p:dict):
    expected_worlds=[f'r66-thermal-timescale-dev-{i:03d}' for i in range(1,5)]
    if (p.get('schema')!='phm.r66.timescale-dev.v1'
        or p.get('research_phase')!='DEVELOPMENT_PILOT_ONLY'
        or p.get('open_calibration_or_blind_worlds') is not False
        or p.get('model_training') is not False
        or p.get('qualification')!='NOT_EVALUATED'
        or p.get('world_ids')!=expected_worlds
        or p.get('arms')!={'A':BASELINE,'B':SLOW}):
        raise ValueError('R66_UNFROZEN_RESEARCH_ROLE_OR_ARMS')
    s=p['simulation']; e=s['excitation']; v=p['evaluation']
    if (s['domain']!='thermal' or s['bsk']!='2.11.0'
        or s['native_module']!='Basilisk.simulation.sensorThermal.SensorThermal'
        or (s['duration_s'],s['sample_s'],s['solver_s'],s['onset_s'])!=(14400,20,2,7200)
        or s['fault_labels']!=[0,1,2,3,4]
        or (s['max_nominal_temperature_k'],s['min_nominal_temperature_k'])!=(450.0,250.0)
        or s['same_physical_world_across_arms'] is not True
        or (e['segment_s'], tuple(e['multipliers']),e['component_b_shift_segments'])!=(SEGMENT_S, LEVELS, 3)
        or e['independent_of_fault_truth'] is not True):
        raise ValueError('R66_PHYSICS_OR_SCHEDULE_CHANGED')
    if (v['development_only'] is not True or v['no_training_or_model_selection'] is not True
        or v['deployment_fields']!=list(HEADER) or v['blind_worlds_opened']!=0
        or v['no_qualification_claim'] is not True
        or v['paired_same_noise_oracle_separate'] is not True
        or v['independent_observation_noise'] is not True
        or v['primary_metric']!='median_scaled_residual_fraction_B_minus_A'
        or v['exploratory_success_threshold_absolute']!=0.02
        or v['support_threshold_pairs_positive']!=6
        or v['min_temperature_span_increase_k']!=5.0
        or v['unit_of_independence']!='physical_world_id'):
        raise ValueError('R66_EVALUATION_CONTRACT_CHANGED')


def evaluate(p:dict, observations:dict)->dict:
    a=observations['A']; b=observations['B']
    pairs=[]
    for x,y in zip(a['components'],b['components']):
        assert x['component']==y['component']
        pairs.append({'component':x['component'],
            'scaled_residual_delta_B_minus_A':y['postfault_scaled_residual_fraction']-x['postfault_scaled_residual_fraction'],
            'prefault_span_delta_k':y['prefault_nominal_temperature_span_k']-x['prefault_nominal_temperature_span_k'],
            'postfault_span_delta_k':y['postfault_nominal_temperature_span_k']-x['postfault_nominal_temperature_span_k']})
    return {'world_id':p['world_id'],'components':pairs}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    if args.output.exists():parser.error('OUTPUT_MUST_BE_NEW')
    raw=PROTOCOL_FILE.read_bytes();plan=json.loads(raw);validate_protocol(plan)
    if md.version('bsk').split('+')[0]!='2.11.0':raise ValueError('BASILISK_VERSION_MISMATCH')
    args.output.mkdir(parents=True)
    (args.output/'PROTOCOL_FROZEN_COPY.json').write_bytes(raw)
    manifest={'PROTOCOL_FROZEN_COPY.json':sha(raw)}
    records=[];per_world=[];params={}
    for world in plan['world_ids']:
        p=parameters(world,'thermal');p['onset_s']=plan['simulation']['onset_s']
        params[world]=p
        runs={}; reports={}
        for arm,profile in plan['arms'].items():
            paired=[]
            for label in plan['simulation']['fault_labels']:
                s=run_native('thermal',p,label,thermal_excitation=profile,thermal_duration_s=14400)
                if len(s['observations'])!=721 or any(z != [int(i*20e9)]*2 for i,z in enumerate(s['output_times_ns'])):
                    raise AssertionError('R66_NATIVE_TIMESTAMP_OR_ROWS_FAILED')
                arr=np.asarray(s['observations'],dtype=float)
                if (not np.all(np.isfinite(arr))
                    or arr[:,4:6].max()>plan['simulation']['max_nominal_temperature_k']
                    or arr[:,4:6].min()<plan['simulation']['min_nominal_temperature_k']):
                    raise AssertionError('R66_TEMPERATURE_SAFETY_BOUND')
                paired.append(s)
                name=f'research_oracle_pairs/{arm}/{world}/L{label}.csv'
                manifest[name]=write_csv(args.output/name,s['observations'])
            report=assess_world(paired,p,arm,world)
            runs[arm]=paired;reports[arm]=report
            for label in plan['simulation']['fault_labels']:
                independent=dict(p)
                independent['measurement_seed']=int(sha(f'r66:{world}:{arm}:L{label}:noise'.encode())[:16],16)
                s=run_native('thermal',independent,label,thermal_excitation=profile,thermal_duration_s=14400)
                name=f'deployment_observations_dev_only/{arm}/{world}/L{label}.csv'
                manifest[name]=write_csv(args.output/name,s['observations'])
        summary=evaluate({'world_id':world},reports)
        summary['arms']=reports
        per_world.append(summary)
        print(f'R66_world_completed={world}',flush=True)
    pairs=[pair for w in per_world for pair in w['components']]
    delta=np.asarray([x['scaled_residual_delta_B_minus_A'] for x in pairs])
    spans=np.asarray([x['prefault_span_delta_k'] for x in pairs])
    def agg(arm,key):
        d=[component[key] for w in per_world for component in next(a for a in w['arms'].values() if a['arm']==arm)['components']]
        return {'median':float(np.median(d)),'min':float(np.min(d)),'max':float(np.max(d))}
    positive=int(np.count_nonzero(delta>0))
    meets=(float(np.median(delta))>=plan['evaluation']['exploratory_success_threshold_absolute']
           and positive>=plan['evaluation']['support_threshold_pairs_positive']
           and float(np.median(spans))>=plan['evaluation']['min_temperature_span_increase_k'])
    report={'schema':'phm.r66.native-timescale-pilot.v1',
        'status':'DEVELOPMENT_ONLY_NOT_QUALIFIED',
        'protocol_sha256':sha(raw),'bsk_version':md.version('bsk'),'python':sys.version.split()[0],
        'development_worlds_per_arm':4, 'paired_native_trajectories':40,
        'independent_noise_native_trajectories':40,'blind_worlds_opened':0,
        'model_training_performed':False,'diagnostic_qualification':'NOT_EVALUATED',
        'worlds':per_world,
        'aggregate':{'A':{key:agg('A',key) for key in ('postfault_mechanism_pearson','postfault_scaled_residual_fraction','prefault_nominal_temperature_span_k','postfault_constant_vs_t4_condition_number')},
                     'B':{key:agg('B',key) for key in ('postfault_mechanism_pearson','postfault_scaled_residual_fraction','prefault_nominal_temperature_span_k','postfault_constant_vs_t4_condition_number')}},
        'comparison':{'median_scaled_residual_delta_B_minus_A':float(np.median(delta)),
                      'positive_component_pairs':positive,'component_pairs':len(pairs),
                      'median_prefault_temperature_span_delta_k':float(np.median(spans)),
                      'exploratory_physics_thresholds_met':bool(meets),
                      'note':'descriptive same-noise oracle counterfactual comparison, NOT diagnostic performance or qualified confidence interval'},
        'limits':['no new training/calibration/blind worlds','known fault time exists in research oracle only',
                  'two uncoupled virtual SensorThermal nodes, no flight qualification']}
    manifest['R66_DEV_RESULTS.json']=write_json(args.output/'R66_DEV_RESULTS.json',report)
    manifest['SIMULATOR_PARAMETERS_ONLY.json']=write_json(args.output/'SIMULATOR_PARAMETERS_ONLY.json',{'role':'SIMULATOR_ONLY_NOT_DEPLOYMENT','parameters':params})
    write_json(args.output/'MANIFEST_SHA256.json',{'schema':'phm.r66.dev-evidence-manifest.v1','files':manifest})
    print(json.dumps({'status':report['status'],'comparison':report['comparison'],'aggregate':report['aggregate']},indent=2),flush=True)

if __name__=='__main__':main()
