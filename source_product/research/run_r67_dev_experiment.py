"""R67 pre-registered, equal-energy, crossed-phase native Basilisk development-only trial.

Run from checkout: PYTHONPATH=src python research/run_r67_dev_experiment.py --output NEW_EMPTY_PATH
This program NEVER trains/qualifies a model or opens calibration/blind worlds.
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
from phm_design.adapters.sat_multicomponent import parameters, run_native
from phm_design.thermal_excitation_r67 import (
    CROSS_AB, CROSS_BA, CONSTANT, SEGMENT_S, LEVELS, DURATION_S, ONSET_S,
    commanded_heat, schedule_energy_audit,
)
try:
    from research.run_r65_dev_experiment import assess_world, HEADER
except ModuleNotFoundError:
    from run_r65_dev_experiment import assess_world, HEADER

PROTOCOL_FILE = Path(__file__).with_name('R67_DEV_PROTOCOL.json')


def sha(b: bytes)->str:
    return hashlib.sha256(b).hexdigest()


def write_json(path: Path, data: dict)->str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(data, indent=2, sort_keys=True, allow_nan=False, ensure_ascii=False)+'\n').encode()
    with path.open('xb') as f: f.write(payload)
    return sha(payload)


def write_csv(path: Path, rows: list)->str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', newline='', encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(HEADER);w.writerows(rows)
    return sha(path.read_bytes())


def validate_protocol(p: dict)->None:
    if (p.get('schema')!='phm.r67.phase-cross-isoenergy-thermal-dev.v1'
            or p.get('research_phase')!='DEVELOPMENT_PILOT_ONLY'
            or p.get('qualification')!='NOT_EVALUATED'
            or p.get('open_calibration_or_blind_worlds') is not False
            or p.get('model_training') is not False
            or p.get('world_ids') != [f'r67-thermal-phase-dev-{i:03d}' for i in range(1,5)]
            or p.get('arms')!={'X':CROSS_AB,'Y':CROSS_BA,'C':CONSTANT}):
        raise ValueError('R67_RESEARCH_ROLE_OR_ARM_TAMPER')
    s=p['simulation']; e=s['excitation']; v=p['evaluation']
    if (s['domain']!='thermal' or s['bsk']!='2.11.0'
            or s['native_module']!='Basilisk.simulation.sensorThermal.SensorThermal'
            or (s['duration_s'],s['solver_s'],s['sample_s'],s['onset_s'])!=(DURATION_S,2,20,ONSET_S)
            or s['fault_labels']!=[0,1,2,3,4] or s['same_physical_world_across_arms'] is not True
            or (s['min_nominal_temperature_k'],s['max_nominal_temperature_k'])!=(250.0,450.0)
            or e['segment_s']!=SEGMENT_S or tuple(e['multipliers'])!=LEVELS
            or e['component_phase_shifts_X']!=[0,3] or e['component_phase_shifts_Y']!=[3,0]
            or e['expected_each_half_normalized_energy_watt_seconds']!=7200.0
            or e['independent_of_fault_truth'] is not True):
        raise ValueError('R67_SIMULATION_OR_ENERGY_TAMPER')
    if (v['development_only'] is not True or v['blind_worlds_opened']!=0
            or v['no_training_or_model_selection'] is not True or v['no_qualification_claim'] is not True
            or v['paired_same_noise_oracle_separate'] is not True or v['independent_observation_noise'] is not True
            or v['deployment_fields']!=list(HEADER)
            or v['primary_metric']!='per_component_world_paired_phase3_minus_phase0_scaled_residual_fraction'
            or v['phase_materiality_absolute']!=0.02 or v['min_same_direction_worlds_out_of_4']!=3
            or v['unit_of_independence']!='physical_world_id'
            or v['require_both_components_opposite_sign_for_r66_asymmetry_attribution'] is not True
            or v['constant_control']!='descriptive_only_not_a_pass_replacement'
            or v['no_pooled_8_pair_inference'] is not True):
        raise ValueError('R67_EVALUATION_TAMPER')
    schedule_energy_audit()


def noise_observation_summary(rows:list)->dict:
    """Single-record-stream observables; no labels, fault time, or counterfactual."""
    x=np.asarray(rows,dtype=float)
    if x.shape != (721,6) or not np.all(np.isfinite(x)):
        raise AssertionError('R67_OBSERVATIONS_SHAPE_OR_FINITE')
    return {
        'temperature_range_k':[float(np.ptp(x[:,4])),float(np.ptp(x[:,5]))],
        'command_energy_ws':[float(20*np.sum(x[1:,2])),float(20*np.sum(x[1:,3]))],
        'temperature_first_difference_std_k':[
            float(np.std(np.diff(x[:,4]))),float(np.std(np.diff(x[:,5])))],
    }


def evaluate_world(world_id:str, reports:dict)->dict:
    comp={a:{c['component']:c for c in v['components']} for a,v in reports.items()}
    effects=[]
    for name,zero,three in [('a','X','Y'),('b','Y','X')]:
        x=comp[zero][name]; y=comp[three][name]; c=comp['C'][name]
        effects.append({'component':name,
            'phase0_arm':zero,'phase3_arm':three,
            'phase0_residual':x['postfault_scaled_residual_fraction'],
            'phase3_residual':y['postfault_scaled_residual_fraction'],
            'constant_control_residual':c['postfault_scaled_residual_fraction'],
            'phase3_minus_phase0_residual':y['postfault_scaled_residual_fraction']-x['postfault_scaled_residual_fraction'],
            'phase3_minus_phase0_nominal_prefault_span_k':y['prefault_nominal_temperature_span_k']-x['prefault_nominal_temperature_span_k'],
            'phase3_minus_phase0_nominal_postfault_span_k':y['postfault_nominal_temperature_span_k']-x['postfault_nominal_temperature_span_k']})
    return {'world_id':world_id,'component_phase_effects':effects,'arms':reports}


def aggregate(worlds:list, protocol:dict)->dict:
    details={}
    for name in ('a','b'):
        d=np.asarray([next(x for x in w['component_phase_effects'] if x['component']==name)['phase3_minus_phase0_residual'] for w in worlds])
        median=float(np.median(d)); sign=int(np.count_nonzero(d>0) if median>0 else np.count_nonzero(d<0)) if median!=0 else 0
        qualifies=(abs(median)>=protocol['evaluation']['phase_materiality_absolute']
                   and sign>=protocol['evaluation']['min_same_direction_worlds_out_of_4'])
        details[name]={'per_world_delta':d.tolist(), 'median_phase3_minus_phase0':median,
                       'same_direction_worlds':sign, 'world_count':len(worlds),
                       'material_phase_effect_thresholds_met':bool(qualifies)}
    a=details['a'];b=details['b']; both=a['material_phase_effect_thresholds_met'] and b['material_phase_effect_thresholds_met']
    opposite=a['median_phase3_minus_phase0']*b['median_phase3_minus_phase0']<0
    return {'per_component':details,
            'any_component_material_phase_effect':bool(a['material_phase_effect_thresholds_met'] or b['material_phase_effect_thresholds_met']),
            'both_components_material_opposite_sign':bool(both and opposite),
            'can_explain_r66_asymmetry_causally':False,
            'interpretation':'Dev oracle descriptive contrasts; four WORLD-level paired samples, not eight independent replicates. Cross-phase may indicate a confound but cannot retrospectively identify the cause of R66.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    if args.output.exists():parser.error('OUTPUT_MUST_BE_NEW')
    raw=PROTOCOL_FILE.read_bytes();plan=json.loads(raw);validate_protocol(plan)
    if md.version('bsk').split('+')[0]!='2.11.0':raise ValueError('BASILISK_VERSION_MISMATCH')
    energy=schedule_energy_audit()
    try:
        frozen_source_commit=subprocess.check_output(['git','-C',str(PROTOCOL_FILE.parent.parent),'rev-parse','HEAD'],text=True,stderr=subprocess.DEVNULL).strip()
    except Exception:
        frozen_source_commit=None
    args.output.mkdir(parents=True)
    (args.output/'PROTOCOL_FROZEN_COPY.json').write_bytes(raw)
    manifest={'PROTOCOL_FROZEN_COPY.json':sha(raw)}
    worlds=[];params={};observation_summaries=[]
    for world_id in plan['world_ids']:
        p=parameters(world_id,'thermal');p['onset_s']=ONSET_S;params[world_id]=p
        arm_reports={}
        for arm,profile in plan['arms'].items():
            reference=[]
            for label in plan['simulation']['fault_labels']:
                s=run_native('thermal',p,label,thermal_excitation=profile,thermal_duration_s=DURATION_S)
                if len(s['observations'])!=721 or any(z!=[int(i*20e9)]*2 for i,z in enumerate(s['output_times_ns'])):
                    raise AssertionError('R67_NATIVE_TIMESTAMPS')
                a=np.asarray(s['observations'],dtype=float)
                if (not np.all(np.isfinite(a)) or np.min(a[:,4:6])<250 or np.max(a[:,4:6])>450):
                    raise AssertionError('R67_TEMPERATURE_SAFETY')
                for component in (0,1):
                    # Real, positive-duration simulation intervals; row 0 is initialization.
                    heat_col=2+component
                    expected=energy[profile][str(component)]
                    for start in (0,360):
                        total=float(20*np.sum(a[start+1:start+361,heat_col]))
                        base=p['heat_a_w' if component==0 else 'heat_b_w']
                        if not np.isclose(total,expected[start//360]*base,atol=1e-7,rtol=1e-10):
                            raise AssertionError('R67_ACTUAL_COMMAND_ENERGY')
                reference.append(s)
                name=f'research_oracle_pairs/{arm}/{world_id}/L{label}.csv'
                manifest[name]=write_csv(args.output/name,s['observations'])
            arm_reports[arm]=assess_world(reference,p,arm,world_id)
            for label in plan['simulation']['fault_labels']:
                alt=dict(p)
                alt['measurement_seed']=int(sha(f'r67:{world_id}:{arm}:L{label}:noise'.encode())[:16],16)
                s=run_native('thermal',alt,label,thermal_excitation=profile,thermal_duration_s=DURATION_S)
                name=f'deployment_observations_dev_only/{arm}/{world_id}/L{label}.csv'
                manifest[name]=write_csv(args.output/name,s['observations'])
                observation_summaries.append({'world_id':world_id,'arm':arm,'case_id_dev_only':f'L{label}',
                                              'features':noise_observation_summary(s['observations'])})
        worlds.append(evaluate_world(world_id,arm_reports))
        print('R67_dev_world_complete='+world_id,flush=True)
    report={'schema':'phm.r67.native-phase-crossover-dev-results.v1',
            'status':'DEVELOPMENT_ONLY_NOT_QUALIFIED',
            'protocol_sha256':sha(raw),'source_commit_sha':frozen_source_commit,
            'bsk_version':md.version('bsk'),'python':sys.version.split()[0],
            'native_trajectories':120,'paired_oracle_trajectories':60,'independent_noise_trajectories':60,
            'physical_worlds':4,'worlds':worlds,'energy_by_arm_component_and_half_ws_per_watt':energy,
            'comparison':aggregate(worlds,plan),
            'blind_worlds_opened':0,'model_training_performed':False,
            'diagnostic_qualification':'NOT_EVALUATED',
            'limitations':['oracle fault time and same-noise counterfactuals are never inference features',
                           '2 uncoupled virtual native SensorThermal components',
                           'single schedule family and four development worlds are not conclusive',
                           'same energy does not imply matched instantaneous heating, state or temperature range']}
    manifest['R67_DEV_RESULTS.json']=write_json(args.output/'R67_DEV_RESULTS.json',report)
    manifest['SIMULATOR_PARAMETERS_ONLY.json']=write_json(args.output/'SIMULATOR_PARAMETERS_ONLY.json',
                             {'role':'SIMULATOR_ONLY_NOT_DEPLOYMENT','parameters':params})
    manifest['OBSERVATION_FEATURES_DEV_ONLY.json']=write_json(args.output/'OBSERVATION_FEATURES_DEV_ONLY.json',
                             {'role':'OBSERVATION_ONLY_NOT_MODEL_SCORE','features':observation_summaries})
    write_json(args.output/'MANIFEST_SHA256.json',{'schema':'phm.r67.dev-evidence-manifest.v1','files':manifest})
    print(json.dumps({'status':report['status'],'comparison':report['comparison']},indent=2),flush=True)

if __name__=='__main__':main()
