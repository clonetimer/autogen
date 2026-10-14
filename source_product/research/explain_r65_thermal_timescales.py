"""R65 POST-HOC physical explanation, not a preregistered acceptance metric.

Estimate local radiative thermal time scales from already generated native
parameters. Never trains a model, reads holdout or changes the frozen pilot.
The formula assumes radiation-only linearization near initial temperature;
Basilisk dynamics also include prescribed sunlight and commanded heaters.
"""
from __future__ import annotations
import argparse
import json
import statistics
from pathlib import Path

STEFAN_BOLTZMANN_W_M2_K4=5.670374419e-8
SPECIFIC_HEAT_J_KG_K=890.0

def explain(parameters:dict)->dict:
    records=[]
    for world in sorted(parameters):
        p=parameters[world]
        for component in ('a','b'):
            mass=p[f'thermal_{component}_mass_kg']
            emissivity=p[f'thermal_{component}_emissivity']
            area=p[f'thermal_{component}_area_m2']
            temp=p[f'thermal_{component}_initial_k']
            if min(mass,emissivity,area,temp)<=0: raise ValueError('NONPHYSICAL_THERMAL_PARAMETER')
            capacity=mass*SPECIFIC_HEAT_J_KG_K
            conductance=4*emissivity*STEFAN_BOLTZMANN_W_M2_K4*area*temp**3
            tau=capacity/conductance
            records.append({'world_id':world,'component':component,
                            'heat_capacity_j_k':capacity,
                            'linearized_radiation_w_k':conductance,
                            'estimated_radiative_tau_s':tau,
                            'segment_to_tau_ratio':220.0/tau})
    taus=[r['estimated_radiative_tau_s'] for r in records]
    return {'schema':'phm.r65.posthoc-radiative-time-scale.v1',
            'role':'POST_HOC_EXPLANATORY_HYPOTHESIS_NOT_PREREGISTERED_ACCEPTANCE',
            'model':'radiative-only small-perturbation linearization near T0',
            'segment_s':220,'records':records,
            'tau_min_s':min(taus),'tau_median_s':statistics.median(taus),
            'tau_max_s':max(taus),
            'limitation':'No proof of structural observability or improved classifier risk; sun input and heaters are omitted from local tau formula.',
            'new_blind_worlds':0,'diagnostic_qualification':'NOT_EVALUATED'}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--simulator-parameters',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('OUTPUT_ALREADY_EXISTS')
    d=json.loads(args.simulator_parameters.read_text())
    args.output.write_text(json.dumps(explain(d['parameters']),indent=2)+'\n')

if __name__=='__main__':main()
