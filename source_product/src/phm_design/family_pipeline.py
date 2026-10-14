"""Versioned non-ADCS normal-design passports and executable preprocessing.

Same original graph designer/trainer; no fault target/onset/severity inputs.
Passports are exported BEFORE native simulation. Their availability and exact
values are research assumptions, not a hardware calibration claim. Old v1
mechanical physical-program compiler and old model hashes remain unchanged.
"""
from __future__ import annotations
import copy
import csv
import math
import shutil
import tempfile
from pathlib import Path
import numpy as np
from .adapters import sat_nonadcs as native
from .data import Dataset,sequence_from_csv,validate_contract,transform
from .engine import load_ensemble,ensemble_scores,deterministic_cpu
from .family_design import design_family
from .io import digest,read_json,write_json,sha_file,seal,verify,safe_child

SCHEMA='phm.normal-design-passport.v1'
TRANSFORM='phm.nonadcs-features.v1'
CONTEXT={
 'eps':[('normal_full_light_power_w','W')],
 'thermal':[('normal_heat_capacity_j_per_k','J/K'),('normal_absorbed_full_light_w','W'),('normal_radiation_coefficient_w_per_k4','W/K^4')],
 'data':[('normal_storage_capacity_bits','bit')],
}
PHYSICAL={
 'eps':[('solar_loss_fraction','1'),('load_innovation_w','W'),('solar_loss_running_mean','1'),('solar_observable','1')],
 'thermal':[('energy_residual_running_w','W'),('radiation_loss_slope','1'),('unexplained_heat_intercept_w','W'),('thermal_regression_computable','1')],
 'data':[('generation_excess_running_bps','bit/s'),('contact_innovation_running_bps','bit/s'),('service_loss_fraction','1'),('queue_balance_computable','1')],
}
CONTROL={
 'eps':[('panel_change_w','W'),('bus_change_w','W'),('panel_rate_w_per_s','W/s'),('illumination_copy','1')],
 'thermal':[('temperature_change_k','K'),('temperature_increment_k','K'),('temperature_four_step_change_k','K'),('heat_command_change_w','W')],
 'data':[('queue_change_bits','bit'),('queue_rate_bps','bit/s'),('generation_change_bps','bit/s'),('contact_copy','1')],
}


def implementation_sha():return sha_file(Path(__file__))


def make_passport(world:str,domain:str,p:dict)->dict:
    """Strict allowlist: never copy all simulator parameters into the passport."""
    if domain=='eps':values=[1361.0*p['panel_area_m2']*p['panel_efficiency']]
    elif domain=='thermal':values=[p['mass_kg']*890.0,1361.0*p['thermal_area_m2']*p['absorptivity'],5.670374419e-8*p['thermal_area_m2']*p['emissivity']]
    elif domain=='data':values=[25000000.0]
    else:raise ValueError('UNKNOWN_FAMILY')
    result={'schema':SCHEMA,'domain':domain,'instance_lineage':world,
      'available_before_fault_injection':True,'source':{'reference':'pre_injection_design_allowlist',
      'exporter_sha256':implementation_sha()},'parameters':{
      n:{'value':v,'unit':u,'role':'normal_design_parameter'} for (n,u),v in zip(CONTEXT[domain],values)},
      'scope':'exact_nominal_simulator_design_known_in_advance; not hardware calibrated'}
    validate_passport(result,domain);return result


def validate_passport(p:dict,domain:str)->list[float]:
    fields={'schema','domain','instance_lineage','available_before_fault_injection','source','parameters','scope'}
    if not isinstance(p,dict) or set(p)!=fields or p['schema']!=SCHEMA or p['domain']!=domain:
        raise ValueError('PASSPORT_SCHEMA_OR_DOMAIN')
    if p['available_before_fault_injection'] is not True or not isinstance(p['instance_lineage'],str) or not p['instance_lineage']:
        raise ValueError('NORMAL_PREINJECTION_PASSPORT_REQUIRED')
    s=p['source']
    if not isinstance(s,dict) or set(s)!={'reference','exporter_sha256'} or not s['reference'] or len(s['exporter_sha256'])!=64:
        raise ValueError('PASSPORT_PROVENANCE_REQUIRED')
    if not isinstance(p['parameters'],dict) or set(p['parameters'])!={n for n,_ in CONTEXT[domain]}:
        raise ValueError('UNDECLARED_PARAMETER_OR_FAULT_TRUTH')
    values=[]
    for n,u in CONTEXT[domain]:
        entry=p['parameters'][n]
        if not isinstance(entry,dict) or set(entry)!={'value','unit','role'} or entry['unit']!=u or entry['role']!='normal_design_parameter':
            raise ValueError('PASSPORT_UNIT_OR_ROLE')
        v=entry['value']
        if type(v) not in (float,int) or not math.isfinite(v) or v<=0:raise ValueError('PASSPORT_FINITE_POSITIVE_VALUES')
        values.append(float(v))
    if not isinstance(p['scope'],str) or not p['scope']:raise ValueError('PASSPORT_SCOPE')
    return values


def feature_contract(domain:str,mode:str)->dict:
    if mode not in {'context','mechanism'}:raise ValueError('UNKNOWN_FEATURE_MODE')
    c=copy.deepcopy(native.contract(domain))
    c['device_family']+=f'.normal_passport.{mode}.v1'
    c['scope']+='; exact normal-design passport assumed available before injection; derived causal window features, uncalibrated; complete observations only'
    for name,unit in CONTEXT[domain]:c['channels'].append({'name':name,'unit':unit,'role':'deployment_input','provenance':'Pre-injection nominal design metadata, idealized availability; not fault-updated state'})
    for name,unit in (PHYSICAL if mode=='mechanism' else CONTROL)[domain]:c['channels'].append({'name':name,'unit':unit,'role':'deployment_input','provenance':'Deterministic causal calculation from this observed window and normal passport only; not simulator truth'})
    validate_contract(c);return c


def feature_binding(domain:str,mode:str)->dict:
    b=copy.deepcopy(native.binding(domain));b['contract']=feature_contract(domain,mode)
    b['groups'] += [{'id':'normal_design','channels':[n for n,_ in CONTEXT[domain]]},
                   {'id':'derived_information','channels':[n for n,_ in (PHYSICAL if mode=='mechanism' else CONTROL)[domain]]}]
    b['mechanism_neural_branch_enabled']=mode=='mechanism'
    b['feature_pipeline']=TRANSFORM
    return b


def _running_mean(value,valid):
    count=np.cumsum(valid.astype(float));total=np.cumsum(np.where(valid,value,0.0))
    return np.divide(total,count,out=np.zeros_like(total),where=count>0),count


def compute_features(domain:str,raw:np.ndarray,passport:dict,mode:str)->np.ndarray:
    """Causal only within supplied window; no fit, labels, onset, pair baseline.

    Native factory stores the forcing held on (t[k-1],t[k]] at row k. Thermal
    interval integration uses that right-indexed forcing; emitted T^4 uses
    endpoint trapezoids. Unit schema is domain-specific, NOT generic SI proof.
    """
    values=validate_passport(passport,domain)
    x=np.asarray(raw,dtype=np.float64)
    if x.ndim!=2 or x.shape[0]<2 or x.shape[1]!=len(native.FAMILIES[domain]['channels']) or not np.isfinite(x).all():
        raise ValueError('INVALID_COMPLETE_OBSERVATION_SHAPE_OR_VALUES')
    if mode not in {'context','mechanism'}:raise ValueError('UNKNOWN_FEATURE_MODE')
    # Boundary and command values are physical, not standardized network data.
    if np.any((x[:,0]<0)|(x[:,0]>1)):raise ValueError('BOUNDARY_VALUE_OUT_OF_RANGE')
    dt=float(native.FAMILIES[domain]['sample_s']);n=len(x)
    delta=x-x[:1];inc=np.vstack((np.zeros((1,x.shape[1])),np.diff(x,axis=0)))
    if mode=='context':
        if domain=='eps':f=np.column_stack((delta[:,2],delta[:,3],inc[:,2]/dt,x[:,0]))
        elif domain=='thermal':
            slow=x[:,2]-x[np.maximum(0,np.arange(n)-4),2]
            f=np.column_stack((delta[:,2],inc[:,2],slow,delta[:,1]))
        else:f=np.column_stack((delta[:,3],inc[:,3]/dt,delta[:,1],x[:,0]))
    elif domain=='eps':
        power=values[0];valid=x[:,0]>.05;expected=power*x[:,0]
        loss=np.divide(expected-x[:,2],expected,out=np.zeros(n),where=valid)
        mean,_=_running_mean(loss,valid)
        f=np.column_stack((loss,x[:,2]-x[:,3]-x[:,1],mean,valid.astype(float)))
    elif domain=='thermal':
        C,solar,rad=values;T=x[:,2]
        if np.any((T<=0)|(T>2000)):raise ValueError('ABSOLUTE_TEMPERATURE_REQUIRED')
        qrad=rad*T**4
        # Nonoverlapping single-step balance residuals, then prefix regression.
        radiation=.5*(qrad[1:]+qrad[:-1])
        residual=C*np.diff(T)/dt-(solar*x[1:,0]+x[1:,1]-radiation)
        count=np.arange(1,n,dtype=float);sumq=np.cumsum(radiation);sumr=np.cumsum(residual)
        qmean=sumq/count;rmean=sumr/count
        var=np.cumsum(radiation**2)/count-qmean**2
        cov=np.cumsum(radiation*residual)/count-qmean*rmean
        good=(count>=8)&(var>np.maximum(.01,1e-4*qmean**2))
        slope=np.divide(cov,var,out=np.zeros_like(var),where=good)
        intercept=np.where(good,rmean-slope*qmean,0.0)
        f=np.vstack((np.zeros((1,4)),np.column_stack((rmean,slope,intercept,good.astype(float)))))
    else:
        q=x[:,3];cap=values[0]
        if np.any((q<0)|(q>cap)):raise ValueError('QUEUE_OUTSIDE_DESIGN_CAPACITY')
        prev=np.r_[q[0],q[:-1]]
        valid=(q>0)&(prev>0)&(q<cap)&(prev<cap);valid[0]=False
        innovation=inc[:,3]/dt-(x[:,1]-x[:,0]*x[:,2])
        off=(x[:,0]<.5)&valid;on=(x[:,0]>=.5)&valid
        gen,ng=_running_mean(innovation,off);con,nc=_running_mean(innovation,on)
        denom=np.maximum(x[:,2],1.0);have=(ng>0)&(nc>0)
        loss=np.where(have,(con-gen)/denom,0.0)
        f=np.column_stack((gen,con,loss,have.astype(float)))
    context=np.broadcast_to(values,(n,len(values)))
    y=np.concatenate((x,context,f),axis=1)
    if not np.isfinite(y).all():raise ValueError('NONFINITE_DERIVED_FEATURE')
    out=y.astype(np.float32)
    if not np.isfinite(out).all():raise ValueError('FEATURE_OUTSIDE_FLOAT32')
    return out


def generate_with_passports(config_path,output,on_case=None):
    config=read_json(config_path);plan=native.plan_campaign(config);out=Path(output)
    out.mkdir(parents=True,exist_ok=False)
    passports={}
    for item in plan:
        key=item['domain']+'/'+item['world_id'];p=make_passport(item['world_id'],item['domain'],item['parameters'])
        rel='passports/'+key+'.json';write_json(out/rel,p)
        passports[key]={'path':rel,'sha256':sha_file(out/rel)}
    write_json(out/'passport_registry.json',{'schema':'phm.passport-registry.v1','config_sha256':digest(config),'passports':passports,'frozen_before_simulation':True,'definition_sha256':implementation_sha()})
    summary=native.generate_nonadcs(config_path,out/'native',on_case=on_case)
    write_json(out/'summary.json',summary)
    seal(out,[str(p.relative_to(out)) for p in out.rglob('*') if p.is_file() and p.name not in {'.lock','manifest.json'}])
    return summary


def prepare_family_section(campaign,domain,section,mode,output):
    if section not in {'development','calibration','holdout'}:raise ValueError('SECTION_INVALID')
    root=Path(campaign);registry=read_json(root/'passport_registry.json')
    if registry['schema']!='phm.passport-registry.v1' or not registry['frozen_before_simulation']:raise ValueError('PASSPORT_NOT_FROZEN')
    roles={'development':{'train','validation'},'calibration':{'calibration'},'holdout':{'test'}}[section]
    data=Dataset(root/'native'/domain/section/'dataset.json',allowed_splits=roles);data.verify_files()
    if data.contract!=native.contract(domain):raise ValueError('RAW_DEVICE_CONTRACT_CHANGED')
    window=native.FAMILIES[domain]['window'];out=Path(output);out.mkdir(parents=True,exist_ok=False)
    samples=[];records=[]
    for s in data.samples:
        pr=registry['passports'][domain+'/'+s['lineage_id']];pfile=safe_child(root,pr['path'])
        if sha_file(pfile)!=pr['sha256']:raise ValueError('PASSPORT_FILE_CHANGED')
        passport=read_json(pfile)
        if passport['instance_lineage']!=s['lineage_id']:raise ValueError('PASSPORT_INSTANCE_MISMATCH')
        raw=sequence_from_csv(safe_child(data.root,s['path']),data.contract,window)
        y=compute_features(domain,raw,passport,mode)
        relative='observations/'+s['id']+'.csv';path=out/relative;path.parent.mkdir(exist_ok=True)
        with path.open('x',newline='') as f:
            w=csv.writer(f);w.writerow(['time_s',*[c['name'] for c in feature_contract(domain,mode)['channels']]])
            for i,row in enumerate(y):w.writerow([i*data.contract['sample_s'],*map(float,row)])
        samples.append({**s,'path':relative,'sha256':sha_file(path)})
        records.append({'id':s['id'],'passport_sha256':pr['sha256'],'raw_sha256':s['sha256']})
    manifest={'schema':'phm.sequence-dataset.v1','dataset_id':digest([data.hash,mode,TRANSFORM,implementation_sha()]),
      'contract':feature_contract(domain,mode),'samples':samples,'provenance':{'raw_dataset_hash':data.hash,'window':window,'transform':TRANSFORM,'transform_sha256':implementation_sha(),'source_role':'trusted_preparation; no learning','passport_role':'pre_injection_nominal','selected_on_test':False}}
    write_json(out/'dataset.json',manifest);write_json(out/'device_binding.json',feature_binding(domain,mode));write_json(out/'source_records.json',records)
    Dataset(out/'dataset.json',allowed_splits=roles).verify_files()
    seal(out,[str(p.relative_to(out)) for p in out.rglob('*') if p.is_file()]);return manifest


def attach_pipeline(selected,domain,mode,output):
    """Export deployable wrapper; never place it inside a sealed source run."""
    selected=Path(selected);verify(selected);out=Path(output);out.mkdir(parents=True,exist_ok=False)
    shutil.copytree(selected,out/'neural',ignore=shutil.ignore_patterns('.lock','evaluation_ledger.json'))
    bundle=read_json(out/'neural/model_bundle.json')
    if bundle['contract']!=feature_contract(domain,mode):raise ValueError('PIPELINE_NEURAL_CONTRACT_MISMATCH')
    specification={'schema':'phm.family-inference-pipeline.v1','domain':domain,'mode':mode,
      'transform':TRANSFORM,'transform_sha256':implementation_sha(),'raw_contract':native.contract(domain),
      'window':bundle['shape']['window'],'neural_manifest_sha256':sha_file(out/'neural/manifest.json'),
      'passport_schema':SCHEMA,'normal_parameters_known_exactly_in_research':True,
      'missing_observations':'REJECTED','production_qualified':False}
    write_json(out/'pipeline.json',specification)
    seal(out,[str(p.relative_to(out)) for p in out.rglob('*') if p.is_file()]);return specification


def predict_pipeline(pipeline,csv_path,passport_path):
    out=Path(pipeline);verify(out);p=read_json(out/'pipeline.json')
    if p['schema']!='phm.family-inference-pipeline.v1' or p['transform_sha256']!=implementation_sha():raise ValueError('PIPELINE_IMPLEMENTATION_CHANGED')
    if p['raw_contract']!=native.contract(p['domain']):raise ValueError('PIPELINE_RAW_CONTRACT_CHANGED')
    raw=sequence_from_csv(Path(csv_path),p['raw_contract'],p['window'])
    features=compute_features(p['domain'],raw,read_json(passport_path),p['mode'])
    with deterministic_cpu():
        bundle,models=load_ensemble(out/'neural');scores=ensemble_scores(models,transform(features[None],bundle['scaler']))[0]
    best=int(scores.argmax());accepted=bool(scores[best]>=bundle['abstention']['min_uncalibrated_score'])
    return {'status':'PREDICTED' if accepted else 'ABSTAIN_LOW_SCORE','prediction':bundle['contract']['classes'][best] if accepted else None,
      'closed_set_argmax':bundle['contract']['classes'][best],'scores':scores.tolist(),'probability_calibrated':False,
      'transform_sha256':p['transform_sha256'],'claim_status':'RESEARCH_ONLY'}
