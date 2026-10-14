"""Explicit, unqualified simulation-domain R57 feature/model API.

Declaring a task is a use-case claim, not an inference from fault truth.
Never claim RC or Stable based solely on the bundled developer models.
"""
from __future__ import annotations
from importlib.resources import files
from pathlib import Path
import csv, json, hashlib
import joblib
import numpy as np
from .platform_features import spec
from .platform_features_r57 import registry
from .open_design_fit import predict,design,TrainingExample

RESOURCES=files("phm_design").joinpath("limited_models")

def contract():
    return json.loads(RESOURCES.joinpath("scope_contract.json").read_text())

def allowed_use_case(case):
    profile=contract()
    if case in profile['proposed_certification_cells']:
        _checked_release_certificate()
        return {'case':case,'status':'SIMULATION_DOMAIN_RELEASE_QUALIFIED','release_tier':contract()['release_tier'],'simulation_only':True}
    if case in profile['out_of_claim_unqualified_cells']:
        return {'case':case,'status':'OUT_OF_CLAIM_NOT_QUALIFIED','simulation_only':True}
    raise ValueError('UNKNOWN_USE_CASE')

def model_manifest():
    return json.loads(RESOURCES.joinpath("model_manifest.json").read_text())

def open_bundled(domain):
    entry=model_manifest()['models'].get(domain)
    if not entry: raise ValueError('NO_TRAINING_SELECTED_BUNDLED_MODEL_FOR_DOMAIN')
    model_path=RESOURCES.joinpath(entry['file'])
    content=model_path.read_bytes()
    if hashlib.sha256(content).hexdigest()!=entry['sha256']:raise ValueError('BUNDLED_MODEL_HASH_MISMATCH')
    # Only trusted package-managed files. Arbitrary user pickle is never opened.
    with model_path.open('rb') as f:
        model=joblib.load(f)
    if model.get('platform_domain')!=domain:raise ValueError('BUNDLED_MODEL_DOMAIN_MISMATCH')
    return model

def rows_from_csv(path,domain):
    with Path(path).open(newline='') as f:
        rdr=csv.DictReader(f)
        required=tuple(spec(domain).fields)
        if tuple(rdr.fieldnames or ())!=required:raise ValueError('EXACT_DEPLOYMENT_FIELDS_REQUIRED')
        rows=[[float(r[k]) for k in required] for r in rdr]
    return spec(domain).array(rows)

def predict_bundle(domain,case,observations,*,healthy_prefix_confirmed):
    if case.split(':',1)[0]!=domain:raise ValueError('USE_CASE_DOMAIN_MISMATCH')
    scope=allowed_use_case(case)
    if scope['status']!='SIMULATION_DOMAIN_RELEASE_QUALIFIED':raise ValueError('OUT_OF_CLAIM_USE_CASE')
    model=open_bundled(domain)
    a=rows_from_csv(observations,domain)
    result=predict(model,a,registry(domain),healthy_prefix_confirmed=healthy_prefix_confirmed)
    if result['status']!='UNKNOWN' and result['prediction'] is not None and result['prediction']>=0:
        result={**result,'qualified':True,'status':'SIMULATION_DOMAIN_QUALIFIED_PREDICTION'}
    return {'use_case':case,'scope_status':scope['status'],
            'qualification':scope['release_tier']+'_SIMULATION_ONLY','model_domain':domain,**result}

def design_extended(dataset,output,*,max_trials=24):
    source=Path(dataset).resolve()
    data=json.loads(source.read_text())
    domain=data['domain'];obs_spec=spec(domain)
    examples=[]
    for item in data['records']:
        if item.get('role')!='design_train':raise ValueError('PROTECTED_ROLE_NOT_TRAINING_DATA')
        rel=Path(item['path'])
        path=(source.parent/rel).resolve()
        if not path.is_relative_to(source.parent) or not path.is_file():raise ValueError('DATASET_PATH_OUTSIDE_ROOT_OR_MISSING')
        rows=rows_from_csv(path,domain)
        examples.append(TrainingExample(rows,int(item['label']),str(item['world_id']),healthy_prefix_confirmed=item.get('healthy_prefix_confirmed',False)))
    model,report=design(examples,obs_spec,tuple(data['classes']),data['healthy_label'],registry=registry(domain),max_trials=max_trials)
    model['platform_domain']=domain
    folder=Path(output)
    folder.mkdir(parents=True,exist_ok=False)
    joblib.dump(model,folder/'model.joblib')
    (folder/'design_report.json').write_text(json.dumps(report,indent=2))
    return {'model':str(folder/'model.joblib'),'qualification':'DEVELOPMENT_ONLY','trials_not_hard_cpu_budget':True}

def _checked_release_certificate():
    import hashlib,json
    p=RESOURCES.joinpath('qualification_certificate.json')
    data=p.read_bytes()
    if hashlib.sha256(data).hexdigest()!=contract()['qualification_certificate_sha256']:
        raise ValueError('RELEASE_QUALIFICATION_CERTIFICATE_TAMPERED')
    cert=json.loads(data)
    if cert.get('contract_id')!=contract()['contract_id'] or cert.get('calibration_passed')!=15 or cert.get('sealed_passed')!=15:
        raise ValueError('RELEASE_QUALIFICATION_CERTIFICATE_INVALID')
    return cert
