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
from . import __version__

RESOURCES=files("phm_design").joinpath("limited_models")

def _release_contract():
    """Read immutable 0.7 contract as historical evidence, not a current claim."""
    return json.loads(RESOURCES.joinpath("scope_contract.json").read_text())


def contract():
    """Return the *active package* scope, with the frozen certificate provenance.

    An embedded 0.7 certificate is not an 0.8 development qualification.
    Stable's bytes are never rewritten and its historical contract stays intact.
    """
    historical = _release_contract()
    active = __version__ == historical['release_version']
    return {**historical,
            'bundled_legacy_release_version': historical['release_version'],
            'bundled_legacy_release_tier': historical['release_tier'],
            'active_package_version': __version__,
            'release_version': __version__,
            'release_tier': historical['release_tier'] if active else 'DEVELOPMENT',
            'deployment_qualification_status': (
                historical['deployment_qualification_status'] if active else 'DEVELOPMENT_NOT_QUALIFIED'),
            'qualification_applies_to_current_package': active,
            'current_package_qualified_cells': (historical['proposed_certification_cells'] if active else []),
            'bundled_legacy_certificate_only': not active}


def allowed_use_case(case):
    profile=contract()
    if case in profile['proposed_certification_cells']:
        _checked_release_certificate()  # authenticate legacy provenance, not active qualification
        if profile['qualification_applies_to_current_package']:
            return {'case':case,'status':'SIMULATION_DOMAIN_RELEASE_QUALIFIED',
                    'release_tier':profile['release_tier'],'simulation_only':True,'qualified':True}
        return {'case':case,'status':'LEGACY_MODEL_DEVELOPMENT_ONLY',
                'release_tier':'DEVELOPMENT','simulation_only':True,'qualified':False,
                'bundled_release_version':profile['bundled_legacy_release_version'],
                'active_package_version':__version__}
    if case in profile['out_of_claim_unqualified_cells']:
        return {'case':case,'status':'OUT_OF_CLAIM_NOT_QUALIFIED','simulation_only':True,'qualified':False}
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
    if scope['status'] not in ('SIMULATION_DOMAIN_RELEASE_QUALIFIED','LEGACY_MODEL_DEVELOPMENT_ONLY'):
        raise ValueError('OUT_OF_CLAIM_USE_CASE')
    model=open_bundled(domain)
    a=rows_from_csv(observations,domain)
    result=predict(model,a,registry(domain),healthy_prefix_confirmed=healthy_prefix_confirmed)
    active=scope['qualified']
    if result['status']!='UNKNOWN' and result['prediction'] is not None and result['prediction']>=0:
        result={**result,'qualified':active,
                'status':'SIMULATION_DOMAIN_QUALIFIED_PREDICTION' if active else 'LEGACY_MODEL_RESEARCH_PREDICTION'}
    else:
        result={**result,'qualified':False}
    return {'use_case':case,'scope_status':scope['status'],
            'qualification':scope['release_tier']+'_SIMULATION_ONLY' if active else 'DEVELOPMENT_ONLY_LEGACY_0_7_MODEL',
            'active_package_version':__version__,
            'bundled_release_version':contract()['bundled_legacy_release_version'],
            'model_domain':domain,**result}

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
    if hashlib.sha256(data).hexdigest()!=_release_contract()['qualification_certificate_sha256']:
        raise ValueError('RELEASE_QUALIFICATION_CERTIFICATE_TAMPERED')
    cert=json.loads(data)
    if cert.get('contract_id')!=_release_contract()['contract_id'] or cert.get('calibration_passed')!=15 or cert.get('sealed_passed')!=15:
        raise ValueError('RELEASE_QUALIFICATION_CERTIFICATE_INVALID')
    return cert
