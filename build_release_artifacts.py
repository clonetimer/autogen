"""Create reproducible RC/Stable packaging from the frozen dev13 algorithm code.

The resulting packages only alter release-state metadata and the narrowly
scoped wrapper status; they cannot retrain or replace frozen models.
"""
from pathlib import Path
import json,hashlib,shutil,subprocess,sys
import setuptools.build_meta
ROOT=Path('/mnt/data/PHM_R60_LIMITED_15_20261011')
DEV=ROOT/'product/phm-design-0.7.0.dev13'
QUAL=ROOT/'qualification'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def get_certificate():
 c=json.loads((ROOT/'APPROVED_15_CELL_CONTRACT.json').read_text())
 cal=json.loads((QUAL/'CALIBRATION_SCORES.json').read_text());seal=json.loads((QUAL/'SEALED_SCORES.json').read_text())
 assert cal['all_claimed_cells_pass'] and seal['all_claimed_cells_pass']
 assert cal['passed_count']==seal['passed_count']==15
 assert cal['worlds']==300 and seal['worlds']==600
 assert all((ROOT/'evidence'/f'{name}.status').read_text().strip()=='PASS' for name in ['311_wheel','311_sdist','312_wheel','312_sdist'])
 manifest=json.loads((DEV/'src/phm_design/limited_models/model_manifest.json').read_text())
 models={d:sha(DEV/'src/phm_design/limited_models'/v['file']) for d,v in manifest['models'].items()}
 assert all(models[d]==v['sha256'] for d,v in manifest['models'].items())
 cert={
   'schema':'phm.simulation.qualification.certificate.v1','contract_id':c['contract_id'],
   'domain':'Basilisk 2.11.0 / Linux x86_64 / offline full-trajectory / trusted initial healthy segment',
   'out_of_scope':'not on-orbit, not causal streaming, not hardware safety certificate',
   'claimed_cells':sorted(c['proposed_certification_cells']),
   'unqualified_cells':sorted(c['out_of_claim_unqualified_cells']),
   'numerical_thresholds':c['qualification_thresholds'],
   'calibration_worlds':300,'sealed_worlds':600,'calibration_passed':15,'sealed_passed':15,
   'development_artifact_hashes':{
    'dev13_wheel':sha(next((DEV/'dist').glob('*.whl'))),
    'dev13_sdist':sha(next((DEV/'dist').glob('*.tar.gz'))),
    'dev13_contract':sha(ROOT/'APPROVED_15_CELL_CONTRACT.json'),
    'calibration_freeze':sha(QUAL/'PREQUALIFICATION_FREEZE.json'),
    'calibration_scores':sha(QUAL/'CALIBRATION_SCORES.json'),
    'sealed_plan':sha(QUAL/'SEALED_PLAN.json'),
    'sealed_freeze':sha(QUAL/'SEALED_FREEZE.json'),
    'sealed_scores':sha(QUAL/'SEALED_SCORES.json'),
   },
   'frozen_models_sha256':models,
   'product_matrix':{name:sha(ROOT/'evidence'/f'{name}_FULL_TEST.log') for name in ['311_wheel','311_sdist','312_wheel','312_sdist']},
   'no_qualification_data_model_refit':True,
   'issued_diagnostic_scope':'SIMULATION_DOMAIN_ONLY',
   'historical_R59_failure_preserved':True,
 }
 if not (ROOT/'QUALIFICATION_CERTIFICATE.json').exists():
  (ROOT/'QUALIFICATION_CERTIFICATE.json').write_text(json.dumps(cert,indent=2,sort_keys=True,ensure_ascii=False)+'\n')
 else: assert json.loads((ROOT/'QUALIFICATION_CERTIFICATE.json').read_text())==cert
 return cert

RELEASE_FUNCTION='''\ndef _checked_release_certificate():\n    import hashlib,json\n    p=RESOURCES.joinpath('qualification_certificate.json')\n    data=p.read_bytes()\n    if hashlib.sha256(data).hexdigest()!=contract()['qualification_certificate_sha256']:\n        raise ValueError('RELEASE_QUALIFICATION_CERTIFICATE_TAMPERED')\n    cert=json.loads(data)\n    if cert.get('contract_id')!=contract()['contract_id'] or cert.get('calibration_passed')!=15 or cert.get('sealed_passed')!=15:\n        raise ValueError('RELEASE_QUALIFICATION_CERTIFICATE_INVALID')\n    return cert\n'''

def build(version,tier):
 cert=get_certificate();dst=ROOT/f'product/phm-design-{version}'
 if dst.exists():raise FileExistsError('RELEASE_VERSION_ALREADY_CREATED')
 shutil.copytree(DEV,dst,ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.egg-info','dist','build','.pytest_cache'))
 c=json.loads((dst/'src/phm_design/limited_models/scope_contract.json').read_text())
 c.update(candidate_version=version,release_version=version,release_tier=tier,
   deployment_qualification_status='SIMULATION_DOMAIN_'+tier,calibration_openings_used=1,sealed_openings_used=1,
   qualification_certificate_sha256=sha(ROOT/'QUALIFICATION_CERTIFICATE.json'))
 cp=dst/'src/phm_design/limited_models/scope_contract.json';cp.write_text(json.dumps(c,indent=2,sort_keys=True,ensure_ascii=False)+'\n')
 certfile=dst/'src/phm_design/limited_models/qualification_certificate.json'
 certfile.write_bytes((ROOT/'QUALIFICATION_CERTIFICATE.json').read_bytes())
 manifest=json.loads((dst/'src/phm_design/limited_models/model_manifest.json').read_text())
 manifest['state']='SIMULATION_DOMAIN_'+tier
 for v in manifest['models'].values():v['qualified']=True
 (dst/'src/phm_design/limited_models/model_manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
 (dst/'src/phm_design/__init__.py').write_text(f'__version__ = "{version}"\n')
 p=dst/'pyproject.toml';p.write_text(p.read_text().replace('version = "0.7.0.dev13"',f'version = "{version}"'))
 # Changes allowed only in wrapper status: feature transformation, estimators,
 # training data, model weights and open_design_fit.predict all remain byte identical.
 wrapper=dst/'src/phm_design/platform_limited.py';s=wrapper.read_text()
 s=s.replace("return {'case':case,'status':'DEVELOPMENT_CANDIDATE_NOT_QUALIFIED','simulation_only':True}",
 "_checked_release_certificate()\n        return {'case':case,'status':'SIMULATION_DOMAIN_RELEASE_QUALIFIED','release_tier':contract()['release_tier'],'simulation_only':True}")
 s=s.replace("if scope['status']!='DEVELOPMENT_CANDIDATE_NOT_QUALIFIED':raise ValueError('OUT_OF_CLAIM_USE_CASE')",
 "if scope['status']!='SIMULATION_DOMAIN_RELEASE_QUALIFIED':raise ValueError('OUT_OF_CLAIM_USE_CASE')")
 s=s.replace("return {'use_case':case,'scope_status':scope['status'],\n            'qualification':'NOT_RC_NOT_STABLE','model_domain':domain,**result}",
 """if result['status']!='UNKNOWN' and result['prediction'] is not None and result['prediction']>=0:\n        result={**result,'qualified':True,'status':'SIMULATION_DOMAIN_QUALIFIED_PREDICTION'}\n    return {'use_case':case,'scope_status':scope['status'],\n            'qualification':scope['release_tier']+'_SIMULATION_ONLY','model_domain':domain,**result}""")
 s+=RELEASE_FUNCTION;wrapper.write_text(s)
 # Tests adapt only eligibility markers, never validation thresholds.
 test=dst/'tests/phm_design/test_limited_scope_dev12.py';txt=test.read_text()
 txt=txt.replace("c['deployment_qualification_status']=='NONE_ISSUED'", "c['deployment_qualification_status']=='SIMULATION_DOMAIN_'+c['release_tier']")
 txt=txt.replace("c['candidate_version'].startswith('0.7.0.dev13')",f"c['candidate_version']=='{version}'")
 txt=txt.replace("assert man[domain]['qualified'] is False", "assert man[domain]['qualified'] is True")
 txt=txt.replace("assert result['scope_status']=='DEVELOPMENT_CANDIDATE_NOT_QUALIFIED'","assert result['scope_status']=='SIMULATION_DOMAIN_RELEASE_QUALIFIED'")
 txt=txt.replace("assert result['qualification']=='NOT_RC_NOT_STABLE'",f"assert result['qualification']=='{tier}_SIMULATION_ONLY'")
 txt=txt.replace("assert result['qualified'] is False", "assert result['qualified'] is True")
 test.write_text(txt)
 t=dst/'tests/phm_design/test_limited_scope_v2.py';txt=t.read_text().replace("assert c['calibration_openings_used']==c['sealed_openings_used']==0","assert c['calibration_openings_used']==c['sealed_openings_used']==1")
 t.write_text(txt)
 t=dst/'tests/phm_design/test_release_qualification.py'
 t.write_text(f'''"""Proof-bound 15-cell simulation release contract."""\nimport hashlib,json,pytest\nfrom phm_design.platform_limited import contract,allowed_use_case,predict_bundle,_checked_release_certificate\nfrom pathlib import Path\nF=Path(__file__).parent/'fixtures/limited_opened'\ndef test_release_provenance():\n c=contract(); cert=_checked_release_certificate()\n assert c['release_tier']=='{tier}' and c['release_version']=='{version}'\n assert cert['sealed_worlds']==600 and cert['calibration_worlds']==300\n assert cert['calibration_passed']==cert['sealed_passed']==15\n assert len(cert['claimed_cells'])==15 and len(cert['unqualified_cells'])==8\n assert c['qualification_thresholds']==cert['numerical_thresholds']\ndef test_qualified_prediction_only_when_valid():\n p=predict_bundle('EPS','EPS:conventional',F/'EPS.csv',healthy_prefix_confirmed=True)\n assert p['qualified'] is True and p['qualification']=='{tier}_SIMULATION_ONLY'\n assert p['status']=='SIMULATION_DOMAIN_QUALIFIED_PREDICTION'\ndef test_out_of_claim_thermal_never_qualified():\n assert allowed_use_case('Thermal:conventional')['status']=='OUT_OF_CLAIM_NOT_QUALIFIED'\n with pytest.raises(ValueError,match='OUT_OF_CLAIM'):\n  predict_bundle('Thermal','Thermal:conventional',F/'Thermal.csv',healthy_prefix_confirmed=True)\n''')
 dist=dst/'dist';dist.mkdir()
 cmd=[sys.executable,'-m','pip','wheel','.','--no-deps','--no-build-isolation','-w','dist']
 with (ROOT/f'BUILD_{tier}_WHEEL.log').open('w') as logfile:
  subprocess.run(cmd,cwd=dst,stdout=logfile,stderr=subprocess.STDOUT,check=True)
 old=Path.cwd()
 try:
  import os
  os.chdir(dst)
  with (ROOT/f'BUILD_{tier}_SDIST.log').open('w') as logfile:
   # backend build writes a lot; capture to file
   from contextlib import redirect_stdout,redirect_stderr
   with redirect_stdout(logfile),redirect_stderr(logfile):print(setuptools.build_meta.build_sdist('dist'))
 finally:os.chdir(old)
 return {'tier':tier,'version':version,'artifact_hashes':{p.name:sha(p) for p in sorted(dist.iterdir())},'certificate_sha256':sha(ROOT/'QUALIFICATION_CERTIFICATE.json')}

if __name__=='__main__':
 tier=sys.argv[1]
 if tier not in ('RC','STABLE'):raise SystemExit('BAD_RELEASE_TIER')
 version='0.7.0rc1' if tier=='RC' else '0.7.0'
 print(json.dumps(build(version,tier),indent=2))
