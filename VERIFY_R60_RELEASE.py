#!/usr/bin/env python3
"""Self-contained metadata, outcome and provenance auditor (no model training)."""
import json, hashlib, sys, zipfile, re
from pathlib import Path

def sha(p):
    h = hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def checked(condition, what):
    if not condition: raise AssertionError(what)
    checks.append(what)

repo=Path(__file__).resolve().parent
checks=[]
file_hashes=json.loads((repo/'DELIVERY_FILES_SHA256.json').read_text())
for rel, expected in file_hashes.items():
    p=repo/rel
    checked(p.is_file() and sha(p)==expected, f'sha256:{rel}')

src=repo/'qualification'
contract=json.loads((repo/'APPROVED_15_CELL_CONTRACT.json').read_text())
cert=json.loads((repo/'QUALIFICATION_CERTIFICATE.json').read_text())
checked(contract['contract_id']==cert['contract_id'], 'contract-id')
checked(len(contract['proposed_certification_cells'])==len(cert['claimed_cells'])==15, '15-supported-cells')
checked(set(contract['proposed_certification_cells'])==set(cert['claimed_cells']), 'claimed-cell-set')
checked(len(contract['out_of_claim_unqualified_cells'])==len(cert['unqualified_cells'])==8, '8-out-of-scope-cells')
checked(set(contract['out_of_claim_unqualified_cells'])==set(cert['unqualified_cells']), 'out-of-scope-set')
checked(set(cert['claimed_cells']).isdisjoint(set(cert['unqualified_cells'])), 'disjoint-certified-and-not-certified')
checked(cert['calibration_passed']==cert['sealed_passed']==15, 'certificate-15-15')
checked(cert['calibration_worlds']==300 and cert['sealed_worlds']==600, 'certificate-worlds')
checked(cert['no_qualification_data_model_refit'] is True, 'zero-refit-certificate')

thresholds=cert['numerical_thresholds']
all_world_ids=set(); all_fingerprints=set()
score_summary={}
for kind,nworld,ntraj,nper in [('CALIBRATION',300,980,20),('SEALED',600,1960,40)]:
    plan=json.loads((src/(kind+'_PLAN.json')).read_text())
    score=json.loads((src/(kind+'_SCORES.json')).read_text())
    checked(len(plan)==nworld and len({w['world_id'] for w in plan})==nworld, f'{kind}-plan-world-count')
    checked(len({w['physical_fingerprint'] for w in plan})==nworld, f'{kind}-plan-fingerprint-unique')
    ids={w['world_id'] for w in plan}; fps={w['physical_fingerprint'] for w in plan}
    checked(ids.isdisjoint(all_world_ids), f'{kind}-no-world-overlap')
    checked(fps.isdisjoint(all_fingerprints), f'{kind}-no-physical-fingerprint-overlap')
    all_world_ids.update(ids); all_fingerprints.update(fps)
    checked(len(score['results'])==score['passed_count']==15 and not score['failed_cells'], f'{kind}-all-pass')
    checked(score['worlds']==nworld and score['trajectories']==ntraj and score['zero_refits'],f'{kind}-evaluation-count')
    checked(score['single_opening'] and (score['sealed_openings_used']==(1 if kind=='SEALED' else 0)),f'{kind}-single-opening')
    grouped={}
    for w in plan: grouped.setdefault(w['qualification_scope_cell'], []).append(w)
    checked(set(grouped)==set(cert['claimed_cells']), f'{kind}-exact-cells')
    for name,result in score['results'].items():
        checked(len(grouped[name])==nper and len(set(w['scenario_family'] for w in grouped[name]))>=4, f'{kind}-{name}-worlds-and-families')
        checked(result['passed_numeric_gate'] and all(result['checks'].values()),f'{kind}-{name}-checks')
        checked(result['accuracy']>=thresholds['overall_accuracy_min'] and result['healthy_far']<=thresholds['healthy_known_fault_false_alarm_max'],f'{kind}-{name}-accuracy-and-far')
        checked(result['unknown_fraction']<=thresholds['unknown_fraction_max'],f'{kind}-{name}-unknown')
        checked(min(result['per_fault_recall'].values())>=thresholds['per_fault_recall_min'],f'{kind}-{name}-recall')
        checked(min(result['per_family_accuracy'].values())>=thresholds['worst_family_accuracy_min'],f'{kind}-{name}-worst-family')
    score_summary[kind]={'worlds':nworld,'trajectories':ntraj,'passed':15,'failed':0,'same_models_no_refit':True}

for filename,digest in cert['development_artifact_hashes'].items():
    resolved={
        'dev13_contract':repo/'APPROVED_15_CELL_CONTRACT.json',
        'calibration_freeze':src/'PREQUALIFICATION_FREEZE.json',
        'calibration_scores':src/'CALIBRATION_SCORES.json',
        'sealed_freeze':src/'SEALED_FREEZE.json',
        'sealed_plan':src/'SEALED_PLAN.json',
        'sealed_scores':src/'SEALED_SCORES.json',
        'dev13_wheel':repo/'artifacts'/'phm_design-0.7.0.dev13-py3-none-any.whl',
        'dev13_sdist':repo/'artifacts'/'phm_design-0.7.0.dev13.tar.gz',
    }.get(filename)
    if resolved: checked(sha(resolved)==digest,f'certificate-provenance:{filename}')

stable_build=json.loads((repo/'STABLE_BUILD_RESULT.json').read_text())
rc_build=json.loads((repo/'RC_BUILD_RESULTS.json').read_text())
checked(stable_build['version']=='0.7.0' and stable_build['tier']=='STABLE','stable-version-and-tier')
checked(rc_build['version']=='0.7.0rc1','rc-version')
checked(stable_build['certificate_sha256']==rc_build['certificate_sha256']==sha(repo/'QUALIFICATION_CERTIFICATE.json'),'stable-rc-same-certificate')
for build in (stable_build,rc_build):
    for name,digest in build['artifact_hashes'].items():
        checked(sha(repo/'artifacts'/name)==digest,f'release-artifact-sha:{name}')
checked(sha(repo/'artifacts'/'phm_design-0.6.0-py3-none-any.whl')=='91c0501f03d88dc92d20d3a0e33dd556dd833d250f1a1027aa7dfa34a01e3421','original-0.6-rollback-wheel')

wheel_names=['phm_design-0.7.0.dev13-py3-none-any.whl','phm_design-0.7.0rc1-py3-none-any.whl','phm_design-0.7.0-py3-none-any.whl']
wheels=[]
for name in wheel_names:
    with zipfile.ZipFile(repo/'artifacts'/name) as z:
        wheel={k:hashlib.sha256(z.read(k)).hexdigest() for k in z.namelist() if k.startswith('phm_design/')}
        for dom,digest in cert['frozen_models_sha256'].items():
            checked(wheel.get(f'phm_design/limited_models/{dom}.joblib')==digest,f'{name}-{dom}-weights-frozen')
        wheels.append(wheel)
changes=[]
for before,after in zip(wheels,wheels[1:]):
    changes.append(sorted(k for k in before.keys()|after.keys() if before.get(k)!=after.get(k)))
allowed_dev_rc={'phm_design/__init__.py','phm_design/limited_models/model_manifest.json','phm_design/limited_models/qualification_certificate.json','phm_design/limited_models/scope_contract.json','phm_design/platform_limited.py'}
allowed_rc_stable={'phm_design/__init__.py','phm_design/limited_models/model_manifest.json','phm_design/limited_models/scope_contract.json'}
checked(set(changes[0])==allowed_dev_rc,'dev13-to-rc1-only-release-state-changes')
checked(set(changes[1])==allowed_rc_stable,'rc1-to-stable-only-release-metadata-changes')

operations=json.loads((repo/'PLAN_100_STABLE_OPERATIONS.json').read_text())
planhash=sha(repo/'PLAN_100_STABLE_OPERATIONS.json')
checked(planhash==(repo/'PLAN_100_STABLE_OPERATIONS.sha256').read_text().split()[0],'operations-plan-hash')
checked(operations['status']=='PREDECLARED_BEFORE_EXECUTION' and len(operations['operations'])==100,'predeclared-100-operations')
planned={x['id']:x for x in operations['operations']}
checked(len(planned)==100,'plan-no-duplicate-operation-id')
observed={};per_env={}
for cell in ('311_wheel','311_sdist','312_wheel','312_sdist'):
    a=json.loads((repo/f'STABLE_OPERATIONS_{cell}.json').read_text())
    checked(a['plan_sha256']==planhash and a['product_version']=='0.7.0',f'{cell}-right-plan')
    checked(a['passed']==25 and a['failed']==0 and a['qualified_ops']==25 and len(a['events'])==25,f'{cell}-25-operations-pass')
    checked((repo/'evidence'/f'{cell}.status').read_text().strip()=='PASS',f'{cell}-qualification-pass-status')
    test=(repo/'evidence'/f'{cell}_FULL_TEST.log').read_text()
    checked(bool(re.search(r'\b738 passed\b',test)) and not re.search(r'\b\d+ (failed|skipped|errors?)\b',test),f'{cell}-738-tests-zero-fail-skip')
    checked('No broken requirements found.' in (repo/'evidence'/f'{cell}_PIP_CHECK.log').read_text(),f'{cell}-pip-check')
    pkg=json.loads((repo/'evidence'/f'{cell}_PACKAGE_CHECK.json').read_text())
    checked(pkg['product']=='0.7.0' and pkg['tier']=='STABLE' and pkg['models']==5,f'{cell}-installed-metadata')
    smoke=json.loads((repo/'evidence'/f'{cell}_BASILISK_SMOKE.json').read_text())
    checked(smoke['status']=='PASS',f'{cell}-native-smoke')
    for event in a['events']:
        eid=event['id']
        checked(eid in planned and planned[eid]['cell']==cell and eid not in observed and event['status']=='PASS',f'{cell}-{eid}-declared-and-passed')
        checked(event['rss_peak_mib']<=planned[eid]['rss_limit_mib'] and event['wall_s']<=planned[eid]['wall_limit_s'],f'{cell}-{eid}-resource-limits')
        observed[eid]=cell
    per_env[cell]={'full_tests_pass':738,'failed':0,'skipped':0,'stable_operations':25,'basilisk_smoke':'PASS','pip_check':'PASS'}
checked(set(observed)==set(planned),'all-100-planned-operations-observed')
pre=json.loads((repo/'evidence'/'312_sdist_ROLLBACK_PRE.json').read_text())
post=json.loads((repo/'evidence'/'312_sdist_ROLLBACK_CHECK.json').read_text())
checked(pre['source_version']=='0.7.0' and pre['target_version']=='0.6.0' and post['rollback_version']=='0.6.0' and post['status']=='PASS','rollback-0.7-to-original-0.6')
checked('Successfully uninstalled phm-design-0.7.0' in (repo/'evidence'/'312_sdist_ROLLBACK_UNINSTALL.log').read_text(),'rollback-uninstall')
checked('Successfully installed phm-design-0.6.0' in (repo/'evidence'/'312_sdist_ROLLBACK_INSTALL.log').read_text(),'rollback-install')
checked('No broken requirements found.' in (repo/'evidence'/'312_sdist_ROLLBACK_PIP_CHECK.log').read_text(),'rollback-deps')

report={'schema':'phm.r60.stable.final_audit.v1','release':'0.7.0','decision':'PASS_SIMULATION_DOMAIN_STABLE','contract_id':contract['contract_id'], 'qualified_cells':15,'not_qualified_cells':8,'calibration':score_summary['CALIBRATION'],'sealed':score_summary['SEALED'], 'rc_test_matrix':'four environments, 738 each (historical locked logs)','stable_test_matrix':per_env,'stable_operations_unique':len(observed), 'rollback':'PASS_0.7.0_TO_0.6.0','previous_stable_0.6_retained':True,'core_changes_dev13_to_rc':changes[0],'core_changes_rc_to_stable':changes[1], 'not_certified':['real flight','physical hardware','causal online diagnosis','out-of-claim eight cells'], 'checks_passed':len(checks)}
(repo/'FINAL_RELEASE_AUDIT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2))
