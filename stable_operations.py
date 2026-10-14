"""Predeclared, bounded, research-data-safe 100-operation Stable acceptance suite.

Reads only previously opened R57 compatibility fixtures; never opens R60
calibration/sealed traces and never refits the frozen qualification models.
Each invocation runs exactly 25 operations in its own fresh installed env.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, resource, subprocess, sys, tempfile, time
from pathlib import Path

DIR=Path(__file__).resolve().parent
W=Path('/mnt/data/PHM_R60_LIMITED_15_20261011')
FIX=W/'release_validation/RC/testbed/tests/phm_design/fixtures/limited_opened'
CELLS=['311_wheel','311_sdist','312_wheel','312_sdist']
DOMAINS=['ADCS','EPS','Data','EPS_localization','Data_localization']
ALL_DOMAINS=['ADCS','EPS','Thermal','Data','EPS_localization','Thermal_localization','Data_localization']


def canonical(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def build_plan():
    from json import loads
    # JSON contract established before any Stable operations are run.
    ct=loads((W/'APPROVED_15_CELL_CONTRACT.json').read_text())
    supported=list(ct['proposed_certification_cells']);unsupported=list(ct['out_of_claim_unqualified_cells'])
    assert len(supported)==15 and len(unsupported)==8
    tasks=[]
    for case in supported+[supported[i] for i in range(5)]:tasks.append({'kind':'supported_prediction','case':case})
    for dom in ALL_DOMAINS*2:tasks.append({'kind':'valid_observation_contract','domain':dom})
    for dom,variant in zip(['ADCS','EPS','Thermal','Data','EPS_localization','Data_localization'],
                           ['extra_truth','missing_field','nonfinite','nonmonotone','reverse_time','empty']):
        tasks.append({'kind':'invalid_observation_contract','domain':dom,'variant':variant})
    for case in supported:tasks.append({'kind':'scope_authorized','case':case})
    tasks.append({'kind':'scope_invalid','case':'ADCS:simulator_injected_fault_truth'})
    for case in unsupported:tasks.append({'kind':'scope_rejected','case':case})
    for dom in DOMAINS*2:tasks.append({'kind':'frozen_model_integrity','domain':dom})
    for dom in DOMAINS*2:tasks.append({'kind':'persist_restore_model','domain':dom})
    for dom in DOMAINS+['ADCS']:tasks.append({'kind':'process_restart_prediction','domain':dom})
    for i in range(6):tasks.append({'kind':'independent_synthetic_design','trial':i})
    for variant in ['healthy_prefix_missing','domain_mismatch','unbundled_model','unknown_case']:
        tasks.append({'kind':'negative_release_boundary','variant':variant})
    assert len(tasks)==100,len(tasks)
    for i,item in enumerate(tasks):item['id']=f'SOP{i+1:03d}';item['cell']=CELLS[i%4];item['wall_limit_s']=30;item['rss_limit_mib']=1536
    counts={c:sum(x['cell']==c for x in tasks) for c in CELLS};assert set(counts.values())=={25}
    return {'schema':'phm.r60.stable.operations.v1','contract_id':ct['contract_id'],
        'status':'PREDECLARED_BEFORE_EXECUTION', 'input_data':'R57_PREVIOUSLY_OPENED_FIXTURES_ONLY',
        'no_qualification_data_access':True,'target_version':'0.7.0',
        'runtime_scope':['3.11','3.12'], 'ops_per_environment':counts,'operations':tasks}

def init_plan():
    p=DIR/'PLAN_100_STABLE_OPERATIONS.json'
    if p.exists():raise FileExistsError('OPERATION_PLAN_ALREADY_FROZEN')
    p.write_text(json.dumps(build_plan(),indent=2,sort_keys=True)+'\n')
    (DIR/'PLAN_100_STABLE_OPERATIONS.sha256').write_text(canonical(p)+'  PLAN_100_STABLE_OPERATIONS.json\n')
    print('FROZEN',len(json.loads(p.read_text())['operations']),canonical(p))

def sample(domain):
    if domain=='Thermal_localization':return DIR/'fixtures/THERMAL_LOCALIZATION_CONTRACT_SYNTHETIC.csv'
    return FIX/(domain+'.csv')

def run_one(t):
    import numpy as np, joblib
    from phm_design.platform_features import spec
    from phm_design.platform_limited import (contract,allowed_use_case,open_bundled,
        model_manifest,predict_bundle,rows_from_csv,registry)
    from phm_design.open_design_fit import predict as raw_predict
    domain=t.get('domain')
    kind=t['kind']
    if kind=='supported_prediction':
        case=t['case'];dom=case.split(':')[0]
        pred=predict_bundle(dom,case,sample(dom),healthy_prefix_confirmed=True)
        assert pred['qualification']=='STABLE_SIMULATION_ONLY' and pred['qualified'] is True
        assert pred['status']=='SIMULATION_DOMAIN_QUALIFIED_PREDICTION' and pred['prediction']>=0
        return {'domain':dom,'prediction':pred['prediction']}
    if kind=='valid_observation_contract':
        a=rows_from_csv(sample(domain),domain)
        assert a.shape[1]==len(spec(domain).fields) and a.shape[0]>=24
        return {'domain':domain,'row_count':a.shape[0]}
    if kind=='invalid_observation_contract':
        rows=list(csv.reader(sample(domain).open(newline='')))
        variant=t['variant']
        if variant=='extra_truth':
            rows[0].append('truth');[r.append('1') for r in rows[1:]]
        elif variant=='missing_field':[r.pop() for r in rows]
        elif variant=='nonfinite':rows[1][1]='nan'
        elif variant=='nonmonotone':rows[3][0]=rows[2][0]
        elif variant=='reverse_time':rows[2],rows[3]=rows[3],rows[2]
        elif variant=='empty':rows=rows[:1]
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'invalid.csv'
            with p.open('w',newline='') as f:csv.writer(f).writerows(rows)
            try: rows_from_csv(p,domain)
            except ValueError as e:return {'rejected':type(e).__name__,'message':str(e)[:100]}
        raise AssertionError('INVALID_OBSERVATION_ACCEPTED')
    if kind=='scope_authorized':
        status=allowed_use_case(t['case'])
        assert status['status']=='SIMULATION_DOMAIN_RELEASE_QUALIFIED' and status['release_tier']=='STABLE'
        return {'case':t['case'],'authorized':True}
    if kind=='scope_invalid':
        try:allowed_use_case(t['case'])
        except ValueError as e:return {'rejected':str(e)}
        raise AssertionError('UNKNOWN_SCOPE_ACCEPTED')
    if kind=='scope_rejected':
        status=allowed_use_case(t['case']);assert status['status']=='OUT_OF_CLAIM_NOT_QUALIFIED'
        return {'case':t['case'],'authorized':False}
    if kind=='frozen_model_integrity':
        entry=model_manifest()['models'][domain]
        from importlib.resources import files
        p=files('phm_design').joinpath('limited_models',entry['file'])
        assert hashlib.sha256(p.read_bytes()).hexdigest()==entry['sha256']
        assert open_bundled(domain)['platform_domain']==domain
        return {'sha256':entry['sha256']}
    if kind=='persist_restore_model':
        model=open_bundled(domain);a=rows_from_csv(sample(domain),domain)
        before=raw_predict(model,a,registry(domain),healthy_prefix_confirmed=True)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'trusted_local_roundtrip.joblib'
            joblib.dump(model,p,compress=0);roundtripped=joblib.load(p)
            after=raw_predict(roundtripped,a,registry(domain),healthy_prefix_confirmed=True)
            assert before==after and p.stat().st_size>100
        return {'domain':domain,'prediction':before['prediction'],'roundtrip':'PASS'}
    if kind=='process_restart_prediction':
        case=domain+(':multi_component_localization' if domain.endswith('_localization') else ':conventional')
        body=('import json,sys\nfrom phm_design.platform_limited import predict_bundle\n'
              'print(json.dumps(predict_bundle(sys.argv[1],sys.argv[2],sys.argv[3],healthy_prefix_confirmed=True)))\n')
        cmd=[sys.executable,'-c',body,domain,case,str(sample(domain))]
        p=subprocess.run(cmd,capture_output=True,text=True,timeout=20,env={**os.environ,'OPENBLAS_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'},cwd=str(DIR))
        assert p.returncode==0,(p.returncode,p.stderr[-400:])
        pred=json.loads(p.stdout.strip().splitlines()[-1]);assert pred['qualified'] is True
        return {'domain':domain,'restart_prediction':pred['prediction']}
    if kind=='independent_synthetic_design':
        from phm_design.open_design import ObservationSpec,generic_registry
        from phm_design.open_design_fit import TrainingExample,design,predict
        trial=t['trial'];obs_spec=ObservationSpec('synthetic_bounded',('time_s','sensor_a','sensor_b'))
        def obs(c,w):
            tm=np.linspace(0,40,48)
            return np.c_[tm,np.sin(tm*(.15+.01*trial))*(1+c)+w*.01,np.cos(tm*.17)+c*.01*tm]
        data=[TrainingExample(obs(c,w),c,'world_'+str(w)) for w in range(4) for c in range(3)]
        model,report=design(data,obs_spec,(0,1,2),0,max_trials=1)
        assert report['attempts'][0]['status']=='COMPLETE'
        assert not report['release_qualified']
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'synthetic_design.joblib';joblib.dump(model,p)
            m=joblib.load(p)
            before=predict(model,obs(0,0),generic_registry())
            after=predict(m,obs(0,0),generic_registry())
            assert before==after
        return {'trial':trial,'synthetic_design':'PASS','unqualified':True}
    if kind=='negative_release_boundary':
        variant=t['variant']
        if variant=='healthy_prefix_missing':
            p=predict_bundle('EPS','EPS:conventional',sample('EPS'),healthy_prefix_confirmed=False)
            assert p['prediction']==-1 and p['status']=='UNKNOWN' and p['qualified'] is False
            return {'rejected':'HEALTHY_REFERENCE_UNCONFIRMED'}
        if variant=='domain_mismatch':
            try:predict_bundle('Thermal','EPS:conventional',sample('Thermal'),healthy_prefix_confirmed=True)
            except ValueError as e:return {'rejected':str(e)}
            raise AssertionError('DOMAIN_MISMATCH_ACCEPTED')
        if variant=='unbundled_model':
            try:open_bundled('Thermal')
            except ValueError as e:return {'rejected':str(e)}
            raise AssertionError('UNSUPPORTED_MODEL_OPENED')
        if variant=='unknown_case':
            try:allowed_use_case('Data:simulator_truth')
            except ValueError as e:return {'rejected':str(e)}
            raise AssertionError('UNKNOWN_CASE_ACCEPTED')
    raise ValueError('UNSUPPORTED_OPERATION_KIND:'+kind)

def run_cell(cell):
    from importlib.metadata import version
    import phm_design
    assert version('phm-design')=='0.7.0' and phm_design.__version__=='0.7.0'
    plan_file=DIR/'PLAN_100_STABLE_OPERATIONS.json'
    recorded_hash=(DIR/'PLAN_100_STABLE_OPERATIONS.sha256').read_text().split()[0]
    assert canonical(plan_file)==recorded_hash,'PREDECLARED_PLAN_TAMPERED'
    plan=json.loads(plan_file.read_text());tasks=[t for t in plan['operations'] if t['cell']==cell]
    assert len(tasks)==25
    out=DIR/f'STABLE_OPERATIONS_{cell}.json'
    if out.exists():raise FileExistsError('CANNOT_OVERWRITE_OPERATION_EVIDENCE')
    events=[];start=time.monotonic();fails=0
    for t in tasks:
        t0=time.monotonic();e={'id':t['id'],'kind':t['kind'],'cell':cell}
        try:
            result=run_one(t)
            dt=time.monotonic()-t0
            rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
            if dt>t['wall_limit_s'] or rss>t['rss_limit_mib']:
                raise RuntimeError(f'BOUNDED_RESOURCE_VIOLATION {dt=:.4f} {rss=:.2f}')
            e.update(status='PASS',result=result,wall_s=dt,rss_peak_mib=rss)
        except Exception as exc:
            fails+=1;e.update(status='FAIL',error=repr(exc),wall_s=time.monotonic()-t0)
        events.append(e)
        print(f"{e['id']} {e['kind']} {e['status']}",flush=True)
    report={'schema':'phm.r60.stable.operation_results.v1','cell':cell,'plan_sha256':recorded_hash,
      'product_version':'0.7.0','qualified_ops':len(tasks),'passed':len(tasks)-fails,'failed':fails,
      'wall_s':time.monotonic()-start,'events':events}
    out.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps({'cell':cell,'count':len(tasks),'passed':len(tasks)-fails,'failed':fails},ensure_ascii=False))
    return 0 if fails==0 else 1

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['freeze','run']);ap.add_argument('--cell',choices=CELLS)
    args=ap.parse_args()
    if args.mode=='freeze':init_plan()
    else:
        if not args.cell:ap.error('--cell required')
        raise SystemExit(run_cell(args.cell))
