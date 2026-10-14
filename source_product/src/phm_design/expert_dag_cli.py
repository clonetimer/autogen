from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path
from .expert_dag import search_multi_environment,records_to_arrays,execute,metrics,_eligible

def read(p): return json.loads(Path(p).read_text())
def write(p,v): Path(p).parent.mkdir(parents=True,exist_ok=True); Path(p).write_text(json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+"\n")
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def named(items):
    out={}
    for x in items:
        if '=' not in x: raise ValueError('EXPERT_DAG_ENVIRONMENT')
        n,p=x.split('=',1); p=Path(p)
        if not n or n in out or not p.is_file(): raise ValueError('EXPERT_DAG_ENVIRONMENT')
        out[n]=p
    return out

def records(p):
    x=read(p); x=x.get('predictions') if isinstance(x,dict) else x
    if not isinstance(x,list): raise ValueError('EXPERT_DAG_RECORDS')
    return x

def main():
    ap=argparse.ArgumentParser(prog='phm-design-expert-dag')
    ap.add_argument('--mode',choices=('multi','evaluate'),required=True); ap.add_argument('--environment',action='append',default=[])
    ap.add_argument('--records'); ap.add_argument('--config',required=True); ap.add_argument('--candidate'); ap.add_argument('--output',required=True)
    a=ap.parse_args(); cfg=read(a.config); out=Path(a.output); out.mkdir(parents=True,exist_ok=False)
    if a.mode=='multi':
        ps=named(a.environment); env={n:records(p) for n,p in ps.items()}; rep=search_multi_environment(env,cfg)
        write(out/'search_report.json',rep)
        if rep['chosen_candidate'] is not None: write(out/'selected_expert_dag.json',rep['chosen_candidate'])
        write(out/'environment_bindings.json',{'data_role':'development','test_or_holdout_used_for_selection':False,'environments':{n:{'path':str(p),'sha256':sha(p)} for n,p in ps.items()}})
        print(json.dumps({'status':rep['status'],'candidate_count':rep['candidate_count'],'eligible_candidate_count':rep['eligible_candidate_count'],'chosen_grammar':None if rep['chosen_candidate'] is None else rep['chosen_candidate']['grammar']},sort_keys=True))
    else:
        if not a.records or not a.candidate: raise ValueError('EXPERT_DAG_EVALUATE_INPUT')
        cand=read(a.candidate); rs=records(a.records); sig,y,_=records_to_arrays(rs,cand['classes'],cand['experts']); runtime=execute(cand,sig); m=metrics(y,runtime['class_index'],cand['classes']); ok,why=_eligible(m,cfg.get('constraints',{}))
        rep={'schema':'phm.expert-dag-frozen-evaluation.v1','status':'PASS' if ok else 'FAIL_CLOSED','candidate_fingerprint':cand['fingerprint'],'candidate_sha256':sha(a.candidate),'records_sha256':sha(a.records),'config_sha256':sha(a.config),'metrics':m,'strict_gate_pass':ok,'strict_gate_reasons':why,'test_used_for_selection':False,'truth_inputs_used_at_runtime':False}
        write(out/'evaluation.json',rep); print(json.dumps({'status':rep['status'],'metrics':m},sort_keys=True))
    return 0
if __name__=='__main__': raise SystemExit(main())
