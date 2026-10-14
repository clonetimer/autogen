"""Design and replay concrete native-adapter datasets using product APIs."""
from pathlib import Path
import argparse,json
import joblib,numpy as np
from .platform_features import FIELDS,registry,spec
from .open_design_fit import TrainingExample,design,predict

def _child(root,relative):
    p=(root/relative).resolve()
    if not p.is_relative_to(root.resolve()) or not p.is_file():raise ValueError('DATASET_PATH_OUTSIDE_ROOT_OR_MISSING')
    return p

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);subs=p.add_subparsers(dest='action',required=True)
    subs.add_parser('inspect')
    d=subs.add_parser('design');d.add_argument('--dataset',required=True);d.add_argument('--output',required=True);d.add_argument('--trials',type=int,default=24)
    pr=subs.add_parser('predict');pr.add_argument('--model',required=True);pr.add_argument('--observations',required=True);pr.add_argument('--healthy-prefix-confirmed',action='store_true')
    a=p.parse_args(argv)
    if a.action=='inspect':
        result={'domains':{d:list(f) for d,f in FIELDS.items()},'qualification':'dataset-specific, not a flight certificate','hard_budget_note':'--trials is a trial count; use an external CPU supervisor for hard resource limits'}
    elif a.action=='design':
        path=Path(a.dataset).resolve();data=json.loads(path.read_text());domain=data['domain'];examples=[]
        for r in data['records']:
            if r.get('role')!='design_train':raise ValueError('PROTECTED_ROLE_NOT_TRAINING_DATA')
            csv=_child(path.parent,r['path']);header=csv.open().readline().strip().split(',')
            if tuple(header)!=spec(domain).fields:raise ValueError('DEPLOYMENT_HEADER_MISMATCH')
            examples.append(TrainingExample(np.loadtxt(csv,delimiter=',',skiprows=1),r['label'],r['world_id'],healthy_prefix_confirmed=r.get('healthy_prefix_confirmed',False)))
        model,report=design(examples,spec(domain),tuple(data['classes']),data['healthy_label'],registry=registry(domain),max_trials=a.trials)
        model['platform_domain']=domain;out=Path(a.output);out.mkdir(parents=True,exist_ok=False);joblib.dump(model,out/'model.joblib');(out/'design_report.json').write_text(json.dumps(report,indent=2));result={'model':str(out/'model.joblib'),'release_qualified':False}
    else:
        # Only load trusted local artifacts; joblib/pickle is executable input.
        model=joblib.load(a.model);domain=model['platform_domain'];csv=Path(a.observations)
        with csv.open() as f:header=tuple(f.readline().strip().split(','))
        if header!=tuple(model['observation_spec']['fields']):raise ValueError('DEPLOYMENT_HEADER_MISMATCH')
        result=predict(model,np.loadtxt(csv,delimiter=',',skiprows=1),registry(domain),healthy_prefix_confirmed=a.healthy_prefix_confirmed)
    print(json.dumps(result,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
