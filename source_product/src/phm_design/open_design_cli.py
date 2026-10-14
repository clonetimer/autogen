"""Experimental standalone task-contract CLI; never promotes an artifact."""
from pathlib import Path
import argparse,csv,json,hashlib,sys
import numpy as np
from .open_design import ObservationSpec,RiskPreference,generic_registry,canonical
from .open_design_fit import TrainingExample,design,predict


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);sp=ap.add_subparsers(dest='action',required=True)
    sp.add_parser('catalog')
    d=sp.add_parser('design');d.add_argument('task',type=Path);d.add_argument('output',type=Path)
    p=sp.add_parser('predict');p.add_argument('model',type=Path);p.add_argument('csv',type=Path)
    p.add_argument('--expected-sha256',required=True);p.add_argument('--confirm-healthy-prefix',action='store_true')
    a=ap.parse_args(argv)
    if a.action=='catalog':
        print(json.dumps({'schema':'phm.open-design.catalog.v1','atoms':generic_registry().names,
            'compositions':['concat','available_stack','available_average'],'scope':'arbitrary numeric multivariate task contracts','release_qualified':False}));return 0
    if a.action=='predict':
        if digest(a.model)!=a.expected_sha256:raise ValueError('MODEL_HASH_MISMATCH_BEFORE_DESERIALIZATION')
        import joblib
        model=joblib.load(a.model)
        with a.csv.open(newline='') as f:rows=list(csv.DictReader(f))
        out=predict(model,rows,generic_registry(),healthy_prefix_confirmed=a.confirm_healthy_prefix)
        print(json.dumps(out));return 2 if out['status']=='UNKNOWN' else 0
    if a.output.exists():raise FileExistsError('REFUSE_TO_OVERWRITE_STUDY')
    task=json.loads(a.task.read_text());s=task['observation_spec'];s['fields']=tuple(s['fields']);spec=ObservationSpec(**s)
    examples=[];root=a.task.resolve().parent
    for row in task['training']:
        if row['role']!='design_train':raise ValueError('PROTECTED_ROLE_NOT_TRAINING_DATA')
        path=(root/row['path']).resolve()
        if not path.is_relative_to(root):raise ValueError('TRAINING_PATH_ESCAPE')
        if digest(path)!=row['sha256']:raise ValueError('TRAINING_HASH_MISMATCH')
        with path.open(newline='') as f:obs=spec.rows(list(csv.DictReader(f)))
        examples.append(TrainingExample(obs,row['label'],row['world_id'],healthy_prefix_confirmed=row.get('healthy_prefix_confirmed',False)))
    model,report=design(examples,spec,tuple(task['classes']),task['healthy_label'],
        max_trials=int(task.get('max_trials',16)),risk=RiskPreference(**task.get('risk_preference',{})),seed=int(task.get('seed',48019)))
    a.output.mkdir(parents=True)
    import joblib
    joblib.dump(model,a.output/'model.joblib');report['model_sha256']=digest(a.output/'model.joblib')
    (a.output/'REPORT.json').write_bytes(canonical(report));(a.output/'GRAPH.json').write_bytes(canonical(model['graph']))
    print(json.dumps({'model_sha256':report['model_sha256'],'attempts':len(report['attempts']),'release_qualified':False}));return 0

if __name__=='__main__':sys.exit(main())
