"""Predeclared synthetic source/OOD probe. DEVELOPMENT evidence only.

Do not tune against the generated heldout source after seeing results.
No Basilisk final or previously revealed R71-world data is used.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np

from phm_design.task_candidate_registry import TorchTaskEstimator
from phm_design.neural_domain_generalization import DGSequenceEstimator

PROTOCOL = {
    'id':'r77-synthetic-source-shift-probe-v1',
    'source_environments':{'source_a':0.6,'source_b':0.35,'source_c':-0.12,'source_d':-0.28},
    'dev_stress_environment':{'heldout_e':-0.65},
    'train_samples_per_class_per_source':24,
    'holdout_samples_per_class':64,
    'samples_per_trajectory':48,
    'fault_amplitude':0.22,
    'sensor_noise':0.075,
    'seeds':[17,23,41],
    'candidates':['torch_tcn_erm','dg_tcn_groupdro'],
    'feature_channels':['weak_temporal_signature','spurious_environment_correlate','domain_baseline'],
    'training_optimizer_updates_approx':{'torch_tcn_erm':48,'dg_tcn_groupdro':48},
    'holdout_role':'DEVELOPMENT_STRESS_ONLY',
    'qualification':False,
}


def generate(shift, samples_per_class, seed):
    rng = np.random.default_rng(seed)
    t = np.linspace(0,1,PROTOCOL['samples_per_trajectory'])
    x,y=[],[]
    for label in (0,1):
        for _ in range(samples_per_class):
            oscillation = (PROTOCOL['fault_amplitude'] * label * np.sin(4*np.pi*t + rng.normal(0,.06)))
            a = oscillation + rng.normal(0, PROTOCOL['sensor_noise'], len(t))
            b = shift * (2*label - 1) + rng.normal(0,.085,len(t))
            c = shift + rng.normal(0,.08,len(t))
            x.append(np.stack((a,b,c), axis=-1))
            y.append(label)
    order=rng.permutation(len(y))
    return np.asarray(x,np.float32)[order],np.asarray(y)[order]


def metrics(model,x,y):
    out=model.predict_proba(x)
    pred=out.argmax(1)
    return {'accuracy':float(np.mean(pred==y)),
            'healthy_false_alarm_rate':float(np.mean(pred[y==0]!=0)),
            'fault_recall':float(np.mean(pred[y==1]==1)),
            'samples':len(y)}


def main(output):
    source_x=[];source_y=[];source_groups=[]
    for i,(group,shift) in enumerate(PROTOCOL['source_environments'].items()):
        x,y=generate(shift,PROTOCOL['train_samples_per_class_per_source'],1000+i)
        source_x.append(x);source_y.append(y);source_groups.extend([group]*len(y))
    training_x=np.concatenate(source_x);training_y=np.concatenate(source_y)
    groups=np.asarray(source_groups)
    frozen=[]
    for seed in PROTOCOL['seeds']:
        # Both objectives and hyperparameters frozen *before* generated dev-stress observations.
        erm=TorchTaskEstimator('torch_tcn',seed,epochs=4,batch_size=16)
        dg=DGSequenceEstimator('dg_tcn',seed)
        erm.fit(training_x, training_y)
        dg.fit(training_x, training_y,domain_groups=groups)
        frozen.append((seed,erm,dg))
    # Produce the fresh synthetic holdout only after both candidates are fitted.
    holdout_x,holdout_y=generate(PROTOCOL['dev_stress_environment']['heldout_e'],
                                PROTOCOL['holdout_samples_per_class'],771700)
    scores=[]
    for seed,erm,dg in frozen:
        scores.append({'seed':seed,'ERM_TCN':metrics(erm,holdout_x,holdout_y),
                       'GroupDRO_TCN':metrics(dg,holdout_x,holdout_y)})
    report={'protocol':PROTOCOL,
            'protocol_sha256':hashlib.sha256(json.dumps(PROTOCOL,sort_keys=True).encode()).hexdigest(),
            'candidate_results':scores,
            'aggregate_mean_accuracy':{name:float(np.mean([item[name]['accuracy'] for item in scores]))
                                       for name in ('ERM_TCN','GroupDRO_TCN')},
            'no_post_holdout_retuning':True,'not_independent_qualification':True}
    Path(output).write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps(report['aggregate_mean_accuracy'],sort_keys=True))

if __name__=='__main__':
    import sys
    main(sys.argv[1] if len(sys.argv)>1 else 'R77_SYNTHETIC_OOD_AUDIT.json')
