#!/usr/bin/env python3
# 用途：对已公开数据进行低成本难度诊断。
# 参数：--evidence --output。
# 输出：训练专属拟合的统计基线、单特征和负对照JSON。
"""Train/validation-fitted diagnostic audit of already-public PHM evidence.

The public holdout is used only to report fixed diagnostics, not to tune the next
experiment. Metadata features and within-group label permutations are negative
controls, not admissible deployment inputs. No leakage-free certificate is issued.
"""
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score,accuracy_score
from phm_design.data import Dataset,statistics
from phm_design.family_pipeline import CONTEXT
from phm_design.adapters.sat_nonadcs import FAMILIES
from phm_design.io import write_json,sha_file

def fit(x,y):
    return make_pipeline(StandardScaler(),LogisticRegression(C=1.,max_iter=3000,class_weight='balanced',random_state=0)).fit(x,y)

def score(model,x,y):
    p=model.predict(x)
    return {'accuracy':float(accuracy_score(y,p)),'macro_f1':float(f1_score(y,p,average='macro',zero_division=0))}

def audit(evidence,output):
    b=Path(evidence);result={'schema':'phm.difficulty-audit.v2','source':'already-public dev3 evidence','diagnostic_only':True,'domains':{}}
    for d,spec in FAMILIES.items():
        raw=Dataset(b/'campaign/native'/d/'development/dataset.json',allowed_splits={'train','validation'})
        ht=Dataset(b/'campaign/native'/d/'holdout/dataset.json',allowed_splits={'test'})
        prep=Dataset(b/'prepared'/d/'development/dataset.json',allowed_splits={'train','validation'})
        hp=Dataset(b/'prepared'/d/'holdout/dataset.json',allowed_splits={'test'})
        for dataset in (raw,ht,prep,hp):dataset.verify_files()
        tr,va,te=raw.load('train',spec['window']),raw.load('validation',spec['window']),ht.load('test',spec['window'])
        mr,mv,mt=prep.load('train',spec['window']),prep.load('validation',spec['window']),hp.load('test',spec['window'])
        for a,c in [(tr,mr),(va,mv),(te,mt)]:assert [x['id'] for x in a.samples]==[x['id'] for x in c.samples]
        k=len(spec['channels'])+len(CONTEXT[d])
        views={'raw_stats':[statistics(x.x) for x in (tr,va,te)],
               'raw_plus_normal_stats':[statistics(x.x[:,:,:k]) for x in (mr,mv,mt)],
               'mechanism_stats':[statistics(x.x) for x in (mr,mv,mt)]}
        scores={}
        for name,(a,c,t) in views.items():
            model=fit(a,tr.y);scores[name]={'validation':score(model,c,va.y),'public_test':score(model,t,te.y)}
        names=[f'{op}:{c["name"]}' for op in ('mean','std','min','max','last','delta','rms') for c in prep.contract['channels']]
        a,c,t=views['mechanism_stats'];single=[]
        for j,name in enumerate(names):
            model=fit(a[:,[j]],tr.y);r=score(model,c[:,[j]],va.y)
            single.append((r['macro_f1'],name,j,model))
        _,name,j,model=sorted(single,key=lambda r:(-r[0],r[1]))[0]
        best={'feature':name,'selected_using':'validation_only','validation':score(model,c[:,[j]],va.y),'public_test':score(model,t[:,[j]],te.y)}
        def meta(block,root):
            return np.asarray([[(root/x['path']).stat().st_size,*bytes.fromhex(x['id'][:16]),*bytes.fromhex(x['group_id'][:16]),*bytes.fromhex(x['sha256'][:16])] for x in block.samples],float)
        model=fit(meta(tr,raw.root),tr.y)
        neg={'validation':score(model,meta(va,raw.root),va.y),'public_test':score(model,meta(te,ht.root),te.y)}
        rng=np.random.default_rng(20261003);groups={}
        for i,s in enumerate(tr.samples):groups.setdefault(s['group_id'],[]).append(i)
        perm=[]
        for _ in range(20):
            y=tr.y.copy()
            for ids in groups.values():y[ids]=rng.permutation(y[ids])
            perm.append(score(fit(a,y),c,va.y)['macro_f1'])
        result['domains'][d]={'counts':{'train':len(tr.y),'validation':len(va.y),'test':len(te.y),'test_groups':len(te.groups)},
            'feature_views':scores,'validation_selected_single_feature':best,'metadata_negative_control':neg,
            'groupwise_permutation_validation_f1':{'runs':perm,'mean':float(np.mean(perm)),'max':max(perm)},
            'dataset_sha256':{'raw':sha_file(raw.path),'prepared':sha_file(prep.path)}}
    write_json(output,result)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--evidence',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    r=audit(a.evidence,a.output)
    print(json.dumps({d:v['feature_views'] for d,v in r['domains'].items()},indent=2))
