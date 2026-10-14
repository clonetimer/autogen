"""Auditable JSON random-forest expert for family Thermal rolling-v2.

No pickle/joblib is loaded at inference time. Fitting is *train-role only*.
Selection of depth/forest size belongs to an independently frozen development
comparison; this module never opens calibration/holdout/test data for fitting.
"""
from __future__ import annotations

import csv
from pathlib import Path
import json
import math

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .data import Dataset, sequence_from_csv
from .adapters import sat_nonadcs as native
from .io import digest, read_json, write_json, sha_file, seal, verify
from .thermal_family_v2 import compute_features, feature_contract

SCHEMA = 'phm.thermal-family-v2-json-forest.v1'
SUMMARIES = ('mean', 'std', 'tail16', 'tail64', 'last')


def summarize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 3 or x.shape[1] != native.FAMILIES['thermal']['window'] or x.shape[2] != len(feature_contract()['channels']) or not np.isfinite(x).all():
        raise ValueError('THERMAL_V2_SUMMARY_SHAPE')
    return np.concatenate((x.mean(1), x.std(1), x[:, -16:].mean(1),
                          x[:, -64:].mean(1), x[:, -1]), axis=1)


def to_json_forest(model: RandomForestClassifier, *, evidence: dict) -> dict:
    if list(model.classes_) != [0, 1, 2]:
        raise ValueError('THERMAL_V2_MODEL_CLASSES')
    names = [x['name'] for x in feature_contract()['channels']]
    trees=[]
    for estimator in model.estimators_:
        t=estimator.tree_
        leaves=np.asarray(t.value)[:, 0, :]
        s=leaves.sum(axis=1, keepdims=True)
        leaves=np.divide(leaves,s,out=np.zeros_like(leaves,dtype=float),where=s>0)
        trees.append({'children_left':t.children_left.astype(int).tolist(),
            'children_right':t.children_right.astype(int).tolist(),
            'feature':t.feature.astype(int).tolist(),
            'threshold':t.threshold.astype(float).tolist(),
            'probabilities':leaves.tolist()})
    return {'schema':SCHEMA,'classes':['NOMINAL','EMISSIVITY_LOSS','EXTRA_INTERNAL_HEAT'],
        'feature_names':names,'summary_ops':list(SUMMARIES),
        'model':{'kind':'json_random_forest','n_estimators':len(trees),
                 'depth_limit':model.max_depth,'min_leaf':model.min_samples_leaf,
                 'trees':trees},
        'training_evidence':evidence,'requires_pretreatment':'family.thermal.rolling.v2',
        'score_calibrated':False,'deployment_qualified':False}


def predict_proba_from_features(features: np.ndarray, package: dict) -> np.ndarray:
    if package.get('schema')!=SCHEMA or package.get('requires_pretreatment')!='family.thermal.rolling.v2':
        raise ValueError('THERMAL_V2_FOREST_SCHEMA')
    if [x['name'] for x in feature_contract()['channels']]!=package.get('feature_names'):
        raise ValueError('THERMAL_V2_FOREST_CHANNELS_CHANGED')
    if package.get('summary_ops') != list(SUMMARIES):
        raise ValueError('THERMAL_V2_FOREST_SUMMARIES_CHANGED')
    xs=np.asarray(summarize(features),dtype=np.float32)
    trees=package['model']['trees']
    if not trees or len(trees)!=package['model']['n_estimators']:
        raise ValueError('THERMAL_V2_FOREST_COUNT')
    proba=np.zeros((len(xs),3),dtype=np.float64)
    for tree in trees:
        left,right,feature,threshold,p=([tree[name] for name in
                              ('children_left','children_right','feature','threshold','probabilities')])
        if not (len(left)==len(right)==len(feature)==len(threshold)==len(p)):
            raise ValueError('THERMAL_V2_FOREST_NODE_LENGTH')
        for i,row in enumerate(xs):
            node=0
            for _ in range(len(left)+1):
                if feature[node]<0:
                    break
                j=int(feature[node])
                if not 0<=j<len(row):raise ValueError('THERMAL_V2_FOREST_BAD_FEATURE')
                node=left[node] if row[j]<=threshold[node] else right[node]
                if not 0<=node<len(left):raise ValueError('THERMAL_V2_FOREST_BAD_CHILD')
            else:raise ValueError('THERMAL_V2_FOREST_CYCLE')
            probs=np.asarray(p[node],dtype=float)
            if probs.shape!=(3,) or not np.isfinite(probs).all() or np.any(probs<0) or probs.sum()<=0:
                raise ValueError('THERMAL_V2_FOREST_BAD_LEAF')
            proba[i]+=probs/probs.sum()
    return proba/len(trees)


def fit_json_forest(paths: list[str], policy_path: str, output: str) -> dict:
    """Fit selected model on train only; policy file references dev validation selection."""
    target=Path(output)
    if target.exists():raise FileExistsError('THERMAL_V2_MODEL_EXISTS')
    policy=read_json(policy_path)
    if policy.get('schema')!='phm.thermal-v2-frozen-fit-policy.v1':
        raise ValueError('THERMAL_V2_FIT_POLICY_SCHEMA')
    spec=policy['selected_model']
    if set(spec)!={'kind','trees','depth','min_leaf'} or spec['kind']!='forest':
        raise ValueError('THERMAL_V2_FIT_MODEL_UNSUPPORTED')
    if len(paths)<2 or len(paths)!=len(policy['train_dataset_sha256']):
        raise ValueError('THERMAL_V2_TRAIN_ENVIRONMENTS_REQUIRED')
    x,y,ids,groups=[],[],[],[]
    for path in paths:
        root=Path(path)
        if any(z.lower() in {'blind','holdout','test','calibration'} for z in root.parts):
            raise ValueError('THERMAL_V2_FORBIDDEN_PATH')
        if sha_file(root/'dataset.json') != policy['train_dataset_sha256'].get(str(root)):
            raise ValueError('THERMAL_V2_TRAIN_DATA_BINDING_CHANGED')
        data=Dataset(root/'dataset.json',allowed_splits={'train','validation'})
        data.verify_files()
        if data.contract!=feature_contract():raise ValueError('THERMAL_V2_TRAIN_CONTRACT_CHANGED')
        train=data.load('train',128)
        val_groups={s['group_id'] for s in data.samples if s['split']=='validation'}
        if {s['group_id'] for s in train.samples}&val_groups:
            raise ValueError('THERMAL_V2_TRAIN_VALIDATION_GROUP_LEAKAGE')
        x.append(summarize(train.x));y.append(train.y)
        ids.extend(s['id'] for s in train.samples)
        groups.extend(s['group_id'] for s in train.samples)
    if len(ids)!=len(set(ids)) or len(groups)!=len(set(groups))*3:
        raise ValueError('THERMAL_V2_DUPLICATE_TRAIN_GROUPS')
    model=RandomForestClassifier(n_estimators=spec['trees'],max_depth=spec['depth'],
        min_samples_leaf=spec['min_leaf'],class_weight='balanced',random_state=19,n_jobs=1)
    tx=np.concatenate(x);ty=np.concatenate(y);model.fit(tx,ty)
    evidence={'role':'train-only','sample_count':int(len(ty)),
        'world_groups':len(set(groups)), 'sample_ids_digest':digest(sorted(ids)),
        'policy_sha256':sha_file(Path(policy_path)),
        'frozen_source_dataset_sha256':policy['train_dataset_sha256'],
        'fit_environment_ids':policy['fit_environment_ids'],
        'selected_from_development_only':True,
        'known_validation_leakage':False,'test_used':False}
    package=to_json_forest(model,evidence=evidence)
    target.mkdir(parents=True)
    write_json(target/'model.json',package)
    source={'schema':'phm.thermal-v2-model-release.v1','model_file_sha256':sha_file(target/'model.json'),
        'feature_transform_sha256':sha_file(Path(__file__).with_name('thermal_family_v2.py')),
        'inference_impl_sha256':sha_file(Path(__file__)),
        'training_data_role':'train-only','selection_data_role':'development-validation',
        'test_used':False,'deployment_qualified':False}
    write_json(target/'model_release.json',source)
    seal(target,[str(f.relative_to(target)) for f in target.rglob('*') if f.is_file()])
    return source


def predict_raw(model_dir: str, raw_csv: str, passport_json: str) -> dict:
    directory=Path(model_dir);verify(directory)
    release=read_json(directory/'model_release.json')
    if sha_file(directory/'model.json')!=release['model_file_sha256']:
        raise ValueError('THERMAL_V2_RELEASE_MUTATION')
    if sha_file(Path(__file__).with_name('thermal_family_v2.py'))!=release['feature_transform_sha256'] or sha_file(Path(__file__))!=release['inference_impl_sha256']:
        raise ValueError('THERMAL_V2_EXECUTOR_MUTATION')
    raw=sequence_from_csv(Path(raw_csv),native.contract('thermal'),128)
    features=compute_features(raw,read_json(passport_json))
    package=read_json(directory/'model.json')
    scores=predict_proba_from_features(features[None],package)[0]
    winner=int(scores.argmax())
    return {'schema':'phm.thermal-family-v2-prediction.v1',
        'prediction':package['classes'][winner], 'scores':scores.tolist(),
        'probability_calibrated':False, 'claim_status':'RESEARCH_ONLY',
        'model_sha256':sha_file(directory/'model.json')}
