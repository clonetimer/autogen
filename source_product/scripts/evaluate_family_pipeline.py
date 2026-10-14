#!/usr/bin/env python3
# 用途：冻结多域模型的独立评价与恢复检查；参数：--campaign、--designs、--pipelines、--output、--protocol；输出：指标、冻结记录和哈希。
"""One-off frozen multi-domain evaluation. No test-fed architecture changes.

Use new --output. All requested model selections must already be sealed.
Derived holdout CSVs are prepared only after the selection commitment is saved.
Thermal context shifts change diagnostic constants, not simulated dynamics.
"""
from __future__ import annotations
import argparse,copy
from pathlib import Path
import numpy as np
from phm_design.io import read_json,write_json,verify,sha_file,seal,digest
from phm_design.family_pipeline import (prepare_family_section,predict_pipeline,compute_features,implementation_sha)
from phm_design.data import Dataset,sequence_from_csv,transform
from phm_design.engine import evaluate,metrics,deterministic_cpu,load_ensemble,ensemble_scores


def run(campaign,designs,pipelines,output,protocol):
    campaign,designs,pipelines,output=map(Path,(campaign,designs,pipelines,output))
    plan=read_json(protocol)
    if plan['feature_source_sha256']!=implementation_sha():raise ValueError('FEATURE_DEFINITION_CHANGED')
    output.mkdir(parents=True,exist_ok=True)
    snapshots={}
    for domain in plan['domains']:
        for mode in plan['arms']:
            for name in plan['predetermined_exports']:
                model=designs/domain/mode/name;verify(model)
                snapshots[f'{domain}/{mode}/{name}']=sha_file(model/'manifest.json')
    commitment={'models':snapshots,'protocol_sha256':digest(plan),
      'evaluation_source_sha256':sha_file(Path(__file__)),'status':'FROZEN_BEFORE_HOLDOUT_PREPARATION'}
    if (output/'selection_commitment.json').exists():
        old=read_json(output/'selection_commitment.json')
        if old['models']!=snapshots or old['protocol_sha256']!=digest(plan):
            raise ValueError('FROZEN_EVALUATION_CHANGED')
        write_json(output/'evaluation_resume_note.json',{'reason':'single versus batch score roundoff; no model, data or threshold changes',
            'original_evaluation_source_sha256':old['evaluation_source_sha256'],'current_evaluation_source_sha256':sha_file(Path(__file__)),
            'score_rtol':1e-5,'score_atol':1e-7,'prior_COMPLETE_evaluations_reused':True})
    else:write_json(output/'selection_commitment.json',commitment)
    result={};rows=[];direct_checks=[]
    for domain in plan['domains']:
        result[domain]={}
        for mode in plan['arms']:
            prepared=output/'prepared_holdout'/domain/mode
            if prepared.exists():verify(prepared)
            else:prepare_family_section(campaign,domain,'holdout',mode,prepared)
            per={}
            for name in plan['predetermined_exports']:
                report=evaluate(designs/domain/mode/name,prepared/'dataset.json',output/'reports'/domain/mode/name)
                per[name]=report
                n=report['neural'];correct=int(np.trace(n['confusion_matrix']))
                rows.append({'domain':domain,'mode':mode,'selection':name,'correct':correct,'cases':n['cases'],
                    'macro_f1':n['macro_f1'],'baseline':report['baseline'],
                    'selective':report['selective_prediction'],'confusion_matrix':n['confusion_matrix']})
                sample=read_json(campaign/'native'/domain/'holdout/dataset.json')['samples'][0]
                raw=campaign/'native'/domain/'holdout'/sample['path']
                passport=campaign/'passports'/domain/(sample['lineage_id']+'.json')
                prediction=predict_pipeline(pipelines/domain/mode/name,raw,passport)
                row=next(r for r in report['predictions'] if r['id']==sample['id'])
                np.testing.assert_allclose(prediction['scores'],row['class_scores'],rtol=1e-5,atol=1e-7)
                assert prediction['closed_set_argmax']==row['prediction']
                direct_checks.append({'domain':domain,'mode':mode,'selection':name,'id':sample['id'],
                    'scores_match_within_tolerance':True,'maximum_absolute_difference':float(np.max(np.abs(np.asarray(prediction['scores'])-row['class_scores']))),
                    'rtol':1e-5,'atol':1e-7,'class_exact_match':True})
                print(domain,mode,name,correct,n['cases'],n['macro_f1'],flush=True)
            result[domain][mode]=per
    # Cluster resampling preserves all three variants of a parameter group.
    comparisons={};rng=np.random.default_rng(plan['bootstrap_seed'])
    for domain in plan['domains']:
        a=result[domain]['mechanism']['selected_composed'];b=result[domain]['context']['selected_composed']
        br={r['id']:r for r in b['predictions']};ar=a['predictions'];classes=list(a['neural']['recall_by_class'])
        groups=sorted({r['group_id'] for r in ar});index={g:[i for i,r in enumerate(ar) if r['group_id']==g] for g in groups}
        y=np.asarray([classes.index(r['truth']) for r in ar]);pa=np.asarray([classes.index(r['prediction']) for r in ar]);pb=np.asarray([classes.index(br[r['id']]['prediction']) for r in ar])
        from sklearn.metrics import f1_score
        differences=[]
        for _ in range(plan['paired_bootstrap_replicates']):
            idx=np.concatenate([index[g] for g in rng.choice(groups,len(groups),replace=True)])
            differences.append(float(f1_score(y[idx],pa[idx],labels=range(len(classes)),average='macro',zero_division=0)-f1_score(y[idx],pb[idx],labels=range(len(classes)),average='macro',zero_division=0)))
        comparisons[domain]={'parameter_groups':len(groups),'macro_f1_difference':a['neural']['macro_f1']-b['neural']['macro_f1'],
            'paired_cluster_bootstrap_percentile95':np.quantile(differences,[.025,.975]).tolist(),
            'scope':'exploratory same-simulator interval; excludes search-process variance and simulator discrepancy',
            'zero_error_class_miss_upper95_if_n16':1-.05**(1/16)}
    # Fixed diagnostic-parameter stress, already declared in the protocol.
    stresses=[]
    d=Dataset(campaign/'native/thermal/holdout/dataset.json',allowed_splits={'test'});d.verify_files()
    for mode in plan['arms']:
        with deterministic_cpu():
            bundle,models=load_ensemble(pipelines/'thermal'/mode/'selected_composed/neural')
            for scale in plan['parameter_stress']['thermal_heat_capacity_scale']:
                feats=[];labels=[]
                for s in d.samples:
                    raw=sequence_from_csv(d.root/s['path'],d.contract,bundle['shape']['window'])
                    p=read_json(campaign/'passports/thermal'/(s['lineage_id']+'.json'))
                    p['parameters']['normal_heat_capacity_j_per_k']['value']*=scale
                    feats.append(compute_features('thermal',raw,p,mode));labels.append(d.contract['classes'].index(s['label']))
                scores=ensemble_scores(models,transform(np.stack(feats),bundle['scaler']))
                y=np.asarray(labels);pred=scores.argmax(1);accepted=scores.max(1)>=plan['threshold']
                stresses.append({'domain':'thermal','mode':mode,'diagnostic_C_multiplier':scale,
                    **metrics(y,pred,d.contract['classes']),'coverage':float(accepted.mean()),
                    'accepted_errors':int(((pred!=y)&accepted).sum()),'simulated_object_changed':False})
    summary={'status':'SIMULATION_RESEARCH_ONLY','rows':rows,'comparisons':comparisons,'thermal_parameter_stresses':stresses,
             'pipeline_raw_inference_checks':direct_checks,'probability_calibrated':False,'production_qualified':False,
             'normal_parameter_availability':'ideal exact pre-injection nominal design metadata in primary test',
             'old_test_used_for_design':False,'calibration_used':False}
    write_json(output/'results.json',summary)
    seal(output,[str(p.relative_to(output)) for p in output.rglob('*') if p.is_file()])
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--campaign',required=True);p.add_argument('--designs',required=True)
    p.add_argument('--pipelines',required=True);p.add_argument('--output',required=True);p.add_argument('--protocol',required=True)
    a=p.parse_args();run(a.campaign,a.designs,a.pipelines,a.output,a.protocol)
