#!/usr/bin/env python3
# 用途：冻结非ADCS设计并比较独立测试；参数：--data、--designs、--output；输出：分域指标、冻结记录与哈希。
"""Research non-ADCS comparison. Freeze models/rule BEFORE opening test data.

Run only after all three family designs are complete. Calibration remains unused.
No cutoff is chosen from holdout. This is not a risk-qualified release gate.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from phm_design.data import Dataset,reject_overlap,sequence_from_csv
from phm_design.engine import evaluate,metrics
from phm_design.io import read_json,write_json,digest,sha_file,verify,seal,safe_child


def run(data_root,design_root,output):
    data_root,design_root,output=map(Path,(data_root,design_root,output))
    if output.exists():raise FileExistsError("SUITE_EVALUATION_ALREADY_EXISTS")
    output.mkdir(parents=True)
    snapshots={}
    for domain in ('eps','thermal','data'):
        for kind in ('composed','control'):
            root=design_root/domain/f'selected_{kind}'
            verify(root)
            snapshots[f'{domain}/{kind}']={'bundle_sha256':sha_file(root/'model_bundle.json'),
                'weights':{str(p.relative_to(root)):sha_file(p) for p in sorted((root/'weights').glob('*.pt'))}}
    dev=Dataset(data_root/'eps/development/dataset.json',allowed_splits={'train','validation'})
    train=dev.load('train',64);nominal=train.x[train.y==0].astype(float)
    # EPS column order is checked rather than assumed for arbitrary datasets.
    if [c['name'] for c in dev.contract['channels']]!=['illumination','load_command_w','panel_meter_w','bus_net_meter_w']:
        raise ValueError('EPS_RULE_INPUT_ORDER')
    r=nominal[:,:,2]-nominal[:,:,3]-nominal[:,:,1]
    rule={'type':'train_only_power_balance_rule','extra_load_threshold_w':float(np.quantile(np.abs(r),.995)),
       'illuminated_panel_lower_w':float(np.quantile(nominal[:,:,2][nominal[:,:,0]>.5],.005)),
       'fit_role':'normal_train_only','validity':'heuristic_on_this_simulator_not_calibrated_risk',
       'extra_load_definition':'mean(panel_meter - bus_net_meter - requested_load)',
       'solar_definition':'mean(panel_meter in observed illuminated samples)',
       'quantiles_predeclared':[.995,.005]}
    write_json(output/'frozen_models_and_rule.json',{'models':snapshots,'eps_rule':rule,'test_read_for_freeze':False,
        'evaluation_source_sha256':sha_file(Path(__file__))})
    reports={}
    for domain in ('eps','thermal','data'):
        reports[domain]={}
        for kind in ('composed','control'):
            reports[domain][kind]=evaluate(design_root/domain/f'selected_{kind}',
                data_root/domain/'holdout/dataset.json',output/domain/kind)
    hold=Dataset(data_root/'eps/holdout/dataset.json',allowed_splits={'test'});hold.verify_files();reject_overlap(dev.samples,hold.samples)
    test=hold.load('test',64);x=test.x.astype(float)
    residual=(x[:,:,2]-x[:,:,3]-x[:,:,1]).mean(1)
    solar=np.array([row[:,2][row[:,0]>.5].mean() if (row[:,0]>.5).any() else np.nan for row in x])
    if not np.isfinite(solar).all():raise ValueError('EPS_RULE_NO_ILLUMINATED_OBSERVATION')
    prediction=np.where(residual>rule['extra_load_threshold_w'],2,np.where(solar<rule['illuminated_panel_lower_w'],1,0))
    extra={'rule':rule,'metrics':metrics(test.y,prediction,hold.contract['classes']),
           'predictions':[{'id':s['id'],'group_id':s['group_id'],'label':int(y),'prediction':int(p)} for s,y,p in zip(test.samples,test.y,prediction)],
           'test_used_for_rule_selection':False}
    write_json(output/'eps/power_balance_rule.json',extra)
    summary={}
    for domain,results in reports.items():
        design=read_json(design_root/domain/'design_report.json');best=design['best_composed'];control=design['best_fixed_neural_control']
        summary[domain]={
            'test_cases':results['composed']['neural']['cases'],
            'test_world_groups':results['composed']['counts']['test']['groups'],
            'generated':results['composed']['neural'],'fixed_neural':results['control']['neural'],
            'non_neural':results['composed']['baseline'],
            'generated_selective':results['composed']['selective_prediction'],
            'fixed_selective':results['control']['selective_prediction'],
            'generated_structure':{'id':best['id'],'parameters_per_model':best['description']['parameter_count'],
                'graph_nodes':len(best['spec']['parameters']['nodes']),'generation':best['generation'],
                'edit':best['edit'],'branches':len(best['genes']['branches']),'fusion':best['genes']['fusion'],
                'validation_macro_f1_seed_mean':best['mean_validation_macro_f1']},
            'fixed_structure':{'id':control['id'],'spec':control['spec'],'parameters_per_model':control['description']['parameter_count'],
                'validation_macro_f1_seed_mean':control['mean_validation_macro_f1']},
            'generated_graphs':design['generated_graphs'],'candidate_count':design['candidate_count'],
        }
    summary['eps']['power_balance_rule']=extra['metrics']
    result={'status':'SIMULATION_RESEARCH_COMPARISON','domains':summary,'test_read_for_selection':False,
        'calibration_used':False,'same_design_policy_for_all_families':True,
        'cross_domain_weight_transfer_demonstrated':False,'production_qualified':False,
        'limitations':['prescribed boundary low-fidelity modules','12 independent world roots in test, not 108',
          '4 generated proposals vs 2 fixed controls; not equal NAS algorithm search compute',
          'complete observations only outside RW robust v0.5 domain','shared simulator and generator',
          'confidence threshold 0.6 is not calibrated','all three domains now development benchmarks for future designer revisions']}
    write_json(output/'results_summary.json',result)
    seal(output,[str(p.relative_to(output)) for p in output.rglob('*') if p.is_file() and p.name!='manifest.json'])
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True);p.add_argument('--designs',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();result=run(a.data,a.designs,a.output)
    import json;print(json.dumps(result,indent=2))
