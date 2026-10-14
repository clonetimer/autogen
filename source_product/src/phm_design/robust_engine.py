"""Quality-aware graph synthesis, training, recovery and sealed stress evaluation.

The train command accepts development manifests ONLY. Stress variants do not
increase independent-case counts. Unsupported samples count as errors/abstentions
in unconditional metrics; they are NEVER dropped from evaluation denominators.
"""
from __future__ import annotations
import copy
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.metrics import f1_score
from .data import Dataset,reject_overlap,reject_numeric_overlap
from .models import integer
from .engine import deterministic_cpu,environment,implementation_hash,_save_weights,_load_weights
from .io import digest,read_json,write_json,seal,verify,run_lock,safe_child,sha_file
from .robust_ops import Observation,number,validate_mechanism
from .robust_model import (generate_spec,mutate_spec,analyze_robust,build_model,fit_robust_scalers,code_binding)
from .robust_data import perturb,check_profile,packet_hash,packet_from_json


def decision_metrics(y,scores,computable,classes,threshold):
    y=np.asarray(y);scores=np.asarray(scores);computable=np.asarray(computable,dtype=bool)
    if scores.shape!=(len(y),len(classes)) or computable.shape!=(len(y),):raise ValueError('EVALUATION_SHAPE')
    if not np.isfinite(scores).all():raise ValueError('NONFINITE_EVALUATION')
    forced=scores.argmax(1);pred=np.where(computable,forced,-1)
    confidence=scores.max(1);accept=computable & (confidence>=threshold)
    correct=pred==y
    matrix=np.zeros((len(classes),len(classes)+1),dtype=int)
    for truth,p in zip(y,pred):matrix[truth,p if p>=0 else len(classes)]+=1
    by={c:{'cases':int((y==i).sum()),'correct':int((correct & (y==i)).sum()),
           'recall':float((correct & (y==i)).sum()/max(1,(y==i).sum()))} for i,c in enumerate(classes)}
    return {'cases':len(y),'correct':int(correct.sum()),'accuracy_all_cases':float(correct.mean()),
      'macro_f1_with_incomputable_as_errors':float(f1_score(y,pred,labels=list(range(len(classes))),average='macro',zero_division=0)),
      'computable_cases':int(computable.sum()),'computable_coverage':float(computable.mean()),
      'accepted_cases':int(accept.sum()),'accepted_correct':int((correct & accept).sum()),
      'accepted_accuracy':float(correct[accept].mean()) if accept.any() else None,
      'coverage':float(accept.mean()),'correct_and_accepted_fraction_all':float((correct & accept).mean()),
      'by_class':by,'confusion_columns':[*classes,'INSUFFICIENT_OBSERVATION'],'confusion_matrix':matrix.tolist(),
      'uncalibrated_threshold':threshold,'probability_calibrated':False,'unknown_fault_recognition_validated':False}


def inference(models,packet,inertia_multiplier=1.0):
    probs=[];computability=[];availability=[]
    with torch.inference_mode():
        for model in models:
            model.eval();r=model.assess(packet,inertia_multiplier=inertia_multiplier)
            probs.append(r['logits'].softmax(-1).numpy());computability.append(r['computable'].numpy());availability.append(r['branch_available'].numpy())
    if any(not np.array_equal(computability[0],v) for v in computability[1:]):raise ValueError('ENSEMBLE_QUALITY_DISAGREEMENT')
    return np.mean(probs,0),computability[0],availability[0]


def evaluate_profiles(models,prepared,y,classes,threshold):
    reports={};raw={}
    for p,obs in prepared:
        s,c,a=inference(models,obs,p['diagnostic_inertia_multiplier'])
        reports[p['id']]=decision_metrics(y,s,c,classes,threshold)
        reports[p['id']]['branches_available_mean']=a.mean(0).tolist()
        raw[p['id']]={'scores':s.tolist(),'computable':c.tolist()}
    return reports,raw


def rank_metrics(profiles,reports):
    chosen=[reports[p['id']]['macro_f1_with_incomputable_as_errors'] for p in profiles if p['role']=='selection']
    if not chosen:raise ValueError('NO_VALIDATION_SELECTION_PROFILES')
    return [float(min(chosen)),float(np.mean(chosen))]


def validate_request(r,contract):
    keys={'schema','window','proposal_seed','initial_candidates','mutations','max_proposals','grammar','seeds','training',
          'budget','quality_policy','mechanism','train_profiles','validation_profiles','evaluation_profiles',
          'train_perturbation_seed','validation_perturbation_seed','evaluation_perturbation_seed','threshold','evidence_scope'}
    if set(r)!=keys or r['schema']!='phm.robust-design.v1':raise ValueError('ROBUST_REQUEST_SCHEMA')
    if r['evidence_scope']!='hypothetical_measurement_stress_not_hardware_specification':raise ValueError('STRESS_EVIDENCE_SCOPE')
    validate_mechanism(contract,r['mechanism'])
    for k,lo,hi in [('window',3,4096),('initial_candidates',1,16),('mutations',0,16),('max_proposals',1,1000),('proposal_seed',0,2**32-1)]:integer(r[k],k,lo,hi)
    if not isinstance(r['seeds'],list) or not 2<=len(r['seeds'])<=6 or len(set(r['seeds']))!=len(r['seeds']):raise ValueError('MULTISEED_CONTRACT')
    for s in r['seeds']:integer(s,'seed',0,2**32-1)
    for k in ('train_perturbation_seed','validation_perturbation_seed','evaluation_perturbation_seed'):integer(r[k],k,0,2**32-1)
    if len({r[k] for k in ('train_perturbation_seed','validation_perturbation_seed','evaluation_perturbation_seed')})!=3:raise ValueError('SEPARATE_ROLE_PERTURBATION_SEEDS_REQUIRED')
    t=r['training'];b=r['budget'];g=r['grammar']
    if set(t)!={'epochs','batch_size','learning_rate','weight_decay','checkpoint_every'}:raise ValueError('ROBUST_TRAINING_SCHEMA')
    for k in ('epochs','batch_size','checkpoint_every'):integer(t[k],k,1,10000)
    number(t['learning_rate'],'learning_rate',1e-12,1);number(t['weight_decay'],'weight_decay',0,1)
    if set(b)!={'max_parameters','max_ensemble_parameters','max_trials','max_trial_seconds','max_neural_macs'}:raise ValueError('ROBUST_BUDGET_SCHEMA')
    for k in b: number(b[k],k,1,1e9)
    for k in ('max_parameters','max_ensemble_parameters','max_trials','max_neural_macs'):integer(b[k],k,1,1000000000)
    if (r['initial_candidates']+r['mutations'])*len(r['seeds'])>b['max_trials']:raise ValueError('TRIAL_BUDGET_NO_SILENT_TRUNCATION')
    required_grammar={'widths','max_depth','kernels','dilations'}
    optional_grammar={'pools','tail_pool_windows'}
    if not required_grammar.issubset(g) or set(g)-(required_grammar|optional_grammar):raise ValueError('ROBUST_GRAMMAR_SCHEMA')
    integer(g['max_depth'],'depth',1,3)
    for name,lo,hi in [('widths',2,64),('kernels',1,9),('dilations',1,4)]:
        if not isinstance(g[name],list) or not g[name] or len(set(g[name]))!=len(g[name]):raise ValueError('GRAMMAR_CHOICES')
        for v in g[name]:integer(v,name,lo,hi)
    if any(v%2==0 for v in g['kernels']):raise ValueError('ODD_KERNEL_REQUIRED')
    pools=g.get('pools',['mean','last','statistics'])
    allowed_pools={'mean','last','statistics','tail_mean'}
    if not isinstance(pools,list) or not pools or len(set(pools))!=len(pools) or any(x not in allowed_pools for x in pools):raise ValueError('ROBUST_POOL_CHOICES')
    if 'tail_mean' in pools:
        windows=g.get('tail_pool_windows')
        if not isinstance(windows,list) or not windows or len(set(windows))!=len(windows):raise ValueError('ROBUST_TAIL_WINDOWS_REQUIRED')
        for w in windows:integer(w,'tail_pool_window',1,r['window'])
    elif 'tail_pool_windows' in g:
        raise ValueError('TAIL_WINDOWS_WITHOUT_TAIL_POOL')
    c=len(contract['channels'])
    for name in ('train_profiles','validation_profiles','evaluation_profiles'):
        ps=r[name]
        if not isinstance(ps,list) or not 1<=len(ps)<=24 or len({p['id'] for p in ps})!=len(ps):raise ValueError('PROFILE_IDS')
        for p in ps:check_profile(p,c,r['window'])
    if not any(p['role']=='selection' for p in r['validation_profiles']):raise ValueError('NO_SELECTION_PROFILE')
    if any(p['role']=='evaluation_only' for p in r['train_profiles']+r['validation_profiles']):raise ValueError('EVALUATION_PROFILE_IN_DEVELOPMENT')
    number(r['threshold'],'threshold',0,1)


def trial(spec,seed,request,train_packets,y,val_packets,vy,classes,dest):
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=True)
    fingerprint=digest({'spec':spec,'seed':seed,'training':request['training'],'code':code_binding()})
    if (dest/'manifest.json').exists():
        verify(dest);old=read_json(dest/'trial.json')
        if old['fingerprint']!=fingerprint:raise ValueError('TRIAL_RESUME_CHANGED')
        return old
    model=build_model(spec,seed,request['budget']['max_parameters']);start=time.perf_counter()
    t=request['training'];optimizer=torch.optim.AdamW(model.parameters(),lr=t['learning_rate'],weight_decay=t['weight_decay'])
    counts=np.bincount(y,minlength=len(classes));loss_fn=nn.CrossEntropyLoss(weight=torch.tensor(len(y)/(len(classes)*counts),dtype=torch.float32))
    rng=np.random.default_rng(seed);best=None;best_rank=(-1.,-1.);history=[];updates=0
    stacked=Observation(*(torch.cat([getattr(o,k) for _,o in train_packets],0) for k in ('values','observed','source_s','time_s')))
    labels_all=np.tile(y,len(train_packets))
    multipliers=np.repeat([p['diagnostic_inertia_multiplier'] for p,_ in train_packets],len(y))
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(torch.Generator(device='cpu').manual_seed(seed).get_state())
        for epoch in range(1,t['epochs']+1):
            model.train();losses=[];skipped=0
            # A deterministic profile schedule preserves equal use of clean and
            # perturbed observations; repeated views are not independent cases.
            order=rng.permutation(len(y)*len(train_packets))
            for a in range(0,len(order),t['batch_size']):
                if time.perf_counter()-start>request['budget']['max_trial_seconds']:raise TimeoutError('ROBUST_TRIAL_BUDGET')
                ids=order[a:a+t['batch_size']];optimizer.zero_grad(set_to_none=True)
                terms=[];denom=0
                for multiplier in np.unique(multipliers[ids]):
                    base=ids[multipliers[ids]==multiplier]
                    r=model.assess(stacked.select(base),inertia_multiplier=float(multiplier))
                    usable=r['computable'];skipped+=int((~usable).sum())
                    if usable.any():
                        labels=torch.from_numpy(labels_all[base])[usable];n=int(usable.sum())
                        terms.append(loss_fn(r['logits'][usable],labels)*n);denom+=n
                if not terms:continue
                loss=sum(terms)/denom
                if not torch.isfinite(loss):raise ValueError('ROBUST_TRAINING_NONFINITE')
                loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);optimizer.step();updates+=1;losses.append(float(loss.detach()))
            if not losses:raise ValueError('NO_COMPUTABLE_TRAINING_BATCH')
            if epoch%t['checkpoint_every']==0 or epoch==t['epochs']:
                reports,_=evaluate_profiles([model],val_packets,vy,classes,request['threshold'])
                rank=tuple(rank_metrics(request['validation_profiles'],reports))
                history.append({'epoch':epoch,'training_loss':float(np.mean(losses)),'validation_rank':list(rank),'uncomputable_training_views':skipped})
                if rank>best_rank:
                    best_rank=rank;best={k:v.detach().clone() for k,v in model.state_dict().items()};best_epoch=epoch
    if best is None:raise ValueError('NO_FINITE_ROBUST_CHECKPOINT')
    model.load_state_dict(best);model.eval();reports,scores=evaluate_profiles([model],val_packets,vy,classes,request['threshold'])
    _save_weights(model,dest/'model.pt')
    record={'status':'COMPLETED','fingerprint':fingerprint,'seed':seed,'best_epoch':best_epoch,'epochs_executed':t['epochs'],
       'updates':updates,'rank':list(best_rank),'validation':reports,'scores':scores,'history':history,
       'elapsed_seconds':time.perf_counter()-start,'weights_sha256':sha_file(dest/'model.pt')}
    write_json(dest/'trial.json',record);seal(dest,['model.pt','trial.json']);return record


def design_robust(dataset_path,request_path,binding_path,output,*,resume=False,on_trial_complete=None):
    data=Dataset(dataset_path,allowed_splits={'train','validation'});data.verify_files();r=read_json(request_path);validate_request(r,data.contract)
    if sha_file(Path(binding_path))!=r['mechanism']['source']['sha256']:raise ValueError('PHYSICAL_SOURCE_HASH_MISMATCH')
    train=data.load('train',r['window']);val=data.load('validation',r['window']);reject_numeric_overlap(train,val)
    tp=[(p,perturb(train.x,train.samples,data.contract['sample_s'],p,r['train_perturbation_seed'])) for p in r['train_profiles']]
    vp=[(p,perturb(val.x,val.samples,data.contract['sample_s'],p,r['validation_perturbation_seed'])) for p in r['validation_profiles']]
    scalers=fit_robust_scalers(train.x,[o for _,o in tp],data.contract,r['mechanism'])
    fingerprint=digest({'request':r,'data':data.hash,'code':implementation_hash(),'environment':environment(),'scalers':scalers})
    out=Path(output)
    with run_lock(out),deterministic_cpu():
        if (out/'plan.json').exists():
            if read_json(out/'plan.json')['fingerprint']!=fingerprint:raise ValueError('ROBUST_RESUME_INPUT_OR_CODE_CHANGED')
            if not resume:raise FileExistsError('RUN_EXISTS_USE_RESUME')
            if (out/'manifest.json').exists():verify(out);return read_json(out/'report.json')
        else:
            write_json(out/'plan.json',{'schema':'phm.robust-plan.v1','fingerprint':fingerprint,'request':r,'environment':environment(),
                'data_hash':data.hash,'source_hash':sha_file(Path(binding_path)),'holdout_used':False,'counts':data.counts(),
                'independent_training_cases':len(train.y),'training_views':len(train.y)*len(tp),'scalers':scalers,
                'train_packet_hashes':{p['id']:packet_hash(o) for p,o in tp},'validation_packet_hashes':{p['id']:packet_hash(o) for p,o in vp}})
        write_json(out/'development_snapshot.json',data.raw)
        write_json(out/'observation_hashes.json',[digest(x.tolist()) for x in np.concatenate([train.x,val.x])])
        write_json(out/'runtime_binding.json',read_json(binding_path))
        rng=random.Random(r['proposal_seed']);candidates=[];seen=set();rejects=[]
        total=r['initial_candidates']+r['mutations'];attempt=0
        while len(candidates)<total and attempt<r['max_proposals']:
            attempt+=1;parent=None
            if len(candidates)<r['initial_candidates']:
                spec,genes=generate_spec(rng,data.contract,r['mechanism'],scalers,r['window'],r['grammar'],r['quality_policy'])
            else:
                # Mutation parents depend ONLY on development performance.
                eligible=sorted([c for c in candidates if c['status']=='COMPLETED'],key=lambda c:(-c['rank'][0],-c['rank'][1],c['resources']['parameter_count'],c['id']))
                if not eligible:break
                parent=eligible[0];spec,genes=mutate_spec(parent['spec'],rng,r['grammar'])
            key=digest(spec)[:20]
            try:
                if key in seen:raise ValueError('DUPLICATE_GRAPH')
                resources=analyze_robust(spec)
                if resources['parameter_count']>r['budget']['max_parameters'] or resources['parameter_count']*len(r['seeds'])>r['budget']['max_ensemble_parameters'] or resources['neural_mac_estimate']>r['budget']['max_neural_macs']:
                    raise ValueError('ROBUST_RESOURCE_BUDGET')
            except ValueError as exc:
                rejects.append({'attempt':attempt,'reason':str(exc)});continue
            seen.add(key);folder=out/'candidates'/key;write_json(folder/'architecture.json',spec);trials=[]
            for seed in r['seeds']:
                try:rec=trial(spec,seed,r,tp,train.y,vp,val.y,data.contract['classes'],folder/f'seed_{seed}')
                except Exception as exc:
                    rec={'status':'FAILED','seed':seed,'reason':str(exc),'type':type(exc).__name__};write_json(folder/f'seed_{seed}'/'failure.json',rec)
                trials.append(rec)
                if on_trial_complete:on_trial_complete({'candidate':key,**rec})
            complete=all(t['status']=='COMPLETED' for t in trials)
            record={'id':key,'spec':spec,'genes':genes,'parent_id':parent['id'] if parent else None,'resources':resources,
               'status':'COMPLETED' if complete else 'INCOMPLETE_SEEDS','trials':trials}
            if complete:record['rank']=np.mean([t['rank'] for t in trials],0).tolist()
            candidates.append(record);write_json(out/'progress.json',{'candidates':candidates,'rejected':rejects})
        complete=len(candidates)==total and all(c['status']=='COMPLETED' for c in candidates)
        eligible=sorted([c for c in candidates if c['status']=='COMPLETED'],key=lambda c:(-c['rank'][0],-c['rank'][1],c['resources']['parameter_count'],c['id']))
        winner=eligible[0] if eligible else None
        report={'schema':'phm.robust-report.v1','status':'COMPLETED' if complete else 'INCOMPLETE_SEARCH','fingerprint':fingerprint,
            'candidates':candidates,'proposal_rejections':rejects,'generated_candidates':len(candidates),'training_trials':sum(len(c['trials']) for c in candidates),
            'selected_id':winner['id'] if complete and winner else None,'selection':'mean_seed_worst_profile_macro_f1_then_mean_then_parameters',
            'test_used':False,'claim_status':'RESEARCH_ONLY','real_device_qualified':False,'production_release_qualified':False,
            'safety_gate':'incomputable -> no predicted class, scores null','same_base_augmentations_are_not_independent_evidence':True}
        if complete and winner:
            weights=[f"candidates/{winner['id']}/seed_{s}/model.pt" for s in r['seeds']]
            bundle={'schema':'phm.robust-ensemble.v1','spec':winner['spec'],'resources':winner['resources'],'fingerprint':fingerprint,
                'seeds':r['seeds'],'weights':weights,'max_parameters':r['budget']['max_parameters'],'threshold':r['threshold'],
                'environment':environment(),'code':code_binding(),'claim_status':'RESEARCH_ONLY','normalization_roles':['train']}
            write_json(out/'model_bundle.json',bundle)
        write_json(out/'report.json',report)
        if complete:seal(out,[str(p.relative_to(out)) for p in out.rglob('*') if p.is_file() and p.name not in {'.lock','manifest.json'} and not p.name.endswith('.tmp')])
        return report


def load_robust(run):
    run=Path(run);files=verify(run)
    if 'model_bundle.json' not in files:raise ValueError('NO_FROZEN_ROBUST_MODEL')
    b=read_json(run/'model_bundle.json')
    if b.get('schema')!='phm.robust-ensemble.v1' or b['environment']['torch']!=str(torch.__version__) or b['code']!=code_binding():raise ValueError('ROBUST_MODEL_IMPLEMENTATION_CHANGED')
    if analyze_robust(b['spec'])!=b['resources']:raise ValueError('ROBUST_ARCHITECTURE_CHANGED')
    models=[]
    for seed,path in zip(b['seeds'],b['weights'],strict=True):
        if path not in files:raise ValueError('UNSEALED_ROBUST_WEIGHT')
        model=build_model(b['spec'],seed,b['max_parameters']);_load_weights(model,safe_child(run,path));models.append(model)
    if not models:raise ValueError('EMPTY_ENSEMBLE')
    return b,models


def predict_robust(run,packet_path):
    with deterministic_cpu():
        b,models=load_robust(run);p=packet_from_json(read_json(packet_path),b['spec']['contract'],b['spec']['window'])
        scores,c,a=inference(models,p);classes=b['spec']['contract']['classes'];records=[]
        for i in range(len(scores)):
            accepted=bool(c[i] and scores[i].max()>=b['threshold'])
            records.append({'status':'PREDICTED' if accepted else ('LOW_SCORE' if c[i] else 'INSUFFICIENT_OBSERVATION'),
                'prediction':classes[int(scores[i].argmax())] if accepted else None,'scores':scores[i].tolist() if c[i] else None,
                'available_branches':a[i].tolist()})
        return {'schema':'phm.robust-prediction.v1','records':records,'probability_calibrated':False,'claim_status':'RESEARCH_ONLY'}


def evaluate_robust(run,holdout,output):
    run=Path(run);out=Path(output)
    with run_lock(run),deterministic_cpu():
        b,models=load_robust(run);plan=read_json(run/'plan.json');r=plan['request']
        data=Dataset(holdout,allowed_splits={'test'})
        if data.contract!=b['spec']['contract']:raise ValueError('ROBUST_TEST_CONTRACT_CHANGED')
        reject_overlap(read_json(run/'development_snapshot.json')['samples'],data.samples)
        key=digest({'run':b['fingerprint'],'holdout':data.hash,'profiles':r['evaluation_profiles']})
        ledger=run/'robust_evaluation_ledger.json'
        if ledger.exists():
            old=read_json(ledger)
            if old['key']!=key:raise ValueError('DIFFERENT_HOLDOUT_ALREADY_OPENED')
            if old['status']=='COMPLETE':return old['report']
            raise ValueError('PREVIOUS_OPENING_INCOMPLETE_REVIEW_REQUIRED')
        if out.exists():raise FileExistsError('EVALUATION_OUTPUT_EXISTS')
        data.verify_files();write_json(ledger,{'key':key,'status':'OPENED','evaluation_profiles':r['evaluation_profiles']})
        block=data.load('test',r['window'])
        if set(read_json(run/'observation_hashes.json')) & {digest(x.tolist()) for x in block.x}:raise ValueError('TEST_NUMERIC_OVERLAP')
        prepared=[(p,perturb(block.x,block.samples,data.contract['sample_s'],p,r['evaluation_perturbation_seed'])) for p in r['evaluation_profiles']]
        reports,scores=evaluate_profiles(models,prepared,block.y,data.contract['classes'],r['threshold'])
        report={'schema':'phm.robust-evaluation.v1','status':'COMPLETE','key':key,'base_case_count':len(block.y),
            'base_parameter_groups':len(block.groups),'profile_reports':reports,'predictions':scores,'classes':data.contract['classes'],
            'profile_packet_hashes':{p['id']:packet_hash(o) for p,o in prepared},'base_samples':block.samples,
            'stress_source':'postprocessed simulated observations; parameter multiplier changes diagnostic model only, NOT plant dynamics',
            'views_do_not_increase_independent_n':True,'real_device_qualified':False,'production_release_qualified':False}
        write_json(out/'evaluation_report.json',report);seal(out,['evaluation_report.json']);write_json(ledger,{'key':key,'status':'COMPLETE','report':report})
        return report
