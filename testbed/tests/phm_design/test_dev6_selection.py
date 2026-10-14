import copy
import csv
import shutil
from pathlib import Path
import numpy as np
import pytest
from phm_design.io import read_json,write_json,sha_file,seal,verify
from phm_design.data import Dataset
from phm_design.reference_design import design_reference
from phm_design.route_selection import (POLICY,route_eligible,route_rank,select_route,evaluate_suite)
from phm_design.evidence_ledger import EvidenceLedger
from tests.phm_design.test_design_system import fixture_dataset
from tests.phm_design.test_reference_dev3 import make_policy
from tests.phm_design.test_compositional_design import request


def metric(f1,recalls,errors=0,correct=8):
    return {'macro_f1':f1,'recall_by_class':dict(zip(['C0','C1','C2'],recalls)),
            'cases':12,'accepted_errors':errors,'accepted_correct':correct,'accepted_cases':errors+correct}


def test_each_label_guard_catches_minimum_recall_identity_swap():
    ref=metric(.75,[.5,.9,.9]);candidate=metric(.80,[.6,.6,1.0])
    weakest=copy.deepcopy(POLICY);weakest['recall_guard']='weakest_class'
    assert route_eligible(candidate,candidate,ref,ref,'raw',weakest)
    assert not route_eligible(candidate,candidate,ref,ref,'raw')


def test_eligible_only_when_known_classes_do_not_degrade():
    ref=metric(.7,[.5,.6,.8]);better=metric(.8,[.6,.7,.9])
    assert route_eligible(better,better,ref,ref,'raw')
    assert not route_eligible(ref,ref,ref,ref,'bounded')
    assert route_eligible(ref,ref,ref,ref,'reference')


@pytest.mark.parametrize('bad',[float('nan'),-1.,2.,True])
def test_invalid_metric_rejected(bad):
    a=metric(.7,[.5,.6,.8]);b=copy.deepcopy(a);b['macro_f1']=bad
    with pytest.raises(ValueError):route_eligible(b,a,a,a,'raw')


def calibration(root,contract):
    rng=np.random.default_rng(9917);samples=[]
    for label in range(3):
        for j in range(2):
            ident=f'cal-{label}-{j}';p=root/(ident+'.csv');p.parent.mkdir(parents=True,exist_ok=True)
            x=rng.normal(label*2,.14,(12,2))
            with p.open('w',newline='') as s:
                w=csv.writer(s);w.writerow(['time_s',*[c['name'] for c in contract['channels']]])
                w.writerows([[i,*v] for i,v in enumerate(x)])
            samples.append({'id':ident,'lineage_id':ident,'group_id':ident,'split':'calibration',
                'label':contract['classes'][label],'path':p.name,'sha256':sha_file(p)})
    f=root/'dataset.json';write_json(f,{'schema':'phm.sequence-dataset.v1','dataset_id':'cal',
        'contract':contract,'samples':samples,'provenance':{'source':'software_fixture'}})
    return f


@pytest.fixture(scope='module')
def trained(tmp_path_factory):
    p=tmp_path_factory.mktemp('dev6-frozen');dev,test=fixture_dataset(p/'data');d=Dataset(dev,allowed_splits={'train','validation'})
    b={'schema':'phm.device-binding.v1','contract':d.contract,'groups':request()['groups'],
       'window_samples':8,'quality_mode':'complete_observation_only_missing_values_rejected'}
    write_json(p/'b.json',b);write_json(p/'p.json',make_policy())
    design_reference(dev,p/'b.json',p/'p.json',p/'run')
    cal=calibration(p/'cal',d.contract)
    return p,dev,test,cal


@pytest.fixture
def inputs(trained,tmp_path):
    p,dev,test,cal=trained
    shutil.copytree(p/'run',tmp_path/'run')
    return tmp_path/'run',dev,test,cal,tmp_path/'db.sqlite',tmp_path/'selection',tmp_path/'eval'


def test_select_then_paired_suite_cache_replay(inputs):
    run,dev,test,cal,db,sel,out=inputs
    r=select_route(run,cal,sel,registry=db)
    assert not r['test_used'] and r['calibration_role_consumed_for_route_selection']
    assert select_route(run,cal,sel,registry=db)==r
    rep=evaluate_suite(run,sel,{'iid':test,'paired_copy':test},out,registry=db)
    assert rep['union_test_groups']==6 and rep['cohort_case_views']==12
    hashes={str(p):sha_file(p) for p in out.iterdir() if p.is_file()}
    assert evaluate_suite(run,sel,{'iid':test,'paired_copy':test},out,registry=db)==rep
    assert hashes=={str(p):sha_file(Path(p)) for p in hashes}


def test_route_selector_rejects_test_role(inputs):
    run,dev,test,cal,db,sel,out=inputs
    with pytest.raises(ValueError,match='DATA_ROLE'):select_route(run,test,sel,registry=db)


def test_role_consumption_blocks_new_policy_even_with_new_output(inputs):
    run,dev,test,cal,db,sel,out=inputs
    select_route(run,cal,sel,registry=db)
    p=copy.deepcopy(POLICY);p['recall_guard']='weakest_class'
    with pytest.raises(ValueError,match='CALIBRATION_ALREADY_CONSUMED'):
        select_route(run,cal,sel.with_name('again'),registry=db,policy=p)


def test_after_holdout_cannot_reselect(inputs):
    run,dev,test,cal,db,sel,out=inputs
    select_route(run,cal,sel,registry=db);evaluate_suite(run,sel,{'iid':test},out,registry=db)
    p=copy.deepcopy(POLICY);p['recall_guard']='weakest_class'
    with pytest.raises(ValueError,match='SOURCE_HOLDOUT_ALREADY_OPENED'):
        select_route(run,cal,sel.with_name('again'),registry=db,policy=p)


def test_changed_suite_cannot_be_opened(inputs):
    run,dev,test,cal,db,sel,out=inputs
    select_route(run,cal,sel,registry=db);evaluate_suite(run,sel,{'iid':test},out,registry=db)
    with pytest.raises(ValueError,match='SOURCE_HOLDOUT_ALREADY_OPENED'):
        evaluate_suite(run,sel,{'other':test},out.with_name('newtest'),registry=db)


def test_result_tampering_detected_on_replay(inputs):
    run,dev,test,cal,db,sel,out=inputs
    select_route(run,cal,sel,registry=db);evaluate_suite(run,sel,{'iid':test},out,registry=db)
    (out/'evaluation_report.json').write_text('{}')
    with pytest.raises(ValueError,match='INTEGRITY'):
        evaluate_suite(run,sel,{'iid':test},out,registry=db)


def test_numerical_copy_with_new_ids_and_new_csv_bytes_rejected(inputs,tmp_path):
    run,dev,test,cal,db,sel,out=inputs
    raw=read_json(dev);data=Dataset(dev,allowed_splits={'train','validation'})
    rows=[];dest=tmp_path/'numeric_copy';dest.mkdir()
    for i,s in enumerate([x for x in raw['samples'] if x['split']=='train']):
        csvrows=list(csv.reader((data.root/s['path']).open()))
        ident=f'renamed-{i}';p=dest/(ident+'.csv')
        with p.open('w',newline='') as f:
            w=csv.writer(f);w.writerow(csvrows[0]);w.writerows([[format(float(x),'.18f') for x in r] for r in csvrows[1:]])
        rows.append({**s,'id':ident,'group_id':ident,'lineage_id':ident,'split':'calibration','path':p.name,'sha256':sha_file(p)})
    raw['samples']=rows;raw['dataset_id']='renamed';write_json(dest/'dataset.json',raw)
    with pytest.raises(ValueError,match='IDENTICAL_OBSERVATIONS'):
        select_route(run,dest/'dataset.json',sel,registry=db)


def test_reservation_is_not_released_after_interruption(tmp_path):
    with EvidenceLedger(tmp_path/'db') as ledger:
        assert ledger.claim('key','selection','source',tmp_path/'out','domain',[]) is None
    with EvidenceLedger(tmp_path/'db') as ledger:
        with pytest.raises(ValueError,match='ALREADY_OPENED'):
            ledger.claim('key','selection','source',tmp_path/'out','domain',[])


def test_calibration_numeric_reuse_blocked_under_renaming(tmp_path):
    with EvidenceLedger(tmp_path/'db') as ledger:
        ledger.claim('a','selection','sa',tmp_path/'a','d',[],[('same','calibration')])
        with pytest.raises(ValueError,match='CALIBRATION_NUMERIC_ALREADY_CONSUMED'):
            ledger.claim('b','selection','sb',tmp_path/'b','d',[],[('same','calibration')])
