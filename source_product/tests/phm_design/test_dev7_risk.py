import csv
import shutil
from pathlib import Path
import numpy as np
import pytest

from phm_design.data import Dataset
from phm_design.io import sha_file, write_json, read_json
from phm_design.reference_design import design_reference
from phm_design.route_selection import select_route
from phm_design.risk_calibration import (
    POLICY, clopper_pearson_upper, selective_report, choose_threshold,
    calibrate_risk, evaluate_risk,
)
from phm_design.evidence_ledger import EvidenceLedger
from tests.phm_design.test_design_system import fixture_dataset
from tests.phm_design.test_reference_dev3 import make_policy
from tests.phm_design.test_compositional_design import request


def calibration(root, contract, prefix, seed, groups_per_class=4):
    rng=np.random.default_rng(seed);samples=[]
    for label in range(3):
        for j in range(groups_per_class):
            ident=f'{prefix}-{label}-{j}';path=root/(ident+'.csv');path.parent.mkdir(parents=True,exist_ok=True)
            x=rng.normal(label*2,.16,(12,2))
            with path.open('w',newline='') as f:
                w=csv.writer(f);w.writerow(['time_s',*[c['name'] for c in contract['channels']]])
                w.writerows([[i,*row] for i,row in enumerate(x)])
            samples.append({'id':ident,'lineage_id':ident,'group_id':ident,'split':'calibration',
                'label':contract['classes'][label],'path':path.name,'sha256':sha_file(path)})
    manifest=root/'dataset.json'
    write_json(manifest,{'schema':'phm.sequence-dataset.v1','dataset_id':prefix,
        'contract':contract,'samples':samples,'provenance':{'source':'software_fixture','purpose':prefix}})
    return manifest


@pytest.fixture(scope='module')
def frozen(tmp_path_factory):
    p=tmp_path_factory.mktemp('dev7-risk')
    dev,test=fixture_dataset(p/'data')
    d=Dataset(dev,allowed_splits={'train','validation'})
    binding={'schema':'phm.device-binding.v1','contract':d.contract,'groups':request()['groups'],
       'window_samples':8,'quality_mode':'complete_observation_only_missing_values_rejected'}
    write_json(p/'binding.json',binding);write_json(p/'policy.json',make_policy())
    design_reference(dev,p/'binding.json',p/'policy.json',p/'run')
    selection_cal=calibration(p/'selection_cal',d.contract,'selection',101,4)
    risk_cal=calibration(p/'risk_cal',d.contract,'risk',202,8)
    return p,dev,test,selection_cal,risk_cal


def test_exact_binomial_upper_limit_basic_properties():
    assert 0 < clopper_pearson_upper(0,50,.05) < .10
    assert clopper_pearson_upper(1,50,.05) > clopper_pearson_upper(0,50,.05)
    assert clopper_pearson_upper(50,50,.05) == 1.0


def test_selective_group_metric_and_threshold_choice():
    # 12 independent groups, one case each. Low-confidence mistakes disappear at .8.
    samples=[{'group_id':f'g{i}'} for i in range(12)]
    y=np.asarray([0]*12)
    scores=np.asarray([[.95,.03,.02]]*8 + [[.45,.50,.05]]*4)
    p={'schema':'phm.selective-risk-policy.v1','thresholds':[.5,.8,.9],
       'alpha':.2,'max_group_error_upper_bound':.5,'min_case_coverage':.5}
    result=choose_threshold(y,scores,samples,['C0','C1','C2'],p)
    assert result['status']=='QUALIFIED_THRESHOLD'
    assert result['threshold'] in {.8,.9}
    chosen=result['selected']
    assert chosen['accepted_errors']==0 and chosen['case_coverage']>=.5


def test_no_qualified_threshold_abstains_all():
    samples=[{'group_id':f'g{i}'} for i in range(8)]
    y=np.zeros(8,dtype=np.int64)
    scores=np.asarray([[.05,.90,.05]]*8)
    p={'schema':'phm.selective-risk-policy.v1','thresholds':[.5,.8,.9],
       'alpha':.05,'max_group_error_upper_bound':.1,'min_case_coverage':.5}
    r=choose_threshold(y,scores,samples,['C0','C1','C2'],p)
    assert r['status']=='NO_THRESHOLD_QUALIFIED_ABSTAIN_ALL' and r['threshold'] is None


def test_selection_risk_calibration_and_holdout_are_distinct(frozen,tmp_path):
    p,dev,test,selection_cal,risk_cal=frozen
    run=tmp_path/'run';shutil.copytree(p/'run',run)
    db=tmp_path/'ledger.sqlite';sel=tmp_path/'selection'
    select_route(run,selection_cal,sel,registry=db)
    policy=dict(POLICY);policy.update(alpha=.2,max_group_error_upper_bound=.8,min_case_coverage=0.0)
    risk=tmp_path/'risk'
    rr=calibrate_risk(run,sel,risk_cal,risk,registry=db,policy=policy)
    assert rr['selection_calibration_reused'] is False and not rr['test_used']
    assert rr['risk_certified'] is False
    assert rr['group_risk_bound_qualified'] == (rr['status']=='QUALIFIED_THRESHOLD')
    report=evaluate_risk(run,sel,risk,test,tmp_path/'evaluation',registry=db)
    assert report['test_used_for_threshold_selection'] is False
    assert report['frozen_threshold_metrics']['cases']==6
    assert report['risk_threshold_metrics']['cases']==6


def test_same_calibration_cannot_be_reused_for_risk(frozen,tmp_path):
    p,dev,test,selection_cal,risk_cal=frozen
    run=tmp_path/'run';shutil.copytree(p/'run',run)
    db=tmp_path/'ledger.sqlite';sel=tmp_path/'selection'
    select_route(run,selection_cal,sel,registry=db)
    with pytest.raises(ValueError,match='HOLDOUT_OVERLAP|IDENTICAL_OBSERVATIONS|CALIBRATION_ALREADY_CONSUMED'):
        calibrate_risk(run,sel,selection_cal,tmp_path/'risk',registry=db,policy=POLICY)


def test_ledger_blocks_cross_kind_calibration_reuse(tmp_path):
    sample={'id':'a','group_id':'g','lineage_id':'l','split':'calibration','label':'C0','path':'x','sha256':'h'}
    with EvidenceLedger(tmp_path/'db') as ledger:
        ledger.claim('s','selection','source-a',tmp_path/'a','domain',[sample],[('sig','calibration')])
        with pytest.raises(ValueError,match='CALIBRATION_ALREADY_CONSUMED|CALIBRATION_NUMERIC_ALREADY_CONSUMED'):
            ledger.claim('r','risk_calibration','source-b',tmp_path/'b','domain',[sample],[('sig','calibration')])
