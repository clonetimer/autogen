"""Security, provenance, metric, release boundary, deterministic output tests."""
import copy
import hashlib
import json
from pathlib import Path
import pytest
from phm_design.design_readiness import SCHEMA, EvidenceError, analyze, sha_file


def sample(tmp_path):
    path=tmp_path/'baseline.json'
    path.write_text('{"readonly":true}',encoding='utf-8')
    return {
      'schema':SCHEMA,'domain':'Thermal','task':'cross_environment_fault_diagnosis',
      'data_role':'DEVELOPMENT_ONLY','protected_splits_opened':False,
      'candidate_status':'NO_ELIGIBLE',
      'source_bindings':[{'path':'baseline.json','sha256':sha_file(path)}],
      'gates':{'min_macro_f1':.90,'min_class_recall':.80,'max_normal_false_alarm':.10,'min_coverage':.75},
      'physics_pair_evidence':{'pair_count':2,'status':'PHYSICAL_PAIRS_PASS'},
      'training_provenance':{'model_family':'bounded','candidate_count':1,'training_count':6,
                             'model_fits':1,'source_report_role':'DEVELOPMENT_ONLY'},
      'observed_channels':['time_s','equipment_temperature_k'],
      'environments':[{'environment_id':'env-a','case_count':6,'world_group_count':2,
         'support_by_class':{'NOMINAL':3,'FAULT':3},
         'metrics':{'macro_f1':.70,'normal_false_alarm':1/3,'coverage':1.,
                    'recall_by_class':{'NOMINAL':2/3,'FAULT':2/3}}}]}


def test_fail_closed_and_recommendations(tmp_path):
    r=analyze(sample(tmp_path),source_root=tmp_path)
    assert r['release_decision']=='HOLD_DO_NOT_PROMOTE'
    assert not r['development_gate_metrics_pass'] and not r['independent_qualification']
    assert {'NORMAL_FALSE_ALARM_EXCEEDS_GATE','WEAK_CLASS','LIMITED_INDEPENDENT_ENVIRONMENTS'} <= {x['code'] for x in r['evidence_gaps']}
    assert all(x['qualification_effect']=='HYPOTHESIS_ONLY' for x in r['proposed_studies'])


def test_reproducible_fingerprint(tmp_path):
    m=sample(tmp_path);a=analyze(m,source_root=tmp_path);b=analyze(m,source_root=tmp_path)
    assert a==b and a['recommendation_fingerprint']==b['recommendation_fingerprint']


def test_reject_calibration_or_blind_role(tmp_path):
    m=sample(tmp_path);m['data_role']='SEALED_BLIND'
    with pytest.raises(EvidenceError,match='PROTECTED_DATA_ROLE_REJECTED'):analyze(m,source_root=tmp_path)
    m['data_role']='DEVELOPMENT_ONLY';m['protected_splits_opened']=True
    with pytest.raises(EvidenceError,match='PROTECTED_DATA_ROLE_REJECTED'):analyze(m,source_root=tmp_path)


def test_missing_file_and_tamper_rejected(tmp_path):
    m=sample(tmp_path);(tmp_path/'baseline.json').write_text('changed')
    with pytest.raises(EvidenceError,match='SOURCE_SHA_MISMATCH'):analyze(m,source_root=tmp_path)
    (tmp_path/'baseline.json').unlink()
    with pytest.raises(EvidenceError,match='UNSAFE_OR_MISSING_SOURCE'):analyze(m,source_root=tmp_path)


def test_path_escape_rejected(tmp_path):
    m=sample(tmp_path);m['source_bindings'][0]['path']='../secret.json'
    with pytest.raises(EvidenceError,match='UNSAFE_SOURCE_PATH'):analyze(m,source_root=tmp_path)


def test_symlink_escape_rejected(tmp_path):
    m=sample(tmp_path)
    external=tmp_path.parent/'leaked_reference.json'
    external.write_text('sensitive')
    (tmp_path/'baseline.json').unlink()
    (tmp_path/'baseline.json').symlink_to(external)
    m['source_bindings'][0]['sha256']=sha_file(external)
    with pytest.raises(EvidenceError,match='UNSAFE_OR_MISSING_SOURCE'):analyze(m,source_root=tmp_path)
    external.unlink()


def test_oracle_channel_rejected(tmp_path):
    m=sample(tmp_path);m['observed_channels'].append('ground_truth_fault_label')
    with pytest.raises(EvidenceError,match='ORACLE_CHANNEL_REJECTED'):analyze(m,source_root=tmp_path)


def test_invalid_rates_or_nan_rejected(tmp_path):
    m=sample(tmp_path);m['environments'][0]['metrics']['coverage']=float('nan')
    with pytest.raises(EvidenceError,match='INVALID_FINITE_NUMBER'):analyze(m,source_root=tmp_path)


def test_support_missing_remains_unverified(tmp_path):
    m=sample(tmp_path);m['environments'][0].pop('support_by_class')
    r=analyze(m,source_root=tmp_path)
    assert 'CLASS_SUPPORT_UNVERIFIED' in {v['code'] for v in r['evidence_gaps']}


def test_support_mismatch_rejected(tmp_path):
    m=sample(tmp_path);m['environments'][0]['support_by_class']['FAULT']=4
    with pytest.raises(EvidenceError,match='INVALID_CLASS_SUPPORT'):analyze(m,source_root=tmp_path)


def test_class_contract_drift_rejected(tmp_path):
    m=sample(tmp_path);second=copy.deepcopy(m['environments'][0]);second['environment_id']='env-b'
    second['metrics']['recall_by_class']['OTHER']=1.0
    m['environments'].append(second)
    with pytest.raises(EvidenceError,match='CLASS_CONTRACT_DRIFT'):analyze(m,source_root=tmp_path)


def test_positive_dev_still_not_release(tmp_path):
    m=sample(tmp_path)
    m['candidate_status']='DEVELOPMENT_ELIGIBLE_NOT_BLIND'
    for i in range(3):
        e=copy.deepcopy(m['environments'][0]);e['environment_id']=f'env-{i}'
        e['metrics']={'macro_f1':1.,'normal_false_alarm':0.,'coverage':1.,'recall_by_class':{'NOMINAL':1.,'FAULT':1.}}
        if i==0:m['environments']=[e]
        else:m['environments'].append(e)
    r=analyze(m,source_root=tmp_path)
    assert r['development_gate_metrics_pass']
    assert not r['independent_qualification'] and r['release_decision']=='HOLD_DO_NOT_PROMOTE'
    assert not any(x['code'] in {'WEAK_CLASS','NORMAL_FALSE_ALARM_EXCEEDS_GATE'} for x in r['evidence_gaps'])


def test_recall_support_consistency_rejected(tmp_path):
    m=sample(tmp_path)
    m['environments'][0]['metrics']['recall_by_class']['FAULT']=0.8  # 0.8*3 is not integer
    with pytest.raises(EvidenceError,match='RECALL_SUPPORT_INCONSISTENT'):
        analyze(m,source_root=tmp_path)


def test_training_cost_negative_rejected(tmp_path):
    m=sample(tmp_path);m['training_provenance']['model_fits']=-1
    with pytest.raises(EvidenceError,match='INVALID_TRAINING_COST'):
        analyze(m,source_root=tmp_path)


def test_small_support_is_flagged(tmp_path):
    m=sample(tmp_path)
    r=analyze(m,source_root=tmp_path)
    assert 'SMALL_CLASS_SUPPORT' in {v['code'] for v in r['evidence_gaps']}
