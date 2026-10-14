"""R59 developer-only scope & original opened-data compatibility tests."""
import csv,json,hashlib
from pathlib import Path
import pytest
from phm_design.platform_limited import contract,allowed_use_case,model_manifest,open_bundled,predict_bundle,rows_from_csv,registry
from phm_design.platform_features import FIELDS
from phm_design.open_design_fit import predict

FIXTURES=Path(__file__).parent/'fixtures/limited_opened'
DOMAINS=('ADCS','EPS','Data','EPS_localization','Data_localization')

def test_authorization_new_contract_not_original_old_cycle():
    c=contract()
    assert c['authorization'].startswith('USER_APPROVED_')
    assert c['historical_old_broad_cycle']=='CLOSED_NO_RELEASE_RETAINED'
    assert len(c['proposed_certification_cells'])==15
    assert len(c['out_of_claim_unqualified_cells'])==8
    assert not (set(c['proposed_certification_cells'])&set(c['out_of_claim_unqualified_cells']))
    assert c['deployment_qualification_status']=='DEVELOPMENT_NOT_QUALIFIED'
    assert c['qualification_applies_to_current_package'] is False

def test_formal_stable_not_promoted():
    c=contract();assert c['formal_stable'].startswith('0.7.0') and c['previous_formal_stable'].startswith('0.6.0')
    assert c['candidate_version']=='0.7.0'
    assert c['active_package_version']==__import__('phm_design').__version__
    assert c['bundled_legacy_release_version']=='0.7.0'

def test_unsupported_cannot_be_declared_supported():
    for s in contract()['out_of_claim_unqualified_cells']:
        assert allowed_use_case(s)['status']=='OUT_OF_CLAIM_NOT_QUALIFIED'

def test_unknown_scope_fail_closed():
    with pytest.raises(ValueError,match='UNKNOWN_USE_CASE'):allowed_use_case('EPS:ground_truth_fault')

def test_six_selected_models_have_hashes_and_development_markers():
    man=model_manifest()['models'];assert set(man)==set(DOMAINS)
    for domain in DOMAINS:
        assert man[domain]['qualified'] is True
        assert man[domain]['source_revision']=='R57_FROZEN_TRAINING_SELECTED'
        assert open_bundled(domain)['platform_domain']==domain

@pytest.mark.parametrize('domain',DOMAINS)
def test_real_opened_r57_trajectory_installed_prediction(domain):
    case=domain+':'+('multi_component_localization' if domain.endswith('_localization') else 'conventional')
    result=predict_bundle(domain,case,FIXTURES/(domain+'.csv'),healthy_prefix_confirmed=True)
    assert result['scope_status']=='LEGACY_MODEL_DEVELOPMENT_ONLY'
    assert result['qualification']=='DEVELOPMENT_ONLY_LEGACY_0_7_MODEL'
    assert result['qualified'] is False
    assert isinstance(result['prediction'],int)

@pytest.mark.parametrize('domain',DOMAINS)
def test_exact_observation_contract_from_real_opened_trace(domain):
    a=rows_from_csv(FIXTURES/(domain+'.csv'),domain)
    assert a.shape[1]==len(FIELDS[domain])
    assert len(a)>=24

def test_known_unqualified_case_rejected_by_prediction():
    with pytest.raises(ValueError,match='OUT_OF_CLAIM'):
        predict_bundle('Thermal','Thermal:weak_fault',FIXTURES/'Thermal.csv',healthy_prefix_confirmed=True)

def test_domain_mismatch_rejected_before_pickle_or_csv():
    with pytest.raises(ValueError,match='DOMAIN_MISMATCH'):
        predict_bundle('Thermal','EPS:conventional',FIXTURES/'Thermal.csv',healthy_prefix_confirmed=True)

def test_truth_column_rejected(tmp_path):
    src=FIXTURES/'EPS.csv';bad=tmp_path/'wrong.csv';lines=src.read_text().splitlines()
    bad.write_text(lines[0]+',truth\n'+'\n'.join(x+',1' for x in lines[1:])+'\n')
    with pytest.raises(ValueError,match='EXACT_DEPLOYMENT_FIELDS'):
        rows_from_csv(bad,'EPS')

def test_nonfinite_rejected(tmp_path):
    src=FIXTURES/'Thermal.csv';bad=tmp_path/'nonfinite.csv'
    lines=src.read_text().splitlines();lines[1]=lines[1].replace(lines[1].split(',')[1],'nan',1)
    bad.write_text('\n'.join(lines)+'\n')
    with pytest.raises(ValueError,match='OBSERVATION_NONFINITE'):
        rows_from_csv(bad,'Thermal')

def test_reverse_timestamp_rejected(tmp_path):
    lines=(FIXTURES/'Data.csv').read_text().splitlines();lines[3]=lines[2]
    f=tmp_path/'reverse.csv';f.write_text('\n'.join(lines)+'\n')
    with pytest.raises(ValueError,match='TIME_NOT_STRICTLY_INCREASING'):
        rows_from_csv(f,'Data')

def test_no_hidden_oracle_feature_names():
    forbidden=('truth','label','fault_id','injection','applied_torque','world_id')
    for domain in DOMAINS:
        for field in FIELDS[domain]:
            assert not any(x in field.lower() for x in forbidden)

def test_no_runtime_router_from_simulator_fault_truth():
    import inspect,phm_design.platform_limited as m
    source=inspect.getsource(m.predict_bundle)
    for suspicious in ('offline_label','fault_truth','injection_envelope','world_id'):
        assert suspicious not in source

def test_augmented_r57_registry_is_explicitly_available():
    assert 'transport' in registry('Thermal').names
    assert 'mechanism_innovation' in registry('Thermal').names
    assert 'mechanism_innovation' not in registry('Data').names
