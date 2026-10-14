"""Newly authorized 15-cell profile: no retrospective certification."""
import pytest
from phm_design.platform_limited import contract,allowed_use_case,open_bundled,predict_bundle

def test_scope_authorized_and_unchanged_thresholds():
 c=contract();assert c['contract_id']=='PHM_0_7_LIMITED_SIM_15_20261011_V2'
 assert len(c['proposed_certification_cells'])==15 and len(c['out_of_claim_unqualified_cells'])==8
 assert c['old_16_profile']=='CLOSED_NO_RELEASE_FOR_THIS_16_CELL_PROFILE'
 assert c['qualification_thresholds']=={'overall_accuracy_min':.9,'healthy_known_fault_false_alarm_max':.1,'per_fault_recall_min':.8,'unknown_fraction_max':.1,'worst_family_accuracy_min':.8}
 assert c['calibration_openings_used']==c['sealed_openings_used']==1

def test_thermal_conventional_rejected_and_not_bundled():
 assert allowed_use_case('Thermal:conventional')['status']=='OUT_OF_CLAIM_NOT_QUALIFIED'
 with pytest.raises(ValueError,match='NO_TRAINING_SELECTED'):
  open_bundled('Thermal')

def test_other_thermal_types_remain_unqualified():
 for x in ('conventional','cross_scenario','weak_fault','progressive_degradation','intermittent_fault'):
  assert allowed_use_case('Thermal:'+x)['status']=='OUT_OF_CLAIM_NOT_QUALIFIED'

def test_supported_domains_and_unknown_fail_closed():
 domains={x.split(':')[0] for x in contract()['proposed_certification_cells']}
 assert domains=={'ADCS','EPS','Data','EPS_localization','Data_localization'}
 with pytest.raises(ValueError,match='UNKNOWN_USE_CASE'):
  allowed_use_case('ADCS:counterfactual_truth')
