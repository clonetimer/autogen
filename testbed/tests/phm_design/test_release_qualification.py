"""Proof-bound 15-cell simulation release contract."""
import hashlib,json,pytest
from phm_design.platform_limited import contract,allowed_use_case,predict_bundle,_checked_release_certificate
from pathlib import Path
F=Path(__file__).parent/'fixtures/limited_opened'
def test_release_provenance():
 c=contract(); cert=_checked_release_certificate()
 assert c['release_tier']=='STABLE' and c['release_version']=='0.7.0'
 assert cert['sealed_worlds']==600 and cert['calibration_worlds']==300
 assert cert['calibration_passed']==cert['sealed_passed']==15
 assert len(cert['claimed_cells'])==15 and len(cert['unqualified_cells'])==8
 assert c['qualification_thresholds']==cert['numerical_thresholds']
def test_qualified_prediction_only_when_valid():
 p=predict_bundle('EPS','EPS:conventional',F/'EPS.csv',healthy_prefix_confirmed=True)
 assert p['qualified'] is True and p['qualification']=='STABLE_SIMULATION_ONLY'
 assert p['status']=='SIMULATION_DOMAIN_QUALIFIED_PREDICTION'
def test_out_of_claim_thermal_never_qualified():
 assert allowed_use_case('Thermal:conventional')['status']=='OUT_OF_CLAIM_NOT_QUALIFIED'
 with pytest.raises(ValueError,match='OUT_OF_CLAIM'):
  predict_bundle('Thermal','Thermal:conventional',F/'Thermal.csv',healthy_prefix_confirmed=True)
