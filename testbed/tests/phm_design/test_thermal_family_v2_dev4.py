"""Pure unit/regression tests: no fresh-blind data or native runtime needed."""
import copy
import numpy as np
import pytest
from phm_design.adapters.sat_nonadcs import parameters, world_id
from phm_design.family_pipeline import compute_features as family_features, make_passport
from phm_design.thermal_family_v2 import (
    compute_features, feature_contract, feature_binding, FEATURES,
    prepare_section, WINDOWS,
)
from phm_design.representation_registry import representation_candidates, codesign_arm_plan


def _sample(n=128):
    w=world_id(77,'train',0)
    p=make_passport(w,'thermal',parameters(w,'thermal'))
    t=np.arange(n, dtype=float)
    raw=np.column_stack((.4 + .2*np.sin(t/12), 5. + np.cos(t/7),
                          301. + 4*np.sin(t/29) + .01*t))
    return raw,p


def test_additive_v1_prefix_bitwise_same():
    raw,passport=_sample()
    v1=family_features('thermal',raw,passport,'mechanism')
    v2=compute_features(raw,passport)
    assert v2.shape==(128,18)
    np.testing.assert_array_equal(v1,v2[:,:10])
    assert np.isfinite(v2).all()
    assert len(FEATURES)==8


def test_all_rolling_outputs_causal():
    raw,passport=_sample()
    before=compute_features(raw,passport)
    changed=raw.copy();changed[90:,2]+=12.;changed[90:,1]+=5
    after=compute_features(changed,passport)
    np.testing.assert_array_equal(before[:90],after[:90])
    assert not np.array_equal(before[90:],after[90:])


def test_constant_thermal_avoids_regression_false_claim():
    raw,passport=_sample()
    raw[:,2]=300.;raw[:,0]=1.;raw[:,1]=5.
    result=compute_features(raw,passport)
    for flag in (13,17):
        assert not result[:,flag].any()
    assert not result[0,10:].any()


def test_schema_units_names_binding_and_preperformance_registry():
    contract=feature_contract()
    assert len(contract['channels'])==18
    assert [c['name'] for c in contract['channels'][-8:]]==[n for n,_ in FEATURES]
    b=feature_binding()
    assert b['contract']==contract
    assert b['groups'][-1]['id']=='causal_thermal_rolling'
    assert 'family.thermal.rolling.v2' in [x.id for x in representation_candidates('thermal',task='family')]
    assert 'family.thermal.rolling.v2' not in [x.id for x in representation_candidates('eps',task='family')]
    arms=codesign_arm_plan('thermal',task='family')
    assert next(x for x in arms if x.representation_id=='family.thermal.rolling.v2').prepare_command == 'prepare-family-thermal-v2'


def test_reject_bad_passport_and_invalid_raw():
    raw,p=_sample()
    poisoned=copy.deepcopy(p)
    poisoned['parameters']['fault_truth']={'value':1.,'unit':'1','role':'normal_design_parameter'}
    with pytest.raises(ValueError,match='UNDECLARED'):compute_features(raw,poisoned)
    bad=raw.copy();bad[2,2]=float('nan')
    with pytest.raises(ValueError):compute_features(bad,p)
    bad=raw.copy();bad[2,2]=-1.
    with pytest.raises(ValueError,match='ABSOLUTE_TEMPERATURE'):compute_features(bad,p)


def test_role_and_missing_campaign_fail_before_output(tmp_path):
    with pytest.raises(ValueError,match='SECTION_INVALID'):
        prepare_section(tmp_path/'absent','unknown',tmp_path/'output')
    assert not (tmp_path/'output').exists()
    with pytest.raises(FileNotFoundError):
        prepare_section(tmp_path/'absent','development',tmp_path/'output')
    assert not (tmp_path/'output').exists()


def test_rolling_window_compilation_values():
    raw,passport=_sample()
    v2=compute_features(raw,passport)
    # Prefix 8-sample minimum for slope; only finite, dimensionless coefficients.
    assert not v2[:8,13].any() and not v2[:8,17].any()
    assert np.all((v2[:,13] == 0) | (v2[:,13] == 1))
    assert np.all((v2[:,17] == 0) | (v2[:,17] == 1))
    assert WINDOWS==(16,64)
