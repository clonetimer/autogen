"""Development-only excitation experiment guards: no oracle inference."""
import json
import numpy as np
import pytest
from phm_design.adapters.sat_multicomponent import parameters,run_native
from phm_design.thermal_excitation_r65 import BASELINE,EXPERIMENT,commanded_heat,LEVELS


def test_legacy_command_exact():
    import math
    for i in (0,1):
        for now in range(0,3601,20):
            base=4.0
            assert commanded_heat(base,now,i,BASELINE) == base*(1+.25*float(math.sin(now/(115+15*i))>0))


def test_profile_levels_fixed_and_time_only():
    assert len(LEVELS)==8 and len(set(LEVELS))==8
    for i in (0,1):
        seen={commanded_heat(4.0,t,i,EXPERIMENT) for t in range(0,1200,20)}
        assert len(seen)>=5


@pytest.mark.parametrize('profile,idx,now,base',[(None,0,20,1),('invalid',0,20,1),
                                               (BASELINE,2,20,1),(EXPERIMENT,0,-20,1),
                                               (BASELINE,0,21,1),(BASELINE,0,20,float('nan'))])
def test_invalid_commands_fail_closed(profile,idx,now,base):
    with pytest.raises(ValueError,match='R65_INVALID'):
        commanded_heat(base,now,idx,profile)


def test_protocol_strictly_development_only():
    from research.run_r65_dev_experiment import PROTOCOL_FILE,validate_protocol
    p=json.loads(PROTOCOL_FILE.read_text())
    validate_protocol(p)
    p['open_blind_worlds']=True
    with pytest.raises(ValueError):validate_protocol(p)


def test_native_baseline_backward_compatible_and_enhanced_changes_command():
    p=parameters('r65-test-native-consistency','thermal')
    old=run_native('thermal',p,0)
    new=run_native('thermal',p,0,thermal_excitation=BASELINE)
    enhanced=run_native('thermal',p,0,thermal_excitation=EXPERIMENT)
    assert old==new  # Original 0.8.0.dev5 native trajectory is unchanged.
    a=np.asarray(old['observations']);b=np.asarray(enhanced['observations'])
    assert a.shape==b.shape==(181,6)
    assert len(set(a[:,2]))==2
    assert len(set(b[:,2]))>=5
    assert np.max(np.abs(a[:,4]-b[:,4]))>0
    assert not np.array_equal(a[:,2:4],b[:,2:4])
    assert np.array_equal(a[:,0],b[:,0])


def test_native_wrong_domain_rejected():
    with pytest.raises(ValueError,match='R65_EXCITATION_ONLY_THERMAL'):
        run_native('eps',parameters('r65-test','eps'),0,thermal_excitation=EXPERIMENT)


def test_oracle_and_deployment_headers_identical_and_limited():
    from research.run_r65_dev_experiment import HEADER,validate_protocol,PROTOCOL_FILE
    p=json.loads(PROTOCOL_FILE.read_text())
    assert tuple(p['evaluation']['deployment_observations_only'])==HEADER
    assert all(not any(secret in field for secret in ('label','onset','internal','fault','remaining','truth')) for field in HEADER)
    with pytest.raises(ValueError,match='R65_EVALUATION_PROTOCOL_CHANGED'):
        p['evaluation']['deployment_observations_only'].append('fault_active')
        validate_protocol(p)
