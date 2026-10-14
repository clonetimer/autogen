"""R66 protocol and command verification, plus native original-path protection."""
import json
import numpy as np
import pytest
from phm_design.thermal_excitation_r66 import commanded_heat,SLOW,SEGMENT_S,LEVELS
from phm_design.adapters.sat_multicomponent import parameters,run_native
from research.run_r66_dev_experiment import validate_protocol,PROTOCOL_FILE,HEADER


def test_slow_command_exact():
    for i in (0,1):
        for t in (0, 20, 2380,2400,7200,14380):
            assert commanded_heat(4.,t,i)==4*LEVELS[(t//SEGMENT_S + 3*i)%len(LEVELS)]

@pytest.mark.parametrize('base,t,i',[(0,0,0),(4.,-20,0),(4.,11,0),(4.,0,2),(float('nan'),0,0),(True,0,0)])
def test_command_rejects_invalid(base,t,i):
    with pytest.raises(ValueError,match='R66_INVALID'):commanded_heat(base,t,i)


def test_frozen_dev_contract_and_reject_tampering():
    p=json.loads(PROTOCOL_FILE.read_text());validate_protocol(p)
    assert tuple(p['evaluation']['deployment_fields'])==HEADER
    p['evaluation']['blind_worlds_opened']=1
    with pytest.raises(ValueError,match='R66_EVALUATION'):validate_protocol(p)


def test_basilisk_native_baseline_and_slow_equal_horizon():
    p=parameters('r66-test-world-new','thermal');p['onset_s']=7200
    baseline=run_native('thermal',p,0,thermal_duration_s=14400)
    legacy=run_native('thermal',p,0,thermal_duration_s=14400,thermal_excitation='legacy_two_level')
    slow=run_native('thermal',p,0,thermal_duration_s=14400,thermal_excitation=SLOW)
    assert baseline==legacy
    a=np.asarray(baseline['observations']);b=np.asarray(slow['observations'])
    assert a.shape==b.shape==(721,6)
    assert np.array_equal(a[:,0],b[:,0])
    assert np.any(a[:,2]!=b[:,2]);assert np.any(a[:,4]!=b[:,4])
    old=run_native('thermal',p,0)
    assert len(old['observations'])==181


def test_duration_restrictions():
    with pytest.raises(ValueError,match='R66_INVALID_THERMAL_DURATION'):
        run_native('thermal',parameters('r66','thermal'),0,thermal_duration_s=14501)
    with pytest.raises(ValueError,match='R66_INVALID_THERMAL_DURATION'):
        run_native('eps',parameters('r66','eps'),0,thermal_duration_s=14400)
