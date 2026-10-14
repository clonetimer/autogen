"""R67 development contract, equal-energy mathematics, native Basilisk and audit tests."""
import json
import numpy as np
import pytest
from phm_design.thermal_excitation_r67 import (
    CROSS_AB,CROSS_BA,CONSTANT,LEVELS,SEGMENT_S,commanded_heat,schedule_energy_audit)
from phm_design.adapters.sat_multicomponent import parameters,run_native
from research.run_r67_dev_experiment import validate_protocol,PROTOCOL_FILE,aggregate,noise_observation_summary


def test_exact_phase_schedule():
    assert len(LEVELS)==6
    for t in (0,20,2380,2400,7180,7200,14380):
        for i in (0,1):
            assert commanded_heat(4.,t,i,CONSTANT)==4.
            assert commanded_heat(4.,t,i,CROSS_AB)==4*LEVELS[(t//SEGMENT_S+3*i)%6]
            assert commanded_heat(4.,t,i,CROSS_BA)==4*LEVELS[(t//SEGMENT_S+3*(1-i))%6]


@pytest.mark.parametrize('b,t,c,profile',[(0,0,0,CROSS_AB),(True,0,0,CROSS_AB),(float('nan'),0,0,CROSS_AB),
    (3,-20,0,CROSS_AB),(3,1,0,CROSS_AB),(3,14400,0,CROSS_AB),(3,0,2,CROSS_AB),(3,0,0,'unknown')])
def test_reject_invalid_heat_command(b,t,c,profile):
    with pytest.raises(ValueError,match='R67_INVALID'):commanded_heat(b,t,c,profile)


def test_exact_half_horizon_equal_energy():
    for arm,comps in schedule_energy_audit().items():
        for value in comps.values():assert value==[7200.,7200.]


def test_contract_frozen_and_rejects_blind_tampering():
    p=json.loads(PROTOCOL_FILE.read_text());validate_protocol(p)
    p['evaluation']['blind_worlds_opened']=1
    with pytest.raises(ValueError,match='R67_EVALUATION'):validate_protocol(p)


def test_aggregate_uses_world_unit_not_pseudo_eight_samples():
    vals=[(.01,-.04),(.03,-.03),(.01,-.01),(.02,-.02)]
    worlds=[{'component_phase_effects':[{'component':comp,'phase3_minus_phase0_residual':v} for comp,v in (('a',a),('b',b))]}
            for a,b in vals]
    results=aggregate(worlds,json.loads(PROTOCOL_FILE.read_text()))
    assert results['per_component']['a']['world_count']==4
    assert results['per_component']['b']['world_count']==4
    assert results['both_components_material_opposite_sign'] is False


def test_observable_feature_does_not_require_truth():
    rows=[[i*20,1,1,1,300+np.sin(i),302+np.cos(i)] for i in range(721)]
    feat=noise_observation_summary(rows)
    assert feat['command_energy_ws']==[14400,14400]
    assert all(x>=0 for x in feat['temperature_range_k'])


def test_native_basilisk_cross_profile_preserves_development_contract():
    p=parameters('r67-test-phase-native-new','thermal');p['onset_s']=7200
    x=run_native('thermal',p,0,thermal_excitation=CROSS_AB,thermal_duration_s=14400)
    y=run_native('thermal',p,0,thermal_excitation=CROSS_BA,thermal_duration_s=14400)
    c=run_native('thermal',p,0,thermal_excitation=CONSTANT,thermal_duration_s=14400)
    xx=np.asarray(x['observations']);yy=np.asarray(y['observations']);cc=np.asarray(c['observations'])
    assert xx.shape==yy.shape==cc.shape==(721,6)
    assert np.array_equal(xx[:,0],cc[:,0])
    assert not np.array_equal(xx[:,2],yy[:,2])
    assert not np.array_equal(xx[:,4],yy[:,4])
    for arr in (xx,yy,cc):
        assert np.all(np.isfinite(arr))
        assert 250<=np.min(arr[:,4:6])<=np.max(arr[:,4:6])<=450
    assert len(run_native('thermal',p,0)['observations'])==181
