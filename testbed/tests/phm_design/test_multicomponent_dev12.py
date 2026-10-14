import json,copy
from pathlib import Path
import numpy as np
import pytest

from phm_design.adapters import sat_multicomponent as mc
from phm_design.multicomponent_features import make_passport,compute_features,feature_contract


def test_multicomponent_contract_separates_component_and_mechanism():
    expected={
      "eps":{"SOLAR_A_EFFICIENCY_LOSS","SOLAR_B_EFFICIENCY_LOSS","LOAD_A_EXTRA","LOAD_B_EXTRA"},
      "thermal":{"THERMAL_A_EMISSIVITY_LOSS","THERMAL_B_EMISSIVITY_LOSS","THERMAL_A_EXTRA_HEAT","THERMAL_B_EXTRA_HEAT"},
      "data":{"SERVICE_A_RATE_LOSS","SERVICE_B_RATE_LOSS","GEN_A_EXCESS","GEN_B_EXCESS"}}
    for d in expected:
        c=mc.contract(d)
        assert c["task"]=="fixed_length_fault_mechanism_and_component_localization"
        assert set(c["classes"][1:])==expected[d]
        assert len(mc.binding(d)["groups"])>=3


def test_multicomponent_parameters_are_deterministic_by_world_and_domain():
    for d in mc.FAMILIES:
        w=mc.world_id(1234,"train",2)
        assert mc.parameters(w,d)==mc.parameters(w,d)
        assert mc.parameters(w,d)!=mc.parameters(mc.world_id(1234,"train",3),d)


def synthetic_raw(domain):
    c=mc.contract(domain);n=mc.FAMILIES[domain]["window"]
    x=np.zeros((n,len(c["channels"])),dtype=float)
    t=np.arange(n,dtype=float)
    if domain=="eps":
        x[:,0]=1;x[:,1]=5;x[:,2]=6;x[:,3]=20;x[:,4]=22;x[:,5]=5;x[:,6]=6;x[:,7]=25
    elif domain=="thermal":
        x[:,0]=(np.sin(t/8)>-.5);x[:,1]=3;x[:,2]=4;x[:,3]=305+.02*t;x[:,4]=307+.015*t
    else:
        x[:,0]=(np.cos(t/7)>.2);x[:,1]=9000;x[:,2]=45000;x[:,3]=1e6+1000*t
        x[:,4]=10000;x[:,5]=47000;x[:,6]=1.2e6+800*t
    return x


def nominal_params(domain):
    w=mc.world_id(99,"train",0);return w,mc.parameters(w,domain)


def test_component_features_have_no_label_or_fault_argument():
    for d in mc.FAMILIES:
        w,p=nominal_params(d);passport=make_passport(w,d,p)
        y=compute_features(d,synthetic_raw(d),passport)
        assert y.shape[1]==len(feature_contract(d)["channels"])
        assert np.isfinite(y).all()


@pytest.mark.parametrize("domain",["eps","thermal","data"])
def test_native_multicomponent_prefault_pair_is_exact_when_basilisk_available(domain):
    pytest.importorskip("Basilisk")
    w=mc.world_id(7401,"train",0);p=mc.parameters(w,domain)
    a=mc.run_native(domain,p,0);b=mc.run_native(domain,p,1)
    audit=mc.pair_audit(a,b,p,domain,1)
    assert audit["prefault_exact_match"] is True
    assert audit["component"] is not None
    assert audit["mechanism"]!="NOMINAL"
