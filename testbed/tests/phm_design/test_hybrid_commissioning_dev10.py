from __future__ import annotations
import copy, importlib.util, json
from pathlib import Path
import numpy as np
import pytest
from phm_design.adapters import sat_nonadcs as native
from phm_design.adapters import nonadcs_challenge as challenge_native
from phm_design.commissioning_hybrid import (_independent_parameters,generate_profiles,prepare_section,
    validate_profile,feature_contract,feature_binding,compute_features,hybrid_feature_contract,hybrid_feature_binding,compute_hybrid_features,prepare_hybrid_section)
from phm_design.family_pipeline import generate_with_passports
from phm_design.family_design import make_request
from phm_design.feature_arm_selection import commissioned_eligible
from phm_design.io import write_json,read_json,verify
from phm_design.data import Dataset,sequence_from_csv
from tests.phm_design.test_nonadcs import policy


def challenge(count=1):
    return {"schema":"phm.native-nonadcs.challenge.v2","seed":34061003,
      "groups_per_role":{"train":count,"validation":0,"calibration":0,"test":0},"domains":["eps","thermal","data"],
      "challenge":{
        "fault_fraction_by_role":{r:[.1,.2] for r in native.ROLES},
        "onset_fraction_by_role":{r:[.35,.45] for r in native.ROLES},
        "nuisance_scale_by_role":{r:[.5,1.] for r in native.ROLES},
        "boundary_cycles_by_role":{r:[2.,3.] for r in native.ROLES}}}


def metric(f1,recalls):
    return {"macro_f1":f1,"recall_by_class":{f"C{i}":v for i,v in enumerate(recalls)}}


def test_arm_guard_requires_per_class_noninferiority_and_strict_gain():
    md=metric(.70,[.8,.6,.7]);mc=metric(.68,[.8,.55,.7])
    assert commissioned_eligible(md,mc,metric(.75,[.82,.62,.72]),metric(.72,[.8,.58,.72]))
    assert not commissioned_eligible(md,mc,metric(.78,[.9,.5,.9]),metric(.75,[.9,.6,.8]))
    assert not commissioned_eligible(md,mc,copy.deepcopy(md),copy.deepcopy(mc))


def test_independent_commissioning_changes_phase_and_noise_but_not_plant_nuisance():
    c=challenge();world=challenge_native.plan_campaign(c)[1]["world_id"]
    p=next(x["parameters"] for x in challenge_native.plan_campaign(c) if x["world_id"]==world and x["domain"]=="thermal")
    q=_independent_parameters(world,"thermal",p)
    assert q["measurement_seed"]!=p["measurement_seed"] and q["phase"]!=p["phase"]
    for k in ("thermal_absorptivity_gain","thermal_emissivity_gain","heat_actuator_gain","temperature_bias_k"):
        assert q[k]==p[k]
    assert q["onset_s"]>native.FAMILIES["thermal"]["duration_s"]


@pytest.fixture(scope="module")
def native_commissioning(tmp_path_factory):
    if importlib.util.find_spec("Basilisk") is None:pytest.skip("Basilisk optional runtime unavailable")
    root=tmp_path_factory.mktemp("dev9-commissioning");cfg=root/"config.json";write_json(cfg,challenge())
    campaign=root/"campaign";generate_with_passports(cfg,campaign)
    commission=root/"commissioning";reg=generate_profiles(campaign,commission)
    return root,campaign,commission,reg


def test_native_commissioning_is_independent_healthy_and_sealed(native_commissioning):
    _,campaign,commission,reg=native_commissioning
    assert reg["healthy_runs"]==3 and reg["fault_labels_used"] is False and reg["diagnosis_trajectory_used"] is False
    verify(commission)
    for rec in reg["records"]:
        assert rec["diagnosis_measurement_seed"]!=rec["commissioning_measurement_seed"]
        assert rec["diagnosis_phase"]!=rec["commissioning_phase"]
    for ref in reg["profiles"].values():
        p=read_json(commission/ref["profile"]);text=json.dumps(p).lower()
        assert "fault_fraction" not in text and "onset_s" not in text and "remaining_fraction" not in text
        validate_profile(p,p["domain"])


@pytest.mark.parametrize("domain",["eps","thermal","data"])
def test_commissioned_features_compile_and_are_causal(native_commissioning,tmp_path,domain):
    _,campaign,commission,_=native_commissioning
    out=tmp_path/domain;manifest=prepare_section(campaign,commission,domain,"development",out)
    data=Dataset(out/"dataset.json",allowed_splits={"train"});block=data.load("train",native.FAMILIES[domain]["window"])
    assert block.x.shape[2]==len(feature_contract(domain)["channels"])
    make_request(feature_binding(domain),policy())
    # A change to the last raw sample cannot alter earlier causal features.
    s=data.samples[0];raw_manifest=Dataset(campaign/"native"/domain/"development"/"dataset.json",allowed_splits={"train"})
    rs=next(x for x in raw_manifest.samples if x["id"]==s["id"])
    raw=sequence_from_csv(raw_manifest.root/rs["path"],raw_manifest.contract,native.FAMILIES[domain]["window"])
    preg=read_json(campaign/"passport_registry.json")["passports"][domain+"/"+s["lineage_id"]]
    creg=read_json(commission/"registry.json")["profiles"][domain+"/"+s["lineage_id"]]
    passport=read_json(campaign/preg["path"]);profile=read_json(commission/creg["profile"])
    a=compute_features(domain,raw,passport,profile);changed=raw.copy();changed[-1,-1]+=1e-3
    b=compute_features(domain,changed,passport,profile)
    np.testing.assert_array_equal(a[:-1],b[:-1])


@pytest.mark.parametrize("domain",["eps","thermal","data"])
def test_hybrid_preserves_mechanism_and_appends_commissioning(native_commissioning,tmp_path,domain):
    _,campaign,commission,_=native_commissioning
    out=tmp_path/(domain+"_hybrid");manifest=prepare_hybrid_section(campaign,commission,domain,"development",out)
    data=Dataset(out/"dataset.json",allowed_splits={"train"});block=data.load("train",native.FAMILIES[domain]["window"])
    names=[c["name"] for c in data.contract["channels"]]
    assert any(n in names for n in ("radiation_loss_slope","generation_excess_running_bps","solar_loss_fraction"))
    assert any(n.startswith("commissioned_") for n in names)
    assert block.x.shape[2]==len(hybrid_feature_contract(domain)["channels"])
    make_request(hybrid_feature_binding(domain),policy())
