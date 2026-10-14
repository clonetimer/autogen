"""Healthy commissioning profiles and causal residual features for non-ADCS PHM.

The commissioning run is a separate healthy experiment on the same simulated
instance. It uses a different excitation phase and measurement-noise seed, so it
is not a paired nominal twin of the later diagnosis trajectory. Only compact
fitted coefficients and residual scales enter diagnosis. Fault labels, fault
onset/severity, the diagnosis trajectory, and simulator internal truth never
enter the profile.

This is RESEARCH_ONLY and assumes a representative healthy commissioning run is
available before fault diagnosis. It is not a hardware calibration claim.
"""
from __future__ import annotations
import copy
import csv
import math
from pathlib import Path
import numpy as np
from .adapters import sat_nonadcs as native
from .data import Dataset, sequence_from_csv, validate_contract
from .family_pipeline import make_passport, validate_passport, CONTEXT
from . import family_pipeline as mechanism_features
from .io import digest, read_json, write_json, sha_file, seal, verify, safe_child

PROFILE_SCHEMA = "phm.healthy-commissioning-profile.v2"
FEATURE_SCHEMA = "phm.commissioned-features.v2"

def implementation_sha() -> str:
    return digest({"hybrid_commissioning": sha_file(Path(__file__)), "mechanism_features": mechanism_features.implementation_sha()})

PROFILE_CONTEXT = {
    "eps": [
        ("healthy_solar_gain", "1"),
        ("healthy_panel_to_bus_gain", "1"),
        ("healthy_load_to_bus_gain", "1"),
    ],
    "thermal": [
        ("healthy_solar_dynamics_gain", "1"),
        ("healthy_heat_dynamics_gain", "1"),
        ("healthy_radiation_dynamics_gain", "1"),
    ],
    "data": [
        ("healthy_generation_dynamics_gain", "1"),
        ("healthy_service_dynamics_gain", "1"),
    ],
}

DERIVED = {
    "eps": [
        ("commissioned_solar_residual_z", "1"),
        ("commissioned_bus_residual_z", "1"),
        ("commissioned_solar_short_mean_z", "1"),
        ("commissioned_solar_long_mean_z", "1"),
        ("commissioned_bus_short_mean_z", "1"),
        ("commissioned_bus_long_mean_z", "1"),
        ("commissioned_solar_observable", "1"),
    ],
    "thermal": [
        ("commissioned_energy_residual_z", "1"),
        ("commissioned_energy_short_mean_z", "1"),
        ("commissioned_energy_medium_mean_z", "1"),
        ("commissioned_energy_long_mean_z", "1"),
        ("commissioned_radiation_slope_medium", "1"),
        ("commissioned_radiation_slope_long", "1"),
        ("commissioned_thermal_excitation_valid", "1"),
    ],
    "data": [
        ("commissioned_queue_residual_z", "1"),
        ("commissioned_off_short_mean_z", "1"),
        ("commissioned_off_long_mean_z", "1"),
        ("commissioned_on_short_mean_z", "1"),
        ("commissioned_on_long_mean_z", "1"),
        ("commissioned_contact_contrast_z", "1"),
        ("commissioned_queue_dynamics_valid", "1"),
    ],
}


def _finite(v, name):
    if type(v) not in (int, float) or not math.isfinite(float(v)):
        raise ValueError("COMMISSIONING_NONFINITE:" + name)
    return float(v)


def _independent_parameters(world: str, domain: str, p: dict) -> dict:
    """Same plant instance, independent healthy excitation/noise realization."""
    q = copy.deepcopy(p)
    # Diagnosis noise/phase must not be replayed in commissioning.
    u = int(digest([world, domain, "commissioning-phase-v2"])[:16], 16) / float(16**16)
    q["phase"] = 2 * math.pi * u
    cycles = 2.15 + .70 * (int(digest([world, domain, "commissioning-cycles-v2"])[:8], 16) / float(16**8))
    q["boundary_period_s"] = native.FAMILIES[domain]["duration_s"] / cycles
    q["measurement_seed"] = int(digest([world, domain, "commissioning-measurement-v2"])[:16], 16)
    q["onset_s"] = native.FAMILIES[domain]["duration_s"] + native.FAMILIES[domain]["sample_s"]
    return q


def _lstsq(X, y, min_rows, rank_required):
    X = np.asarray(X, np.float64); y = np.asarray(y, np.float64)
    if X.ndim != 2 or y.ndim != 1 or len(X) != len(y) or len(y) < min_rows:
        raise ValueError("COMMISSIONING_INSUFFICIENT_ROWS")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("COMMISSIONING_NONFINITE_FIT")
    beta, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    if rank < rank_required:
        raise ValueError("COMMISSIONING_DESIGN_RANK_DEFICIENT")
    residual = y - X @ beta
    scale = max(float(np.sqrt(np.mean(residual**2))), 1e-6)
    return beta.astype(float).tolist(), scale, int(rank), int(len(y))


def _fit_profile(domain: str, raw: np.ndarray, passport: dict, world: str, source_sha: str) -> dict:
    values = validate_passport(passport, domain)
    x = np.asarray(raw, np.float64)
    if x.ndim != 2 or x.shape[0] < 16 or x.shape[1] != len(native.FAMILIES[domain]["channels"]) or not np.isfinite(x).all():
        raise ValueError("COMMISSIONING_INPUT_INVALID")
    dt = float(native.FAMILIES[domain]["sample_s"])
    details = {}
    if domain == "eps":
        nominal_power = values[0]
        expected = nominal_power * x[:, 0]
        # Include dark and illuminated intervals so gain and meter bias are identifiable.
        b1, solar_scale, rank1, n1 = _lstsq(np.column_stack((expected, np.ones(len(x)))), x[:, 2], 16, 2)
        b2, bus_scale, rank2, n2 = _lstsq(np.column_stack((x[:, 2], x[:, 1], np.ones(len(x)))), x[:, 3], 16, 3)
        context = [b1[0], b2[0], -b2[1]]
        details = {"solar_model": b1, "bus_model": b2, "solar_residual_scale_w": solar_scale,
                   "bus_residual_scale_w": bus_scale, "fit_rank": [rank1, rank2], "fit_rows": [n1, n2]}
    elif domain == "thermal":
        C, solar, rad = values; T = x[:, 2]
        if np.any((T <= 0) | (T > 2000)):
            raise ValueError("COMMISSIONING_THERMAL_TEMPERATURE")
        tm = .5 * (T[1:] + T[:-1])
        target = C * np.diff(T) / dt
        X = np.column_stack((solar * x[1:, 0], x[1:, 1], -rad * tm**4, np.ones(len(target))))
        beta, scale, rank, rows = _lstsq(X, target, 24, 4)
        context = beta[:3]
        details = {"energy_model": beta, "energy_residual_scale_w": scale, "fit_rank": rank, "fit_rows": rows}
    elif domain == "data":
        cap = values[0]; q = x[:, 3]
        dq = np.diff(q) / dt
        prev = q[:-1]; cur = q[1:]
        valid = (prev > 1.) & (cur > 1.) & (prev < cap - 1.) & (cur < cap - 1.)
        Xall = np.column_stack((x[1:, 1], -x[1:, 0] * x[1:, 2], np.ones(len(dq))))
        # Boundary clipping can remove many contact-on rows. Use interior rows first;
        # only fall back to all finite intervals if the healthy trajectory lacks rank.
        try:
            beta, scale, rank, rows = _lstsq(Xall[valid], dq[valid], 12, 3)
            fit_scope = "interior_queue_only"
        except ValueError:
            beta, scale, rank, rows = _lstsq(Xall, dq, 24, 3)
            fit_scope = "all_intervals_fallback_boundary_effect_possible"
        context = [beta[0], beta[1]]
        details = {"queue_model": beta, "queue_residual_scale_bps": scale, "fit_rank": rank,
                   "fit_rows": rows, "fit_scope": fit_scope}
    else:
        raise ValueError("UNKNOWN_FAMILY")
    result = {
        "schema": PROFILE_SCHEMA, "domain": domain, "instance_lineage": world,
        "available_before_fault_diagnosis": True,
        "source": {"reference": "independent_healthy_commissioning_run", "observation_sha256": source_sha},
        "context": {n: {"value": _finite(v, n), "unit": u, "role": "healthy_commissioning_parameter"}
                    for (n, u), v in zip(PROFILE_CONTEXT[domain], context)},
        "details": details,
        "scope": "same simulated instance; independent healthy excitation/noise; no fault label/onset/severity or paired diagnosis trajectory",
    }
    validate_profile(result, domain)
    return result


def validate_profile(p: dict, domain: str) -> list[float]:
    fields = {"schema", "domain", "instance_lineage", "available_before_fault_diagnosis", "source", "context", "details", "scope"}
    if not isinstance(p, dict) or set(p) != fields or p["schema"] != PROFILE_SCHEMA or p["domain"] != domain:
        raise ValueError("COMMISSIONING_PROFILE_SCHEMA")
    if p["available_before_fault_diagnosis"] is not True or not p["instance_lineage"]:
        raise ValueError("COMMISSIONING_PROFILE_TIMING")
    if set(p["source"]) != {"reference", "observation_sha256"} or len(p["source"]["observation_sha256"]) != 64:
        raise ValueError("COMMISSIONING_PROFILE_PROVENANCE")
    expected = {n for n, _ in PROFILE_CONTEXT[domain]}
    if set(p["context"]) != expected:
        raise ValueError("COMMISSIONING_PROFILE_CONTEXT")
    vals = []
    for name, unit in PROFILE_CONTEXT[domain]:
        e = p["context"][name]
        if set(e) != {"value", "unit", "role"} or e["unit"] != unit or e["role"] != "healthy_commissioning_parameter":
            raise ValueError("COMMISSIONING_PROFILE_CONTEXT_ENTRY")
        vals.append(_finite(e["value"], name))
    if not isinstance(p["details"], dict) or not p["details"] or not p["scope"]:
        raise ValueError("COMMISSIONING_PROFILE_DETAILS")
    return vals


def feature_contract(domain: str) -> dict:
    c = copy.deepcopy(native.contract(domain))
    c["device_family"] += ".healthy_commissioning.v2"
    c["scope"] += "; independent healthy commissioning profile assumed available before diagnosis; causal residual features; complete observations only"
    for name, unit in CONTEXT[domain]:
        c["channels"].append({"name": name, "unit": unit, "role": "deployment_input",
            "provenance": "Pre-injection nominal design metadata; not fault-updated state"})
    for name, unit in PROFILE_CONTEXT[domain]:
        c["channels"].append({"name": name, "unit": unit, "role": "deployment_input",
            "provenance": "Fitted from a separate healthy commissioning trajectory before diagnosis"})
    for name, unit in DERIVED[domain]:
        c["channels"].append({"name": name, "unit": unit, "role": "deployment_input",
            "provenance": "Causal calculation from current observation window plus frozen nominal/commissioning profile"})
    validate_contract(c)
    return c


def feature_binding(domain: str) -> dict:
    b = copy.deepcopy(native.binding(domain)); b["contract"] = feature_contract(domain)
    b["groups"] += [
        {"id": "normal_design", "channels": [n for n, _ in CONTEXT[domain]]},
        {"id": "healthy_commissioning", "channels": [n for n, _ in PROFILE_CONTEXT[domain]]},
        {"id": "commissioned_residuals", "channels": [n for n, _ in DERIVED[domain]]},
    ]
    b["mechanism_neural_branch_enabled"] = True
    b["feature_pipeline"] = FEATURE_SCHEMA
    b["commissioning_required"] = True
    return b


def _rolling_mean(v, width):
    v = np.asarray(v, np.float64); out = np.zeros_like(v)
    cs = np.r_[0., np.cumsum(v)]
    for i in range(len(v)):
        a = max(0, i + 1 - width); out[i] = (cs[i + 1] - cs[a]) / (i + 1 - a)
    return out


def _rolling_masked_mean(v, mask, width):
    v = np.asarray(v, np.float64); mask = np.asarray(mask, bool); out = np.zeros_like(v); count = np.zeros(len(v))
    sv = np.r_[0., np.cumsum(np.where(mask, v, 0.))]; sc = np.r_[0, np.cumsum(mask.astype(int))]
    for i in range(len(v)):
        a = max(0, i + 1 - width); n = sc[i + 1] - sc[a]; count[i] = n
        out[i] = (sv[i + 1] - sv[a]) / n if n else 0.
    return out, count


def _rolling_slope(y, x, width, min_rows=6):
    y = np.asarray(y, np.float64); x = np.asarray(x, np.float64); out = np.zeros(len(y)); valid = np.zeros(len(y), bool)
    for i in range(len(y)):
        a = max(0, i + 1 - width); xx = x[a:i+1]; yy = y[a:i+1]
        if len(xx) < min_rows: continue
        var = float(np.var(xx))
        if var <= max(1e-12, 1e-6 * float(np.mean(np.abs(xx)))**2): continue
        out[i] = float(np.mean((xx-xx.mean())*(yy-yy.mean())) / var); valid[i] = True
    return out, valid


def compute_features(domain: str, raw: np.ndarray, passport: dict, profile: dict) -> np.ndarray:
    normal = validate_passport(passport, domain); context = validate_profile(profile, domain)
    if profile["instance_lineage"] != passport["instance_lineage"]:
        raise ValueError("COMMISSIONING_INSTANCE_MISMATCH")
    x = np.asarray(raw, np.float64)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] != len(native.FAMILIES[domain]["channels"]) or not np.isfinite(x).all():
        raise ValueError("COMMISSIONED_INPUT_INVALID")
    if np.any((x[:, 0] < 0) | (x[:, 0] > 1)):
        raise ValueError("BOUNDARY_VALUE_OUT_OF_RANGE")
    n = len(x); dt = float(native.FAMILIES[domain]["sample_s"]); d = profile["details"]
    if domain == "eps":
        power = normal[0]; bsolar = np.asarray(d["solar_model"], float); bbus = np.asarray(d["bus_model"], float)
        expected = power * x[:,0]
        solar_res = x[:,2] - (bsolar[0]*expected + bsolar[1])
        bus_res = x[:,3] - (bbus[0]*x[:,2] + bbus[1]*x[:,1] + bbus[2])
        sz = solar_res / max(float(d["solar_residual_scale_w"]), 1e-6)
        bz = bus_res / max(float(d["bus_residual_scale_w"]), 1e-6)
        f = np.column_stack((sz,bz,_rolling_mean(sz,4),_rolling_mean(sz,16),_rolling_mean(bz,4),_rolling_mean(bz,16),(x[:,0]>.05).astype(float)))
    elif domain == "thermal":
        C, solar, rad = normal; T = x[:,2]
        if np.any((T<=0)|(T>2000)): raise ValueError("ABSOLUTE_TEMPERATURE_REQUIRED")
        beta = np.asarray(d["energy_model"], float); tm=.5*(T[1:]+T[:-1])
        target=C*np.diff(T)/dt; qrad=rad*tm**4
        X=np.column_stack((solar*x[1:,0],x[1:,1],-qrad,np.ones(len(target))))
        res=target-X@beta; rz=np.r_[0.,res/max(float(d["energy_residual_scale_w"]),1e-6)]
        q=np.r_[qrad[0] if len(qrad) else 0.,qrad]
        s16,v16=_rolling_slope(rz,q,16);s32,v32=_rolling_slope(rz,q,32)
        # Dimensionless slope after z-normalization; excitation validity requires either scale to be estimable.
        f=np.column_stack((rz,_rolling_mean(rz,4),_rolling_mean(rz,16),_rolling_mean(rz,32),s16,s32,(v16|v32).astype(float)))
    else:
        cap=normal[0];q=x[:,3];beta=np.asarray(d["queue_model"],float)
        dq=np.r_[0.,np.diff(q)/dt]
        pred=beta[0]*x[:,1]+beta[1]*(-x[:,0]*x[:,2])+beta[2]
        rz=(dq-pred)/max(float(d["queue_residual_scale_bps"]),1e-6);rz[0]=0.
        prev=np.r_[q[0],q[:-1]];valid=(q>1)&(prev>1)&(q<cap-1)&(prev<cap-1);valid[0]=False
        off=valid&(x[:,0]<.5);on=valid&(x[:,0]>=.5)
        off8,no8=_rolling_masked_mean(rz,off,8);off32,no32=_rolling_masked_mean(rz,off,32)
        on8,nn8=_rolling_masked_mean(rz,on,8);on32,nn32=_rolling_masked_mean(rz,on,32)
        contrast=np.where((no32>0)&(nn32>0),on32-off32,0.)
        have=((no32>0)&(nn32>0)).astype(float)
        f=np.column_stack((rz,off8,off32,on8,on32,contrast,have))
    nominal=np.broadcast_to(normal,(n,len(normal)));healthy=np.broadcast_to(context,(n,len(context)))
    y=np.concatenate((x,nominal,healthy,f),axis=1)
    if not np.isfinite(y).all():raise ValueError("NONFINITE_COMMISSIONED_FEATURE")
    out=y.astype(np.float32)
    if not np.isfinite(out).all():raise ValueError("COMMISSIONED_FEATURE_OUTSIDE_FLOAT32")
    return out


def generate_profiles(campaign, output):
    root=Path(campaign); verify(root); plan=read_json(root/"native"/"generation_plan.json")
    scenarios=plan.get("scenarios")
    if not isinstance(scenarios,list) or not scenarios:raise ValueError("COMMISSIONING_CAMPAIGN_PLAN")
    out=Path(output);out.mkdir(parents=True,exist_ok=False);profiles={};records=[]
    registry=read_json(root/"passport_registry.json")
    for item in scenarios:
        domain=item["domain"];world=item["world_id"];key=domain+"/"+world
        if key in profiles:continue
        pref=registry["passports"][key];pfile=safe_child(root,pref["path"])
        if sha_file(pfile)!=pref["sha256"]:raise ValueError("PASSPORT_FILE_CHANGED")
        passport=read_json(pfile);q=_independent_parameters(world,domain,item["parameters"])
        run=native.run_native(domain,q,0)
        relative=Path("observations")/domain/(world+".csv");path=out/relative;path.parent.mkdir(parents=True,exist_ok=True)
        with path.open("x",newline="",encoding="utf-8") as f:
            w=csv.writer(f);w.writerow(["time_s",*[n for n,_ in native.FAMILIES[domain]["channels"]]]);w.writerows(run["observations"])
        raw=np.asarray([r[1:] for r in run["observations"]],np.float64)
        profile=_fit_profile(domain,raw,passport,world,sha_file(path))
        pro_rel=Path("profiles")/domain/(world+".json");write_json(out/pro_rel,profile)
        profiles[key]={"profile":str(pro_rel),"profile_sha256":sha_file(out/pro_rel),"observation":str(relative),"observation_sha256":sha_file(path)}
        records.append({"domain":domain,"world_id":world,"role":item["role"],"diagnosis_measurement_seed":item["parameters"]["measurement_seed"],
                        "commissioning_measurement_seed":q["measurement_seed"],"diagnosis_phase":item["parameters"]["phase"],"commissioning_phase":q["phase"]})
    result={"schema":"phm.commissioning-registry.v2","source_campaign_manifest_sha256":sha_file(root/"manifest.json"),
            "profiles":profiles,"records":records,"healthy_runs":len(profiles),"fault_labels_used":False,"diagnosis_trajectory_used":False}
    write_json(out/"registry.json",result);seal(out,[str(p.relative_to(out)) for p in out.rglob('*') if p.is_file()])
    return result


def prepare_section(campaign, commissioning, domain, section, output):
    if section not in {"development","calibration","holdout"}:raise ValueError("SECTION_INVALID")
    root=Path(campaign);creg=Path(commissioning);verify(root);verify(creg)
    registry=read_json(creg/"registry.json")
    if registry["source_campaign_manifest_sha256"]!=sha_file(root/"manifest.json"):raise ValueError("COMMISSIONING_CAMPAIGN_CHANGED")
    roles={"development":{"train","validation"},"calibration":{"calibration"},"holdout":{"test"}}[section]
    data=Dataset(root/"native"/domain/section/"dataset.json",allowed_splits=roles);data.verify_files()
    if data.contract!=native.contract(domain):raise ValueError("RAW_DEVICE_CONTRACT_CHANGED")
    preg=read_json(root/"passport_registry.json");window=native.FAMILIES[domain]["window"]
    out=Path(output);out.mkdir(parents=True,exist_ok=False);samples=[];source=[]
    for s in data.samples:
        key=domain+"/"+s["lineage_id"];pr=preg["passports"][key];cr=registry["profiles"][key]
        pfile=safe_child(root,pr["path"]);cfile=safe_child(creg,cr["profile"])
        if sha_file(pfile)!=pr["sha256"] or sha_file(cfile)!=cr["profile_sha256"]:raise ValueError("COMMISSIONING_OR_PASSPORT_CHANGED")
        raw=sequence_from_csv(safe_child(data.root,s["path"]),data.contract,window)
        y=compute_features(domain,raw,read_json(pfile),read_json(cfile))
        rel=Path("observations")/(s["id"]+".csv");path=out/rel;path.parent.mkdir(exist_ok=True)
        with path.open("x",newline="",encoding="utf-8") as f:
            w=csv.writer(f);w.writerow(["time_s",*[c["name"] for c in feature_contract(domain)["channels"]]])
            for i,row in enumerate(y):w.writerow([i*data.contract["sample_s"],*map(float,row)])
        samples.append({**s,"path":str(rel),"sha256":sha_file(path)})
        source.append({"id":s["id"],"raw_sha256":s["sha256"],"passport_sha256":pr["sha256"],"commissioning_profile_sha256":cr["profile_sha256"]})
    manifest={"schema":"phm.sequence-dataset.v1","dataset_id":digest([data.hash,FEATURE_SCHEMA,sha_file(Path(__file__))]),
              "contract":feature_contract(domain),"samples":samples,
              "provenance":{"raw_dataset_hash":data.hash,"feature_pipeline":FEATURE_SCHEMA,"implementation_sha256":sha_file(Path(__file__)),
                            "commissioning_registry_sha256":sha_file(creg/"manifest.json"),"source_role":"trusted_preparation_no_learning",
                            "fault_labels_used_for_profile":False,"selected_on_test":False}}
    write_json(out/"dataset.json",manifest);write_json(out/"device_binding.json",feature_binding(domain));write_json(out/"source_records.json",source)
    Dataset(out/"dataset.json",allowed_splits=roles).verify_files();seal(out,[str(p.relative_to(out)) for p in out.rglob('*') if p.is_file()]);return manifest

HYBRID_SCHEMA = "phm.mechanism-plus-commissioning.v1"


def hybrid_feature_contract(domain: str) -> dict:
    """Keep all dev7 mechanism features and add healthy commissioning information."""
    from .family_pipeline import feature_contract as mechanism_contract
    c=copy.deepcopy(mechanism_contract(domain,"mechanism"))
    c["device_family"] += ".healthy_commissioning_hybrid.v1"
    c["scope"] += "; preserves original mechanism features and appends independent healthy-commissioning context/residuals"
    for name,unit in PROFILE_CONTEXT[domain]:
        c["channels"].append({"name":name,"unit":unit,"role":"deployment_input",
            "provenance":"Fitted from a separate healthy commissioning trajectory before diagnosis"})
    for name,unit in DERIVED[domain]:
        c["channels"].append({"name":name,"unit":unit,"role":"deployment_input",
            "provenance":"Causal calculation from current observations plus frozen nominal/healthy profile"})
    validate_contract(c);return c


def hybrid_feature_binding(domain: str) -> dict:
    from .family_pipeline import feature_binding as mechanism_binding
    b=copy.deepcopy(mechanism_binding(domain,"mechanism"));b["contract"]=hybrid_feature_contract(domain)
    b["groups"] += [
        {"id":"healthy_commissioning","channels":[n for n,_ in PROFILE_CONTEXT[domain]]},
        {"id":"commissioned_residuals","channels":[n for n,_ in DERIVED[domain]]},
    ]
    b["feature_pipeline"]=HYBRID_SCHEMA;b["commissioning_required"]=True;return b


def compute_hybrid_features(domain: str, raw: np.ndarray, passport: dict, profile: dict) -> np.ndarray:
    from .family_pipeline import compute_features as mechanism_features
    mech=mechanism_features(domain,raw,passport,"mechanism")
    comm=compute_features(domain,raw,passport,profile)
    base=len(native.FAMILIES[domain]["channels"])+len(CONTEXT[domain])
    # commissioned tail = healthy context + commissioning residuals. Mechanism's
    # original physical features remain intact and are not recomputed/replaced.
    out=np.concatenate((mech,comm[:,base:]),axis=1).astype(np.float32)
    if out.shape[1]!=len(hybrid_feature_contract(domain)["channels"]):raise ValueError("HYBRID_FEATURE_WIDTH")
    if not np.isfinite(out).all():raise ValueError("HYBRID_NONFINITE_FEATURE")
    return out


def prepare_hybrid_section(campaign, commissioning, domain, section, output):
    if section not in {"development","calibration","holdout"}:raise ValueError("SECTION_INVALID")
    root=Path(campaign);creg=Path(commissioning);verify(root);verify(creg)
    registry=read_json(creg/"registry.json")
    if registry["source_campaign_manifest_sha256"]!=sha_file(root/"manifest.json"):raise ValueError("COMMISSIONING_CAMPAIGN_CHANGED")
    roles={"development":{"train","validation"},"calibration":{"calibration"},"holdout":{"test"}}[section]
    data=Dataset(root/"native"/domain/section/"dataset.json",allowed_splits=roles);data.verify_files()
    if data.contract!=native.contract(domain):raise ValueError("RAW_DEVICE_CONTRACT_CHANGED")
    preg=read_json(root/"passport_registry.json");window=native.FAMILIES[domain]["window"]
    out=Path(output);out.mkdir(parents=True,exist_ok=False);samples=[];source=[]
    for s in data.samples:
        key=domain+"/"+s["lineage_id"];pr=preg["passports"][key];cr=registry["profiles"][key]
        pfile=safe_child(root,pr["path"]);cfile=safe_child(creg,cr["profile"])
        if sha_file(pfile)!=pr["sha256"] or sha_file(cfile)!=cr["profile_sha256"]:raise ValueError("COMMISSIONING_OR_PASSPORT_CHANGED")
        raw=sequence_from_csv(safe_child(data.root,s["path"]),data.contract,window)
        y=compute_hybrid_features(domain,raw,read_json(pfile),read_json(cfile))
        rel=Path("observations")/(s["id"]+".csv");path=out/rel;path.parent.mkdir(exist_ok=True)
        with path.open("x",newline="",encoding="utf-8") as f:
            w=csv.writer(f);w.writerow(["time_s",*[c["name"] for c in hybrid_feature_contract(domain)["channels"]]])
            for i,row in enumerate(y):w.writerow([i*data.contract["sample_s"],*map(float,row)])
        samples.append({**s,"path":str(rel),"sha256":sha_file(path)})
        source.append({"id":s["id"],"raw_sha256":s["sha256"],"passport_sha256":pr["sha256"],"commissioning_profile_sha256":cr["profile_sha256"]})
    manifest={"schema":"phm.sequence-dataset.v1","dataset_id":digest([data.hash,HYBRID_SCHEMA,sha_file(Path(__file__))]),
      "contract":hybrid_feature_contract(domain),"samples":samples,
      "provenance":{"raw_dataset_hash":data.hash,"feature_pipeline":HYBRID_SCHEMA,"implementation_sha256":sha_file(Path(__file__)),
                    "commissioning_registry_sha256":sha_file(creg/"manifest.json"),"source_role":"trusted_preparation_no_learning",
                    "preserves_dev7_mechanism_features":True,"fault_labels_used_for_profile":False,"selected_on_test":False}}
    write_json(out/"dataset.json",manifest);write_json(out/"device_binding.json",hybrid_feature_binding(domain));write_json(out/"source_records.json",source)
    Dataset(out/"dataset.json",allowed_splits=roles).verify_files();seal(out,[str(p.relative_to(out)) for p in out.rglob('*') if p.is_file()]);return manifest

def predict_hybrid_selected(run, selection, csv_path, passport_path, profile_path, domain):
    from .route_selection import _selection, selected_scores
    from .data import transform
    from .engine import deterministic_cpu
    run, dep, _, rec = _selection(run, selection)
    if dep.get("raw_feature_transform") != HYBRID_SCHEMA or dep.get("raw_feature_transform_sha256") != implementation_sha():
        raise ValueError("HYBRID_DEPLOYMENT_TRANSFORM_CHANGED")
    if dep["contract"] != hybrid_feature_contract(domain):
        raise ValueError("HYBRID_DEPLOYMENT_CONTRACT")
    raw = sequence_from_csv(Path(csv_path), native.contract(domain), dep["shape"]["window"])
    features = compute_hybrid_features(domain, raw, read_json(passport_path), read_json(profile_path))
    x = transform(features[None], read_json(run / "preprocessing.json"))
    with deterministic_cpu():
        scores = selected_scores(run, selection, x)[0]
    j = int(scores.argmax()); accepted = bool(scores[j] >= dep["threshold"])
    return {"route": rec["chosen_route"], "status": "PREDICTED" if accepted else "ABSTAIN_LOW_SCORE",
        "prediction": dep["contract"]["classes"][j] if accepted else None,
        "closed_set_argmax": dep["contract"]["classes"][j], "scores": scores.tolist(),
        "commissioning_profile_required": True, "probability_calibrated": False, "claim_status": "RESEARCH_ONLY"}


def predict_hybrid_risk(run, selection, risk, csv_path, passport_path, profile_path, domain):
    from .risk_calibration import _risk
    from .route_selection import selected_scores
    from .data import transform
    from .engine import deterministic_cpu
    run, dep, _, rec, rr = _risk(run, selection, risk)
    if dep.get("raw_feature_transform") != HYBRID_SCHEMA or dep.get("raw_feature_transform_sha256") != implementation_sha():
        raise ValueError("HYBRID_DEPLOYMENT_TRANSFORM_CHANGED")
    if dep["contract"] != hybrid_feature_contract(domain):
        raise ValueError("HYBRID_DEPLOYMENT_CONTRACT")
    raw = sequence_from_csv(Path(csv_path), native.contract(domain), dep["shape"]["window"])
    features = compute_hybrid_features(domain, raw, read_json(passport_path), read_json(profile_path))
    x = transform(features[None], read_json(run / "preprocessing.json"))
    with deterministic_cpu():
        scores = selected_scores(run, selection, x)[0]
    j = int(scores.argmax())
    accepted = bool(rr["status"] == "QUALIFIED_THRESHOLD" and scores[j] >= rr["threshold"])
    return {"route": rec["chosen_route"], "risk_status": rr["status"],
        "status": "PREDICTED" if accepted else "ABSTAIN_RISK_THRESHOLD",
        "prediction": dep["contract"]["classes"][j] if accepted else None,
        "closed_set_argmax": dep["contract"]["classes"][j], "scores": scores.tolist(), "threshold": rr["threshold"],
        "commissioning_profile_required": True, "probability_calibrated": False, "risk_certified": False,
        "claim_status": "RESEARCH_ONLY"}
