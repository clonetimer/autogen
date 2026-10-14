"""Causal component-wise mechanism features for sat_multicomponent.

Only normal design parameters available before fault injection are exported.
No fault label, onset, severity, paired nominal trajectory, or simulator truth is
used by the feature transform.
"""
from __future__ import annotations
import copy,csv,math
from pathlib import Path
import numpy as np
from .io import write_json,read_json,sha_file,safe_child,digest
from .data import Dataset,sequence_from_csv,validate_contract
from .adapters import sat_multicomponent as native

SCHEMA="phm.multicomponent-normal-passport.v1"
THERMAL_ROLLING_WINDOW = 64

CONTEXT={
 "eps":[("normal_full_light_power_a_w","W"),("normal_full_light_power_b_w","W")],
 "thermal":[
   ("normal_heat_capacity_a_j_per_k","J/K"),("normal_absorbed_full_light_a_w","W"),("normal_radiation_a_w_per_k4","W/K^4"),
   ("normal_heat_capacity_b_j_per_k","J/K"),("normal_absorbed_full_light_b_w","W"),("normal_radiation_b_w_per_k4","W/K^4")],
 "data":[("normal_storage_capacity_a_bits","bit"),("normal_storage_capacity_b_bits","bit")],
}
PHYSICAL={
 "eps":[
   ("solar_loss_a_fraction","1"),("solar_loss_b_fraction","1"),
   ("load_a_innovation_w","W"),("load_b_innovation_w","W"),
   ("solar_loss_a_running_mean","1"),("solar_loss_b_running_mean","1"),
   ("solar_a_observable","1"),("solar_b_observable","1")],
 "thermal":[
   ("energy_residual_a_running_w","W"),("radiation_loss_a_slope","1"),("unexplained_heat_a_intercept_w","W"),("thermal_a_regression_computable","1"),
   ("energy_residual_b_running_w","W"),("radiation_loss_b_slope","1"),("unexplained_heat_b_intercept_w","W"),("thermal_b_regression_computable","1")],
 "data":[
   ("generation_a_excess_running_bps","bit/s"),("contact_a_innovation_running_bps","bit/s"),("service_a_loss_fraction","1"),("queue_a_balance_computable","1"),
   ("generation_b_excess_running_bps","bit/s"),("contact_b_innovation_running_bps","bit/s"),("service_b_loss_fraction","1"),("queue_b_balance_computable","1")],
}

THERMAL_ROLLING_PHYSICAL=[
  ("radiation_loss_a_rolling64_slope","1"),
  ("unexplained_heat_a_rolling64_intercept_w","W"),
  ("thermal_a_rolling64_regression_computable","1"),
  ("radiation_loss_b_rolling64_slope","1"),
  ("unexplained_heat_b_rolling64_intercept_w","W"),
  ("thermal_b_rolling64_regression_computable","1"),
]


def make_passport(world,domain,p):
    if domain=="eps":
        vals=[1361*p["panel_a_area_m2"]*p["panel_a_efficiency"],
              1361*p["panel_b_area_m2"]*p["panel_b_efficiency"]]
    elif domain=="thermal":
        vals=[]
        for k in ("a","b"):
            vals += [p[f"thermal_{k}_mass_kg"]*890.0,
                     1361*p[f"thermal_{k}_area_m2"]*p[f"thermal_{k}_absorptivity"],
                     5.670374419e-8*p[f"thermal_{k}_area_m2"]*p[f"thermal_{k}_emissivity"]]
    elif domain=="data":
        vals=[p["capacity_a_bits"],p["capacity_b_bits"]]
    else: raise ValueError("UNKNOWN_MULTICOMPONENT_DOMAIN")
    return {"schema":SCHEMA,"domain":domain,"instance_lineage":world,
      "available_before_fault_injection":True,
      "parameters":{n:{"value":float(v),"unit":u,"role":"normal_design_parameter"}
                    for (n,u),v in zip(CONTEXT[domain],vals)},
      "scope":"exact nominal simulator design known before injection; no fault truth or paired nominal diagnosis trajectory"}


def validate_passport(p,domain):
    if not isinstance(p,dict) or set(p)!={"schema","domain","instance_lineage","available_before_fault_injection","parameters","scope"}:
        raise ValueError("MULTICOMPONENT_PASSPORT_SCHEMA")
    if p["schema"]!=SCHEMA or p["domain"]!=domain or p["available_before_fault_injection"] is not True:
        raise ValueError("MULTICOMPONENT_PASSPORT_SCHEMA")
    expected={n for n,_ in CONTEXT[domain]}
    if set(p["parameters"])!=expected:raise ValueError("MULTICOMPONENT_PASSPORT_PARAMETERS")
    vals=[]
    for n,u in CONTEXT[domain]:
        e=p["parameters"][n]
        if set(e)!={"value","unit","role"} or e["unit"]!=u or e["role"]!="normal_design_parameter":
            raise ValueError("MULTICOMPONENT_PASSPORT_PARAMETER_ROLE")
        v=e["value"]
        if type(v) not in (float,int) or not math.isfinite(v) or v<=0:raise ValueError("MULTICOMPONENT_PASSPORT_VALUE")
        vals.append(float(v))
    return vals


def feature_contract(domain, representation_version="v1"):
    if representation_version not in {"v1","v2"}:
        raise ValueError("MULTICOMPONENT_REPRESENTATION_VERSION")
    c=copy.deepcopy(native.contract(domain))
    suffix=".mechanism.v2" if representation_version=="v2" and domain=="thermal" else ".mechanism.v1"
    c["device_family"]+=suffix
    c["scope"]+="; component-wise causal mechanism residuals from observed window and pre-injection normal design passport"
    if representation_version=="v2" and domain=="thermal":
        c["scope"]+=f"; causal trailing-{THERMAL_ROLLING_WINDOW}-sample thermal residual regression"
    for n,u in CONTEXT[domain]:
        c["channels"].append({"name":n,"unit":u,"role":"deployment_input",
          "provenance":"Normal design parameter available before fault injection"})
    physical=list(PHYSICAL[domain])
    if representation_version=="v2" and domain=="thermal":
        physical+=THERMAL_ROLLING_PHYSICAL
    for n,u in physical:
        c["channels"].append({"name":n,"unit":u,"role":"deployment_input",
          "provenance":"Deterministic causal feature from this observed window plus normal design passport"})
    validate_contract(c);return c


def feature_binding(domain, representation_version="v1"):
    b=copy.deepcopy(native.binding(domain));b["contract"]=feature_contract(domain,representation_version)
    physical=list(PHYSICAL[domain])
    if representation_version=="v2" and domain=="thermal": physical+=THERMAL_ROLLING_PHYSICAL
    b["groups"] += [
      {"id":"normal_design","channels":[n for n,_ in CONTEXT[domain]]},
      {"id":"component_mechanism_residuals","channels":[n for n,_ in physical]}]
    b["mechanism_neural_branch_enabled"]=True
    b["mechanism_representation_version"]="v2" if representation_version=="v2" and domain=="thermal" else "v1"
    return b


def _running_mean(v,valid):
    count=np.cumsum(valid.astype(float));total=np.cumsum(np.where(valid,v,0.))
    return np.divide(total,count,out=np.zeros_like(total),where=count>0),count


def _thermal_features(T,light,command,C,solar,rad,dt):
    qrad=rad*T**4
    radiation=.5*(qrad[1:]+qrad[:-1])
    residual=C*np.diff(T)/dt-(solar*light[1:]+command[1:]-radiation)
    count=np.arange(1,len(T),dtype=float);sumq=np.cumsum(radiation);sumr=np.cumsum(residual)
    qmean=sumq/count;rmean=sumr/count
    var=np.cumsum(radiation**2)/count-qmean**2
    cov=np.cumsum(radiation*residual)/count-qmean*rmean
    good=(count>=8)&(var>np.maximum(.01,1e-4*qmean**2))
    slope=np.divide(cov,var,out=np.zeros_like(var),where=good)
    intercept=np.where(good,rmean-slope*qmean,0.)
    return np.vstack((np.zeros((1,4)),np.column_stack((rmean,slope,intercept,good.astype(float)))))


def _thermal_rolling_regression(T,light,command,C,solar,rad,dt,window=THERMAL_ROLLING_WINDOW):
    """Causal trailing-window residual-vs-radiation regression; no label/onset information."""
    qrad=rad*T**4
    radiation=.5*(qrad[1:]+qrad[:-1])
    residual=C*np.diff(T)/dt-(solar*light[1:]+command[1:]-radiation)
    out=np.zeros((len(T),3),dtype=float)
    for row in range(1,len(T)):
        hi=row;lo=max(0,hi-window)
        q=radiation[lo:hi];r=residual[lo:hi]
        if len(q)<8: continue
        qmean=float(np.mean(q));rmean=float(np.mean(r));var=float(np.var(q))
        good=var>max(.01,1e-4*qmean*qmean)
        if not good: continue
        cov=float(np.mean((q-qmean)*(r-rmean)))
        slope=cov/var;intercept=rmean-slope*qmean
        out[row]=(slope,intercept,1.0)
    return out


def _data_features(q,contact,generation,service,capacity,dt):
    if np.any((q<0)|(q>capacity)):raise ValueError("QUEUE_OUTSIDE_DESIGN_CAPACITY")
    inc=np.r_[0.,np.diff(q)]/dt
    prev=np.r_[q[0],q[:-1]]
    valid=(q>0)&(prev>0)&(q<capacity)&(prev<capacity);valid[0]=False
    innovation=inc-(generation-contact*service)
    off=(contact<.5)&valid;on=(contact>=.5)&valid
    gen,ng=_running_mean(innovation,off);con,nc=_running_mean(innovation,on)
    denom=np.maximum(service,1.);have=(ng>0)&(nc>0)
    loss=np.where(have,(con-gen)/denom,0.)
    return np.column_stack((gen,con,loss,have.astype(float)))


def compute_features(domain,raw,passport,representation_version="v1"):
    if representation_version not in {"v1","v2"}: raise ValueError("MULTICOMPONENT_REPRESENTATION_VERSION")
    if representation_version=="v2" and domain!="thermal": raise ValueError("MULTICOMPONENT_V2_THERMAL_ONLY")
    vals=validate_passport(passport,domain)
    x=np.asarray(raw,dtype=np.float64)
    if x.ndim!=2 or x.shape[1]!=len(native.FAMILIES[domain]["channels"]) or len(x)<2 or not np.isfinite(x).all():
        raise ValueError("MULTICOMPONENT_FEATURE_SHAPE")
    dt=float(native.FAMILIES[domain]["sample_s"]);n=len(x)
    if domain=="eps":
        pa,pb=vals;light=x[:,0]
        va=light>.05;vb=va.copy()
        ea=pa*light;eb=pb*light
        la=np.divide(ea-x[:,3],ea,out=np.zeros(n),where=va)
        lb=np.divide(eb-x[:,4],eb,out=np.zeros(n),where=vb)
        ma,_=_running_mean(la,va);mb,_=_running_mean(lb,vb)
        ia=x[:,5]-x[:,1];ib=x[:,6]-x[:,2]
        f=np.column_stack((la,lb,ia,ib,ma,mb,va.astype(float),vb.astype(float)))
    elif domain=="thermal":
        a=_thermal_features(x[:,3],x[:,0],x[:,1],*vals[:3],dt)
        b=_thermal_features(x[:,4],x[:,0],x[:,2],*vals[3:],dt)
        f=np.concatenate((a,b),axis=1)
        if representation_version=="v2":
            ar=_thermal_rolling_regression(x[:,3],x[:,0],x[:,1],*vals[:3],dt)
            br=_thermal_rolling_regression(x[:,4],x[:,0],x[:,2],*vals[3:],dt)
            f=np.concatenate((f,ar,br),axis=1)
    else:
        a=_data_features(x[:,3],x[:,0],x[:,1],x[:,2],vals[0],dt)
        b=_data_features(x[:,6],x[:,0],x[:,4],x[:,5],vals[1],dt)
        f=np.concatenate((a,b),axis=1)
    context=np.broadcast_to(vals,(n,len(vals)))
    y=np.concatenate((x,context,f),axis=1)
    if not np.isfinite(y).all():raise ValueError("MULTICOMPONENT_NONFINITE_FEATURE")
    return y.astype(np.float32)


def prepare_section(campaign,domain,section,output,representation_version="v1"):
    campaign=Path(campaign);out=Path(output)
    if out.exists():raise FileExistsError("MULTICOMPONENT_FEATURE_OUTPUT_EXISTS")
    data=Dataset(campaign/"native"/domain/"dataset.json",allowed_splits={"train","validation","calibration","test"})
    role={"development":{"train","validation"},"calibration":{"calibration"},"holdout":{"test"}}.get(section)
    if role is None:raise ValueError("MULTICOMPONENT_FEATURE_SECTION")
    samples=[]
    for s in data.samples:
        if s["split"] not in role:continue
        raw=sequence_from_csv(safe_child(data.root,s["path"]),data.contract,native.FAMILIES[domain]["window"])
        pp=campaign/"normal_passports"/s["group_id"]/f"{domain}.json"
        passport=read_json(pp)
        y=compute_features(domain,raw,passport,representation_version)
        dest=out/"observations"/f"{s['id']}.csv";dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open("w",newline="",encoding="utf-8") as f:
            w=csv.writer(f);w.writerow(["time_s",*[c["name"] for c in feature_contract(domain,representation_version)["channels"]]])
            for i,row in enumerate(y):w.writerow([i*data.contract["sample_s"],*map(float,row)])
        samples.append({**s,"path":str(dest.relative_to(out)),"sha256":sha_file(dest)})
    out.mkdir(parents=True,exist_ok=True)
    transform="multicomponent_mechanism.v2" if representation_version=="v2" else "multicomponent_mechanism.v1"
    manifest={"schema":"phm.sequence-dataset.v1","dataset_id":digest([data.hash,domain,section,transform]),
      "contract":feature_contract(domain,representation_version),"samples":samples,
      "provenance":{"raw_dataset_hash":data.hash,"section":section,"transform":transform,
                    "transform_sha256":sha_file(Path(__file__)),
                    "fault_labels_used_for_features":False,"fault_onset_used_for_features":False,
                    "pair_baseline_used":False,"rolling_window_samples":THERMAL_ROLLING_WINDOW if representation_version=="v2" else None}}
    write_json(out/"dataset.json",manifest);write_json(out/"device_binding.json",feature_binding(domain,representation_version))
    return manifest
