"""Native Basilisk multi-component localization benchmark.

Two distinguishable components are instantiated per subsystem. Fault mechanism
and fault location are separate target dimensions. This remains low-fidelity,
instrumented subsystem simulation; it is not a coupled whole-spacecraft model.
"""
from __future__ import annotations
import csv, gc, importlib.metadata as md, math, random, time, traceback
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from ..io import digest,write_json,sha_file,seal,run_lock
from ..data import validate_contract

PROTOCOL="phm.native-multicomponent.v1"
ROLES=("train","validation","calibration","test")

FAMILIES={
 "eps":{
   "duration_s":1200,"sample_s":10,"solver_s":2,"window":64,
   "classes":["NOMINAL","SOLAR_A_EFFICIENCY_LOSS","SOLAR_B_EFFICIENCY_LOSS","LOAD_A_EXTRA","LOAD_B_EXTRA"],
   "channels":[("illumination","1"),("load_a_command_w","W"),("load_b_command_w","W"),
               ("panel_a_meter_w","W"),("panel_b_meter_w","W"),
               ("load_a_meter_w","W"),("load_b_meter_w","W"),("bus_net_meter_w","W")],
   "groups":[("boundary",[0]),("load_a",[1,5]),("load_b",[2,6]),("solar_a",[3]),("solar_b",[4]),("bus",[7])],
   "truth":{
      0:(None,"NOMINAL"),1:("solar_a","solar_conversion_efficiency_loss"),
      2:("solar_b","solar_conversion_efficiency_loss"),
      3:("load_a","uncommanded_or_excess_load"),4:("load_b","uncommanded_or_excess_load")},
   "native_modules":["SimpleSolarPanel","SimpleBattery"]},
 "thermal":{
   "duration_s":3600,"sample_s":20,"solver_s":2,"window":128,
   "classes":["NOMINAL","THERMAL_A_EMISSIVITY_LOSS","THERMAL_B_EMISSIVITY_LOSS","THERMAL_A_EXTRA_HEAT","THERMAL_B_EXTRA_HEAT"],
   "channels":[("illumination","1"),("heat_a_command_w","W"),("heat_b_command_w","W"),
               ("temperature_a_k","K"),("temperature_b_k","K")],
   "groups":[("boundary",[0]),("thermal_a",[1,3]),("thermal_b",[2,4])],
   "truth":{
      0:(None,"NOMINAL"),1:("thermal_a","effective_emissivity_loss"),
      2:("thermal_b","effective_emissivity_loss"),
      3:("thermal_a","uncommanded_internal_heat"),4:("thermal_b","uncommanded_internal_heat")},
   "native_modules":["SensorThermal"]},
 "data":{
   "duration_s":1200,"sample_s":10,"solver_s":2,"window":64,
   "classes":["NOMINAL","SERVICE_A_RATE_LOSS","SERVICE_B_RATE_LOSS","GEN_A_EXCESS","GEN_B_EXCESS"],
   "channels":[("contact_allowed","1"),
       ("generation_a_command_bps","bit/s"),("service_a_command_bps","bit/s"),("queue_a_bits","bit"),
       ("generation_b_command_bps","bit/s"),("service_b_command_bps","bit/s"),("queue_b_bits","bit")],
   "groups":[("boundary",[0]),("data_a",[1,2,3]),("data_b",[4,5,6])],
   "truth":{
      0:(None,"NOMINAL"),1:("service_a","service_rate_loss"),2:("service_b","service_rate_loss"),
      3:("instrument_a","excess_data_generation"),4:("instrument_b","excess_data_generation")},
   "native_modules":["SimpleStorageUnit"]},
}


def check_config(c):
    if not isinstance(c,dict) or set(c)!={"schema","seed","groups_per_role","domains"} or c.get("schema")!=PROTOCOL:
        raise ValueError("MULTICOMPONENT_CONFIG_SCHEMA")
    if type(c["seed"]) is not int or not 0<=c["seed"]<2**32:raise ValueError("INVALID_FACTORY_SEED")
    if not isinstance(c["groups_per_role"],dict) or set(c["groups_per_role"])!=set(ROLES):
        raise ValueError("FOUR_ROLES_REQUIRED")
    if any(type(v) is not int or not 0<=v<=10000 for v in c["groups_per_role"].values()) or not sum(c["groups_per_role"].values()):
        raise ValueError("INVALID_GROUP_COUNTS")
    if not isinstance(c["domains"],list) or not c["domains"] or any(d not in FAMILIES for d in c["domains"]) or len(c["domains"])!=len(set(c["domains"])):
        raise ValueError("INVALID_DOMAIN_SET")


def world_id(seed,role,index):
    if role not in ROLES or type(index) is not int or index<0:raise ValueError("INVALID_WORLD")
    return digest([PROTOCOL,seed,role,index])


def parameters(world,domain):
    s=FAMILIES[domain];r=random.Random(int(digest([world,domain])[:16],16));dt=s["sample_s"]
    q={
      "phase":r.uniform(0,2*math.pi),"boundary_period_s":s["duration_s"]/r.uniform(2.,3.5),
      "onset_s":dt*round(s["duration_s"]*r.uniform(.35,.55)/dt),
      "remaining_fraction":r.uniform(.55,.8),"measurement_seed":int(digest([world,domain,"measurement"])[:16],16)}
    if domain=="eps":
        for k in ("a","b"):
            q[f"panel_{k}_area_m2"]=r.uniform(.07,.13);q[f"panel_{k}_efficiency"]=r.uniform(.20,.28)
            q[f"load_{k}_w"]=r.uniform(3.,8.);q[f"extra_load_{k}_w"]=r.uniform(1.5,4.)
    elif domain=="thermal":
        for k in ("a","b"):
            q[f"thermal_{k}_area_m2"]=r.uniform(.012,.025);q[f"thermal_{k}_absorptivity"]=r.uniform(.4,.6)
            q[f"thermal_{k}_emissivity"]=r.uniform(.7,.9);q[f"thermal_{k}_mass_kg"]=r.uniform(.25,.6)
            q[f"thermal_{k}_initial_k"]=r.uniform(300.,315.);q[f"heat_{k}_w"]=r.uniform(2.,6.)
            q[f"extra_heat_{k}_w"]=r.uniform(3.,7.)
    else:
        for k in ("a","b"):
            q[f"generation_{k}_bps"]=r.uniform(6000.,14000.);q[f"service_{k}_bps"]=r.uniform(35000.,65000.)
            q[f"capacity_{k}_bits"]=25000000.
    return q


def contract(domain):
    s=FAMILIES[domain]
    c={"device_family":f"satellite.{domain}.multicomponent_native.v1",
       "task":"fixed_length_fault_mechanism_and_component_localization",
       "channels":[{"name":n,"unit":u,"role":"deployment_input",
          "provenance":"Prescribed command/boundary or virtual component meter; hardware sensing unvalidated"}
          for n,u in s["channels"]],
       "classes":s["classes"],"sample_s":s["sample_s"],"evidence_type":"simulation",
       "scope":"Two instrumented parallel components inside one low-fidelity subsystem; mechanism and location are separable; no whole-spacecraft coupling or hardware qualification."}
    validate_contract(c);return c


def binding(domain):
    s=FAMILIES[domain]
    return {"schema":"phm.device-binding.v1","contract":contract(domain),
      "groups":[{"id":name,"channels":[s["channels"][i][0] for i in ids]} for name,ids in s["groups"]],
      "window_samples":s["window"],"native_modules":s["native_modules"],
      "input_frame":"physical_units_fixed_sampling","quality_mode":"complete_observation_only_missing_values_rejected",
      "scope":"two-component native localization research benchmark",
      "mechanism_neural_branch_enabled":False}


def plan_campaign(c):
    check_config(c)
    return [{"world_id":(w:=world_id(c["seed"],role,i)),"role":role,"index_in_role":i,
      "domain":d,"parameters":parameters(w,d)}
      for role in ROLES for i in range(c["groups_per_role"][role]) for d in c["domains"]]


def load_runtime():
    version=md.version("bsk")
    if version.split("+")[0]!="2.11.0":raise ValueError("UNREVIEWED_BASILISK_VERSION:"+version)
    from Basilisk.architecture import messaging
    from Basilisk.utilities import SimulationBaseClass
    from Basilisk.simulation import simpleSolarPanel,simpleBattery,sensorThermal,simpleStorageUnit
    return SimpleNamespace(version=version,msg=messaging,sim=SimulationBaseClass,
      solar=simpleSolarPanel,battery=simpleBattery,thermal=sensorThermal,storage=simpleStorageUnit)


def run_native(domain,p,label,*,solver_s=None,thermal_excitation=None,thermal_duration_s=None):
    s=FAMILIES[domain]
    if thermal_duration_s is not None:
        if (domain != "thermal" or type(thermal_duration_s) is not int
                or thermal_duration_s < 3600 or thermal_duration_s > 28800
                or thermal_duration_s % s["sample_s"]):
            raise ValueError("R66_INVALID_THERMAL_DURATION")
        if (type(p.get("onset_s")) is not int or not 0 < p["onset_s"] < thermal_duration_s):
            raise ValueError("R66_ONSET_OUTSIDE_DURATION")
    if domain not in FAMILIES or type(label) is not int or not 0<=label<len(s["classes"]):raise ValueError("UNKNOWN_DOMAIN_OR_LABEL")
    if thermal_excitation is not None:
        if domain != "thermal": raise ValueError("R65_EXCITATION_ONLY_THERMAL")
        from ..thermal_excitation_r65 import BASELINE,EXPERIMENT
        from ..thermal_excitation_r66 import SLOW
        from ..thermal_excitation_r67 import PROFILES as R67_PROFILES
        if thermal_excitation not in (BASELINE,EXPERIMENT,SLOW,*R67_PROFILES):
            raise ValueError("R65_UNKNOWN_EXCITATION_PROFILE")
    B=load_runtime();step=s["solver_s"] if solver_s is None else solver_s
    if type(step) is not int or step<=0 or s["sample_s"]%step:raise ValueError("INVALID_NATIVE_STEP")
    sim=B.sim.SimBaseClass();proc=sim.CreateNewProcess("multicomponent")
    proc.addTask(sim.CreateNewTask("tick",int(step*1e9)))
    def add(model,tag,priority):
        model.ModelTag=tag;sim.AddModelToTask("tick",model,priority)
    state=B.msg.SCStatesMsgPayload();state.r_BN_N=[0.,0.,0.];state.sigma_BN=[0.,0.,0.]
    state_msg=B.msg.SCStatesMsg().write(state)
    sun=B.msg.SpicePlanetStateMsgPayload();sun.PositionVector=[149597870700.,0.,0.]
    sun_msg=B.msg.SpicePlanetStateMsg().write(sun);eclipse_msg=B.msg.EclipseMsg()
    ports=[]

    if domain=="eps":
        load_msgs=[B.msg.PowerNodeUsageMsg(),B.msg.PowerNodeUsageMsg()]
        panels=[]
        for i,k in enumerate(("a","b")):
            panel=B.solar.SimpleSolarPanel();panel.stateInMsg.subscribeTo(state_msg);panel.sunInMsg.subscribeTo(sun_msg)
            panel.sunEclipseInMsg.subscribeTo(eclipse_msg);add(panel,f"panel_{k}",40-i);panels.append(panel)
        battery=B.battery.SimpleBattery();battery.storageCapacity=160000.;battery.storedCharge_Init=80000.
        for panel in panels:battery.addPowerNodeToModel(panel.nodePowerOutMsg)
        for msg in load_msgs:battery.addPowerNodeToModel(msg)
        add(battery,"shared_battery",20)
        ports=[panels[0].nodePowerOutMsg,panels[1].nodePowerOutMsg,battery.batPowerOutMsg]
    elif domain=="thermal":
        thermals=[]
        for i,k in enumerate(("a","b")):
            t=B.thermal.SensorThermal();t.nHat_B=[1.,0.,0.];t.sensorPowerStatus=1
            t.stateInMsg.subscribeTo(state_msg);t.sunInMsg.subscribeTo(sun_msg);t.sunEclipseInMsg.subscribeTo(eclipse_msg)
            add(t,f"thermal_{k}",30-i);thermals.append(t)
        ports=[t.temperatureOutMsg for t in thermals]
    else:
        source=[B.msg.DataNodeUsageMsg(),B.msg.DataNodeUsageMsg()]
        sink=[B.msg.DataNodeUsageMsg(),B.msg.DataNodeUsageMsg()]
        stores=[]
        for i,k in enumerate(("a","b")):
            st=B.storage.SimpleStorageUnit();st.storageCapacity=p[f"capacity_{k}_bits"]
            st.addDataNodeToModel(source[i]);st.addDataNodeToModel(sink[i]);add(st,f"storage_{k}",30-i);stores.append(st)
        ports=[st.storageUnitDataOutMsg for st in stores]

    readers=[port.addSubscriber() for port in ports]
    observations=[];truth=[];times=[];noise=random.Random(p["measurement_seed"]);sample=s["sample_s"]
    for now in range(0,(thermal_duration_s if thermal_duration_s is not None else s["duration_s"])+1,sample):
        left=max(0,now-sample);active=(label!=0 and left>=p["onset_s"])
        phase=2*math.pi*left/p["boundary_period_s"]+p["phase"]
        light=float(math.sin(phase)>-.25);contact=float(math.cos(phase+.7)>.25)
        ns=int(now*1e9);write_ns=int(left*1e9)
        e=B.msg.EclipseMsgPayload();e.illuminationFactor=light;eclipse_msg.write(e,write_ns)
        aux=[]
        if domain=="eps":
            commands=[];consumptions=[]
            for i,k in enumerate(("a","b")):
                cmd=p[f"load_{k}_w"]*(1+.2*math.sin(left/(65+10*i)+p["phase"]));commands.append(cmd)
                fault_solar=(label==1+i and active);eta=p[f"panel_{k}_efficiency"]*(p["remaining_fraction"] if fault_solar else 1)
                panels[i].setPanelParameters([1.,0.,0.],p[f"panel_{k}_area_m2"],eta)
                fault_load=(label==3+i and active);cons=cmd+(p[f"extra_load_{k}_w"] if fault_load else 0.);consumptions.append(cons)
                q=B.msg.PowerNodeUsageMsgPayload();q.netPower=-cons;load_msgs[i].write(q,write_ns)
        elif domain=="thermal":
            commands=[]
            for i,k in enumerate(("a","b")):
                t=thermals[i];t.sensorArea=p[f"thermal_{k}_area_m2"];t.sensorAbsorptivity=p[f"thermal_{k}_absorptivity"]
                t.sensorMass=p[f"thermal_{k}_mass_kg"];t.sensorSpecificHeat=890.;t.T_0=p[f"thermal_{k}_initial_k"]-273.15
                if thermal_excitation is None:
                    # Retain original R62/R64 trajectory path exactly.
                    cmd=p[f"heat_{k}_w"]*(1+.25*float(math.sin(left/(115+15*i))>0))
                else:
                    if thermal_excitation == "slow_thermal_plateaus_r66":
                        from ..thermal_excitation_r66 import commanded_heat as r66_command
                        cmd=r66_command(p[f"heat_{k}_w"],left,i)
                    elif thermal_excitation in R67_PROFILES:
                        from ..thermal_excitation_r67 import commanded_heat as r67_command
                        cmd=r67_command(p[f"heat_{k}_w"],left,i,thermal_excitation)
                    else:
                        from ..thermal_excitation_r65 import commanded_heat
                        cmd=commanded_heat(p[f"heat_{k}_w"],left,i,thermal_excitation)
                commands.append(cmd)
                emiss_loss=(label==1+i and active);t.sensorEmissivity=p[f"thermal_{k}_emissivity"]*(p["remaining_fraction"] if emiss_loss else 1)
                extra=(label==3+i and active);t.sensorPowerDraw=cmd+(p[f"extra_heat_{k}_w"] if extra else 0.)
        else:
            commands=[]
            for i,k in enumerate(("a","b")):
                gen=float(round(p[f"generation_{k}_bps"]*(1+.3*float(math.sin(left/(40+5*i))>0))))
                service=float(round(p[f"service_{k}_bps"]));commands.append((gen,service))
                src=B.msg.DataNodeUsageMsgPayload();src.dataName=f"instrument_{k}"
                src.baudRate=gen*(1.7 if label==3+i and active else 1.)
                snk=B.msg.DataNodeUsageMsgPayload();snk.dataName=f"service_{k}"
                snk.baudRate=-contact*service*(p["remaining_fraction"] if label==1+i and active else 1.)
                source[i].write(src,write_ns);sink[i].write(snk,write_ns)
        if now==0:sim.InitializeSimulation()
        sim.ConfigureStopTime(ns);sim.ExecuteSimulation()
        nt=[int(r.timeWritten()) for r in readers]
        if any(t!=ns for t in nt):raise ValueError("NATIVE_OUTPUT_NOT_UPDATED")
        times.append(nt)
        if domain=="eps":
            pa=float(panels[0].nodePowerOutMsg.read().netPower);pb=float(panels[1].nodePowerOutMsg.read().netPower)
            bo=battery.batPowerOutMsg.read();net=float(bo.currentNetPower)
            values=[light,commands[0],commands[1],pa+noise.gauss(0,.1),pb+noise.gauss(0,.1),
                    consumptions[0]+noise.gauss(0,.08),consumptions[1]+noise.gauss(0,.08),net+noise.gauss(0,.1)]
            internal=float(bo.storageLevel);aux=[pa,pb,net]
        elif domain=="thermal":
            ta=float(thermals[0].temperatureOutMsg.read().temperature)+273.15
            tb=float(thermals[1].temperatureOutMsg.read().temperature)+273.15
            values=[light,commands[0],commands[1],ta+noise.gauss(0,.1),tb+noise.gauss(0,.1)]
            internal=(ta+tb)/2;aux=[ta,tb]
        else:
            qa=float(stores[0].storageUnitDataOutMsg.read().storageLevel);qb=float(stores[1].storageUnitDataOutMsg.read().storageLevel)
            values=[contact,commands[0][0],commands[0][1],qa,commands[1][0],commands[1][1],qb]
            internal=qa+qb;aux=[qa,qb]
        if not all(math.isfinite(v) for v in [*values,internal,*aux]):raise ValueError("NONFINITE_NATIVE_VALUE")
        observations.append([now,*values]);truth.append([now,int(active),internal,*aux])
    del sim;gc.collect()
    return {"observations":observations,"truth":truth,"output_times_ns":times}


def _csv(path,header,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="") as f:
        w=csv.writer(f);w.writerow(header);w.writerows(rows)


def pair_audit(nom,fault,p,domain,label):
    a=np.asarray(nom["observations"]);b=np.asarray(fault["observations"]);active=np.asarray(fault["truth"])[:,1].astype(bool)
    if not np.array_equal(a[~active],b[~active]):raise ValueError("PREFAULT_PAIR_MISMATCH")
    delta=float(np.max(np.abs(a[active,1:]-b[active,1:]))) if active.any() else 0.
    comp,mech=FAMILIES[domain]["truth"][label]
    return {"label":FAMILIES[domain]["classes"][label],"component":comp,"mechanism":mech,
      "prefault_exact_match":True,"max_active_observation_delta":delta,
      "effect_status":"NUMERICAL_DIFFERENCE_PRESENT" if delta>0 else "NO_NUMERICAL_DIFFERENCE",
      "diagnostic_observability":"NOT_QUALIFIED"}


def generate(config_path,output,on_case=None):
    c=__import__("json").loads(Path(config_path).read_text());check_config(c);out=Path(output)
    out.mkdir(parents=True,exist_ok=False)
    with run_lock(out):
      root=out/"native";root.mkdir()
      completed=[];samples={d:{r:[] for r in ROLES} for d in c["domains"]};start=time.monotonic()
      try:
        for item in plan_campaign(c):
          w,role,d,p=item["world_id"],item["role"],item["domain"],item["parameters"]
          from ..multicomponent_features import make_passport
          passport=make_passport(w,d,p)
          pp=out/"normal_passports"/w/f"{d}.json";pp.parent.mkdir(parents=True,exist_ok=True);write_json(pp,passport)
          nominal=run_native(d,p,0)
          for label in range(len(FAMILIES[d]["classes"])):
            data=nominal if label==0 else run_native(d,p,label)
            audit=pair_audit(nominal,data,p,d,label);case=digest([w,d,label])
            domain_root=root/d;domain_root.mkdir(parents=True,exist_ok=True)
            obs=domain_root/"observations"/f"{case}.csv";_csv(obs,["time_s",*[x[0] for x in FAMILIES[d]["channels"]]],data["observations"])
            comp,mech=FAMILIES[d]["truth"][label]
            _csv(out/"evaluation_truth"/role/d/f"{case}.csv",["time_s","fault_active","internal_state",*[f"aux_{i}" for i in range(len(data["truth"][0])-3)]],data["truth"])
            sample={"id":case,"group_id":w,"lineage_id":w,"split":role,"label":FAMILIES[d]["classes"][label],
                    "path":str(obs.relative_to(domain_root)),"sha256":sha_file(obs)}
            samples[d][role].append(sample)
            rec={"case_id":case,"world_id":w,"role":role,"domain":d,"sample":sample,"pair_audit":audit,
                 "root_cause_truth":{"component":comp,"mechanism":mech},
                 "case_execution":"NATIVE_COMPLETED"}
            write_json(out/"case_records"/f"{case}.json",rec);completed.append(rec)
            if on_case:on_case(rec)
        for d in c["domains"]:
          all_samples=[s for r in ROLES for s in samples[d][r]]
          write_json(root/d/"dataset.json",{"schema":"phm.sequence-dataset.v1","dataset_id":digest([c,d]),
            "contract":contract(d),"samples":all_samples,
            "provenance":{"protocol":PROTOCOL,"split_unit":"world_id","root_cause_truth_separate":True}})
          write_json(root/d/"device_binding.json",binding(d))
        summary={"status":"MULTICOMPONENT_NATIVE_DATA_GENERATED","cases":len(completed),
          "world_roots":sum(c["groups_per_role"].values()),"domains":c["domains"],
          "counts_by_role":{r:sum(len(samples[d][r]) for d in c["domains"]) for r in ROLES},
          "fault_pairs_with_numeric_difference":sum(x["sample"]["label"]!="NOMINAL" and x["pair_audit"]["effect_status"]=="NUMERICAL_DIFFERENCE_PRESENT" for x in completed),
          "elapsed_seconds":time.monotonic()-start,"production_qualified":False}
        write_json(out/"summary.json",summary);seal(out,[str(p.relative_to(out)) for p in out.rglob("*") if p.is_file() and p.name!="manifest.json"])
        return summary
      except Exception as exc:
        write_json(out/"status.json",{"status":"FAILED","completed_cases":len(completed),"reason":str(exc),"traceback":traceback.format_exc()})
        raise
