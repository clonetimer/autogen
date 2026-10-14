"""Controlled-boundary, native Basilisk non-ADCS research data factory.

Optional runtime adapter. No dependency on Basilisk at module import time.
No TaskSpec/Capability certification, orbit propagation, RF chain, battery
chemistry, or hardware validation is implied. Faults affect native parameters
or input messages during execution, never post-processed output arrays.
"""
from __future__ import annotations
import copy
import csv
import gc
import importlib.metadata as md
import math
from pathlib import Path
import platform
import random
import time
import traceback
from types import SimpleNamespace
import numpy as np
from ..io import digest, write_json, read_json, sha_file, seal, run_lock, verify, safe_child
from ..data import Dataset, validate_contract, reject_numeric_overlap

PROTOCOL = "phm.native-nonadcs.v1"
ROLES = ("train", "validation", "calibration", "test")
FAMILIES = {
    "eps": {
        "duration_s":1200, "sample_s":10, "solver_s":2, "window":64,
        "classes":["NOMINAL", "SOLAR_EFFICIENCY_LOSS", "EXTRA_BUS_LOAD"],
        "channels":[("illumination", "1"), ("load_command_w", "W"),
                    ("panel_meter_w", "W"), ("bus_net_meter_w", "W")],
        "components":[("solar", [0,2]), ("commanded_load", [1]), ("bus", [3])],
        "native_modules":["SimpleSolarPanel", "SimpleBattery"],
    },
    "thermal": {
        "duration_s":3600, "sample_s":20, "solver_s":2, "window":128,
        "classes":["NOMINAL", "EMISSIVITY_LOSS", "EXTRA_INTERNAL_HEAT"],
        "channels":[("illumination", "1"), ("heat_command_w", "W"), ("temperature_k", "K")],
        "components":[("environment", [0]), ("commanded_heat", [1]), ("thermal_sensor", [2])],
        "native_modules":["SensorThermal"],
    },
    "data": {
        "duration_s":1200, "sample_s":10, "solver_s":2, "window":64,
        "classes":["NOMINAL", "SERVICE_RATE_LOSS", "EXCESS_GENERATION"],
        "channels":[("contact_allowed", "1"), ("generation_command_bps", "bit/s"),
                    ("service_command_bps", "bit/s"), ("queue_counter_bits", "bit")],
        "components":[("instrument", [1]), ("service", [0,2]), ("storage", [3])],
        "native_modules":["SimpleStorageUnit"],
    },
}


def check_config(config:dict)->None:
    if isinstance(config,dict) and config.get("schema")=="phm.native-nonadcs.challenge.v2":
        from .nonadcs_challenge import check_config as check_challenge_config
        check_challenge_config(config);return
    if set(config)!={"schema","seed","groups_per_role","domains"} or config["schema"]!=PROTOCOL:
        raise ValueError("NONADCS_CONFIG_SCHEMA")
    if type(config["seed"]) is not int or not 0<=config["seed"]<2**32:
        raise ValueError("INVALID_FACTORY_SEED")
    counts=config["groups_per_role"]
    if not isinstance(counts,dict) or set(counts)!=set(ROLES):
        raise ValueError("FOUR_ROLES_REQUIRED")
    if any(type(v) is not int or not 0<=v<=10000 for v in counts.values()) or not sum(counts.values()):
        raise ValueError("INVALID_GROUP_COUNTS")
    ds=config["domains"]
    if not isinstance(ds,list) or not ds or any(x not in FAMILIES for x in ds) or len(ds)!=len(set(ds)):
        raise ValueError("INVALID_DOMAIN_SET")


def world_id(seed:int,role:str,index:int)->str:
    if role not in ROLES or type(index) is not int or index<0:
        raise ValueError("INVALID_WORLD_ROLE_OR_INDEX")
    return digest([PROTOCOL,seed,role,index])


def parameters(world:str,domain:str)->dict:
    if domain not in FAMILIES: raise ValueError("UNKNOWN_DOMAIN")
    r=random.Random(int(digest([world,domain])[:16],16));spec=FAMILIES[domain];dt=spec["sample_s"]
    return {
        "phase":r.uniform(0,2*math.pi),
        "boundary_period_s":spec["duration_s"]/r.uniform(2.,3.),
        "onset_s":dt*round(spec["duration_s"]*r.uniform(.25,.4)/dt),
        "remaining_fraction":r.uniform(.45,.7),
        "panel_area_m2":r.uniform(.09,.14),"panel_efficiency":r.uniform(.20,.28),
        "load_w":r.uniform(6.,12.),"extra_load_w":r.uniform(3.,6.),
        "thermal_area_m2":r.uniform(.015,.025),"absorptivity":r.uniform(.4,.6),
        "emissivity":r.uniform(.7,.9),"mass_kg":r.uniform(.3,.6),
        "initial_temperature_k":r.uniform(300.,315.),"heat_w":r.uniform(2.,7.),
        "extra_heat_w":r.uniform(4.,8.),"generation_bps":r.uniform(8000.,15000.),
        "service_bps":r.uniform(45000.,65000.),
        "measurement_seed":int(digest([world,domain,"measurement"])[:16],16),
    }


def contract(domain:str)->dict:
    spec=FAMILIES[domain]
    c={
        "device_family":f"satellite.{domain}.prescribed_native.v1",
        "task":"fixed_length_case_classification",
        "channels":[{"name":name,"unit":unit,"role":"deployment_input",
            "provenance":("Externally prescribed, logged boundary or command" if i < (3 if domain=="data" else 2)
              else "Virtual meter over a native output; hardware sensing availability/error unvalidated")}
            for i,(name,unit) in enumerate(spec["channels"])],
        "classes":spec["classes"],"sample_s":spec["sample_s"],"evidence_type":"simulation",
        "scope":"Low-fidelity native subsystem, prescribed illumination/contact, no orbit/RF propagation; research measurement assumptions, no hardware, RUL or online alarm qualification.",
    }
    validate_contract(c);return c


def binding(domain:str)->dict:
    s=FAMILIES[domain]
    return {"schema":"phm.device-binding.v1","contract":contract(domain),
        "groups":[{"id":n,"channels":[s["channels"][i][0] for i in ids]} for n,ids in s["components"]],
        "window_samples":s["window"],"native_modules":s["native_modules"],
        "input_frame":"physical_units_fixed_sampling", "quality_mode":"complete_observation_only_missing_values_rejected",
        "scope":"synthetic instrumented native subsystem; not approved real device sensing",
        "mechanism_neural_branch_enabled":False,
        "measurement_settings":{"power_meter_white_noise_std_w":.1,"temperature_white_noise_std_k":.1,
             "data_counter":"ideal","source":"engineering_sensitivity_assumptions_not_hardware_specifications"}}


def plan_campaign(config:dict)->list[dict]:
    check_config(config)
    if config["schema"]=="phm.native-nonadcs.challenge.v2":
        from .nonadcs_challenge import plan_campaign as challenge_plan
        return challenge_plan(config)
    return [{"world_id":(world:=world_id(config["seed"],role,i)),"role":role,"index_in_role":i,
            "domain":domain,"parameters":parameters(world,domain)}
            for role in ROLES for i in range(config["groups_per_role"][role]) for domain in config["domains"]]


def load_runtime():
    version=md.version("bsk")
    if version.split("+")[0]!="2.11.0":raise ValueError("UNREVIEWED_BASILISK_VERSION:"+version)
    from Basilisk.architecture import messaging
    from Basilisk.utilities import SimulationBaseClass
    from Basilisk.simulation import simpleSolarPanel,simpleBattery,sensorThermal,simpleStorageUnit
    return SimpleNamespace(version=version,msg=messaging,sim=SimulationBaseClass,
        solar=simpleSolarPanel,battery=simpleBattery,thermal=sensorThermal,storage=simpleStorageUnit)


def run_native(domain:str,p:dict,label:int,*,solver_s:int|None=None)->dict:
    """Boundaries are held over each recorded interval; not a full spacecraft."""
    if domain not in FAMILIES or type(label) is not int or label not in (0,1,2):
        raise ValueError("UNKNOWN_DOMAIN_OR_LABEL")
    B=load_runtime();s=FAMILIES[domain];step=s["solver_s"] if solver_s is None else solver_s
    if type(step) is not int or step<=0 or s["sample_s"]%step:raise ValueError("INVALID_NATIVE_STEP")
    sim=B.sim.SimBaseClass();proc=sim.CreateNewProcess("nonadcs")
    proc.addTask(sim.CreateNewTask("tick",int(step*1e9)))
    def add(model,tag,priority):
        model.ModelTag=tag;sim.AddModelToTask("tick",model,priority)
    state=B.msg.SCStatesMsgPayload();state.r_BN_N=[0.,0.,0.];state.sigma_BN=[0.,0.,0.]
    state_msg=B.msg.SCStatesMsg().write(state)
    sun=B.msg.SpicePlanetStateMsgPayload();sun.PositionVector=[149597870700.,0.,0.]
    sun_msg=B.msg.SpicePlanetStateMsg().write(sun)
    eclipse_msg=B.msg.EclipseMsg();load_msg=B.msg.PowerNodeUsageMsg()
    source_msg=B.msg.DataNodeUsageMsg();sink_msg=B.msg.DataNodeUsageMsg()
    if domain=="eps":
        panel=B.solar.SimpleSolarPanel();panel.stateInMsg.subscribeTo(state_msg)
        panel.sunInMsg.subscribeTo(sun_msg);panel.sunEclipseInMsg.subscribeTo(eclipse_msg)
        add(panel,"research_panel",30)
        store=B.battery.SimpleBattery();store.storageCapacity=120000.;store.storedCharge_Init=60000.
        store.addPowerNodeToModel(panel.nodePowerOutMsg);store.addPowerNodeToModel(load_msg)
        add(store,"research_battery",20);ports=[panel.nodePowerOutMsg,store.batPowerOutMsg]
    elif domain=="thermal":
        thermal=B.thermal.SensorThermal();thermal.nHat_B=[1.,0.,0.]
        thermal.sensorArea=p["thermal_area_m2"];thermal.sensorAbsorptivity=p["absorptivity"]
        thermal.sensorMass=p["mass_kg"];thermal.sensorSpecificHeat=890.
        thermal.T_0=p["initial_temperature_k"]-273.15;thermal.sensorPowerStatus=1
        thermal.stateInMsg.subscribeTo(state_msg);thermal.sunInMsg.subscribeTo(sun_msg)
        thermal.sunEclipseInMsg.subscribeTo(eclipse_msg)
        add(thermal,"research_thermal",20);ports=[thermal.temperatureOutMsg]
    else:
        store=B.storage.SimpleStorageUnit();store.storageCapacity=25000000
        store.addDataNodeToModel(source_msg);store.addDataNodeToModel(sink_msg)
        add(store,"research_data_storage",20);ports=[store.storageUnitDataOutMsg]
    readers=[port.addSubscriber() for port in ports]
    observations=[];truth=[];times=[];noise=random.Random(p["measurement_seed"])
    sample=s["sample_s"]
    for now in range(0,s["duration_s"]+1,sample):
        left=max(0,now-sample);active=(label!=0 and left>=p["onset_s"])
        phase=2*math.pi*left/p["boundary_period_s"]+p["phase"]
        light=float(math.sin(phase)>-.25);contact=float(math.cos(phase+.7)>.25)
        ns=int(now*1e9)
        # Write at the beginning of the held interval, not a future time.
        write_ns=int(left*1e9)
        eclipse=B.msg.EclipseMsgPayload();eclipse.illuminationFactor=light;eclipse_msg.write(eclipse,write_ns)
        if domain=="eps":
            command=p["load_w"]*(1+.25*math.sin(left/70+p["phase"]))
            eta=p["panel_efficiency"]*p.get("solar_gain",1.0)*(p["remaining_fraction"] if active and label==1 else 1)
            panel.setPanelParameters([1.,0.,0.],p["panel_area_m2"],eta)
            consumption=command*p.get("load_actuator_gain",1.0)+(p["extra_load_w"] if active and label==2 else 0.)
            q=B.msg.PowerNodeUsageMsgPayload();q.netPower=-consumption;load_msg.write(q,write_ns)
        elif domain=="thermal":
            command=p["heat_w"]*(1+.3*float(math.sin(left/130)>0))
            thermal.sensorAbsorptivity=p["absorptivity"]*p.get("thermal_absorptivity_gain",1.0)
            thermal.sensorEmissivity=p["emissivity"]*p.get("thermal_emissivity_gain",1.0)*(p["remaining_fraction"] if active and label==1 else 1)
            thermal.sensorPowerDraw=command*p.get("heat_actuator_gain",1.0)+(p["extra_heat_w"] if active and label==2 else 0.)
        else:
            generation=float(round(p["generation_bps"]*(1+.35*float(math.sin(left/45)>0))))
            service=float(round(p["service_bps"]))
            producer=B.msg.DataNodeUsageMsgPayload();producer.dataName="instrument"
            producer.baudRate=generation*p.get("generation_gain",1.0)*(p.get("excess_generation_multiplier",1.8) if active and label==2 else 1.)
            consumer=B.msg.DataNodeUsageMsgPayload();consumer.dataName="service"
            consumer.baudRate=-contact*service*p.get("service_gain",1.0)*(p["remaining_fraction"] if active and label==1 else 1.)
            source_msg.write(producer,write_ns);sink_msg.write(consumer,write_ns)
        if now==0:sim.InitializeSimulation()
        sim.ConfigureStopTime(ns);sim.ExecuteSimulation()
        native_times=[int(reader.timeWritten()) for reader in readers]
        if any(t!=ns for t in native_times):raise ValueError("NATIVE_OUTPUT_NOT_UPDATED")
        times.append(native_times)
        if domain=="eps":
            power=float(panel.nodePowerOutMsg.read().netPower);out=store.batPowerOutMsg.read()
            internal=float(out.storageLevel);net=float(out.currentNetPower)
            if not -1e-6<=internal<=120000.+1e-6:raise ValueError("BATTERY_BOUND_VIOLATION")
            values=[light,command,power*p.get("panel_meter_gain",1.0)+p.get("panel_meter_bias_w",0.0)+noise.gauss(0,p.get("power_noise_std_w",.1)),net*p.get("bus_meter_gain",1.0)+p.get("bus_meter_bias_w",0.0)+noise.gauss(0,p.get("power_noise_std_w",.1))]
            auxiliary=[power,net]
        elif domain=="thermal":
            internal=float(thermal.temperatureOutMsg.read().temperature)+273.15
            if now==0 and not math.isclose(internal,p["initial_temperature_k"],abs_tol=1e-7):
                raise ValueError("INITIAL_TEMPERATURE_UNIT_CHECK_FAILED")
            if not 0<internal<2000:raise ValueError("THERMAL_NUMERICAL_SANITY_FAILED")
            values=[light,command,internal+p.get("temperature_bias_k",0.0)+noise.gauss(0,p.get("temperature_noise_std_k",.1))];auxiliary=[]
        else:
            internal=float(store.storageUnitDataOutMsg.read().storageLevel)
            if not 0<=internal<=25000000:raise ValueError("DATA_BOUND_VIOLATION")
            meter=min(25000000.0,max(0.0,internal*p.get("queue_meter_gain",1.0)+p.get("queue_bias_bits",0.0)))
            values=[contact,generation,service,meter];auxiliary=[]
        if not all(math.isfinite(v) for v in [*values,internal,*auxiliary]):raise ValueError("NONFINITE_NATIVE_VALUE")
        observations.append([now,*values]);truth.append([now,int(active),internal,*auxiliary])
    return {"observations":observations,"truth":truth,"output_times_ns":times,
            "native_modules":s["native_modules"],"solver_s":step,"bsk_version":B.version}


def pair_audit(nominal:dict,fault:dict,p:dict,domain:str,label:int)->dict:
    a,b=np.asarray(nominal["observations"]),np.asarray(fault["observations"])
    if a.shape!=b.shape or not np.array_equal(a[:,0],b[:,0]):raise ValueError("PAIR_TIME_MISMATCH")
    active=np.asarray(fault["truth"])[:,1].astype(bool)
    if not np.array_equal(a[~active],b[~active]):raise ValueError("PREFAULT_PAIR_MISMATCH")
    delta=np.max(np.abs(a[:,1:]-b[:,1:]),axis=0)
    return {"label":FAMILIES[domain]["classes"][label],"prefault_exact_match":True,
            "first_active_output_s":float(b[active,0][0]) if active.any() else None,
            "maximum_difference":dict(zip([n for n,_ in FAMILIES[domain]["channels"]],delta.tolist())),
            "effect_status":"NUMERICAL_DIFFERENCE_PRESENT" if delta.any() else "NUMERICALLY_IDENTICAL_TO_NOMINAL",
            "diagnostic_observability":"NOT_QUALIFIED"}


def _csv(path:Path,fields:list[str],rows:list)->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",newline="",encoding="utf-8") as f:
        writer=csv.writer(f);writer.writerow(fields);writer.writerows(rows)


def generate_nonadcs(config_path:str|Path,output:str|Path,*,on_case=None)->dict:
    config=read_json(config_path);plan=plan_campaign(config);output=Path(output)
    # Failed output is preserved; this factory does not append to partial data.
    output.mkdir(parents=True,exist_ok=False);start=time.perf_counter();completed=[]
    with run_lock(output):
        write_json(output/"status.json",{"status":"RUNNING"})
        try:
            B=load_runtime()
            native_bindings={}
            for module in (B.solar,B.battery,B.thermal,B.storage):
                file=Path(module.__file__);binary=next(file.parent.glob('_'+file.stem+'*.so'),None)
                native_bindings[file.stem]={"wrapper_sha256":sha_file(file),"binary_sha256":sha_file(binary) if binary else None}
            write_json(output/"generation_plan.json",{"schema":config["schema"],"config":config,"scenarios":plan,
                "scope":"prescribed_boundary_low_fidelity_native_subsystems_not_orbit_propagated",
                "parameters_source":"engineering_research_settings_not_hardware_specs",
                "factory_sha256":sha_file(Path(__file__)),"bsk_version":B.version,"native_bindings":native_bindings,
                "python":platform.python_version(),"same_world_across_domains_is_grouped":True})
            for domain in config["domains"]:
                bd=binding(domain)
                if config["schema"]!="phm.native-nonadcs.v1":
                    from .nonadcs_challenge import measurement_assumptions
                    bd["measurement_settings"]=measurement_assumptions()
                    bd["generator_protocol"]=config["schema"]
                write_json(output/domain/"device_binding.json",bd)
            for item in plan:
                domain,role,p,world=(item[k] for k in ("domain","role","parameters","world_id"))
                section="development" if role in {"train","validation"} else ("holdout" if role=="test" else "calibration")
                root=output/domain/section;nominal=None
                for label in range(3):
                    case=digest([world,domain,label]);data=run_native(domain,p,label)
                    if nominal is None:nominal=data
                    audit=pair_audit(nominal,data,p,domain,label)
                    obs=root/"observations"/(case+".csv")
                    _csv(obs,["time_s",*[n for n,_ in FAMILIES[domain]["channels"]]],data["observations"])
                    extras=["native_panel_w","native_net_w"] if domain=="eps" else []
                    _csv(output/"evaluation_truth"/role/domain/(case+".csv"),
                        ["time_s","fault_active","internal_state",*extras],data["truth"])
                    sample={"id":case,"group_id":world,"lineage_id":world,"split":role,
                        "label":FAMILIES[domain]["classes"][label],"path":str(obs.relative_to(root)),"sha256":sha_file(obs)}
                    record={"sample":sample,"domain":domain,"section":section,"pair_audit":audit,
                        "native_modules":data["native_modules"],"all_output_timestamps_verified":True,
                        "final_output_times_ns":data["output_times_ns"][-1],"case_execution":"NATIVE_COMPLETED"}
                    write_json(output/"case_records"/(case+".json"),record);completed.append(record)
                    write_json(output/"progress.json",{"completed_cases":len(completed),"last_case":case})
                    if on_case:on_case(record)
                gc.collect()
            for domain in config["domains"]:
                for section in ("development","calibration","holdout"):
                    samples=[r["sample"] for r in completed if r["domain"]==domain and r["section"]==section]
                    if not samples:continue
                    manifest={"schema":"phm.sequence-dataset.v1","dataset_id":digest([config,domain,section]),
                       "contract":contract(domain),"samples":samples,
                       "provenance":{"schema":config["schema"],"factory_sha256":sha_file(Path(__file__)),
                        "scope":"native_subsystem_research_only","split_unit":"world_id_shared_by_domain_and_fault_variants",
                        "seed":config["seed"],"generation_config_sha256":digest(config)}}
                    path=output/domain/section/"dataset.json";write_json(path,manifest)
                    d=Dataset(path,allowed_splits={s["split"] for s in samples});d.verify_files()
                    blocks=[d.load(split,FAMILIES[domain]["window"]) for split in sorted({s["split"] for s in samples})]
                    if len(blocks)==2:reject_numeric_overlap(*blocks)
            summary={"status":"NATIVE_DATA_GENERATED","cases":len(completed),
                "world_roots":sum(config["groups_per_role"].values()),
                "domain_parameter_groups":len(plan),"counts_by_role":{r:3*len(config["domains"])*n for r,n in config["groups_per_role"].items()},
                "counts_per_domain":3*sum(config["groups_per_role"].values()),
                "fault_pairs_with_numeric_difference":sum(r["sample"]["label"]!="NOMINAL" and r["pair_audit"]["effect_status"]=="NUMERICAL_DIFFERENCE_PRESENT" for r in completed),
                "elapsed_seconds":time.perf_counter()-start,"training_executed":False,
                "production_release_qualified":False,"hardware_validated":False,
                "capability_registry_qualification":False}
            write_json(output/"summary.json",summary);write_json(output/"status.json",{"status":"COMPLETE"})
            seal(output,[str(p.relative_to(output)) for p in output.rglob('*') if p.is_file() and p.name!='.lock'])
            return summary
        except Exception as exc:
            (output/"failure.log").write_text(traceback.format_exc(),encoding="utf-8")
            write_json(output/"status.json",{"status":"FAILED","completed_cases":len(completed),"reason":str(exc)})
            raise
