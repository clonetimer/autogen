"""Additional native stress envelopes; no injected fault is a telemetry edit.

This is a new adapter derived from the frozen non-ADCS native adapter. Existing
run_native behavior and evidence remain unchanged. Illumination/contact remain
prescribed boundary messages, not propagated orbit/communication physics.
"""
import math,random
from .sat_nonadcs import FAMILIES,load_runtime

def fault_envelope(t,parameters):
    onset=float(parameters["onset_s"])
    if t<onset:return 0.
    mode=parameters.get("fault_profile","step")
    if mode=="step":return 1.
    if mode=="progressive":
        ramp=float(parameters.get("ramp_s",0))
        if not math.isfinite(ramp) or ramp<=0:raise ValueError("INVALID_RAMP_DURATION")
        return min(1.,(t-onset)/ramp)
    if mode=="intermittent":
        period=float(parameters.get("intermittent_period_s",0));duty=float(parameters.get("duty_cycle",0))
        if not math.isfinite(period) or period<=0 or not 0<duty<1:raise ValueError("INVALID_INTERMITTENT_PROFILE")
        return float((t-onset)%period<period*duty)
    raise ValueError("UNKNOWN_FAULT_PROFILE")

def run_stress_native(domain:str,p:dict,label:int,*,solver_s:int|None=None)->dict:
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
        left=max(0,now-sample); amplitude=fault_envelope(left,p) if label else 0.;active=amplitude>0
        phase=2*math.pi*left/p["boundary_period_s"]+p["phase"]
        light=float(math.sin(phase)>-.25);contact=float(math.cos(phase+.7)>.25)
        ns=int(now*1e9)
        # Write at the beginning of the held interval, not a future time.
        write_ns=int(left*1e9)
        eclipse=B.msg.EclipseMsgPayload();eclipse.illuminationFactor=light;eclipse_msg.write(eclipse,write_ns)
        if domain=="eps":
            command=p["load_w"]*(1+.25*math.sin(left/70+p["phase"]))
            eta=p["panel_efficiency"]*p.get("solar_gain",1.0)*(1-amplitude*(1-p["remaining_fraction"]) if label==1 else 1)
            panel.setPanelParameters([1.,0.,0.],p["panel_area_m2"],eta)
            consumption=command*p.get("load_actuator_gain",1.0)+(amplitude*p["extra_load_w"] if label==2 else 0.)
            q=B.msg.PowerNodeUsageMsgPayload();q.netPower=-consumption;load_msg.write(q,write_ns)
        elif domain=="thermal":
            command=p["heat_w"]*(1+.3*float(math.sin(left/130)>0))
            thermal.sensorAbsorptivity=p["absorptivity"]*p.get("thermal_absorptivity_gain",1.0)
            thermal.sensorEmissivity=p["emissivity"]*p.get("thermal_emissivity_gain",1.0)*(1-amplitude*(1-p["remaining_fraction"]) if label==1 else 1)
            thermal.sensorPowerDraw=command*p.get("heat_actuator_gain",1.0)+(amplitude*p["extra_heat_w"] if label==2 else 0.)
        else:
            generation=float(round(p["generation_bps"]*(1+.35*float(math.sin(left/45)>0))))
            service=float(round(p["service_bps"]))
            producer=B.msg.DataNodeUsageMsgPayload();producer.dataName="instrument"
            producer.baudRate=generation*p.get("generation_gain",1.0)*(1+amplitude*(p.get("excess_generation_multiplier",1.8)-1) if label==2 else 1.)
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
        observations.append([now,*values]);truth.append([now,int(active),internal,*auxiliary,amplitude])
    return {"observations":observations,"truth":truth,"output_times_ns":times,
            "native_modules":s["native_modules"],"solver_s":step,"bsk_version":B.version}

