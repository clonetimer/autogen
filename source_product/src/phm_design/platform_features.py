"""Deployment-only feature programs for the PHM native-adapter test matrix.

These are candidate numerical representations, not fault decision rules. The
linear response parameters are estimated from an explicitly healthy reference
prefix. Simulator injection parameters, truth and world IDs are never accepted.
"""
from __future__ import annotations
import numpy as np
from .open_design import FeatureRegistry,ObservationSpec,temporal_bins,random_temporal_kernels

FIELDS={
 'ADCS':('time_s','adcs.pointing_error_deg',*(f'adcs.rw.speed_rad_s_{i}' for i in range(3)),*(f'adcs.rw.command_torque_nm_{i}' for i in range(3))),
 'EPS':('time_s','illumination','load_command_w','panel_meter_w','bus_net_meter_w'),
 'Thermal':('time_s','illumination','heat_command_w','temperature_k'),
 'Data':('time_s','contact_allowed','generation_command_bps','service_command_bps','queue_counter_bits'),
 'EPS_localization':('time_s','illumination','load_a_command_w','load_b_command_w','panel_a_meter_w','panel_b_meter_w','load_a_meter_w','load_b_meter_w','bus_net_meter_w'),
 'Thermal_localization':('time_s','illumination','heat_a_command_w','heat_b_command_w','temperature_a_k','temperature_b_k'),
 'Data_localization':('time_s','contact_allowed','generation_a_command_bps','service_a_command_bps','queue_a_bits','generation_b_command_bps','service_b_command_bps','queue_b_bits'),
}

def spec(domain):
    if domain not in FIELDS:raise ValueError('UNKNOWN_PLATFORM_DOMAIN')
    return ObservationSpec('platform.'+domain,FIELDS[domain])

def _summary(z):
    z=np.asarray(z,float)
    if z.ndim==1:z=z[:,None]
    blocks=np.array_split(z,6)
    parts=[]
    for b in blocks:
        t=np.linspace(-1.,1.,len(b));parts.extend((b.mean(0),b.std(0),np.quantile(b,.1,axis=0),np.quantile(b,.9,axis=0),(t@b)/max(float(t@t),1.)))
    return np.arcsinh(np.concatenate(parts))

def _response(X,y,n):
    X=np.asarray(X,float);y=np.asarray(y,float)
    center=X[:n].mean(0);scale=X[:n].std(0);scale=np.where(scale>1e-12,scale,1.)
    z=np.column_stack(((X-center)/scale,np.ones(len(X))))
    regularizer=np.eye(z.shape[1])*1e-4;regularizer[-1,-1]=1e-8
    beta=np.linalg.solve(z[:n].T@z[:n]+regularizer,z[:n].T@y[:n])
    r=y-z@beta
    # Scale follows reference target magnitude, not a fault amplitude or truth.
    units=max(float(np.std(y[:n])),float(np.mean(np.abs(y[:n])))*.05,1e-8)
    return r/units

def self_reference(a):
    x=a[:,1:];n=max(8,len(a)//5);center=np.median(x[:n],axis=0)
    scale=np.maximum(x[:n].std(0),np.maximum(np.mean(np.abs(x[:n]),axis=0)*.02,1e-6))
    return _summary((x-center)/scale)

def _eps(a):
    n=max(8,len(a)//5);light,command,panel,net=a[:,1:].T
    response=np.column_stack((_response(light[:,None],panel,n),_response(np.column_stack((light,command)),net,n),
        _response(command[:,None],panel-net,n)))
    return _summary(response)

def _thermal(a):
    t=a[:,0];light,heat,temp=a[:,1:].T;stride=3
    # Finite differences over measured temperatures and logged input commands.
    dt=t[stride:]-t[:-stride];rate=(temp[stride:]-temp[:-stride])/dt
    X=np.column_stack((light[:-stride],heat[:-stride],(temp[:-stride]/300.)**4))
    r=_response(X,rate,max(8,len(rate)//5))
    return np.concatenate((_summary(r),self_reference(a)))

def _data(a):
    t,contact,gen,service,queue=a.T;dt=np.diff(t);rate=np.diff(queue)/dt
    X=np.column_stack((gen[:-1],-service[:-1]*contact[:-1]))
    r=_response(X,rate,max(8,len(rate)//5))
    return np.concatenate((_summary(r),_summary(np.column_stack((rate/(np.abs(gen[:-1])+1),queue[:-1]/(np.mean(gen[:max(8,len(a)//5)])*max(t[-1],1)+1))))))

def _adcs(a):
    t=a[:,0];wheel=a[:,2:5];command=a[:,5:8];n=max(8,(len(t)-1)//5);parts=[]
    for j in range(3):
        speed=wheel[:,j];rate=np.diff(speed)/np.diff(t)
        fits=[]
        for lag in (0,1):
            u=command[:-1,j] if lag==0 else command[1:,j]
            X=np.column_stack((u,speed[:-1]));r=_response(X,rate,n);fits.append(r)
        r=min(fits,key=lambda v:float(np.mean(v[:n]**2)))
        parts.append((float(np.mean(np.abs(r[n:]))),_summary(r)))
    # Diagnose mechanism independently of wheel location in this contract.
    return np.concatenate([v for _,v in sorted(parts,key=lambda q:q[0])]+[_summary(a[:,1])])

def balance_features(a,domain):
    if domain=='ADCS':return _adcs(a)
    if domain=='EPS':return _eps(a)
    if domain=='Thermal':return _thermal(a)
    if domain=='Data':return _data(a)
    if domain=='EPS_localization':
        parts=[]
        for cmd,panel,meter in ((2,4,6),(3,5,7)):
            parts.append(_eps(a[:,[0,1,cmd,panel,8]]))
            parts.append(_summary(_response(a[:,cmd:cmd+1],a[:,meter],max(8,len(a)//5))))
        return np.concatenate(parts)
    if domain=='Thermal_localization':return np.concatenate([_thermal(a[:,[0,1,2,4]]),_thermal(a[:,[0,1,3,5]])])
    if domain=='Data_localization':return np.concatenate([_data(a[:,[0,1,2,3,4]]),_data(a[:,[0,1,5,6,7]])])
    raise ValueError('UNKNOWN_PLATFORM_DOMAIN')

def registry(domain,*,raw_only=False):
    spec(domain);r=FeatureRegistry()
    r.register('temporal',temporal_bins)
    r.register('kernels',lambda a:random_temporal_kernels(a))
    r.register('difference_kernels',lambda a:random_temporal_kernels(a,differences=True))
    if not raw_only:
        r.register('self_reference',self_reference,requires_healthy_prefix=True)
        r.register('balance',lambda a:balance_features(a,domain),requires_healthy_prefix=True)
    return r
