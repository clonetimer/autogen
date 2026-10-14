"""R57 preregistered, observation-only learning candidates.

Do not use task kind, simulator faults, injection envelopes, world/family IDs,
truth, applied wheel torque, or the future evaluation label in inference.
The first fifth of a record must be independently confirmed healthy.
All features are continuous; none is an alarm threshold or rule classifier.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import lsq_linear
from .platform_features import FIELDS
from .platform_features_r56 import registry as previous_registry, _stats, _ridge, _array


def _thermal_single(a):
    a=_array(a)
    if a.shape[1]!=4:raise ValueError('THERMAL_DEPLOYMENT_SCHEMA')
    t,light,heat,T=a.T
    dt=np.diff(t)
    n=max(12,len(t)//5)
    # Physics-inspired innovation models fitted exclusively to the confirmed
    # healthy prefix. Allow intercept to absorb unknown orbital flux/offset.
    temp_mid=(T[1:]+T[:-1])*.5
    inputs=np.column_stack((light[1:],heat[1:],-(temp_mid/300.)**4,np.ones(len(dt))))
    dy=np.diff(T)/dt
    q=inputs[:n-1]
    center=np.median(q,axis=0)
    spread=np.maximum(np.std(q,axis=0),np.array([.12,.12,.025,1.]))
    z=(q-center)/spread
    # Normalised ridge to avoid falsely interpreting rank-deficient excitation
    # as an exact physical coefficient. Coefficients are learned from telemetry.
    qfit=np.column_stack((z,np.ones(len(z))))
    coef=np.linalg.solve(qfit.T@qfit+.25*np.eye(5),qfit.T@dy[:n-1])
    forecast=np.column_stack(((inputs-center)/spread,np.ones(len(inputs))))@coef
    innovations=(dy-forecast)*dt
    noise=max(float(np.std(innovations[:n-1])),.02)
    res=innovations/noise
    pre=np.std(T[:n]);scale=max(float(pre),.25)
    parts=[_stats(np.column_stack((res,np.cumsum(innovations)/scale))) ]
    for width in (4,12,24):
        c=np.convolve(res,np.ones(width)/np.sqrt(width),mode='valid')
        parts.append(_stats(c))
    # Fit the scale-normalized *difference* between early and rolling local
    # response: additive heat and changed state-dependent cooling compete.
    deltas=[]
    for ind in np.array_split(np.arange(len(dt)),6):
        x=inputs[ind]
        y=dy[ind]
        xfit=np.column_stack(((x-center)/spread,np.ones(len(x))))
        local=np.linalg.solve(xfit.T@xfit+1.0*np.eye(5),xfit.T@y)
        deltas.extend(np.arcsinh((local-coef)/max(float(np.std(dy[:n-1])),.0002)).tolist())
    # Competing mechanisms: unexplained heating is constant-like; missing
    # emissivity follows absolute temperature to fourth power. Neither is
    # forcibly classified without learned evidence.
    post=np.arange(n-1,len(dy))
    qpost=np.column_stack((np.ones(len(post)),(T[post+1]/300.)**4,heat[post+1],light[post+1]))
    rx=dy[post]-forecast[post]
    diag=_ridge(qpost,rx,alpha=0.2)
    return np.concatenate(parts+[np.asarray(deltas),np.arcsinh(diag),
        np.arcsinh(np.array([noise,scale,np.linalg.matrix_rank(qfit)/5,
            float(np.std(dy[:n-1])),float(np.mean(res[n-1:])),
            float(np.median(res[n-1:]))]))])


def _thermal_localization(a):
    if a.shape[1]!=6:raise ValueError('THERMAL_LOCALIZATION_SCHEMA')
    l=_thermal_single(a[:,[0,1,2,4]])
    r=_thermal_single(a[:,[0,1,3,5]])
    return np.concatenate((l,r,l-r,np.abs(l-r)))


def _eps_location(a):
    if a.shape[1]!=9:raise ValueError('EPS_LOCALIZATION_SCHEMA')
    # Per-component continuous power flow residual, trained/selected by models.
    # No component's command/power is silently replaced by shared bus readings.
    t,light,c1,c2,p1,p2,l1,l2,bus=a.T
    n=max(12,len(t)//5)
    per=[]
    for command,power,local in ((c1,p1,l1),(c2,p2,l2)):
        X=np.column_stack((light,command,np.ones(len(t))))
        base=_ridge(X[:n],power[:n]-local[:n],alpha=.02)
        res=(power-local-X@base)/max(float(np.std(power[:n]-local[:n])),.2)
        per.append(np.concatenate((_stats(res),np.arcsinh(base))))
    a1,a2=per
    return np.concatenate((a1,a2,a1-a2,np.abs(a1-a2)))


def _adcs_location(a):
    if a.shape[1]!=8:raise ValueError('ADCS_SCHEMA')
    t=a[:,0];u=a[:,5:8];w=a[:,2:5];dt=np.diff(t)
    n=max(12,len(t)//5)
    wheel=[]
    for j in range(3):
        v=(w[1:,j]-w[:-1,j])/dt
        x=np.column_stack((u[1:,j],w[:-1,j],np.sign(w[:-1,j]),np.ones(len(v))))
        beta=_ridge(x[:n-1],v[:n-1],alpha=.1)
        r=(v-x@beta)*dt
        unit=max(float(np.std(r[:n-1])),.001)
        stats=_stats(np.column_stack((r/unit,np.cumsum(r)/max(np.std(w[:n,j]),.3))))
        wheel.append((float(np.mean(np.abs(r[n-1:]))/unit),stats))
    return np.concatenate([s for _,s in sorted(wheel,key=lambda q:q[0])])


def mechanism_innovation(a,domain):
    a=_array(a)
    if a.shape[1]!=len(FIELDS[domain]):raise ValueError('DEPLOYMENT_COLUMNS_MISMATCH')
    if domain=='Thermal':return _thermal_single(a)
    if domain=='Thermal_localization':return _thermal_localization(a)
    if domain=='EPS_localization':return _eps_location(a)
    if domain=='ADCS':return _adcs_location(a)
    # No invented task-specific mechanism for other domains.
    raise ValueError('R57_MECHANISM_NOT_REGISTERED')


def registry(domain,*,raw_only=False):
    reg=previous_registry(domain,raw_only=raw_only)
    if not raw_only and domain in ('Thermal','Thermal_localization','EPS_localization','ADCS'):
        reg.register('mechanism_innovation',lambda a:mechanism_innovation(a,domain),requires_healthy_prefix=True)
    return reg
