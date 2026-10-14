"""Observation-only candidate representations, not fixed fault classifiers.

Piecewise command alignment follows the *supplied* adapters' logged interval
contract. Healthy-prefix coefficients are learned per trajectory. Projection
and constrained coefficients are physical model assumptions, not diagnostic
thresholds. Full-trajectory/offline only. No task, world ID, injection or truth.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import nnls
from .platform_features import registry as old_registry, FIELDS


def _array(a):
    a=np.asarray(a,dtype=float)
    if a.ndim!=2 or len(a)<24 or not np.isfinite(a).all() or np.any(np.diff(a[:,0])<=0):
        raise ValueError('INVALID_DEPLOYMENT_ARRAY')
    return a


def _stats(x):
    x=np.asarray(x,float)
    if x.ndim==1:x=x[:,None]
    parts=[]
    for b in np.array_split(x,6):
        t=np.linspace(-1.,1.,len(b));parts.extend([np.mean(b,axis=0),np.std(b,axis=0),
            np.quantile(b,.1,axis=0),np.quantile(b,.5,axis=0),np.quantile(b,.9,axis=0),
            (t@b)/max(t@t,1.)])
    out=np.arcsinh(np.concatenate(parts))
    if not np.isfinite(out).all():raise ValueError('NONFINITE_RESPONSE_FEATURE')
    return out


def _ridge(X,y,prior=None,alpha=1e-5):
    X=np.asarray(X,float);y=np.asarray(y,float)
    prior=np.zeros(X.shape[1]) if prior is None else np.asarray(prior,float)
    if len(X)==0:return prior.copy()
    scale=np.maximum(np.sqrt(np.mean(X*X,axis=0)),1e-10)
    z=X/scale
    lhs=z.T@z+alpha*np.eye(z.shape[1]);rhs=z.T@y+alpha*prior*scale
    return np.linalg.solve(lhs,rhs)/scale


def _thermal_fit(a,start,end):
    # Temperature increment over integrated held commands. Recorded command k
    # applies on (t[k-1],t[k]], unlike a forward-shifted legacy difference.
    t,light,heat,temp=a.T
    dt=np.diff(t[start:end]);light=light[start+1:end];heat=heat[start+1:end]
    rad=.5*((temp[start:end-1]/300.)**4+(temp[start+1:end]/300.)**4)
    flux=np.column_stack((light,heat,-rad))
    X=np.cumsum(flux*dt[:,None],axis=0);y=temp[start+1:end]-temp[start]
    # Fit nonnegative absorption, inverse heat capacity, radiative coefficient.
    scale=np.maximum(np.linalg.norm(X,axis=0),1e-8)
    beta=nnls(X/scale,y,maxiter=200)[0]/scale
    return beta,flux,np.linalg.matrix_rank(X)


def thermal_integral(a):
    a=_array(a);n=max(12,len(a)//5)
    beta,_,rank=_thermal_fit(a,0,n);t,light,heat,temp=a.T
    dt=np.diff(t);flux=np.column_stack((light[1:],heat[1:],-.5*((temp[:-1]/300.)**4+(temp[1:]/300.)**4)))
    residual=np.diff(temp)-flux@beta*dt
    reference_noise=max(float(np.std(residual[:n-1])),1e-5)
    pieces=[_stats(np.cumsum(residual)/max(np.std(temp[:n]),.01))]
    for width in (3,8,16):
        integ=np.convolve(residual,np.ones(width),mode='valid')
        denom=reference_noise*np.sqrt(width)
        pieces.append(_stats(integ/denom))
    # Time-local parameter changes and fit support are learned features, never labels.
    coeff=[]
    for b in np.array_split(np.arange(len(a)),6):
        lo=max(0,int(b[0])-2);hi=min(len(a),int(b[-1])+3)
        q,_,r=_thermal_fit(a,lo,hi)
        unit=np.maximum(np.abs(beta),np.array([.005,.0002,.005]))
        coeff.extend(np.arcsinh((q-beta)/unit));coeff.append(r/3.)
    # Competing additive-vs-state-dependent signatures of unexplained heating.
    mass_scale=max(beta[1],.0002)
    rate=residual/dt/mass_scale
    z=np.column_stack((np.ones(len(rate)),(temp[1:]/300.)**4,light[1:],heat[1:]))
    signed=_ridge(z[n:],rate[n:],alpha=.01)
    return np.concatenate(pieces+[np.array(coeff),np.arcsinh(signed),np.array([rank/3.])])


def _data_fit(a,start,end,prior):
    t,contact,gen,service,q=a.T
    dt=np.diff(t);X=np.column_stack((gen[1:],-service[1:]*contact[1:]))
    y=np.diff(q)/dt
    indices=np.arange(len(y));use=(indices>=start)&(indices<end)&(q[:-1]>0)&(q[1:]>0)
    # Exact zero-width plateau carries censoring information rather than a
    # fictitious zero flow coefficient. No assumed storage capacity is supplied.
    use &= (np.diff(q)!=0)
    beta=_ridge(X[use],y[use],prior=prior,alpha=.002)
    return beta,int(np.linalg.matrix_rank(X[use])) if use.any() else 0


def data_projected(a):
    a=_array(a);n=max(10,len(a)//5)
    t,contact,gen,service,q=a.T;dt=np.diff(t)
    beta,rank=_data_fit(a,0,n-1,np.ones(2))
    X=np.column_stack((gen[1:],-service[1:]*contact[1:]))
    predicted=np.maximum(0.,q[:-1]+dt*(X@beta))
    residual=(q[1:]-predicted)/(dt*np.maximum(np.abs(gen[1:]),1.))
    out=[_stats(np.column_stack((residual,residual*contact[1:],residual*(1-contact[1:]))))]
    coeff=[]
    for b in np.array_split(np.arange(len(dt)),6):
        bta,r=_data_fit(a,int(b[0]),int(b[-1])+1,beta)
        coeff.extend((bta-beta)/np.maximum(np.abs(beta),.05));coeff.append(r/2.)
    out.extend([np.arcsinh(np.array(coeff)),np.array([rank/2.,np.mean(q[:n]==0),np.mean(q[n:]==0)])])
    return np.concatenate(out)


def eps_separate(a):
    a=_array(a);n=max(10,len(a)//5);t,light,command,panel,net=a.T
    # Power streams are algebraic at the same sample; no differencing shift.
    X=np.column_stack((light,np.ones(len(a))))
    solar=_ridge(X[:n],panel[:n],alpha=1e-6)
    Y=np.column_stack((command,np.ones(len(a))))
    load=panel-net;ld=_ridge(Y[:n],load[:n],alpha=1e-6)
    p=(panel-X@solar)/max(abs(solar[0]),np.mean(abs(panel[:n])),1.)
    l=(load-Y@ld)/max(np.mean(abs(command[:n])),1.)
    r=np.column_stack((p,l,p*light,p*(1-light)))
    return _stats(r)


def adcs_integral(a):
    a=_array(a);t=a[:,0];w=a[:,2:5];u=a[:,5:8];dt=np.diff(t);n=max(12,len(a)//5)
    outputs=[]
    for j in range(3):
        fitted=[]
        for align in ('left','right','mid'):
            cmd=u[:-1,j] if align=='left' else u[1:,j] if align=='right' else (u[:-1,j]+u[1:,j])/2
            X=np.column_stack((cmd,.5*(w[:-1,j]+w[1:,j]),np.sign(.5*(w[:-1,j]+w[1:,j])),np.ones(len(dt))))*dt[:,None]
            # Smooth temporal integration avoids a speed-noise derivative amplifier.
            width=3;xi=np.stack([np.convolve(X[:,k],np.ones(width),mode='valid') for k in range(4)],1)
            yi=w[width:,j]-w[:-width,j]
            b=_ridge(xi[:n-width],yi[:n-width],alpha=.002)
            err=yi-xi@b;loss=np.mean(err[:n-width]**2)
            fitted.append((loss,X,b,err))
        _,X,b,err=min(fitted,key=lambda v:v[0]);unit=max(np.std(err[:n-3]),.0005)
        residual=np.diff(w[:,j])-X@b;parts=[]
        for width in (3,8,16):
            r=np.convolve(residual,np.ones(width),mode='valid')/(unit*np.sqrt(width/3))
            parts.append(_stats(r))
        # Learning can distinguish support deficiencies from actual response changes.
        diag=[np.linalg.matrix_rank(X[:n])/4.,np.std(u[:n,j]),np.std(w[:n,j])]
        feature=np.concatenate(parts+[np.arcsinh(diag)])
        outputs.append((np.mean(abs(err[n:]))/unit,feature))
    return np.concatenate([p for _,p in sorted(outputs,key=lambda v:v[0])])


def transport_features(a,domain):
    a=_array(a)
    if a.shape[1]!=len(FIELDS[domain]):raise ValueError('DEPLOYMENT_COLUMNS_MISMATCH')
    if domain=='ADCS':return adcs_integral(a)
    if domain=='EPS':return eps_separate(a)
    if domain=='Thermal':return thermal_integral(a)
    if domain=='Data':return data_projected(a)
    if domain=='Thermal_localization':return np.concatenate([thermal_integral(a[:,[0,1,2,4]]),thermal_integral(a[:,[0,1,3,5]])])
    if domain=='Data_localization':return np.concatenate([data_projected(a[:,[0,1,2,3,4]]),data_projected(a[:,[0,1,5,6,7]])])
    if domain=='EPS_localization':
        # Both component loads are explicitly observed; reconstruct each local net
        # rather than incorrectly using the shared bus for both component models.
        left=np.column_stack((a[:,0],a[:,1],a[:,2],a[:,4],a[:,4]-a[:,6]))
        right=np.column_stack((a[:,0],a[:,1],a[:,3],a[:,5],a[:,5]-a[:,7]))
        return np.concatenate([eps_separate(left),eps_separate(right)])
    raise ValueError('UNKNOWN_PLATFORM_DOMAIN')


def registry(domain,*,raw_only=False):
    r=old_registry(domain,raw_only=raw_only)
    if not raw_only:r.register('transport',lambda a:transport_features(a,domain),requires_healthy_prefix=True)
    return r
