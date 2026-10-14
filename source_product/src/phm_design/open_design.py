"""Experimental, domain-neutral feedback-driven diagnostic program design.

No fault-specific physical threshold is part of the search controller. A task
supplies measurement and label contracts, available feature atoms and explicit
risk preferences. The controller mutates executable programs using completed
training-only feedback and keeps a non-dominated archive. Clinical, flight and
other deployment qualifications are not implied by this research module.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, replace
from typing import Callable, Mapping, Sequence
import hashlib
import json
import numpy as np


def canonical(value) -> bytes:
    return (json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':'))+'\n').encode()


@dataclass(frozen=True)
class ObservationSpec:
    """An exact deployment schema. Labels/roles never enter ``values``."""
    name: str
    fields: tuple[str, ...]
    time_field: str = 'time_s'
    min_rows: int = 16

    def __post_init__(self):
        if not self.name or len(set(self.fields)) != len(self.fields) or len(self.fields) < 2:
            raise ValueError('INVALID_OBSERVATION_SPEC')
        if self.fields[0] != self.time_field or self.min_rows < 4:
            raise ValueError('TIME_MUST_BE_FIRST_FIELD')
        forbidden=('truth','label','fault_id','world_id','onset','injection','applied_torque')
        if any(any(q in f.lower() for q in forbidden) for f in self.fields):
            raise ValueError('NON_DEPLOYMENT_FIELD')

    def array(self, values) -> np.ndarray:
        a=np.asarray(values, dtype=float)
        if a.ndim != 2 or a.shape[1] != len(self.fields) or len(a)<self.min_rows:
            raise ValueError('OBSERVATION_SHAPE')
        if not np.isfinite(a).all(): raise ValueError('OBSERVATION_NONFINITE')
        if np.any(np.diff(a[:,0])<=0): raise ValueError('TIME_NOT_STRICTLY_INCREASING')
        return a

    def rows(self, rows: Sequence[Mapping[str, float]]) -> np.ndarray:
        if not rows or any(tuple(r.keys())!=self.fields for r in rows):
            raise ValueError('EXACT_DEPLOYMENT_FIELDS_REQUIRED')
        return self.array([[r[f] for f in self.fields] for r in rows])


@dataclass(frozen=True)
class RiskPreference:
    """Declared task loss, not a universal RC threshold.

    Mean error treats classes equally. Tail error considers the least-recalled
    class. Preferences can be changed only by starting a new study, never by
    revising the objective after the evaluation cohort has been opened.
    """
    mean_class_error: float = 0.6
    worst_class_error: float = 0.4

    def __post_init__(self):
        w=np.asarray([self.mean_class_error,self.worst_class_error],float)
        if not np.isfinite(w).all() or (w<0).any() or not np.isclose(w.sum(),1.):
            raise ValueError('RISK_WEIGHTS_MUST_SUM_TO_ONE')

    def loss(self, errors: Sequence[float]) -> float:
        e=np.asarray(errors,float)
        if e.ndim!=1 or not len(e) or not np.isfinite(e).all() or (e<0).any() or (e>1).any():
            raise ValueError('INVALID_CLASS_ERRORS')
        return float(self.mean_class_error*e.mean()+self.worst_class_error*e.max())


class FeatureUnavailable(ValueError):
    """A declared component support precondition is absent, not invalid input.

    Only this typed exception permits a branch to be omitted. Programming errors,
    nonfinite outputs, unregistered atoms and invalid telemetry remain hard errors.
    """


@dataclass(frozen=True)
class Program:
    atoms: tuple[str,...]
    learner: str
    strength: float = 1.
    healthy_weight: float = 1.
    depth: int = 5
    seed: int = 48019
    augmentation: bool = False
    composition: str = "concat"

    def __post_init__(self):
        if not 1<=len(self.atoms)<=3 or len(set(self.atoms))!=len(self.atoms):
            raise ValueError('UNIQUE_ONE_TO_THREE_ATOMS_REQUIRED')
        if not np.isfinite(self.strength) or self.strength<=0 or self.healthy_weight<=0 or not np.isfinite(self.healthy_weight):
            raise ValueError('INVALID_MODEL_HYPERPARAMETER')
        if self.depth<1 or self.seed<0: raise ValueError('INVALID_DEPTH_OR_SEED')
        if self.composition not in ('concat','available_stack','available_average'):
            raise ValueError('INVALID_COMPOSITION')
        if self.composition!='concat' and self.learner in ('group_stack','group_average'):
            raise ValueError('NESTED_GROUP_COMPOSITION_NOT_SUPPORTED')

    @property
    def key(self) -> str:
        d=asdict(self)
        if self.learner in ('forest','extra','knn'): d['strength']=1.
        if self.learner not in ('forest','extra','group_stack','group_average'): d['depth']=5
        if self.learner in ('knn','gnb','svm'): d['seed']=0
        if self.learner=='knn': d['healthy_weight']=1.
        return hashlib.sha256(canonical(d)).hexdigest()


class FeatureRegistry:
    """Explicit runtime registry. No arbitrary import/eval or hidden truth access."""
    def __init__(self): self._atoms={}

    def register(self,name: str,function: Callable[[np.ndarray], np.ndarray],*,requires_healthy_prefix=False):
        if not name or name in self._atoms or not callable(function):
            raise ValueError('DUPLICATE_OR_INVALID_FEATURE_ATOM')
        self._atoms[name]=(function,bool(requires_healthy_prefix))

    @property
    def names(self): return tuple(self._atoms)

    def transform(self,program: Program,values,spec: ObservationSpec,*,healthy_prefix_confirmed=False):
        a=spec.array(values);out=[]
        for name in program.atoms:
            if name not in self._atoms: raise ValueError('UNKNOWN_FEATURE_ATOM:'+name)
            fn,needs_prefix=self._atoms[name]
            if needs_prefix and not healthy_prefix_confirmed:
                raise FeatureUnavailable('HEALTHY_PREFIX_NOT_CONFIRMED')
            # Atom implementations receive measurement arrays only, never a Record.
            v=np.asarray(fn(a.copy()),float)
            if v.ndim!=1 or not len(v) or not np.isfinite(v).all():
                raise ValueError('INVALID_FEATURE_OUTPUT:'+name)
            out.append(v)
        return np.concatenate(out)

    def transform_parts(self,program: Program,values,spec: ObservationSpec,*,healthy_prefix_confirmed=False):
        """Return supported component vectors and explicit unavailable reasons."""
        a=spec.array(values);parts=[];reasons={}
        for atom in program.atoms:
            # Registry lookup, invalid input/output and arbitrary ValueErrors must
            # never silently masquerade as a supported availability fallback.
            if atom not in self._atoms:raise ValueError('UNKNOWN_FEATURE_ATOM:'+atom)
            try:
                parts.append(self.transform(Program((atom,),'linear'),a,spec,
                    healthy_prefix_confirmed=healthy_prefix_confirmed))
            except FeatureUnavailable as exc:
                parts.append(None);reasons[atom]=str(exc)
        return parts,reasons

    def graph(self,program: Program,spec: ObservationSpec):
        if any(x not in self._atoms for x in program.atoms): raise ValueError('UNKNOWN_FEATURE_ATOM')
        return {'schema':'phm.open-design.program.v1','observation_spec':asdict(spec),
            'program':asdict(program),'nodes':[
                {'id':'input','op':'exact_observation_contract'},
                *[{'id':f'atom{i}','op':x,'inputs':['input']} for i,x in enumerate(program.atoms)],
                {'id':'concat','op':('concatenate' if program.composition=='concat' else 'per_component_classifiers_and_availability'),'inputs':[f'atom{i}' for i in range(len(program.atoms))]},
                {'id':'learner','op':program.learner,'inputs':['concat']},
                {'id':'decision','op':'learned_classifier','inputs':['learner']}],
            'requires_initial_healthy_prefix':(any(self._atoms[x][1] for x in program.atoms) if program.composition=='concat' else all(self._atoms[x][1] for x in program.atoms)),
            'component_prefix_requirements':{x:self._atoms[x][1] for x in program.atoms},
            'fixed_fault_threshold':None,'causal_streaming_qualified':False,
            'composition':program.composition,
            'component_availability': ({'typed_support_errors_only':True,
                'all_unavailable_returns_unknown':True,'meta_training':'physical_group_oof',
                'mask_training':'all_nonempty_subsets_of_oof_supported_branches',
                'dropout_creates_new_physical_worlds':False,
                'meta':'learned_logistic' if program.composition=='available_stack' else 'available_mean'}
                if program.composition!='concat' else None),
            'ensemble_training': ({'base_learners':['linear','extra','gnb'],
              'meta':'group_oof_logistic' if program.learner=='group_stack' else 'fixed_average',
              'group_ids_are_predictors':False, 'inner_group_folds':3,
              'scalers_fit_inside_inner_folds':True}
              if program.learner in ('group_stack','group_average') else None)}


def temporal_bins(a: np.ndarray, bins: int=8) -> np.ndarray:
    """Generic temporal mean/std/slope bank; no prefix or domain assumptions."""
    x=a[:,1:];parts=np.array_split(x,bins);features=[]
    for p in parts:
        t=np.linspace(-1,1,len(p));features.extend((p.mean(0),p.std(0),(t@p)/(t@t+np.finfo(float).eps)))
    features.extend((np.quantile(x,.1,axis=0),np.quantile(x,.9,axis=0)))
    return np.arcsinh(np.concatenate(features))


def random_temporal_kernels(a: np.ndarray, *, differences: bool=False, seed: int=48019, kernels: int=24) -> np.ndarray:
    """Small original random multichannel convolution bank, inspired by ROCKET.

    This is NOT an implementation or performance reproduction of ROCKET. It uses
    signed pooling and a small 24-kernel bank, and is merely another candidate.
    Per-record centering does not assume an initially healthy reference.
    """
    t=a[:,0];x=a[:,1:]
    target=np.linspace(t[0],t[-1],96)
    y=np.vstack([np.interp(target,t,x[:,j]) for j in range(x.shape[1])]).T
    if differences: y=np.diff(y,axis=0)
    mu=np.mean(y,axis=0);scale=np.std(y,axis=0);z=(y-mu)/np.maximum(scale,1e-8)
    rng=np.random.default_rng(seed);features=[]
    for k in range(kernels):
        size=int(rng.choice((3,5,7)));d=int(rng.choice((1,2,4,8)))
        length=len(z)-d*(size-1)
        w=rng.normal(size=(size,z.shape[1]));w-=w.mean(0);w/=np.sqrt(np.sum(w*w)+1e-12)
        v=sum(z[j*d:j*d+length]@w[j] for j in range(size))
        for section in np.array_split(v,3):
            features.extend((float(np.max(section)),float(np.mean(np.tanh(section))),float(np.mean(np.abs(section)))))
    return np.r_[features,np.arcsinh(mu),np.arcsinh(scale)]


def generic_registry() -> FeatureRegistry:
    r=FeatureRegistry();r.register('temporal_bins',temporal_bins)
    r.register('random_kernels',random_temporal_kernels)
    from functools import partial
    r.register('difference_kernels',partial(random_temporal_kernels,differences=True))
    return r


def pareto_indices(errors: Sequence[Sequence[float]]) -> list[int]:
    x=np.asarray(errors,float)
    if x.ndim!=2 or not np.isfinite(x).all():raise ValueError('INVALID_PARETO_METRICS')
    # Exact dominance: no arbitrary diagnostic tolerance/threshold.
    return [i for i,v in enumerate(x) if not any(np.all(u<=v) and np.any(u<v) for j,u in enumerate(x) if j!=i)]


class FeedbackPlanner:
    """Mutable program grammar driven by prior training-only outcomes.

    'adaptive' mutates archive parents selected by task loss; 'expert' uses a
    pre-generated static schedule with the same atoms and learners. Initial seed
    programs are shared, not withheld from the expert. No guarantee of optimum.
    """
    learners=('linear','svm','extra','forest','gnb','knn','mlp','probability_mix')
    def __init__(self,atoms,*,mode='adaptive',seeds=(),risk=RiskPreference(),seed=48019):
        if mode not in ('adaptive','expert') or not atoms or len(set(atoms))!=len(atoms):raise ValueError('PLANNER_SPEC')
        self.atoms=tuple(atoms);self.mode=mode;self.risk=risk;self.rng=np.random.default_rng(seed)
        self.learners=type(self).learners+('group_stack','group_average')
        self.seeds=list(seeds);self.seen=set();self.history=[];self.proposals=[]

    def _random(self):
        n=int(self.rng.choice((1,1,1,2,2,3)));n=min(n,len(self.atoms))
        atoms=tuple(sorted(self.rng.choice(self.atoms,n,replace=False).tolist()))
        learner=str(self.rng.choice(self.learners))
        composition='concat' if learner in ('group_stack','group_average') else str(self.rng.choice(('concat','concat','available_stack','available_average')))
        return Program(atoms,learner,float(10**self.rng.uniform(-2,2)),
            float(self.rng.choice((1.,2.,4.))),int(self.rng.choice((3,5,8))),int(self.rng.integers(1,2**30)),bool(self.rng.integers(2)),composition)

    def propose(self):
        for _ in range(1000):
            parent=None
            if self.seeds: p=self.seeds.pop(0);reason='shared_seed'
            elif self.mode=='adaptive' and self.history and self.rng.random()<.7:
                # Fitness affects parent probability, while exploration remains possible.
                good=sorted(self.history,key=lambda h:(h['loss'],h['program'].key))[:max(1,len(self.history)//3)]
                h=good[int(self.rng.integers(len(good)))];p=h['program'];parent=p.key
                op=str(self.rng.choice(('atoms','learner','strength','weight','depth','augmentation','seed','composition')))
                if op=='atoms':
                    q=list(p.atoms);new=str(self.rng.choice(self.atoms));q[int(self.rng.integers(len(q)))]=new
                    if len(q)<3 and self.rng.random()<.4:q.append(str(self.rng.choice(self.atoms)))
                    p=replace(p,atoms=tuple(sorted(set(q))))
                elif op=='learner':
                    learner=str(self.rng.choice(self.learners))
                    p=replace(p,learner=learner,composition='concat' if learner in ('group_stack','group_average') else p.composition)
                elif op=='composition':
                    composition='concat' if p.learner in ('group_stack','group_average') else str(self.rng.choice(('concat','available_stack','available_average')))
                    p=replace(p,composition=composition)
                elif op=='strength':p=replace(p,strength=float(np.clip(p.strength*10**self.rng.uniform(-1,1),1e-4,1e4)))
                elif op=='weight':p=replace(p,healthy_weight=float(self.rng.choice((1.,2.,4.))))
                elif op=='depth':p=replace(p,depth=int(self.rng.choice((3,5,8))))
                elif op=='augmentation':p=replace(p,augmentation=not p.augmentation)
                else:p=replace(p,seed=int(self.rng.integers(1,2**30)))
                reason='feedback_mutation:'+op
            else:p=self._random();reason='static_expert' if self.mode=='expert' else 'exploration'
            if p.learner=='knn':p=replace(p,healthy_weight=1.)
            if p.key in self.seen:continue
            self.seen.add(p.key);self.proposals.append({'program':asdict(p),'key':p.key,'parent':parent,'reason':reason,'feedback_count':len(self.history)})
            return p
        raise RuntimeError('PROGRAM_SPACE_EXHAUSTED')

    def observe(self,program,errors):
        if program.key not in self.seen:raise ValueError('UNPROPOSED_PROGRAM_FEEDBACK')
        if any(x['program'].key==program.key for x in self.history):raise ValueError('DUPLICATE_FEEDBACK')
        loss=self.risk.loss(errors);self.history.append({'program':program,'errors':list(map(float,errors)),'loss':loss})

    def archive(self):
        if not self.history:return []
        return [{'program':asdict(self.history[i]['program']),'errors':self.history[i]['errors'],'loss':self.history[i]['loss']}
                for i in pareto_indices([h['errors'] for h in self.history])]
