"""Experimental task-aware diagnosis architecture selection (0.8 research).

Training labels and world groups are accepted *only* in ``design``. Prediction
accepts deployment-only observations with exact schemas. This is an offline
whole-trajectory research API, NOT qualified online/on-orbit diagnosis.

Competing architectures: direct joint class model; factorized fault-detection,
component and mechanism heads; training-side OOF-weighted probability fusion.
Every output is projected onto an explicit legal ontology. Only group-held-out
training folds select the method and the optional abstention policy.
"""
from __future__ import annotations
from dataclasses import dataclass
from itertools import product
from typing import Mapping, Sequence
import hashlib
import json
import time

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .multicomponent_diagnosis import ONTOLOGY
from .platform_features import FIELDS, balance_features
from .open_design import temporal_bins, ObservationSpec
from .task_candidate_registry import TaskCandidateRegistry, deployment_sequence


@dataclass(frozen=True)
class TaskOntology:
    domain: str
    labels: tuple[str, ...]
    components: tuple[str | None, ...]
    mechanisms: tuple[str, ...]

    @classmethod
    def from_domain(cls, domain: str) -> 'TaskOntology':
        name = domain.removesuffix('_localization').lower()
        if name not in ONTOLOGY or not domain.endswith('_localization'):
            raise ValueError('UNSUPPORTED_TASK_ONTOLOGY')
        entries=ONTOLOGY[name]
        labels=tuple(entries)
        components=tuple(entries[k][0] for k in labels)
        mechanisms=tuple(entries[k][2] for k in labels)
        if labels[0]!='NOMINAL' or components[0] is not None or len(labels)!=5:
            raise ValueError('ONTOLOGY_CONTRACT_CHANGED')
        return cls(domain, labels, components, mechanisms)

    def __post_init__(self):
        if (len(self.labels)!=5 or len(set(self.labels))!=5 or
            self.labels[0]!='NOMINAL' or self.components[0] is not None or
            self.mechanisms[0]!='NOMINAL' or
            len(set(self.components[1:]))!=2 or len(set(self.mechanisms[1:]))!=2 or
            len({(c,m) for c,m in zip(self.components[1:],self.mechanisms[1:])})!=4):
            raise ValueError('INVALID_ONTOLOGY')

    def heads(self, y: np.ndarray) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
        components=tuple(sorted(set(self.components[1:])))
        mechanisms=tuple(sorted(set(self.mechanisms[1:])))
        return (np.asarray(y!=0,dtype=int),
                np.asarray([components.index(self.components[int(i)]) if int(i) else -1 for i in y]),
                np.asarray([mechanisms.index(self.mechanisms[int(i)]) if int(i) else -1 for i in y]))

    def project(self, p_fault: np.ndarray, p_component: np.ndarray, p_mechanism: np.ndarray) -> np.ndarray:
        components=tuple(sorted(set(self.components[1:])))
        mechanisms=tuple(sorted(set(self.mechanisms[1:])))
        out=np.zeros((len(p_fault),len(self.labels)),dtype=float)
        out[:,0]=1-p_fault[:,1]
        for j in range(1,len(self.labels)):
            out[:,j]=p_fault[:,1]*p_component[:,components.index(self.components[j])]*p_mechanism[:,mechanisms.index(self.mechanisms[j])]
        return _probabilities(out)


@dataclass(frozen=True)
class TaskAcceptancePolicy:
    """Predeclared development-stage grouped-OOF design feasibility thresholds.

    Numerical defaults are conservative project screening gates, NOT a
    release qualification or statistical confidence bound. Changing these
    thresholds is an explicit new design protocol.
    """
    accuracy_min: float = 0.90
    healthy_false_alarm_max: float = 0.10
    worst_fault_recall_min: float = 0.80
    worst_world_accuracy_min: float = 0.80
    min_independent_worlds: int = 8

    def __post_init__(self):
        for field in ('accuracy_min','healthy_false_alarm_max','worst_fault_recall_min','worst_world_accuracy_min'):
            v=getattr(self,field)
            if isinstance(v,bool) or not isinstance(v,(float,int)) or not np.isfinite(v) or not 0 <= v <= 1:
                raise ValueError('INVALID_ACCEPTANCE_POLICY:'+field)
        if type(self.min_independent_worlds) is not int or self.min_independent_worlds < 4:
            raise ValueError('INVALID_ACCEPTANCE_POLICY:min_independent_worlds')

    def evaluate(self,score: Mapping) -> dict:
        requirements=(
            ('accuracy','min',self.accuracy_min),
            ('healthy_false_alarm_rate','max',self.healthy_false_alarm_max),
            ('worst_fault_recall','min',self.worst_fault_recall_min),
            ('worst_world_accuracy','min',self.worst_world_accuracy_min),
            ('independent_worlds','min',self.min_independent_worlds))
        failures=[]
        for key,op,limit in requirements:
            val=score.get(key)
            if not isinstance(val,(int,float)) or not np.isfinite(val) or (val < limit if op=='min' else val > limit):
                failures.append({'metric':key,'observed':val,'operator':'>=' if op=='min' else '<=',
                                 'required':limit})
        return {'eligible':not failures,'failed_gates':failures,'evidence':'TRAIN_GROUP_OOF_ONLY',
                'deployment_qualified':False}


class TaskDesignRejected(RuntimeError):
    """No candidate satisfies the predeclared feasibility requirements.

    ``audit`` contains the original diagnostics; there is no fitted deliverable
    and no way to relabel this as a qualified prediction.
    """
    def __init__(self,reason: str,audit: dict):
        super().__init__(reason)
        self.reason=reason
        self.audit=audit


def _probabilities(p):
    z=np.asarray(p,dtype=float)
    if z.ndim!=2 or not len(z) or not np.isfinite(z).all() or (z < -1.e-9).any():
        raise ValueError('INVALID_CLASS_PROBABILITIES')
    z=np.maximum(z,0.)
    sums=z.sum(axis=1,keepdims=True)
    if (sums<=0).any(): raise ValueError('EMPTY_CLASS_PROBABILITIES')
    return z/sums


def _model(kind: str, seed: int, feature: str = 'temporal'):
    """Single vetted estimator contract for sklearn and opt-in torch CPU."""
    return TaskCandidateRegistry().build(feature, kind, seed)


def _proba(model, x: np.ndarray, expected: Sequence[int]):
    raw=model.predict_proba(x)
    known=np.asarray(model.classes_,dtype=int)
    aligned=np.zeros((len(x),len(expected)),float)
    for j,label in enumerate(expected):
        indices=np.flatnonzero(known==label)
        if len(indices):aligned[:,j]=raw[:,indices[0]]
    return _probabilities(aligned)


class _Fitted:
    def __init__(self,ontology,architecture,feature,learner,seed,weight=0.5):
        self.ontology=ontology;self.architecture=architecture;self.feature=feature
        self.learner=learner;self.seed=seed;self.weight=weight
        self.joint=None;self.detector=None;self.component=None;self.mechanism=None

    def fit(self,x,y):
        fault,component,mechanism=self.ontology.heads(y)
        if set(y)!=set(range(len(self.ontology.labels))):raise ValueError('FULL_ONTOLOGY_CLASS_SUPPORT_REQUIRED')
        # Always fit the two heads for disagreement audits, even when the
        # final decision is a direct joint classifier. Included in resource cost.
        self.joint=_model(self.learner,self.seed,self.feature).fit(x,y)
        self.detector=_model(self.learner,self.seed+1,self.feature).fit(x,fault)
        self.component=_model(self.learner,self.seed+2,self.feature).fit(x[fault==1],component[fault==1])
        self.mechanism=_model(self.learner,self.seed+3,self.feature).fit(x[fault==1],mechanism[fault==1])
        return self

    def both(self,x):
        direct=_proba(self.joint,x,range(len(self.ontology.labels)))
        p_f=_proba(self.detector,x,(0,1))
        p_c=_proba(self.component,x,(0,1))
        p_m=_proba(self.mechanism,x,(0,1))
        fact=self.ontology.project(p_f,p_c,p_m)
        return direct,fact

    def predict_proba(self,x):
        direct,fact=self.both(x)
        if self.architecture=='direct':return direct
        if self.architecture=='factorized':return fact
        if self.architecture=='fusion':return _probabilities(self.weight*direct+(1-self.weight)*fact)
        raise ValueError('UNKNOWN_ARCHITECTURE')


def _feature(values: np.ndarray, domain: str, name: str):
    a=np.asarray(values,float)
    if name=='sequence':result=deployment_sequence(a)
    elif name=='temporal':result=temporal_bins(a)
    elif name=='physical':result=balance_features(a,domain)
    else:raise ValueError('UNKNOWN_FEATURE')
    z=np.asarray(result,float)
    if z.ndim not in (1,2) or not len(z) or not np.isfinite(z).all():raise ValueError('INVALID_FEATURE_VECTOR')
    return z


def _transform(rows: Sequence[np.ndarray],domain: str,feature: str) -> np.ndarray:
    vec=[_feature(x,domain,feature) for x in rows]
    if len(set(v.shape for v in vec))!=1:raise ValueError('INCONSISTENT_FEATURE_LENGTH')
    return np.stack(vec)


def _split(groups,folds=4):
    g=np.asarray(groups)
    unique=np.unique(g)
    if len(unique)<folds:raise ValueError('INSUFFICIENT_INDEPENDENT_WORLDS')
    return list(GroupKFold(n_splits=folds).split(np.zeros(len(g)),groups=g))


def _fusion_weight(x,y,groups,ontology,feature,learner,seed):
    # Inner grouped OOF fitting prevents reusing in-sample predictions to
    # optimize fusion weight in an outer fold.
    splits=_split(groups, min(3,len(set(groups))))
    pd=np.zeros((len(y),len(ontology.labels)));pf=np.zeros_like(pd)
    for tr,va in splits:
        m=_Fitted(ontology,'direct',feature,learner,seed).fit(x[tr],y[tr])
        pd[va],pf[va]=m.both(x[va])
    costs=[]
    for w in (0.,.25,.5,.75,1.):
        pred=(w*pd+(1-w)*pf).argmax(axis=1)
        loss=float(np.mean(pred!=y)+np.mean((y==0)&(pred!=0)))
        costs.append((loss,abs(w-.5),w))
    return min(costs)[-1]


def _score(y: np.ndarray, p: np.ndarray, groups, unknown_cost=0.5):
    pred=p.argmax(axis=1)
    correct=np.asarray(pred==y)
    health=(y==0)
    per_class=[float(np.mean(correct[y==i])) for i in range(p.shape[1])]
    by_world=[float(np.mean(correct[np.asarray(groups)==g])) for g in sorted(set(groups))]
    return {'accuracy':float(correct.mean()),'healthy_false_alarm_rate':float(np.mean(pred[health]!=0)),
        'worst_fault_recall':float(min(per_class[1:])), 'worst_world_accuracy':float(min(by_world)),
        'average_decision_loss':float(np.mean(pred!=y)+np.mean(health&(pred!=0))),
        'independent_worlds':len(set(groups))}


@dataclass
class TaskDesignArtifact:
    ontology: TaskOntology
    spec_fields: tuple[str,...]
    candidate: dict
    selected: _Fitted
    audit: dict
    threshold: float
    abstention_cost: float
    max_unknown_fraction: float

    def _read(self,observations,fields,healthy_prefix_confirmed: bool):
        if tuple(fields)!=self.spec_fields:raise ValueError('EXACT_DEPLOYMENT_FIELDS_REQUIRED')
        if type(healthy_prefix_confirmed) is not bool or not healthy_prefix_confirmed:
            raise ValueError('HEALTHY_PREFIX_NOT_CONFIRMED')
        return ObservationSpec('task-aware:'+self.ontology.domain,self.spec_fields).array(observations)

    def diagnose(self,observations,*,fields,healthy_prefix_confirmed=False):
        a=self._read(observations,fields,healthy_prefix_confirmed)
        v=_transform([a],self.ontology.domain,self.selected.feature)
        direct,factorized=self.selected.both(v)
        p=self.selected.predict_proba(v)[0]
        raw=int(np.argmax(p));confidence=float(np.max(p));unknown=bool(confidence<self.threshold)
        direct_class=int(np.argmax(direct[0]));head_class=int(np.argmax(factorized[0]))
        return {'prediction':None if unknown else self.ontology.labels[raw],
                'status':'UNKNOWN' if unknown else 'RESEARCH_PREDICTION',
                'confidence':confidence,'unknown_reason':'LEARNED_TRAIN_OOF_SELECTIVE_RISK' if unknown else None,
                'head_conflict':direct_class!=head_class,
                'direct_head':self.ontology.labels[direct_class],
                'factorized_head':self.ontology.labels[head_class],
                'proba':{k:float(p[j]) for j,k in enumerate(self.ontology.labels)},
                'architecture':self.selected.architecture,'feature':self.selected.feature,
                'qualified':False,'training_role_fields_used_at_inference':False}


def design(*,domain: str,observations: Sequence,fields: Sequence[str],labels: Sequence[int],
           world_ids: Sequence[str],roles: Sequence[str],healthy_prefix_confirmed: Sequence[bool],
           seed=8202,max_unknown_fraction=0.15,abstention_cost=0.5,
           candidate_specs=None,acceptance_policy: TaskAcceptancePolicy | None = None):
    """Fit and select only using design_train worlds; never inspect eval data.

    Call sites must supply the true role and exact deployment schema; role,
    world and label metadata stay outside the numerical feature extraction.
    """
    policy=TaskAcceptancePolicy() if acceptance_policy is None else acceptance_policy
    if not isinstance(policy,TaskAcceptancePolicy):raise ValueError('INVALID_ACCEPTANCE_POLICY_TYPE')
    ontology=TaskOntology.from_domain(domain)
    schema=tuple(FIELDS[domain])
    if tuple(fields)!=schema:raise ValueError('EXACT_DEPLOYMENT_FIELDS_REQUIRED')
    n=len(observations)
    if not(n==len(labels)==len(world_ids)==len(roles)==len(healthy_prefix_confirmed)) or n<20:
        raise ValueError('INVALID_TRAINING_LENGTH')
    if any(r!='design_train' for r in roles):raise ValueError('PROTECTED_OR_EVALUATION_ROLE')
    if any(type(b) is not bool or not b for b in healthy_prefix_confirmed):raise ValueError('HEALTHY_PREFIX_NOT_CONFIRMED')
    if any(not isinstance(w,str) or not w for w in world_ids):raise ValueError('WORLD_ID_REQUIRED')
    if not(0 <= max_unknown_fraction <= 0.5) or not(0<=abstention_cost<=1):raise ValueError('INVALID_RISK_CONTRACT')
    arrays=[ObservationSpec('task-aware:'+domain,schema).array(a) for a in observations]
    y=np.asarray(labels,dtype=int)
    if not np.array_equal(y,np.asarray(labels)) or set(y)!=set(range(len(ontology.labels))):
        raise ValueError('FULL_ONTOLOGY_CLASS_SUPPORT_REQUIRED')
    # Identical physical worlds cannot occupy independent folds.
    g=np.asarray(world_ids)
    splits=_split(g)
    for tr,va in splits:
        if set(g[tr]) & set(g[va]):raise AssertionError('WORLD_LEAKAGE')
        if set(y[tr])!=set(range(len(ontology.labels))):raise ValueError('FOLD_MISSING_CLASSES')
    programs=candidate_specs if candidate_specs is not None else [(f,l,a) for f,l,a in product(('temporal','physical'),('linear','extra'),('direct','factorized','fusion'))]
    # Reject malformed search contracts *before* fitting any candidate.
    # Opt-in CPU Torch candidates require the raw time-series feature and do
    # not silently join the legacy default candidate space.
    if not isinstance(programs,(tuple,list)) or not programs:
        raise ValueError('EMPTY_TASK_CANDIDATE_SPACE')
    registry=TaskCandidateRegistry()
    for entry in programs:
        if not isinstance(entry,(list,tuple)) or len(entry)!=3:
            raise ValueError('INVALID_TASK_CANDIDATE_SPEC')
        feature,learner,architecture=entry
        registry.describe(feature,learner)
        if architecture not in ('direct','factorized','fusion'):
            raise ValueError('UNREGISTERED_TASK_ARCHITECTURE')
    if len(set(tuple(x) for x in programs))!=len(programs):
        raise ValueError('DUPLICATE_TASK_CANDIDATE')
    feats=tuple(dict.fromkeys(x[0] for x in programs))
    matrix={k:_transform(arrays,domain,k) for k in feats}
    rows=[];winning=None;winner_oof=None
    for feature,learner,arch in programs:
        t0=time.process_time();failure=None
        try:
            x=matrix[feature];oof=np.zeros((n,len(ontology.labels)))
            inner_weights=[]
            for tr,va in splits:
                w=_fusion_weight(x[tr],y[tr],g[tr],ontology,feature,learner,seed) if arch=='fusion' else .5
                m=_Fitted(ontology,arch,feature,learner,seed,weight=w).fit(x[tr],y[tr])
                oof[va]=m.predict_proba(x[va]);inner_weights.append(w)
            score=_score(y,oof,g)
            # Bounds affect only model selection, never assign a fault class.
            # Preserve a Pareto archive for later risk policies.
            record={'candidate':{'feature':feature,'learner':learner,'architecture':arch},
                    'backend':registry.describe(feature,learner).backend,
                    'oof':score,'inner_fusion_weights':inner_weights,
                    'selection_loss':score['average_decision_loss']+0.15*(1-score['worst_fault_recall'])+0.10*(1-score['worst_world_accuracy']),
                    'status':'COMPLETE','acceptance':policy.evaluate(score)}
            rows.append(record)
            ranking=(record['selection_loss'],score['healthy_false_alarm_rate'], -score['worst_fault_recall'],feature,learner,arch)
            if record['acceptance']['eligible'] and (winning is None or ranking<winning):
                winning=ranking;winner_oof=oof;chosen=record['candidate'];chosen_weights=inner_weights
        except Exception as e:
            failure=f'{type(e).__name__}: {e}'
            rows.append({'candidate':{'feature':feature,'learner':learner,'architecture':arch},'status':'FAILED','reason':failure})
        finally:
            rows[-1]['cpu_s']=round(time.process_time()-t0,6)
    valid=[r for r in rows if r['status']=='COMPLETE']
    # A trainable candidate is not necessarily a feasible design. Reject before
    # any final model fitting, while retaining every failed gate and candidate.
    audit_base={'schema':'phm.0.8.task-design.v2','stage':'DEVELOPMENT_ONLY',
        'method':'grouped_nested_oof_task_architecture_competition_with_feasibility_gate',
        'candidate_registry':'r68-v1', 'backend_mode':'sklearn_or_torch_cpu_opt_in',
        'independent_training_worlds':len(set(world_ids)),
        'unique_training_rows':n,'group_fold_intersection':0,
        'acceptance_policy':vars(policy).copy(),
        'candidate_results':rows,'eligible_candidates':sum(r['acceptance']['eligible'] for r in valid),
        'not_deployment_qualified':True,'classifier_sees_simulation_truth':False,
        'training_only_screen_no_sealed_validation':True}
    if winning is None:
        audit={**audit_base,'decision':'REJECT_NO_ELIGIBLE_CANDIDATE',
               'selected_train_only':None,'selection_oof':None,'model_delivered':False}
        raise TaskDesignRejected('NO_CANDIDATE_MEETS_TASK_ACCEPTANCE_POLICY',audit)
    # Selective threshold learned from training-only grouped OOF and limited
    # to a declared rejection budget, with zero-reject always admissible.
    peak=np.max(winner_oof,axis=1)
    options=sorted(set([0.]+[float(q) for q in np.quantile(peak,[.05,.1,.15,.2,.3,.4,.5,.6,.7,.8])]))
    threshold=0.;best=None;cost_records=[]
    for cut in options:
        reject=(peak<cut)
        if float(reject.mean())>max_unknown_fraction+1e-12:continue
        pred=winner_oof.argmax(axis=1)
        risk=float(np.mean((~reject)&(pred!=y)) + np.mean((~reject)&(y==0)&(pred!=0)) + abstention_cost*np.mean(reject))
        cost_records.append({'threshold':cut,'unknown_fraction':float(reject.mean()),'cost':risk})
        # For equal cost, favor coverage rather than gratuitous abstention.
        key=(risk,float(reject.mean()),cut)
        if best is None or key<best:best=key;threshold=cut
    weight=float(np.median(chosen_weights)) if chosen['architecture']=='fusion' else .5
    x=matrix[chosen['feature']]
    model=_Fitted(ontology,chosen['architecture'],chosen['feature'],chosen['learner'],seed,weight).fit(x,y)
    pareto=[]
    for r in valid:
        metrics=r['oof'];v=(1-metrics['accuracy'],metrics['healthy_false_alarm_rate'],1-metrics['worst_fault_recall'],1-metrics['worst_world_accuracy'])
        dominated=False
        for other in valid:
            if other is r:continue
            m=other['oof'];u=(1-m['accuracy'],m['healthy_false_alarm_rate'],1-m['worst_fault_recall'],1-m['worst_world_accuracy'])
            if all(a<=b+1e-12 for a,b in zip(u,v)) and any(a<b-1e-12 for a,b in zip(u,v)):
                dominated=True;break
        if not dominated:pareto.append(r['candidate'])
    audit={**audit_base,'decision':'RESEARCH_CANDIDATE_ELIGIBLE',
       'model_delivered':True,'pareto_candidates':pareto,
       'selected_train_only':chosen,'selection_oof':next(r['oof'] for r in valid if r['candidate']==chosen),
       'threshold_options':cost_records,'threshold_trained_without_evaluation':True}

    return TaskDesignArtifact(ontology,schema,chosen,model,audit,threshold,abstention_cost,max_unknown_fraction)
