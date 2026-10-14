"""Fit/evaluate learned diagnostic programs with grouped validation.

Validation scores are selection-only. Physical worlds, not observation variants,
are the split unit. Feature scalers and classifiers are fitted inside each fold.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, replace
import warnings
import hashlib
from numbers import Integral
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.base import BaseEstimator,ClassifierMixin
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
from .open_design import Program,ObservationSpec,FeatureRegistry,RiskPreference,FeedbackPlanner
from .robust_design_policy import RobustDesignPolicy, RobustDesignRejected
from .leakage_control import WorldLineage, authorize_design


@dataclass
class TrainingExample:
    observations: np.ndarray
    label: int
    group: str
    role: str = 'design_train'
    profile: str = 'identity'
    healthy_prefix_confirmed: bool = False


class ProbabilityMixture(ClassifierMixin,BaseEstimator):
    def __init__(self,strength=1.,seed=48019):self.strength=strength;self.seed=seed
    def fit(self,X,y,sample_weight=None):
        self.classes_=np.unique(y)
        self.linear_=LogisticRegression(C=self.strength,max_iter=300,random_state=self.seed).fit(X,y,sample_weight=sample_weight)
        self.normal_=GaussianNB(var_smoothing=1e-5).fit(X,y,sample_weight=sample_weight)
        return self
    def predict_proba(self,X):return .5*self.linear_.predict_proba(X)+.5*self.normal_.predict_proba(X)
    def predict(self,X):return self.classes_[self.predict_proba(X).argmax(1)]


def estimator(program: Program):
    p=program
    if p.learner=='linear':m=LogisticRegression(C=p.strength,max_iter=300,random_state=p.seed)
    elif p.learner=='svm':m=SVC(C=p.strength)
    elif p.learner=='forest':m=RandomForestClassifier(n_estimators=24,max_depth=p.depth,random_state=p.seed,n_jobs=1)
    elif p.learner=='extra':m=ExtraTreesClassifier(n_estimators=32,max_depth=p.depth,random_state=p.seed,n_jobs=1)
    elif p.learner=='gnb':m=GaussianNB(var_smoothing=1e-6*p.strength)
    elif p.learner=='knn':m=KNeighborsClassifier(n_neighbors=3,weights='distance')
    elif p.learner=='mlp':m=MLPClassifier(hidden_layer_sizes=(16,8),alpha=.1*p.strength,max_iter=150,random_state=p.seed)
    elif p.learner=='probability_mix':m=ProbabilityMixture(p.strength,p.seed)
    else:raise ValueError('UNKNOWN_LEARNER')
    return make_pipeline(StandardScaler(),m)


def fit_estimator(p,X,y,healthy_label,groups=None,atom_sizes=None):
    if p.composition!='concat':
        from .open_design_availability import AvailableComponentEnsemble
        m=AvailableComponentEnsemble(p,tuple(atom_sizes or ()))
        weights=np.where(y==healthy_label,p.healthy_weight,1.)
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always');m.fit(X,y,groups=groups,sample_weight=weights)
        return m,[str(w.message) for w in ws]
    if p.learner in ('group_stack','group_average'):
        from .open_design_ensemble import GroupProbabilityEnsemble
        m=GroupProbabilityEnsemble(p.strength,p.depth,p.seed,
            'learned' if p.learner=='group_stack' else 'average')
        weights=np.where(y==healthy_label,p.healthy_weight,1.)
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always');m.fit(X,y,groups=groups,sample_weight=weights)
        return m,[str(w.message) for w in ws]
    m=estimator(p)
    if p.learner=='knn':
        if p.healthy_weight!=1.:raise ValueError('KNN_DOES_NOT_SUPPORT_SAMPLE_WEIGHTS')
        kw={}
    else:
        weights=np.where(y==healthy_label,p.healthy_weight,1.)
        kw={list(m.named_steps)[-1]+'__sample_weight':weights}
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter('always');m.fit(X,y,**kw)
    return m,[str(w.message) for w in ws]


def validate_training(examples,spec,classes):
    if (not examples or len(classes)!=len(set(classes)) or len(classes)<2 or
        any(isinstance(c,bool) or not isinstance(c,Integral) or c==-1 for c in classes)):
        raise ValueError('INVALID_CLASSES')
    if len({x.group for x in examples})<3:raise ValueError('AT_LEAST_THREE_PHYSICAL_GROUPS_REQUIRED')
    for e in examples:
        if e.role!='design_train':raise ValueError('PROTECTED_ROLE_NOT_TRAINING_DATA')
        if isinstance(e.label,bool) or not isinstance(e.label,Integral) or e.label not in classes or not isinstance(e.group,str) or not e.group:raise ValueError('INVALID_LABEL_OR_GROUP')
        if e.profile not in ('identity','drift_quant','delay_noise'):raise ValueError('INVALID_TRAINING_PROFILE')
        spec.array(e.observations)


def feature_matrix(examples,spec,program,registry,cache):
    # A cache reused across tasks, reordered rows, schema changes, or mutations
    # must not silently supply features from a previous dataset.
    digest=hashlib.sha256()
    digest.update(repr((spec.name,spec.fields,spec.time_field,spec.min_rows)).encode())
    for e in examples:
        a=np.ascontiguousarray(spec.array(e.observations),dtype='<f8')
        digest.update(repr((a.shape,bool(e.healthy_prefix_confirmed))).encode())
        digest.update(a.tobytes())
    namespace=(program.composition!='concat',registry,tuple((n,id(v[0]),v[1]) for n,v in registry._atoms.items()),digest.hexdigest())
    if cache.get('__dataset_context_v2__')!=namespace:
        cache.clear();cache['__dataset_context_v2__']=namespace
    features=[];valids=[]
    for atom in program.atoms:
        if atom not in cache:
            rows=[];ok=[];single=Program((atom,),'linear')
            for e in examples:
                try:
                    f=registry.transform(single,e.observations,spec,healthy_prefix_confirmed=e.healthy_prefix_confirmed);rows.append(f);ok.append(True)
                except ValueError as exc:
                    from .open_design import FeatureUnavailable
                    if program.composition!='concat' and not isinstance(exc,FeatureUnavailable):raise
                    rows.append(None);ok.append(False)
            size=next((len(f) for f in rows if f is not None),None)
            if size is None:raise ValueError('NO_VALID_FEATURE_ROWS:'+atom)
            cache[atom]=(np.asarray([f if f is not None else np.zeros(size) for f in rows]),np.asarray(ok))
        X,v=cache[atom];features.append(X);valids.append(v)
    if program.composition!='concat':
        return np.concatenate(features+[np.asarray(valids,dtype=float).T],axis=1),np.logical_or.reduce(valids)
    return np.concatenate(features,axis=1),np.logical_and.reduce(valids)


def metrics(y,p,groups,classes,healthy_label):
    y=np.asarray(y);p=np.asarray(p);g=np.asarray(groups)
    recall=[float(np.mean(p[y==c]==c)) if np.any(y==c) else 0. for c in classes]
    # Missing class support is conservative; no healthy-only F1 inflation.
    pergroup={str(w):float(f1_score(y[g==w],p[g==w],labels=list(classes),average='macro',zero_division=0)) for w in sorted(set(g))}
    return {'total':len(y),'correct':int(np.sum(y==p)),'classes':list(classes),'recall':recall,
        'class_errors':[1-r for r in recall],'macro_f1':float(f1_score(y,p,labels=list(classes),average='macro',zero_division=0)),
        'healthy_count':int(np.sum(y==healthy_label)),
        'healthy_false_alarms':int(np.sum((y==healthy_label)&(p!=healthy_label)&np.isin(p,classes))),
        'healthy_unknown':int(np.sum((y==healthy_label)&~np.isin(p,classes))),
        'healthy_nonacceptances':int(np.sum((y==healthy_label)&(p!=healthy_label))),
        'unknown':int(np.sum(~np.isin(p,classes))),'worlds':len(pergroup),'world_macro_f1':pergroup,
        'worst_world_macro_f1':min(pergroup.values()),
        'class_support':{str(c):int(np.sum(y==c)) for c in classes}}


def evaluate_program(examples,spec,classes,healthy_label,program,registry,cache,*,validation_families=None):
    validate_training(examples,spec,classes)
    if healthy_label not in classes:raise ValueError('HEALTHY_LABEL_NOT_IN_CLASSES')
    X,valid=feature_matrix(examples,spec,program,registry,cache)
    y=np.asarray([e.label for e in examples]);g=np.asarray([e.group for e in examples]);pf=np.asarray([e.profile for e in examples])
    predictions=np.full(len(y),-1,int);folds=[]
    from .open_design_validation import validation_splits
    split_plan = validation_splits(g, validation_families)
    for tr,te,split_audit in split_plan:
        tr=tr[valid[tr]&((pf[tr]=='identity')|program.augmentation)]
        if set(y[tr])!=set(classes):raise ValueError('MISSING_TRAIN_CLASS')
        model,ws=fit_estimator(program,X[tr],y[tr],healthy_label,groups=g[tr],atom_sizes=[cache[a][0].shape[1] for a in program.atoms])
        te_valid=te[valid[te]]
        if len(te_valid):predictions[te_valid]=model.predict(X[te_valid])
        folds.append({'train_groups':sorted(set(g[tr])),'validation_groups':sorted(set(g[te])),
            'disjoint':not bool(set(g[tr])&set(g[te])),'warnings':ws,
            'inner_group_audit':getattr(model,'group_audit_',[]), **split_audit})
    result=metrics(y,predictions,g,classes,healthy_label)
    result['validation_mode']='physical_world' if validation_families is None else 'leave_environment_family_out'
    if validation_families is not None:
        fam=np.asarray([validation_families[w] for w in g])
        result['family_metrics']={str(q):metrics(y[fam==q],predictions[fam==q],g[fam==q],classes,healthy_label) for q in sorted(set(fam))}
    result['profiles']={q:metrics(y[pf==q],predictions[pf==q],g[pf==q],classes,healthy_label) for q in sorted(set(pf))}
    result['robust_class_errors']=np.max([v['class_errors'] for v in result['profiles'].values()],axis=0).tolist()
    return result,predictions.tolist(),folds


def refit(examples,spec,classes,healthy_label,program,registry,cache):
    validate_training(examples,spec,classes)
    if healthy_label not in classes:raise ValueError('HEALTHY_LABEL_NOT_IN_CLASSES')
    X,v=feature_matrix(examples,spec,program,registry,cache)
    use=v&np.asarray([e.profile=='identity' or program.augmentation for e in examples])
    y=np.asarray([e.label for e in examples]);g=np.asarray([e.group for e in examples])
    if set(y[use])!=set(classes):raise ValueError('MISSING_REFIT_CLASS')
    m,ws=fit_estimator(program,X[use],y[use],healthy_label,groups=g[use],atom_sizes=[cache[a][0].shape[1] for a in program.atoms])
    return {'schema':'phm.open-design.fitted.v1','observation_spec':asdict(spec),'classes':list(classes),
        'healthy_label':healthy_label,'program':asdict(program),'estimator':m,'graph':registry.graph(program,spec),
        'atom_sizes':[cache[a][0].shape[1] for a in program.atoms],
        'refit_warnings':ws,'inner_group_audit':getattr(m,'group_audit_',[]),'release_qualified':False}


def predict(model,values,registry,*,healthy_prefix_confirmed=False):
    s=dict(model['observation_spec']);s['fields']=tuple(s['fields']);spec=ObservationSpec(**s)
    p=dict(model['program']);p['atoms']=tuple(p['atoms']);program=Program(**p)
    try:
        a=spec.rows(values) if isinstance(values,list) and values and isinstance(values[0],dict) else spec.array(values)
        if program.composition=='concat':
            f=registry.transform(program,a,spec,healthy_prefix_confirmed=healthy_prefix_confirmed)
            details={}
        else:
            parts,reasons=registry.transform_parts(program,a,spec,healthy_prefix_confirmed=healthy_prefix_confirmed)
            sizes=model['atom_sizes']
            if len(sizes)!=len(parts):raise ValueError('COMPONENT_DIMENSION_CONTRACT')
            for part,size in zip(parts,sizes):
                if part is not None and len(part)!=size:raise ValueError('COMPONENT_DIMENSION_CONTRACT')
            mask=np.asarray([v is not None for v in parts],float)
            if not mask.any():raise ValueError('NO_AVAILABLE_COMPONENT')
            f=np.concatenate([v if v is not None else np.zeros(size) for v,size in zip(parts,sizes)]+[mask])
            details={'available_components':[n for n,v in zip(program.atoms,parts) if v is not None],
                     'unavailable_components':reasons,'composition':program.composition}
        label=int(model['estimator'].predict(f.reshape(1,-1))[0])
        if label==-1:raise ValueError('NO_AVAILABLE_TRAINED_COMPONENT_OR_OOF_PATTERN')
        if label not in model['classes']:raise ValueError('PREDICTED_CLASS_OUTSIDE_CONTRACT')
        return {'prediction':label,'status':'RESEARCH_PREDICTION','qualified':False,**details}
    except ValueError as e:return {'prediction':-1,'status':'UNKNOWN','reason':str(e),'qualified':False}


def design(examples,spec,classes,healthy_label,*,max_trials=16,registry=None,
           mode='adaptive',risk=RiskPreference(),seed=48019,validation_families=None,
           robust_policy: RobustDesignPolicy | None = None,
           training_lineage: list[WorldLineage] | None = None,
           feedback_used_for_design: bool = False):
    """Small in-process API; production hard CPU enforcement belongs to a supervisor."""
    from .open_design import generic_registry
    if max_trials<1:raise ValueError('INVALID_TRIAL_LIMIT')
    if feedback_used_for_design:
        raise ValueError('FINAL_FEEDBACK_TAINTED_DESIGN')
    receipt = None
    if training_lineage is not None and robust_policy is None:
        if len(training_lineage) != len(examples):
            raise ValueError('TRAINING_LINEAGE_LENGTH_MISMATCH')
        receipt = authorize_design(training_lineage)
        if any(rec.world_id != e.group for rec, e in zip(training_lineage, examples)):
            raise ValueError('TRAINING_LINEAGE_GROUP_MISMATCH')
        if validation_families is not None:
            family_by_world = {}
            for rec in training_lineage:
                previous = family_by_world.setdefault(rec.world_id, rec.environment_family)
                if previous != rec.environment_family:
                    raise ValueError('INCONSISTENT_ENVIRONMENT_LINEAGE')
            if validation_families != family_by_world:
                raise ValueError('ENVIRONMENT_PROVENANCE_MISMATCH')
    if robust_policy is not None:
        if not isinstance(robust_policy, RobustDesignPolicy):
            raise ValueError('INVALID_ROBUST_POLICY_TYPE')
        if training_lineage is None or len(training_lineage) != len(examples):
            raise ValueError('ROBUST_SEARCH_REQUIRES_VERIFIED_LINEAGE')
        receipt = authorize_design(training_lineage)
        if any(rec.world_id != e.group for rec, e in zip(training_lineage,examples)):
            raise ValueError('TRAINING_LINEAGE_GROUP_MISMATCH')
        original_map = {}
        for rec in training_lineage:
            original_map.setdefault(rec.world_id, rec.environment_family)
            if original_map[rec.world_id] != rec.environment_family:
                raise ValueError('INCONSISTENT_ENVIRONMENT_LINEAGE')
        if validation_families is not None and validation_families != original_map:
            raise ValueError('ENVIRONMENT_PROVENANCE_MISMATCH')
        family_map = {}
        for rec in training_lineage:
            previous = family_map.setdefault(rec.generation_family_id,rec.environment_family)
            if previous != rec.environment_family:
                raise ValueError('GENERATION_FAMILY_CROSSES_ENVIRONMENTS')
        examples = [replace(e, group=rec.generation_family_id) for e,rec in zip(examples,training_lineage)]
        validation_families = family_map
    validate_training(examples,spec,classes)
    registry=registry or generic_registry();seeds=[Program((a,),l) for a,l in zip(registry.names,('linear','extra','svm'))]
    planner=FeedbackPlanner(registry.names,mode=mode,seeds=seeds,risk=risk,seed=seed);cache={};attempts=[];best=None
    for _ in range(max_trials):
        p=planner.propose()
        try:
            m,rows,folds=evaluate_program(examples,spec,classes,healthy_label,p,registry,cache,validation_families=validation_families)
            planner.observe(p,m['robust_class_errors'])
            acceptance = robust_policy.evaluate(m, list(classes), healthy_label) if robust_policy else None
            loss = (acceptance['robust_selection_loss'] if acceptance and 'robust_selection_loss' in acceptance
                    else risk.loss(m['robust_class_errors']))
            attempts.append({'program':asdict(p),'metrics':m,'folds':folds,'loss':loss,
                             'status':'COMPLETE' if not acceptance or acceptance['eligible'] else 'REJECTED_BY_POLICY',
                             'acceptance':acceptance})
            if (acceptance is None or acceptance['eligible']) and (best is None or loss<best[0]):
                best=(loss,p)
        except Exception as e:attempts.append({'program':asdict(p),'status':'FAILED','error':repr(e)})
    if best is None:
        if robust_policy is not None:
            raise RobustDesignRejected({'schema':'phm.robust-design.audit.v1',
                'decision':'REJECT_ALL_CANDIDATES', 'model_delivered':False,
                'training_lineage_receipt':receipt, 'attempts':attempts,
                'independent_qualification':False,
                'training_world_design_screen_only':True})
        raise RuntimeError('NO_VALID_PROGRAM')
    model=refit(examples,spec,classes,healthy_label,best[1],registry,cache)
    return model,{'attempts':attempts,'pareto_archive':planner.archive(),'proposals':planner.proposals,
        'risk_preference':asdict(risk),'selection_only':True,'release_qualified':False,
        'robust_policy':asdict(robust_policy) if robust_policy else None,
        'training_lineage_receipt':receipt,
        'independent_qualification':False}
