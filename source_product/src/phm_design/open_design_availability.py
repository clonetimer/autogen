"""Physical-group cross-fitted late fusion with explicit component availability.

Only registered support failures may disable components. Telemetry corruption,
unknown fields and programming errors are not learnable missingness. Missing
features are never imputed as observations; zeros only encode absent meta inputs,
accompanied by an explicit availability mask. All mask augmentation comes from
out-of-fold predictions and does not increase the number of independent worlds.
"""
from __future__ import annotations
from dataclasses import replace
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array


class AvailableComponentEnsemble(ClassifierMixin, BaseEstimator):
    def __init__(self, program, atom_sizes):
        self.program = program
        self.atom_sizes = atom_sizes

    def _unpack(self, X):
        X = check_array(X, dtype=float)
        if X.shape[1] != sum(self.atom_sizes) + len(self.atom_sizes):
            raise ValueError('AVAILABILITY_DIMENSION_CONTRACT')
        mask = X[:, sum(self.atom_sizes):]
        if not np.isin(mask, (0., 1.)).all():
            raise ValueError('AVAILABILITY_MASK_NOT_BINARY')
        bounds = np.r_[0, np.cumsum(self.atom_sizes)]
        return [X[:, bounds[i]:bounds[i+1]] for i in range(len(self.atom_sizes))], mask.astype(bool)

    def _fit_branch(self, X, y, weights):
        from .open_design_fit import estimator
        p = replace(self.program, composition='concat')
        model = estimator(p)
        name = list(model.named_steps)[-1]
        kwargs = {} if p.learner=='knn' else {name+'__sample_weight': weights}
        if p.learner=='knn' and p.healthy_weight!=1.:
            raise ValueError('KNN_DOES_NOT_SUPPORT_SAMPLE_WEIGHTS')
        model.fit(X, y, **kwargs)
        return model

    def _probability(self, model, X):
        if not np.array_equal(model.classes_, self.classes_):
            raise ValueError('AVAILABILITY_CLASS_ALIGNMENT')
        if hasattr(model, 'predict_proba'):
            p = model.predict_proba(X)
        else:
            # SVC decision scores converted to relative evidence, not calibrated
            # probabilities. Meta learning remains strictly group-out-of-fold.
            d = model.decision_function(X)
            if d.ndim==1:d = np.column_stack((-d, d))
            d = d - d.max(axis=1, keepdims=True)
            p = np.exp(d);p /= p.sum(axis=1, keepdims=True)
        if not np.isfinite(p).all() or np.any(p<0) or not np.allclose(p.sum(axis=1),1):
            raise ValueError('AVAILABILITY_INVALID_PROBABILITY')
        return p

    @staticmethod
    def _meta(p, mask):
        # No log(0) and no median/mean fitted across validation observations.
        return np.c_[p.reshape(len(p), -1), mask.astype(float)]

    def fit(self, X, y, *, groups=None, sample_weight=None):
        X, y = check_X_y(X, y, dtype=float)
        if not 1<=len(self.atom_sizes)<=3 or any(n<=0 for n in self.atom_sizes):
            raise ValueError('AVAILABILITY_ATOM_SIZES')
        if self.program.composition not in ('available_stack','available_average'):
            raise ValueError('AVAILABILITY_COMPOSITION')
        blocks, available = self._unpack(X)
        g = np.asarray(groups) if groups is not None else np.array([])
        if g.ndim!=1 or len(g)!=len(y) or len(set(g))<3:
            raise ValueError('AVAILABILITY_REQUIRES_THREE_PHYSICAL_GROUPS')
        self.classes_ = np.unique(y)
        if len(self.classes_)<2 or -1 in self.classes_:
            raise ValueError('AVAILABILITY_CLASS_CONTRACT')
        weights = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight,float)
        if weights.shape!=y.shape or not np.isfinite(weights).all() or np.any(weights<=0):
            raise ValueError('AVAILABILITY_SAMPLE_WEIGHT')
        self.n_features_in_ = X.shape[1]
        n, b, k = len(y), len(blocks), len(self.classes_)
        oof = np.zeros((n,b,k));mask = np.zeros((n,b),bool)
        self.group_audit_ = []
        for tr, va in GroupKFold(3).split(X, y, g):
            if set(g[tr]) & set(g[va]):raise ValueError('AVAILABILITY_GROUP_LEAKAGE')
            for j, block in enumerate(blocks):
                use = tr[available[tr,j]];test = va[available[va,j]]
                supported = np.array_equal(np.unique(y[use]), self.classes_)
                self.group_audit_.append({'component':j,
                    'base_train_groups':sorted(map(str,set(g[use]))),
                    'meta_oof_groups':sorted(map(str,set(g[va]))),
                    'disjoint':True,'train_count':len(use),'full_class_support':supported,
                    'scalers_fitted_inside_base_train':True})
                if not supported:continue
                model = self._fit_branch(block[use], y[use], weights[use])
                if len(test):oof[test,j] = self._probability(model,block[test]);mask[test,j] = True
        # All nonempty subsets of actually supported OOF branches. Counts are
        # normalized per original row, preserving original class/sample weight.
        meta_x=[];meta_y=[];meta_w=[];patterns=set();source=[]
        counts=np.zeros(n,int)
        for bits in range(1, 2**b):
            subset=np.array([bool(bits&(1<<j)) for j in range(b)])
            rows=np.flatnonzero(np.all(~subset|mask,axis=1));counts[rows]+=1
        for bits in range(1, 2**b):
            subset=np.array([bool(bits&(1<<j)) for j in range(b)])
            rows=np.flatnonzero(np.all(~subset|mask,axis=1))
            if not len(rows):continue
            m=np.tile(subset,(len(rows),1));q=oof[rows]*m[:,:,None]
            meta_x.append(self._meta(q,m));meta_y.append(y[rows]);meta_w.append(weights[rows]/counts[rows]);source.extend(rows.tolist());patterns.add(tuple(subset.tolist()))
        if not meta_x or not np.array_equal(np.unique(np.concatenate(meta_y)),self.classes_):
            raise ValueError('AVAILABILITY_INSUFFICIENT_OOF_SUPPORT')
        self.meta_ = make_pipeline(StandardScaler(),LogisticRegression(
            C=self.program.strength,max_iter=300,random_state=self.program.seed))
        self.meta_.fit(np.concatenate(meta_x),np.concatenate(meta_y),
                       logisticregression__sample_weight=np.concatenate(meta_w))
        self.bases_=[]
        for j,block in enumerate(blocks):
            use=np.flatnonzero(available[:,j])
            if np.array_equal(np.unique(y[use]),self.classes_):
                self.bases_.append(self._fit_branch(block[use],y[use],weights[use]))
            else:self.bases_.append(None)
        if not any(m is not None for m in self.bases_):raise ValueError('AVAILABILITY_NO_TRAINED_COMPONENT')
        self.mask_patterns_ = sorted(patterns)
        self.meta_training_audit_={'physical_worlds':len(set(g)),
            'original_rows':n,'oof_supported_rows':int(np.sum(mask.any(axis=1))),
            'augmented_meta_rows':len(source),'new_physical_worlds':0,
            'source':'group_oof_only','per_original_row_weight_preserved':True,
            'pattern_count':len(patterns),'probabilities_independently_calibrated':False}
        return self

    def _evidence(self, X):
        check_is_fitted(self,'bases_')
        blocks,mask=self._unpack(X);p=np.zeros((len(mask),len(blocks),len(self.classes_)))
        for j,model in enumerate(self.bases_):
            if model is None:mask[:,j]=False;continue
            rows=np.flatnonzero(mask[:,j])
            if len(rows):p[rows,j]=self._probability(model,blocks[j][rows])
        return p,mask

    def predict_with_mode(self, X, mode):
        if mode not in ('learned','available_average','strict_all'):
            raise ValueError('AVAILABILITY_PREDICTION_MODE')
        p,mask=self._evidence(X);out=np.full(len(mask),-1,dtype=int)
        valid=mask.all(axis=1) if mode=='strict_all' else mask.any(axis=1)
        # Never manufacture an answer for an availability pattern absent from
        # group-OOF support. This is a data support contract, not fault threshold.
        if mode in ('learned','strict_all'):
            valid &= np.array([tuple(row.tolist()) in self.mask_patterns_ for row in mask])
            if valid.any():out[valid]=self.meta_.predict(self._meta(p[valid],mask[valid]))
        elif valid.any():
            avg=p[valid].sum(axis=1)/mask[valid].sum(axis=1,keepdims=True)
            out[valid]=self.classes_[avg.argmax(axis=1)]
        return out

    def predict(self, X):
        mode='learned' if self.program.composition=='available_stack' else 'available_average'
        return self.predict_with_mode(X,mode)
