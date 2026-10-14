"""Train-only frozen statistical reference plus a generated residual DAG.

The reference is not evidence of a novel neural architecture. Bounds protect
reference *predictions*, not correctness or calibrated risk. Float64 reference
buffers intentionally survive CPU compilation to reproduce training features.
"""
from __future__ import annotations
import copy
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.linear_model import LogisticRegression
from .data import statistics
from .graph import GraphNetwork, analyze_graph
from .models import Shape
from .io import digest, sha_file

SCHEMA = "phm.train-only-reference.v1"


def fit_reference(x: np.ndarray, y: np.ndarray, contract: dict, samples: list[dict]) -> dict:
    if set(np.unique(y).tolist()) != set(range(len(contract["classes"]))):
        raise ValueError("REFERENCE_CLASS_COVERAGE")
    fx = statistics(x)
    mean, scale = fx.mean(0), np.maximum(fx.std(0), 1e-8)
    clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced", random_state=0)
    clf.fit((fx-mean)/scale, y)
    coef, intercept = clf.coef_, clf.intercept_
    if coef.shape[0] == 1:
        coef = np.vstack((np.zeros_like(coef), coef))
        intercept = np.r_[0.0, intercept]
    return {"schema":SCHEMA, "feature_view":"seven_statistics_float64",
        "fit_role":"train_only", "contract_sha256":digest(contract),
        "training_samples_sha256":digest(samples),
        "training_arrays_sha256":digest({"x":x.tolist(),"y":y.tolist()}),
        "class_indices":clf.classes_.tolist(), "mean":mean.tolist(), "scale":scale.tolist(),
        "coef":coef.tolist(), "intercept":intercept.tolist(),
        "estimator":{"kind":"logistic_regression","C":1.0,"max_iter":2000,"class_weight":"balanced"}}


def validate_reference(ref: dict, shape: Shape) -> None:
    fields={"schema","feature_view","fit_role","contract_sha256","training_samples_sha256",
            "training_arrays_sha256","class_indices","mean","scale","coef","intercept","estimator"}
    if not isinstance(ref,dict) or set(ref)!=fields or ref["schema"]!=SCHEMA:
        raise ValueError("REFERENCE_SCHEMA")
    if ref["fit_role"]!="train_only" or ref["feature_view"]!="seven_statistics_float64":
        raise ValueError("REFERENCE_TRAIN_ONLY")
    if ref["class_indices"] != list(range(shape.classes)):
        raise ValueError("REFERENCE_CLASS_ORDER")
    expected={"mean":(7*shape.channels,),"scale":(7*shape.channels,),
              "coef":(shape.classes,7*shape.channels),"intercept":(shape.classes,)}
    for key,dims in expected.items():
        a=np.asarray(ref[key],dtype=np.float64)
        if a.shape!=dims or not np.isfinite(a).all():raise ValueError("REFERENCE_SHAPE_OR_NONFINITE")
    if np.any(np.asarray(ref["scale"])<=0):raise ValueError("REFERENCE_SCALE")
    for key in ("contract_sha256","training_samples_sha256","training_arrays_sha256"):
        v=ref[key]
        if not isinstance(v,str) or len(v)!=64 or any(c not in "0123456789abcdef" for c in v):
            raise ValueError("REFERENCE_PROVENANCE_HASH")
    if ref["estimator"]!={"kind":"logistic_regression","C":1.0,"max_iter":2000,"class_weight":"balanced"}:
        raise ValueError("REFERENCE_ESTIMATOR")


def reference_logits(ref: dict,x: np.ndarray) -> np.ndarray:
    return ((statistics(x)-np.asarray(ref["mean"]))/np.asarray(ref["scale"]))@np.asarray(ref["coef"]).T+np.asarray(ref["intercept"])


class FrozenReference(nn.Module):
    def __init__(self,shape:Shape,ref:dict):
        super().__init__();validate_reference(ref,shape);self.shape=shape
        for key in ("mean","scale","coef","intercept"):
            self.register_buffer(key,torch.tensor(ref[key],dtype=torch.float64))
    def forward(self,x):
        # Reference is frozen even when the residual graph trains.
        with torch.no_grad():
            x=x.double()
            f=torch.cat((x.mean(1),x.std(1,correction=0),x.amin(1),x.amax(1),
                         x[:,-1],x[:,-1]-x[:,0],x.square().mean(1).sqrt()),1)
            return (((f-self.mean)/self.scale)@self.coef.T+self.intercept).float()


def validate_parameters(shape,p):
    if not isinstance(p,dict) or set(p)!={"reference","graph","mode","bound_logits"}:
        raise ValueError("REFERENCE_DAG_SCHEMA")
    validate_reference(p["reference"],shape)
    if p["mode"] not in {"bounded","unbounded"}:raise ValueError("REFERENCE_CORRECTION_MODE")
    if type(p["bound_logits"]) not in (int,float) or not np.isfinite(p["bound_logits"]) or not 0<p["bound_logits"]<=10:
        raise ValueError("REFERENCE_CORRECTION_BOUND")
    return analyze_graph(shape,p["graph"])


class ReferenceResidual(nn.Module):
    def __init__(self,shape,p):
        super().__init__();validate_parameters(shape,p)
        self.reference=FrozenReference(shape,p["reference"])
        self.residual=GraphNetwork(shape,p["graph"])
        self.gain=nn.Parameter(torch.tensor(0.0,dtype=torch.float32))
        self.bound=float(p["bound_logits"]);self.mode=p["mode"];self.shape=shape
    def parts(self,x):
        if x.ndim!=3 or x.shape[1:]!=(self.shape.window,self.shape.channels) or not torch.isfinite(x).all():
            raise ValueError("REFERENCE_INPUT_SHAPE_OR_NONFINITE")
        base=self.reference(x);raw=self.residual(x)
        correction=(self.bound*torch.tanh(self.gain)*torch.tanh(raw) if self.mode=="bounded" else self.gain*raw)
        return base,correction
    def forward(self,x):
        a,b=self.parts(x);return a+b


def residual_count(shape,p):return validate_parameters(shape,p)["parameter_count"]+1

def build_residual(shape,p):return ReferenceResidual(shape,p)


def verify_training_binding(specs,shape,x,y,contract,samples):
    selected=[s for s in specs if s["family"]=="reference_dag"]
    if not selected:return
    expected=fit_reference(x,y,contract,samples)
    for s in selected:
        if s["parameters"]["reference"]!=expected:
            raise ValueError("REFERENCE_NOT_FITTED_ON_THIS_TRAIN_SPLIT")
