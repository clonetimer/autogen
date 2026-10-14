"""2025–2026 spacecraft time-series architecture baselines.

These are *architecture-faithful adaptations* for supervised fixed-window
classification on PHM Design datasets. They are not claimed to reproduce the
papers' original datasets, losses, anomaly scoring, event post-processing, or
reported benchmark numbers exactly.

References are metadata only; no external network access is required at runtime.
"""
from __future__ import annotations
import math
import torch
from torch import nn

CATALOG={
 "grid3d_res":{
   "name":"Temporal-Spatial Residual 3D-CNN adaptation",
   "basis":"Wang et al., Small-Satellite System Fault Diagnosis via a Temporal–Spatial 3D-CNN with Imbalanced-Aware Training",
   "year":2026,"doi":None,
   "task_difference":"Original paper uses telemetry-to-grid 3D-CNN on Lumelite 23-fault diagnosis; this implementation keeps the 3D residual idea but uses automatic near-square channel grids."},
 "multiscale_conv_transformer":{
   "name":"Multi-scale Conv + Transformer adaptation",
   "basis":"Lu et al., Self-Supervised Intelligent Autoencoder with Multi-Scale Convolutional Contrastive Learning for Multivariate Anomaly Detection",
   "year":2025,"doi":"10.1177/18758967251407728",
   "task_difference":"Original method is self-supervised reconstruction/contrastive anomaly detection; this adaptation uses its 3/5/7 multi-scale temporal front-end plus Transformer for supervised classification."},
 "dynamic_graph_gru":{
   "name":"Dynamic graph + GRU attention adaptation",
   "basis":"Lai et al., STGLR: A Spacecraft Anomaly Detection Method Based on Spatio-Temporal Graph Learning",
   "year":2025,"doi":None,
   "task_difference":"Original STGLR is unsupervised graph-reconstruction anomaly detection; this is a supervised graph-message-passing/GRU adaptation and must not be presented as exact STGLR reproduction."},
 "transformer":{
   "name":"Transformer encoder control",
   "basis":"Modern multivariate time-series Transformer control",
   "year":2026,"doi":None,
   "task_difference":"Generic supervised control; not a reproduction of one paper."},
}


class Grid3DResidualClassifier(nn.Module):
    """Near-square telemetry grid + residual 3D convolutions."""
    def __init__(self,channels:int,classes:int,width:int=16):
        super().__init__()
        h=max(2,int(math.floor(math.sqrt(channels))));w=int(math.ceil(channels/h))
        while h*w<channels:w+=1
        self.grid=(h,w)
        self.stem=nn.Sequential(nn.Conv3d(1,width,3,padding=1),nn.BatchNorm3d(width),nn.GELU())
        self.res1=nn.Sequential(nn.Conv3d(width,width,3,padding=1),nn.BatchNorm3d(width),nn.GELU(),
                                nn.Conv3d(width,width,3,padding=1),nn.BatchNorm3d(width))
        self.down=nn.Sequential(nn.Conv3d(width,2*width,3,stride=(2,1,1),padding=1),
                                nn.BatchNorm3d(2*width),nn.GELU())
        self.res2=nn.Sequential(nn.Conv3d(2*width,2*width,3,padding=1),nn.BatchNorm3d(2*width),nn.GELU(),
                                nn.Conv3d(2*width,2*width,3,padding=1),nn.BatchNorm3d(2*width))
        self.pool=nn.AdaptiveAvgPool3d(1);self.drop=nn.Dropout(.2);self.head=nn.Linear(2*width,classes)
    def forward(self,x):
        if x.ndim!=3:raise ValueError("LITERATURE_BASELINE_EXPECTS_BTC")
        b,t,c=x.shape;h,w=self.grid
        if h*w<c:raise ValueError("LITERATURE_GRID_CAPACITY")
        if h*w>c:x=torch.nn.functional.pad(x,(0,h*w-c))
        z=x.reshape(b,t,h,w).unsqueeze(1)
        z=self.stem(z);z=torch.nn.functional.gelu(z+self.res1(z))
        z=self.down(z);z=torch.nn.functional.gelu(z+self.res2(z))
        return self.head(self.drop(self.pool(z).flatten(1)))


class TransformerClassifier(nn.Module):
    def __init__(self,channels:int,classes:int,d_model:int=64,heads:int=4,layers:int=2):
        super().__init__()
        self.proj=nn.Linear(channels,d_model)
        self.pos=nn.Parameter(torch.zeros(1,512,d_model))
        layer=nn.TransformerEncoderLayer(d_model,heads,2*d_model,.15,batch_first=True,norm_first=True,activation="gelu")
        self.encoder=nn.TransformerEncoder(layer,layers);self.norm=nn.LayerNorm(d_model);self.head=nn.Linear(d_model,classes)
    def forward(self,x):
        if x.ndim!=3 or x.shape[1]>self.pos.shape[1]:raise ValueError("LITERATURE_TRANSFORMER_SHAPE")
        z=self.proj(x)+self.pos[:,:x.shape[1]]
        return self.head(self.norm(self.encoder(z).mean(1)))


class MultiScaleConvTransformer(nn.Module):
    def __init__(self,channels:int,classes:int,d_model:int=64):
        super().__init__()
        self.branches=nn.ModuleList([nn.Conv1d(channels,24,k,padding=k//2) for k in (3,5,7)])
        self.mix=nn.Linear(72,d_model)
        layer=nn.TransformerEncoderLayer(d_model,4,2*d_model,.15,batch_first=True,norm_first=True,activation="gelu")
        self.encoder=nn.TransformerEncoder(layer,2);self.norm=nn.LayerNorm(d_model);self.head=nn.Linear(d_model,classes)
    def forward(self,x):
        if x.ndim!=3:raise ValueError("LITERATURE_MULTISCALE_SHAPE")
        a=x.transpose(1,2)
        z=torch.cat([torch.nn.functional.gelu(m(a)) for m in self.branches],1).transpose(1,2)
        z=self.encoder(self.mix(z)).mean(1)
        return self.head(self.norm(z))


class DynamicGraphGRUClassifier(nn.Module):
    """Small supervised graph-temporal control inspired by STGLR concepts."""
    def __init__(self,channels:int,classes:int,d_model:int=32):
        super().__init__();self.channels=channels;self.d_model=d_model
        self.node_embedding=nn.Parameter(torch.randn(channels,d_model)*.05)
        self.value=nn.Linear(1,d_model);self.message=nn.Linear(2*d_model,d_model)
        self.gru=nn.GRU(d_model,2*d_model,batch_first=True)
        self.attention=nn.Linear(2*d_model,1);self.head=nn.Linear(2*d_model,classes)
    def forward(self,x):
        if x.ndim!=3 or x.shape[2]!=self.channels:raise ValueError("LITERATURE_GRAPH_SHAPE")
        A=torch.softmax(torch.relu(self.node_embedding@self.node_embedding.T/math.sqrt(self.d_model)),dim=-1)
        h=torch.nn.functional.gelu(self.value(x.unsqueeze(-1)))
        neigh=torch.einsum("ij,btjd->btid",A,h)
        z=torch.nn.functional.gelu(self.message(torch.cat((h,neigh),-1))).mean(2)
        z,_=self.gru(z)
        w=torch.softmax(self.attention(z).squeeze(-1),1).unsqueeze(-1)
        return self.head((z*w).sum(1))


def class_balanced_focal(logits,target,class_counts,beta:float=.999,gamma:float=2.0):
    counts=torch.as_tensor(class_counts,dtype=logits.dtype,device=logits.device)
    if counts.ndim!=1 or len(counts)!=logits.shape[1] or torch.any(counts<=0):
        raise ValueError("CLASS_BALANCED_FOCAL_COUNTS")
    eff=1-torch.pow(torch.as_tensor(beta,dtype=logits.dtype,device=logits.device),counts)
    weight=(1-beta)/torch.clamp(eff,min=1e-8);weight=weight/weight.sum()*len(counts)
    plain=torch.nn.functional.cross_entropy(logits,target,reduction="none")
    weighted=torch.nn.functional.cross_entropy(logits,target,reduction="none",weight=weight)
    pt=torch.exp(-plain)
    return (((1-pt)**gamma)*weighted).mean()


def parameter_count(model:nn.Module)->int:
    return sum(p.numel() for p in model.parameters())


def _positive_int(p:dict,name:str,lo:int,hi:int)->int:
    v=p.get(name)
    if type(v) is not int or not lo<=v<=hi:raise ValueError("LITERATURE_BASELINE_PARAMETER:"+name)
    return v


def _exact(p:dict,keys:set[str]):
    if set(p)!=keys:raise ValueError("LITERATURE_BASELINE_PARAMETER_SCHEMA")


def build_grid3d_res(shape,p):
    _exact(p,{"width"});return Grid3DResidualClassifier(shape.channels,shape.classes,_positive_int(p,"width",4,64))

def grid3d_res_count(shape,p):
    return parameter_count(build_grid3d_res(shape,p))


def build_transformer_ts(shape,p):
    _exact(p,{"d_model","heads","layers"})
    d=_positive_int(p,"d_model",16,256);h=_positive_int(p,"heads",1,16);layers=_positive_int(p,"layers",1,6)
    if d%h:raise ValueError("LITERATURE_TRANSFORMER_HEAD_DIVISIBILITY")
    return TransformerClassifier(shape.channels,shape.classes,d,h,layers)

def transformer_ts_count(shape,p):
    return parameter_count(build_transformer_ts(shape,p))


def build_multiscale_conv_transformer(shape,p):
    _exact(p,{"d_model"});return MultiScaleConvTransformer(shape.channels,shape.classes,_positive_int(p,"d_model",16,256))

def multiscale_conv_transformer_count(shape,p):
    return parameter_count(build_multiscale_conv_transformer(shape,p))


def build_dynamic_graph_gru(shape,p):
    _exact(p,{"d_model"});return DynamicGraphGRUClassifier(shape.channels,shape.classes,_positive_int(p,"d_model",8,128))

def dynamic_graph_gru_count(shape,p):
    return parameter_count(build_dynamic_graph_gru(shape,p))
