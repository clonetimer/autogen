"""Contract-aware resource planning for fixed neural controls.

A fixed control is a *family*, not a promise that one width/d_model fits every
input contract.  This planner deterministically shrinks only reviewed monotone
capacity knobs before training so the strongest feasible member competes under
exactly the same frozen parameter/ensemble/MAC budgets as generated DAGs.

The preferred policy spec is never overwritten: callers persist both preferred
and effective specs plus the static estimates that caused adaptation.
"""
from __future__ import annotations
import copy
import math
import time
import torch
from torch import nn
from .models import Shape
from .graph import compositional_registry


def _grid_cells(channels:int)->int:
    h=max(2,int(math.floor(math.sqrt(channels))))
    w=int(math.ceil(channels/h))
    while h*w<channels:w+=1
    return h*w


def fixed_control_mac_per_case(shape:Shape,spec:dict)->int|None:
    """Deterministic multiply-accumulate estimate for reviewed fixed controls.

    Estimates intentionally count dominant linear/conv/attention matmuls only;
    activations, normalisation and softmax are excluded, matching the generated
    graph audit's role as a *static feasibility* proxy rather than wall-clock
    certification.  Runtime max_trial_seconds remains the empirical backstop.
    """
    fam=spec.get('family'); p=spec.get('parameters',{})
    t,c,k=shape.window,shape.channels,shape.classes
    if fam=='grid3d_res':
        w=int(p['width']); cells=_grid_cells(c); t2=(t+1)//2
        stem=t*cells*w*27
        res1=2*t*cells*w*w*27
        down=t2*cells*(2*w)*w*27
        res2=2*t2*cells*(2*w)*(2*w)*27
        head=2*w*k
        return int(stem+res1+down+res2+head)
    if fam=='multiscale_conv_transformer':
        d=int(p['d_model'])
        temporal=t*24*c*(3+5+7)
        mix=t*72*d
        # 2 encoder layers: QKV+out + FFN + attention score/value products.
        encoder=2*(8*t*d*d + 2*t*t*d)
        head=d*k
        return int(temporal+mix+encoder+head)
    if fam=='transformer_ts':
        d=int(p['d_model']); layers=int(p['layers'])
        proj=t*c*d
        encoder=layers*(8*t*d*d + 2*t*t*d)
        head=d*k
        return int(proj+encoder+head)
    if fam=='dynamic_graph_gru':
        d=int(p['d_model'])
        # Dynamic adjacency + value/message projections + GRU (hidden=2d).
        graph=c*c*d + t*c*d + t*c*(2*d)*d
        h=2*d
        gru=t*(3*d*h + 3*h*h)
        head=h*k
        return int(graph+gru+head)
    if fam=='mlp':
        # Parameter count is a conservative MAC proxy for one forward pass.
        return None
    return None


def _resource_record(shape:Shape,spec:dict,budget:dict,seeds:int)->dict:
    reg=compositional_registry(); desc=reg.describe(spec,shape)
    mac=fixed_control_mac_per_case(shape,spec)
    reasons=[]
    if desc['parameter_count']>budget['max_parameters_per_model']:
        reasons.append('PARAMETER_BUDGET_EXCEEDED')
    if desc['parameter_count']*seeds>budget['max_ensemble_parameters']:
        reasons.append('ENSEMBLE_PARAMETER_BUDGET_EXCEEDED')
    cap=budget.get('max_multiply_accumulate_per_case')
    if cap is not None and mac is not None and mac>cap:
        reasons.append('MAC_BUDGET_EXCEEDED')
    return {'description':desc,'multiply_accumulate_estimate_per_case':mac,
            'feasible':not reasons,'reasons':reasons}


def _variants(spec:dict):
    """Yield strongest-to-smallest reviewed variants, preferred first."""
    fam=spec['family']; p=copy.deepcopy(spec['parameters'])
    if fam=='grid3d_res':
        for v in range(int(p['width']),3,-1):
            q=copy.deepcopy(p);q['width']=v;yield {'family':fam,'parameters':q}
        return
    if fam=='multiscale_conv_transformer':
        start=int(p['d_model']);start-=start%4
        for v in range(start,15,-4):
            q=copy.deepcopy(p);q['d_model']=v;yield {'family':fam,'parameters':q}
        return
    if fam=='transformer_ts':
        heads=int(p['heads']);start=int(p['d_model']);start-=start%heads
        floor=max(16,heads)
        for v in range(start,floor-1,-heads):
            q=copy.deepcopy(p);q['d_model']=v;yield {'family':fam,'parameters':q}
        return
    if fam=='dynamic_graph_gru':
        for v in range(int(p['d_model']),7,-1):
            q=copy.deepcopy(p);q['d_model']=v;yield {'family':fam,'parameters':q}
        return
    if fam=='mlp' and isinstance(p.get('widths'),list) and len(p['widths'])==1:
        for v in range(int(p['widths'][0]),1,-1):
            q=copy.deepcopy(p);q['widths']=[v];yield {'family':fam,'parameters':q}
        return
    yield copy.deepcopy(spec)


def plan_fixed_control(shape:Shape,preferred:dict,budget:dict,seed_count:int)->dict:
    """Return the strongest feasible reviewed variant or fail closed."""
    attempts=[]; seen=set()
    for candidate in _variants(preferred):
        key=repr(candidate)
        if key in seen:continue
        seen.add(key)
        rec=_resource_record(shape,candidate,budget,seed_count)
        attempts.append({'spec':candidate,**rec})
        if rec['feasible']:
            return {'schema':'phm.fixed-control-resource-plan.v1',
                    'preferred_spec':copy.deepcopy(preferred),'effective_spec':candidate,
                    'adapted':candidate!=preferred,'attempt_count':len(attempts),
                    'effective_resources':rec,'attempts':attempts}
    return {'schema':'phm.fixed-control-resource-plan.v1',
            'preferred_spec':copy.deepcopy(preferred),'effective_spec':None,
            'adapted':False,'attempt_count':len(attempts),'effective_resources':None,
            'attempts':attempts,'status':'NO_FEASIBLE_REVIEWED_VARIANT'}


def plan_fixed_controls(shape:Shape,controls:list[dict],budget:dict,seed_count:int)->list[dict]:
    plans=[plan_fixed_control(shape,c,budget,seed_count) for c in controls]
    if any(p['effective_spec'] is None for p in plans):
        raise ValueError('NO_FEASIBLE_FIXED_CONTROL_VARIANT')
    effective=[p['effective_spec'] for p in plans]
    if len({repr(x) for x in effective})!=len(effective):
        raise ValueError('ADAPTED_FIXED_CONTROL_COLLISION')
    return plans


RUNTIME_PREFLIGHT_SCHEMA = "phm.fixed-control-runtime-preflight.v1"
RUNTIME_PREFLIGHT_BUDGET_FRACTION = 0.85


def _project_runtime_seconds(train_step_seconds: float, validation_seconds: float,
                             training: dict, train_cases: int) -> float:
    """Project one full trial from a measured training step and full validation pass."""
    batches = int(math.ceil(train_cases / int(training['batch_size'])))
    return float((train_step_seconds * batches + validation_seconds) * int(training['epochs']))


def _measure_runtime(shape: Shape, spec: dict, training: dict, budget: dict,
                     validation_cases: int, *, seed: int = 0,
                     warmup_steps: int = 2, measured_steps: int = 5) -> dict:
    """Short CPU-only benchmark matching the real model/build/optimizer path.

    The benchmark is deliberately label/data-value agnostic: zeros with the real
    batch/window/channel shape are sufficient to measure operator wall time.
    The persisted result is an environment-specific resource artifact, not a
    portable architecture property.
    """
    reg = compositional_registry()
    model = reg.build(spec, shape, seed, int(budget['max_parameters_per_model']))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training['learning_rate']),
                                  weight_decay=float(training['weight_decay']))
    criterion = nn.CrossEntropyLoss()
    batch = int(training['batch_size'])
    x = torch.zeros(batch, shape.window, shape.channels, dtype=torch.float32)
    y = torch.zeros(batch, dtype=torch.long)
    vx = torch.zeros(int(validation_cases), shape.window, shape.channels, dtype=torch.float32)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for _ in range(warmup_steps):
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward(); optimizer.step()
        step_times=[]
        for _ in range(measured_steps):
            start=time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss=criterion(model(x),y)
            loss.backward(); optimizer.step()
            step_times.append(time.perf_counter()-start)
        model.eval(); validation_times=[]
        with torch.inference_mode():
            for _ in range(3):
                start=time.perf_counter(); model(vx); validation_times.append(time.perf_counter()-start)
    finally:
        torch.set_num_threads(previous_threads)
    return {
        'schema': RUNTIME_PREFLIGHT_SCHEMA,
        'seed': seed,
        'warmup_steps': warmup_steps,
        'measured_steps': measured_steps,
        'train_step_seconds': step_times,
        'validation_seconds': validation_times,
        'conservative_train_step_seconds': max(step_times),
        'conservative_validation_seconds': max(validation_times),
    }


def plan_fixed_control_runtime(shape: Shape, preferred: dict, budget: dict, seed_count: int,
                               training: dict, train_cases: int, validation_cases: int,
                               *, measure_fn=None,
                               budget_fraction: float = RUNTIME_PREFLIGHT_BUDGET_FRACTION) -> dict:
    """Select strongest statically feasible variant that also passes runtime preflight."""
    if not 0 < float(budget_fraction) <= 1:
        raise ValueError('RUNTIME_PREFLIGHT_BUDGET_FRACTION')
    measure_fn = measure_fn or _measure_runtime
    attempts=[]; seen=set(); limit=float(budget['max_trial_seconds'])*float(budget_fraction)
    for candidate in _variants(preferred):
        key=repr(candidate)
        if key in seen: continue
        seen.add(key)
        static=_resource_record(shape,candidate,budget,seed_count)
        rec={'spec':candidate,**static}
        if not static['feasible']:
            rec['runtime_preflight']=None; attempts.append(rec); continue
        measured=measure_fn(shape,candidate,training,budget,validation_cases)
        projected=_project_runtime_seconds(
            float(measured['conservative_train_step_seconds']),
            float(measured['conservative_validation_seconds']), training, train_cases)
        runtime={**measured,'train_cases':int(train_cases),'validation_cases':int(validation_cases),
                 'projected_trial_seconds':projected,'allowed_projected_seconds':limit,
                 'budget_fraction':float(budget_fraction),'runtime_feasible':projected<=limit}
        rec['runtime_preflight']=runtime; attempts.append(rec)
        if runtime['runtime_feasible']:
            return {'schema':'phm.fixed-control-resource-plan.v2','preferred_spec':copy.deepcopy(preferred),
                    'effective_spec':candidate,'adapted':candidate!=preferred,'attempt_count':len(attempts),
                    'effective_resources':static,'runtime_preflight':runtime,'attempts':attempts}
    return {'schema':'phm.fixed-control-resource-plan.v2','preferred_spec':copy.deepcopy(preferred),
            'effective_spec':None,'adapted':False,'attempt_count':len(attempts),'effective_resources':None,
            'runtime_preflight':None,'attempts':attempts,'status':'NO_FEASIBLE_RUNTIME_REVIEWED_VARIANT'}


def plan_fixed_controls_runtime(shape: Shape, controls: list[dict], budget: dict, seed_count: int,
                                training: dict, train_cases: int, validation_cases: int,
                                *, measure_fn=None,
                                budget_fraction: float = RUNTIME_PREFLIGHT_BUDGET_FRACTION) -> list[dict]:
    plans=[plan_fixed_control_runtime(shape,c,budget,seed_count,training,train_cases,validation_cases,
                                      measure_fn=measure_fn,budget_fraction=budget_fraction) for c in controls]
    if any(p['effective_spec'] is None for p in plans):
        raise ValueError('NO_FEASIBLE_RUNTIME_FIXED_CONTROL_VARIANT')
    effective=[p['effective_spec'] for p in plans]
    if len({repr(x) for x in effective})!=len(effective):
        raise ValueError('ADAPTED_FIXED_CONTROL_COLLISION')
    return plans
