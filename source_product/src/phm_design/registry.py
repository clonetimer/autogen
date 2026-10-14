"""Combined registry; v0.3 graph implementation stays unchanged for replay."""
from pathlib import Path
from .graph import compositional_registry
from .models import Registry
from .io import sha_file
from . import physics, graph
from . import reference
import torch

class MechanismRegistry(Registry):
    def describe(self,spec,shape):
        result=super().describe(spec,shape)
        if spec['family']=='mechanism_dag':
            result['transitive_source_sha256']={
                'physics.py':sha_file(Path(physics.__file__)),
                'graph.py':sha_file(Path(graph.__file__)),
            }
        if spec['family']=='reference_dag':
            ref=spec['parameters']['reference']
            result['transitive_source_sha256']={'reference.py':sha_file(Path(reference.__file__)),
                'graph.py':sha_file(Path(graph.__file__))}
            result['frozen_reference_scalars']=sum(torch.tensor(ref[k]).numel() for k in ('mean','scale','coef','intercept'))
            result['frozen_reference_bytes']=8*result['frozen_reference_scalars']
            result['parameter_budget_scope']='trainable DAG parameters plus gate; frozen reference bytes reported separately'
        return result

    def build(self,spec,shape,seed,max_parameters):
        if spec['family']!='reference_dag':return super().build(spec,shape,seed,max_parameters)
        from .models import integer
        integer(seed,'seed',0,2**32-1)
        desc=self.describe(spec,shape)
        if desc['parameter_count']>max_parameters:raise ValueError('PARAMETER_BUDGET_EXCEEDED')
        with torch.random.fork_rng(devices=[]),torch.device('cpu'):
            torch.set_rng_state(torch.Generator(device='cpu').manual_seed(seed).get_state())
            model=reference.build_residual(shape,spec['parameters']).cpu()
            if sum(p.numel() for p in model.parameters())!=desc['parameter_count']:raise ValueError('PARAMETER_ESTIMATE_MISMATCH')
            model.eval()
            with torch.inference_mode():y=model(torch.zeros(2,shape.window,shape.channels))
            if tuple(y.shape)!=(2,shape.classes) or not torch.isfinite(y).all():raise ValueError('INVALID_NETWORK_OUTPUT')
        return model.train()

def default_registry():
    original=compositional_registry()
    registry=MechanismRegistry()
    registry._families.update(original._families)
    registry.register('mechanism_dag','1',physics.build_mechanism,physics.mechanism_count)
    registry.register('reference_dag','1',reference.build_residual,reference.residual_count)
    return registry
