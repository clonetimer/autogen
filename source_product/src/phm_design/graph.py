"""Typed, compositional network IR and compiler (offline sequence classification).

Graphs are data, never executable Python. Approved primitives compose arbitrary
acyclic fan-out/fan-in networks, with optional explicit parameter sharing.
Differences operate on normalized observations; they are NOT physical residuals.
"""
from __future__ import annotations
from dataclasses import dataclass
import re
from typing import Any
import torch
from torch import nn
from torch.nn import functional as F
from .io import digest
from .models import Shape, integer, default_registry as fixed_registry

SCHEMA = "phm.typed-dag.v1"
PARAMETRIC = {"linear", "conv", "res_tcn", "grid_res3d", "grid_st_res", "self_attention", "norm", "gated_add"}
UNARY = {"select", "difference", "linear", "conv", "res_tcn", "grid_res3d", "grid_st_res", "self_attention", "norm", "relu", "silu", "tanh", "mean", "last", "tail_mean", "statistics"}
OPS = UNARY | {"concat", "add", "gated_add"}

@dataclass(frozen=True)
class TensorType:
    layout: str  # sequence: B,T,C; vector: B,C
    width: int
    steps: int

    def record(self) -> dict:
        return {"layout": self.layout, "width": self.width, "steps": self.steps}


def _keys(value: Any, expected: set[str], reason: str) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(reason)


def _infer(op: str, args: dict, inputs: list[TensorType]) -> tuple[TensorType, int, int]:
    if op not in OPS:
        raise ValueError("UNREGISTERED_GRAPH_PRIMITIVE")
    if (op in UNARY and len(inputs) != 1) or (op in {"add", "gated_add"} and len(inputs) != 2) or (op == "concat" and not 2 <= len(inputs) <= 16):
        raise ValueError("GRAPH_ARITY_MISMATCH")
    first = inputs[0]
    if op == "select":
        _keys(args, {"indices"}, "SELECT_ARGUMENTS")
        idx = args["indices"]
        if first.layout != "sequence" or not isinstance(idx, list) or not idx or len(set(idx)) != len(idx):
            raise ValueError("INVALID_CHANNEL_SELECTION")
        for i in idx:
            integer(i, "channel_index", 0, first.width - 1)
        return TensorType("sequence", len(idx), first.steps), 0, 0
    if op == "linear":
        _keys(args, {"width"}, "LINEAR_ARGUMENTS")
        width = integer(args["width"], "width", 1, 512)
        return TensorType(first.layout, width, first.steps), first.width * width + width, first.steps * first.width * width
    if op == "conv":
        _keys(args, {"width", "kernel", "dilation"}, "CONV_ARGUMENTS")
        width = integer(args["width"], "width", 1, 512)
        kernel = integer(args["kernel"], "kernel", 1, 9)
        dilation = integer(args["dilation"], "dilation", 1, 8)
        if first.layout != "sequence" or kernel % 2 != 1:
            raise ValueError("CONV_REQUIRES_SEQUENCE_AND_ODD_KERNEL")
        return TensorType("sequence", width, first.steps), first.width * width * kernel + width, first.steps * first.width * width * kernel
    if op == "res_tcn":
        _keys(args, {"width", "kernel", "dilation"}, "RES_TCN_ARGUMENTS")
        width = integer(args["width"], "width", 1, 512)
        kernel = integer(args["kernel"], "kernel", 1, 9)
        dilation = integer(args["dilation"], "dilation", 1, 8)
        if first.layout != "sequence" or kernel % 2 != 1:
            raise ValueError("RES_TCN_REQUIRES_SEQUENCE_AND_ODD_KERNEL")
        projection = (first.width * width + width) if first.width != width else 0
        conv1 = first.width * width * kernel + width
        conv2 = width * width * kernel + width
        norm = 2 * width
        macs = first.steps * (first.width * width * kernel + width * width * kernel + (first.width * width if first.width != width else 0))
        return TensorType("sequence", width, first.steps), projection + conv1 + conv2 + norm, macs
    if op == "grid_res3d":
        _keys(args, {"width", "kernel", "dilation"}, "GRID_RES3D_ARGUMENTS")
        width = integer(args["width"], "width", 8, 64)
        kernel = integer(args["kernel"], "kernel", 1, 9)
        dilation = integer(args["dilation"], "dilation", 1, 8)
        if first.layout != "sequence" or first.width < 4 or width % 2 or kernel % 2 != 1:
            raise ValueError("GRID_RES3D_REQUIRES_SEQUENCE_CHANNEL_GRID_EVEN_WIDTH_ODD_KERNEL")
        base = width // 2
        # Conv3d stages use a near-square channel grid while retaining the
        # temporal axis. BatchNorm affine parameters are included exactly.
        conv = lambda ci, co: ci * co * kernel * 9 + co
        count = (conv(1, base) + 2 * base +
                 2 * (conv(base, base) + 2 * base) +
                 conv(base, width) + 2 * width +
                 2 * (conv(width, width) + 2 * width))
        import math
        h=max(2,int(math.floor(math.sqrt(first.width))));w=int(math.ceil(first.width/h))
        while h*w<first.width:w+=1
        spatial=h*w
        macs = first.steps * spatial * kernel * 9 * (base + 2*base*base + base*width + 2*width*width)
        return TensorType("sequence", width, first.steps), count, macs
    if op == "grid_st_res":
        _keys(args, {"width", "kernel", "dilation"}, "GRID_ST_RES_ARGUMENTS")
        width = integer(args["width"], "width", 8, 64)
        kernel = integer(args["kernel"], "kernel", 1, 9)
        dilation = integer(args["dilation"], "dilation", 1, 8)
        if first.layout != "sequence" or first.width < 4 or width % 2 or kernel % 2 != 1:
            raise ValueError("GRID_ST_RES_REQUIRES_SEQUENCE_CHANNEL_GRID_EVEN_WIDTH_ODD_KERNEL")
        base = width // 2
        # Factorized spatial-grid + temporal residual encoder. 2-D spatial
        # convolutions run independently per time sample; temporal mixing is
        # depthwise-separable after spatial pooling. This preserves B,T,W while
        # avoiding the high CPU overhead of stacked full Conv3d kernels.
        conv2 = lambda ci, co: ci * co * 9 + co
        dw1 = lambda c: c * kernel + c
        pw1 = lambda ci, co: ci * co + co
        count = (conv2(1, base) + 2 * base +
                 2 * (conv2(base, base) + 2 * base) +
                 dw1(base) + 2 * base + pw1(base, width) + 2 * width +
                 dw1(width) + 2 * width + pw1(width, width) + 2 * width)
        import math
        h=max(2,int(math.floor(math.sqrt(first.width))));w=int(math.ceil(first.width/h))
        while h*w<first.width:w+=1
        spatial=h*w
        macs = first.steps * (spatial * (9*base + 18*base*base) +
                              base*kernel + base*width + width*kernel + width*width)
        return TensorType("sequence", width, first.steps), count, macs
    if op == "self_attention":
        _keys(args, {"width", "heads"}, "SELF_ATTENTION_ARGUMENTS")
        width = integer(args["width"], "width", 2, 512)
        heads = integer(args["heads"], "heads", 1, 16)
        if first.layout != "sequence" or width % heads:
            raise ValueError("SELF_ATTENTION_SEQUENCE_OR_HEAD_DIVISIBILITY")
        projection = (first.width * width + width) if first.width != width else 0
        # PyTorch MHA: in_proj 3W*W + 3W, out_proj W*W + W; plus LayerNorm.
        mha = 4 * width * width + 4 * width
        norm = 2 * width
        macs = first.steps * (projection + 4 * width * width) + 2 * first.steps * first.steps * width
        return TensorType("sequence", width, first.steps), projection + mha + norm, macs
    if op == "tail_mean":
        _keys(args, {"window"}, "TAIL_MEAN_ARGUMENTS")
        window = integer(args["window"], "tail_mean_window", 1, first.steps)
        if first.layout != "sequence":
            raise ValueError("TAIL_MEAN_REQUIRES_SEQUENCE")
        return TensorType("vector", first.width, 1), 0, first.width * window
    _keys(args, set(), "UNEXPECTED_PRIMITIVE_ARGUMENTS")
    if op in {"relu", "silu", "tanh", "difference", "norm"}:
        if op == "difference" and first.layout != "sequence":
            raise ValueError("DIFFERENCE_REQUIRES_SEQUENCE")
        return first, 2 * first.width if op == "norm" else 0, 0
    if op in {"mean", "last", "tail_mean", "statistics"}:
        if first.layout != "sequence":
            raise ValueError("POOL_REQUIRES_SEQUENCE")
        return TensorType("vector", first.width * (7 if op == "statistics" else 1), 1), 0, 0
    if any(x.layout != first.layout or x.steps != first.steps for x in inputs):
        raise ValueError("FUSION_LAYOUT_MISMATCH")
    if op == "concat":
        return TensorType(first.layout, sum(x.width for x in inputs), first.steps), 0, 0
    if any(x != first for x in inputs):
        raise ValueError("ELEMENTWISE_FUSION_SHAPE_MISMATCH")
    if op == "gated_add":
        return first, 2 * first.width * first.width + first.width, first.steps * 2 * first.width * first.width
    return first, 0, 0


def analyze_graph(shape: Shape, graph: dict) -> dict:
    _keys(graph, {"schema", "nodes", "output"}, "GRAPH_SCHEMA")
    if graph["schema"] != SCHEMA or not isinstance(graph["nodes"], list) or not 1 <= len(graph["nodes"]) <= 96:
        raise ValueError("GRAPH_NODE_BUDGET_OR_SCHEMA")
    types = {"input": TensorType("sequence", shape.channels, shape.window)}
    descriptions, parents, modules = [], {}, {}
    params = macs = edges = elements = 0
    shared = {}
    for node in graph["nodes"]:
        _keys(node, {"id", "op", "inputs", "args", "share"}, "GRAPH_NODE_SCHEMA")
        name = node["id"]
        if not isinstance(name, str) or not re.fullmatch(r"n[0-9]+", name) or name in types:
            raise ValueError("GRAPH_NODE_ID")
        refs = node["inputs"]
        if not isinstance(refs, list) or not refs or any(not isinstance(ref, str) or ref not in types for ref in refs):
            raise ValueError("GRAPH_FORWARD_REFERENCE_OR_CYCLE")
        if len(set(refs)) != len(refs):
            raise ValueError("DUPLICATE_NODE_INPUT")
        if not isinstance(node["op"], str) or not isinstance(node["args"], dict):
            raise ValueError("GRAPH_PRIMITIVE_SCHEMA")
        output_type, count, node_macs = _infer(node["op"], node["args"], [types[ref] for ref in refs])
        if output_type.width > 4096 or output_type.steps * output_type.width > 1_048_576:
            raise ValueError("GRAPH_TENSOR_BUDGET")
        share = node["share"]
        if not isinstance(share, str) or (share and (not re.fullmatch(r"s[0-9]+", share) or node["op"] not in PARAMETRIC)):
            raise ValueError("INVALID_PARAMETER_SHARING")
        module_key = f"shared_{share}" if share else f"node_{name}"
        signature = {"op": node["op"], "args": node["args"], "types": [types[ref].record() for ref in refs]}
        if module_key in shared and shared[module_key] != signature:
            raise ValueError("INCOMPATIBLE_SHARED_OPERATOR")
        if module_key not in shared:
            params += count
        shared[module_key] = signature
        types[name] = output_type
        parents[name] = refs
        edges += len(refs)
        macs += node_macs
        elements += output_type.steps * output_type.width
        descriptions.append({**node, "type": output_type.record(), "operator_parameters": count, "module_key": module_key})
        modules[module_key] = {"op": node["op"], "args": node["args"], "input_type": types[refs[0]].record()}
    output = graph["output"]
    if not isinstance(output, str) or output not in parents or types[output] != TensorType("vector", shape.classes, 1):
        raise ValueError("GRAPH_CLASSIFICATION_OUTPUT_CONTRACT")
    visited = set()
    stack = [output]
    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        stack.extend(parents.get(current, []))
    if visited != set(types):
        raise ValueError("GRAPH_DEAD_NODES_OR_UNUSED_INPUT")
    if edges > 256 or not 1 <= params <= 2_000_000:
        raise ValueError("GRAPH_RESOURCE_LIMIT")
    return {"nodes": descriptions, "modules": modules, "parameter_count": params,
            "node_count": len(descriptions), "edge_count": edges,
            "shared_parameter_groups": len({n["share"] for n in graph["nodes"] if n["share"]}),
            "multiply_accumulate_estimate_per_case": macs,
            "forward_node_output_bytes_fp32_per_case": 4 * elements,
            "memory_estimate_excludes": ["input", "operator_workspace", "autograd", "optimizer", "batch", "framework_overhead"],
            "offline_window_classifier": True, "physical_equations_inferred": False}


def graph_identity(graph: dict, *, topology_only: bool = False) -> str:
    """Normalize node/share names; topology hash discards widths/kernel values."""
    names, shares, nodes = {"input": "input"}, {}, []
    for i, node in enumerate(graph["nodes"]):
        names[node["id"]] = f"n{i}"
        share = node["share"]
        if share and share not in shares:
            shares[share] = f"s{len(shares)}"
        args = node["args"] if not topology_only else ({"indices": node["args"]["indices"]} if node["op"] == "select" else {})
        nodes.append({"id": names[node["id"]], "op": node["op"], "args": args,
                      "inputs": [names[r] for r in node["inputs"]], "share": shares.get(share, "")})
    return digest({"nodes": nodes, "output": names[graph["output"]]})


class CausalConvolution(nn.Module):
    def __init__(self, channels: int, args: dict):
        super().__init__()
        self.left = (args["kernel"] - 1) * args["dilation"]
        self.conv = nn.Conv1d(channels, args["width"], args["kernel"], dilation=args["dilation"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x.transpose(1, 2), (self.left, 0))).transpose(1, 2)


class ResidualTCN(nn.Module):
    def __init__(self, channels: int, args: dict):
        super().__init__()
        width,kernel,dilation=args["width"],args["kernel"],args["dilation"]
        self.left=(kernel-1)*dilation
        self.proj=nn.Identity() if channels==width else nn.Linear(channels,width)
        self.conv1=nn.Conv1d(channels,width,kernel,dilation=dilation)
        self.conv2=nn.Conv1d(width,width,kernel,dilation=dilation)
        self.norm=nn.LayerNorm(width)
    def forward(self,x):
        skip=self.proj(x)
        y=self.conv1(F.pad(x.transpose(1,2),(self.left,0))).transpose(1,2)
        y=F.gelu(y)
        y=self.conv2(F.pad(y.transpose(1,2),(self.left,0))).transpose(1,2)
        return F.gelu(self.norm(skip+y))


class GridResidual3DSequence(nn.Module):
    """Near-square channel-grid residual 3D encoder that returns B,T,W.

    This is an approved topology primitive for offline fixed-window classifiers.
    It uses symmetric temporal padding (not a causal streaming operator), keeps
    the full time axis, and averages only over the channel-grid spatial axes.
    """
    def __init__(self, channels:int,args:dict):
        super().__init__()
        import math
        width,kernel,dilation=args["width"],args["kernel"],args["dilation"]
        base=width//2
        h=max(2,int(math.floor(math.sqrt(channels))));w=int(math.ceil(channels/h))
        while h*w<channels:w+=1
        self.channels=channels;self.grid=(h,w)
        pad_t=((kernel-1)//2)*dilation
        conv_args=dict(kernel_size=(kernel,3,3),padding=(pad_t,1,1),dilation=(dilation,1,1))
        self.stem=nn.Sequential(nn.Conv3d(1,base,**conv_args),nn.BatchNorm3d(base),nn.GELU())
        self.res1=nn.Sequential(nn.Conv3d(base,base,**conv_args),nn.BatchNorm3d(base),nn.GELU(),
                                nn.Conv3d(base,base,**conv_args),nn.BatchNorm3d(base))
        self.expand=nn.Sequential(nn.Conv3d(base,width,**conv_args),nn.BatchNorm3d(width),nn.GELU())
        self.res2=nn.Sequential(nn.Conv3d(width,width,**conv_args),nn.BatchNorm3d(width),nn.GELU(),
                                nn.Conv3d(width,width,**conv_args),nn.BatchNorm3d(width))
    def forward(self,x):
        if x.ndim!=3 or x.shape[-1]!=self.channels:raise ValueError("GRID_RES3D_INPUT_SHAPE")
        b,t,c=x.shape;h,w=self.grid
        if h*w>c:x=F.pad(x,(0,h*w-c))
        z=x.reshape(b,t,h,w).unsqueeze(1)
        z=self.stem(z);z=F.gelu(z+self.res1(z))
        z=self.expand(z);z=F.gelu(z+self.res2(z))
        return z.mean(dim=(-1,-2)).transpose(1,2)


class GridFactorizedSTSequence(nn.Module):
    """Factorized channel-grid spatial encoder + temporal residual mixer.

    This is an offline fixed-window primitive. It preserves the time axis and
    uses the same near-square channel grid as ``grid_res3d``, but replaces the
    stack of full Conv3d layers with per-time Conv2d spatial encoding followed
    by depthwise-separable temporal Conv1d mixing. The goal is predictable CPU
    feasibility without removing grid-aware inductive bias.
    """
    def __init__(self, channels:int,args:dict):
        super().__init__()
        import math
        width,kernel,dilation=args["width"],args["kernel"],args["dilation"]
        base=width//2
        h=max(2,int(math.floor(math.sqrt(channels))));w=int(math.ceil(channels/h))
        while h*w<channels:w+=1
        self.channels=channels;self.grid=(h,w)
        self.spatial_stem=nn.Sequential(
            nn.Conv2d(1,base,3,padding=1),nn.BatchNorm2d(base),nn.GELU())
        self.spatial_res=nn.Sequential(
            nn.Conv2d(base,base,3,padding=1),nn.BatchNorm2d(base),nn.GELU(),
            nn.Conv2d(base,base,3,padding=1),nn.BatchNorm2d(base))
        pad_t=((kernel-1)//2)*dilation
        self.temporal_in=nn.Sequential(
            nn.Conv1d(base,base,kernel,padding=pad_t,dilation=dilation,groups=base),
            nn.BatchNorm1d(base),nn.GELU(),
            nn.Conv1d(base,width,1),nn.BatchNorm1d(width),nn.GELU())
        self.temporal_res=nn.Sequential(
            nn.Conv1d(width,width,kernel,padding=pad_t,dilation=dilation,groups=width),
            nn.BatchNorm1d(width),nn.GELU(),
            nn.Conv1d(width,width,1),nn.BatchNorm1d(width))

    def forward(self,x):
        if x.ndim!=3 or x.shape[-1]!=self.channels:
            raise ValueError("GRID_ST_RES_INPUT_SHAPE")
        b,t,c=x.shape;h,w=self.grid
        if h*w>c:x=F.pad(x,(0,h*w-c))
        z=x.reshape(b*t,1,h,w)
        z=self.spatial_stem(z);z=F.gelu(z+self.spatial_res(z))
        z=z.mean(dim=(-1,-2)).reshape(b,t,-1).transpose(1,2)
        z=self.temporal_in(z);z=F.gelu(z+self.temporal_res(z))
        return z.transpose(1,2)


class TemporalSelfAttention(nn.Module):
    def __init__(self, channels:int,args:dict):
        super().__init__()
        width=args["width"]
        self.proj=nn.Identity() if channels==width else nn.Linear(channels,width)
        self.attn=nn.MultiheadAttention(width,args["heads"],dropout=0.1,batch_first=True)
        self.norm=nn.LayerNorm(width)
    def forward(self,x):
        z=self.proj(x)
        y,_=self.attn(z,z,z,need_weights=False)
        return self.norm(z+y)


class Gate(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.gate = nn.Linear(2 * width, width)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        weight = torch.sigmoid(self.gate(torch.cat((a, b), dim=-1)))
        return weight * a + (1 - weight) * b


class Stateless(nn.Module):
    def __init__(self, op: str, args: dict):
        super().__init__()
        self.op, self.args = op, args

    def forward(self, *xs: torch.Tensor) -> torch.Tensor:
        x = xs[0]
        if self.op == "select":
            return x[..., self.args["indices"]]
        if self.op == "difference":
            return torch.cat((torch.zeros_like(x[:, :1]), x[:, 1:] - x[:, :-1]), dim=1)
        if self.op == "mean":
            return x.mean(1)
        if self.op == "last":
            return x[:, -1]
        if self.op == "tail_mean":
            return x[:, -self.args["window"]:].mean(1)
        if self.op == "statistics":
            return torch.cat((x.mean(1), x.std(1, correction=0), x.amin(1), x.amax(1),
                              x[:, -1], x[:, -1] - x[:, 0], (x.square().mean(1) + 1e-12).sqrt()), dim=1)
        if self.op == "concat":
            return torch.cat(xs, dim=-1)
        if self.op == "add":
            return xs[0] + xs[1]
        raise RuntimeError("UNCOMPILED_STATELESS_OPERATOR")


class GraphNetwork(nn.Module):
    def __init__(self, shape: Shape, graph: dict):
        super().__init__()
        audit = analyze_graph(shape, graph)
        self.shape = shape
        self.output = graph["output"]
        self.execution = [(n["id"], n["module_key"], n["inputs"]) for n in audit["nodes"]]
        modules = {}
        for key, meta in audit["modules"].items():
            op, args, channels = meta["op"], meta["args"], meta["input_type"]["width"]
            if op == "linear":
                module = nn.Linear(channels, args["width"])
            elif op == "conv":
                module = CausalConvolution(channels, args)
            elif op == "res_tcn":
                module = ResidualTCN(channels, args)
            elif op == "grid_res3d":
                module = GridResidual3DSequence(channels, args)
            elif op == "grid_st_res":
                module = GridFactorizedSTSequence(channels, args)
            elif op == "self_attention":
                module = TemporalSelfAttention(channels, args)
            elif op == "norm":
                module = nn.LayerNorm(channels)
            elif op == "gated_add":
                module = Gate(channels)
            elif op in {"relu", "silu", "tanh"}:
                module = {"relu": nn.ReLU, "silu": nn.SiLU, "tanh": nn.Tanh}[op]()
            else:
                module = Stateless(op, args)
            modules[key] = module
        self.operators = nn.ModuleDict(modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != self.shape.window or x.shape[2] != self.shape.channels:
            raise ValueError("GRAPH_INPUT_SHAPE_MISMATCH")
        values = {"input": x}
        for name, module, sources in self.execution:
            values[name] = self.operators[module](*(values[s] for s in sources))
        return values[self.output]


def graph_count(shape: Shape, parameters: dict) -> int:
    return analyze_graph(shape, parameters)["parameter_count"]


def build_graph(shape: Shape, parameters: dict) -> nn.Module:
    return GraphNetwork(shape, parameters)


def compositional_registry():
    registry = fixed_registry()
    registry.register("composed_dag", "1", build_graph, graph_count)
    # Fixed literature-inspired controls participate as baselines; generated DAGs
    # remain a separate family and must earn selection against them.
    from . import literature_baselines as lit
    registry.register("grid3d_res", "2026-adaptation", lit.build_grid3d_res, lit.grid3d_res_count)
    registry.register("transformer_ts", "2026-control", lit.build_transformer_ts, lit.transformer_ts_count)
    registry.register("multiscale_conv_transformer", "2025-adaptation",
                      lit.build_multiscale_conv_transformer, lit.multiscale_conv_transformer_count)
    registry.register("dynamic_graph_gru", "2025-adaptation", lit.build_dynamic_graph_gru, lit.dynamic_graph_gru_count)
    return registry


def graph_to_dot(graph: dict) -> str:
    """Text-only Graphviz artifact. No external process is executed."""
    rows = ['digraph architecture {', 'rankdir=LR;', 'input [label="observations B,T,C"];']
    for n in graph["nodes"]:
        label = n["op"] + (" / " + n["share"] if n["share"] else "")
        rows.append(f'{n["id"]} [label="{label}"];')
        for parent in n["inputs"]:
            rows.append(f'{parent} -> {n["id"]};')
    rows.append('}')
    return '\n'.join(rows) + '\n'
