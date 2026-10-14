"""Versioned architecture registry. Plugins are trusted, reviewed Python code."""
from __future__ import annotations
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Any
import torch
from torch import nn
from .io import sha_file


def integer(value: Any, name: str, lower: int, upper: int) -> int:
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError(f"{name} must be an integer in [{lower}, {upper}]")
    return value


@dataclass(frozen=True)
class Shape:
    window: int
    channels: int
    classes: int

    def __post_init__(self) -> None:
        integer(self.window, "window", 2, 4096)
        integer(self.channels, "channels", 1, 512)
        integer(self.classes, "classes", 2, 256)


@dataclass(frozen=True)
class Family:
    version: str
    builder: Callable[[Shape, dict], nn.Module]
    estimator: Callable[[Shape, dict], int]


class Registry:
    def __init__(self) -> None:
        self._families: dict[str, Family] = {}

    def register(self, name: str, version: str, builder: Callable, estimator: Callable) -> None:
        if not name.isidentifier() or name in self._families or not version:
            raise ValueError("INVALID_OR_DUPLICATE_FAMILY")
        if not callable(builder) or not callable(estimator):
            raise TypeError("family builder and estimator must be callable")
        self._families[name] = Family(version, builder, estimator)

    def describe(self, spec: dict, shape: Shape) -> dict:
        if set(spec) != {"family", "parameters"} or spec["family"] not in self._families:
            raise ValueError("UNREGISTERED_OR_INVALID_ARCHITECTURE")
        f = self._families[spec["family"]]
        count = f.estimator(shape, spec["parameters"])
        integer(count, "parameter_count", 1, 1_000_000_000)
        hashes = []
        for fn in (f.builder, f.estimator):
            path = inspect.getsourcefile(fn)
            if not path or not Path(path).is_file():
                raise ValueError("PLUGIN_REQUIRES_VERSIONED_SOURCE_FILE")
            hashes.append(sha_file(Path(path)))
        return {**spec, "version": f.version, "source_sha256": hashes,
                "parameter_count": count, "fp32_weight_bytes": 4 * count}

    def build(self, spec: dict, shape: Shape, seed: int, max_parameters: int) -> nn.Module:
        integer(seed, "seed", 0, 2**32 - 1)
        description = self.describe(spec, shape)
        if description["parameter_count"] > max_parameters:
            raise ValueError("PARAMETER_BUDGET_EXCEEDED")
        with torch.random.fork_rng(devices=[]), torch.device("cpu"):
            torch.set_rng_state(torch.Generator(device="cpu").manual_seed(seed).get_state())
            model = self._families[spec["family"]].builder(shape, spec["parameters"]).cpu().float()
            count = sum(p.numel() for p in model.parameters())
            if count != description["parameter_count"]:
                raise ValueError("PARAMETER_ESTIMATE_MISMATCH")
            model.eval()
            with torch.inference_mode():
                result = model(torch.zeros(2, shape.window, shape.channels))
            if tuple(result.shape) != (2, shape.classes) or not torch.isfinite(result).all():
                raise ValueError("INVALID_NETWORK_OUTPUT")
        return model.train()


def _widths(p: dict) -> tuple[int, ...]:
    if not isinstance(p.get("widths"), (list, tuple)) or not 1 <= len(p["widths"]) <= 4:
        raise ValueError("widths must have one to four entries")
    return tuple(integer(v, "width", 2, 512) for v in p["widths"])


def mlp_count(s: Shape, p: dict) -> int:
    if set(p) != {"widths", "representation"} or p["representation"] not in {"raw", "statistics"}:
        raise ValueError("MLP_PARAMETER_CONTRACT")
    dimensions = ((s.window if p["representation"] == "raw" else 7) * s.channels, *_widths(p), s.classes)
    return sum(a*b+b for a,b in zip(dimensions, dimensions[1:]))


class SequenceMLP(nn.Module):
    def __init__(self, s: Shape, p: dict) -> None:
        super().__init__()
        mlp_count(s, p)
        self.statistics = p["representation"] == "statistics"
        prev = (7 if self.statistics else s.window) * s.channels
        layers = []
        for width in _widths(p):
            layers += [nn.Linear(prev, width), nn.ReLU()]
            prev = width
        self.layers = nn.Sequential(*layers, nn.Linear(prev, s.classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.statistics:
            x = torch.cat((x.mean(1), x.std(1, correction=0), x.amin(1), x.amax(1),
                           x[:, -1], x[:, -1]-x[:, 0], x.square().mean(1).sqrt()), dim=1)
        return self.layers(x.flatten(start_dim=1))


def build_mlp(s: Shape, p: dict) -> nn.Module:
    return SequenceMLP(s, p)


def cnn_count(s: Shape, p: dict) -> int:
    if set(p) != {"widths", "kernel"}:
        raise ValueError("CNN_PARAMETER_CONTRACT")
    k = integer(p["kernel"], "kernel", 1, 9)
    if k % 2 != 1:
        raise ValueError("kernel must be odd")
    prev, count = s.channels, 0
    for w in _widths(p):
        count += prev*w*k+w
        prev = w
    return count+prev*s.classes+s.classes


class SequenceCNN(nn.Module):
    """Offline sequence classifier, NOT a causal streaming TCN."""
    def __init__(self, s: Shape, p: dict) -> None:
        super().__init__()
        cnn_count(s, p)
        layers, prev = [], s.channels
        for w in _widths(p):
            layers += [nn.Conv1d(prev, w, p["kernel"], padding=p["kernel"]//2), nn.ReLU()]
            prev = w
        self.encoder = nn.Sequential(*layers, nn.AdaptiveAvgPool1d(1))
        self.head = nn.Linear(prev, s.classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x.transpose(1, 2)).squeeze(-1))


def build_cnn(s: Shape, p: dict) -> nn.Module:
    return SequenceCNN(s, p)


def gru_count(s: Shape, p: dict) -> int:
    if set(p) != {"hidden", "layers"}:
        raise ValueError("GRU_PARAMETER_CONTRACT")
    h = integer(p["hidden"], "hidden", 2, 512)
    layers = integer(p["layers"], "layers", 1, 4)
    return sum(3*h*(s.channels if n == 0 else h)+3*h*h+6*h for n in range(layers))+h*s.classes+s.classes


class SequenceGRU(nn.Module):
    def __init__(self, s: Shape, p: dict) -> None:
        super().__init__()
        gru_count(s, p)
        self.encoder = nn.GRU(s.channels, p["hidden"], p["layers"], batch_first=True)
        self.head = nn.Linear(p["hidden"], s.classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.encoder(x)
        return self.head(out[:, -1])


def build_gru(s: Shape, p: dict) -> nn.Module:
    return SequenceGRU(s, p)


def default_registry() -> Registry:
    r = Registry()
    r.register("mlp", "1", build_mlp, mlp_count)
    r.register("cnn1d", "1", build_cnn, cnn_count)
    r.register("gru", "1", build_gru, gru_count)
    return r
