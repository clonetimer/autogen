"""R68 opt-in task-design candidate registry with a common estimator contract.

This is a research-only bridge between grouped-OOF task selection and the
existing versioned PyTorch network families. It does NOT retrofit the separate
open_design search entry, enable CUDA, or authorize deployment qualification.
No training labels, world IDs, fault onset or simulated truth enter features.
"""
from __future__ import annotations
from dataclasses import dataclass
from contextlib import contextmanager
from typing import Sequence
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Import torch lazily to keep the registry metadata query lightweight.
# All torch training is explicitly CPU-only in R68.

SEQ_SAMPLES = 48
_LEARNERS = {
    'linear': ('sklearn', 'tabular'),
    'extra': ('sklearn', 'tabular'),
    'torch_cnn1d': ('torch_cpu', 'sequence'),
    'torch_gru': ('torch_cpu', 'sequence'),
    'torch_transformer': ('torch_cpu', 'sequence'),
}


@dataclass(frozen=True)
class TaskCandidateDefinition:
    learner: str
    backend: str
    input_kind: str
    feature: str
    version: str = 'r68-v1'


class TaskCandidateRegistry:
    """Explicit allowlist; prevents arbitrary imports or executable search specs."""
    def describe(self, feature: str, learner: str) -> TaskCandidateDefinition:
        if not isinstance(feature, str) or not isinstance(learner, str) or learner not in _LEARNERS:
            raise ValueError('UNREGISTERED_TASK_CANDIDATE')
        backend, input_kind = _LEARNERS[learner]
        expected = 'sequence' if input_kind == 'sequence' else ('temporal', 'physical')
        if (feature != expected if isinstance(expected, str) else feature not in expected):
            raise ValueError('TASK_FEATURE_BACKEND_MISMATCH')
        return TaskCandidateDefinition(learner, backend, input_kind, feature)

    def build(self, feature: str, learner: str, seed: int):
        record = self.describe(feature, learner)
        if record.learner == 'linear':
            return make_pipeline(StandardScaler(), LogisticRegression(C=1., max_iter=800, random_state=seed))
        if record.learner == 'extra':
            return ExtraTreesClassifier(n_estimators=48, min_samples_leaf=1, n_jobs=1, random_state=seed)
        return TorchTaskEstimator(record.learner, seed)


def deployment_sequence(values: np.ndarray, samples: int = SEQ_SAMPLES) -> np.ndarray:
    """Interpolate deployment-observed sensor values on a per-trajectory time grid.

    The zero-index time column supplies *only* interpolation coordinates. No
    healthy counterfactual, onset, class or world metadata is ever consulted.
    Normalization is deliberately postponed to fit() of the training fold.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 3 or len(arr) < 4:
        raise ValueError('INVALID_SEQUENCE_OBSERVATIONS')
    if not np.isfinite(arr).all() or np.any(np.diff(arr[:, 0]) <= 0):
        raise ValueError('INVALID_SEQUENCE_TIME_AXIS')
    if type(samples) is not int or not 8 <= samples <= 512:
        raise ValueError('INVALID_SEQUENCE_SAMPLES')
    grid = np.linspace(arr[0, 0], arr[-1, 0], samples)
    return np.stack([np.interp(grid, arr[:, 0], arr[:, j]) for j in range(1, arr.shape[1])], axis=1)


@contextmanager
def _cpu_training_context(seed: int):
    import torch
    old_threads = torch.get_num_threads()
    # No CUDA access or global RNG alteration, for deterministic CPU baselines.
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            yield
    finally:
        torch.set_num_threads(old_threads)


class TorchTaskEstimator:
    """Small CPU Torch sequence classifier exposing sklearn-like fit/proba.

    Preprocessing is fitted on the *current training partition only*. The
    enclosing TaskDesign still owns GroupKFold and feasibility evaluation.
    """
    def __init__(self, learner: str, seed: int, *, epochs: int = 8,
                 batch_size: int = 16, max_parameters: int = 200_000):
        if learner not in {'torch_cnn1d', 'torch_gru', 'torch_transformer'}:
            raise ValueError('UNREGISTERED_TORCH_TASK_MODEL')
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError('INVALID_TORCH_TASK_SEED')
        if type(epochs) is not int or not 1 <= epochs <= 200:
            raise ValueError('INVALID_TORCH_EPOCHS')
        self.learner = learner
        self.seed = seed
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_parameters = max_parameters
        self.classes_ = None
        self.model_ = None
        self.mean_ = None
        self.scale_ = None
        self.training_ = None

    def _network(self, steps: int, channels: int, classes: int):
        from .models import Shape, default_registry
        from .literature_baselines import build_transformer_ts
        shape = Shape(steps, channels, classes)
        if self.learner == 'torch_transformer':
            return build_transformer_ts(shape, {'d_model': 16, 'heads': 2, 'layers': 1})
        registry = default_registry()
        if self.learner == 'torch_cnn1d':
            spec = {'family': 'cnn1d', 'parameters': {'widths': [12], 'kernel': 3}}
        else:
            spec = {'family': 'gru', 'parameters': {'hidden': 12, 'layers': 1}}
        return registry.build(spec, shape, self.seed, self.max_parameters)

    @staticmethod
    def _check(x):
        a = np.asarray(x, dtype=np.float32)
        if a.ndim != 3 or len(a) == 0 or a.shape[1] < 8 or a.shape[2] == 0 or not np.isfinite(a).all():
            raise ValueError('INVALID_SEQUENCE_TENSOR')
        return a

    def fit(self, x, y):
        import torch
        x = self._check(x)
        labels = np.asarray(y, dtype=int)
        if labels.ndim != 1 or len(labels) != len(x) or len(set(labels)) < 2:
            raise ValueError('INSUFFICIENT_TORCH_CLASSES')
        if self.batch_size < 1 or self.max_parameters < 1:
            raise ValueError('INVALID_TORCH_TRAINING_BUDGET')
        self.classes_ = np.unique(labels)
        encoded = np.searchsorted(self.classes_, labels)
        self.mean_ = x.mean(axis=(0, 1), keepdims=True)
        self.scale_ = np.maximum(x.std(axis=(0, 1), keepdims=True), 1e-5)
        transformed = np.clip((x - self.mean_) / self.scale_, -20., 20.)
        tx = torch.from_numpy(np.asarray(transformed, dtype=np.float32))
        ty = torch.from_numpy(encoded.astype(np.int64))
        with _cpu_training_context(self.seed):
            model = self._network(x.shape[1], x.shape[2], len(self.classes_)).cpu()
            count = sum(p.numel() for p in model.parameters())
            if count > self.max_parameters:
                raise ValueError('PARAMETER_BUDGET_EXCEEDED')
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-3)
            rng = np.random.default_rng(self.seed)
            for _ in range(self.epochs):
                model.train()
                for selection in np.array_split(rng.permutation(len(x)), max(1, int(np.ceil(len(x) / self.batch_size)))):
                    optimizer.zero_grad(set_to_none=True)
                    loss = torch.nn.functional.cross_entropy(model(tx[selection]), ty[selection])
                    if not torch.isfinite(loss):
                        raise ValueError('NONFINITE_TASK_TRAINING_LOSS')
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
                    optimizer.step()
            model.eval()
        self.model_ = model
        self.training_ = {'backend': 'torch_cpu', 'model': self.learner,
                          'epochs': self.epochs, 'parameter_count': count,
                          'normalization': 'training_partition_only',
                          'seed': self.seed, 'cuda_used': False}
        return self

    def predict_proba(self, x):
        import torch
        if self.model_ is None:
            raise ValueError('TASK_MODEL_NOT_FITTED')
        a = self._check(x)
        if a.shape[2] != self.mean_.shape[2]:
            raise ValueError('TASK_SEQUENCE_CHANNEL_MISMATCH')
        norm = np.clip((a - self.mean_) / self.scale_, -20., 20.)
        with _cpu_training_context(self.seed):
            self.model_.eval()
            with torch.inference_mode():
                pred = self.model_(torch.from_numpy(np.asarray(norm, dtype=np.float32))).softmax(1).numpy()
        if not np.isfinite(pred).all():
            raise ValueError('NONFINITE_TASK_PROBABILITIES')
        return pred
