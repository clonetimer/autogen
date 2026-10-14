"""R77: CPU PyTorch domain-generalization training for PHM sequences.

Data ownership:
    x/y  : solely the *current training fold*
    groups: independently validated TRAIN environment-family metadata
No evaluation/FINAL observations, held-out group identifiers, fault-onset labels,
or simulator truth are used as input features. A separately held-out environment
is required by the task-graph caller to rank DG against ERM baselines.

The algorithm is a GroupDRO-inspired training implementation, not a claimed
reproduction of any paper or evidence that it improves real PHM robustness.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from .task_candidate_registry import TorchTaskEstimator, _cpu_training_context


@dataclass(frozen=True)
class DGTrainingPolicy:
    epochs: int = 12
    batch_per_environment: int = 12
    learning_rate: float = 0.003
    group_step_size: float = 0.2
    weight_decay: float = 0.01
    noise_std: float = 0.015
    gain_std: float = 0.025
    offset_std: float = 0.015
    consistency_weight: float = 0.10
    min_training_environments: int = 2
    max_training_environments: int = 32

    def __post_init__(self):
        for name in ('epochs', 'batch_per_environment', 'min_training_environments',
                     'max_training_environments'):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError('INVALID_DG_TRAINING_POLICY:' + name)
        if self.min_training_environments < 2 or self.max_training_environments < self.min_training_environments:
            raise ValueError('INVALID_DG_TRAINING_ENVIRONMENT_BUDGET')
        if self.epochs > 200 or self.batch_per_environment > 128:
            raise ValueError('DG_TRAINING_RESOURCE_LIMIT')
        for name in ('learning_rate', 'group_step_size', 'weight_decay',
                     'noise_std', 'gain_std', 'offset_std', 'consistency_weight'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError('INVALID_DG_TRAINING_POLICY:' + name)
        if not 0 < self.learning_rate <= 0.03 or not 0 < self.group_step_size <= 2:
            raise ValueError('INVALID_DG_TRAINING_RATE')
        if self.consistency_weight > 1:
            raise ValueError('DG_CONSISTENCY_BUDGET_EXCEEDED')
        if max(self.noise_std, self.gain_std, self.offset_std) > 0.1:
            raise ValueError('DG_AUGMENTATION_BUDGET_EXCEEDED')


def _channelsafe_perturb(x, policy: DGTrainingPolicy):
    """Training-fold-only mild sensor calibration/noise augmentation.

    Time positions and channel identities are preserved; labels and groups never
    become model inputs. Augmentation is explicitly disabled at inference.
    """
    import torch
    n, _, c = x.shape
    gain = 1 + policy.gain_std * torch.randn((n, 1, c), device=x.device)
    offset = policy.offset_std * torch.randn((n, 1, c), device=x.device)
    noise = policy.noise_std * torch.randn_like(x)
    return (x * gain + offset + noise).clamp(-20., 20.)


class DGSequenceEstimator(TorchTaskEstimator):
    """Sklearn-compatible estimator with environment-conditional GroupDRO.

    Small balanced per-environment stochastic batches avoid majority-domain
    domination. Group log-weights emphasize currently high-loss source domains;
    stronger L2 and finite epochs regularize small training folds.
    """
    def __init__(self, learner: str, seed: int, *, policy: DGTrainingPolicy | None = None,
                 max_parameters: int = 200_000):
        if learner not in ('dg_cnn1d', 'dg_gru', 'dg_transformer', 'dg_tcn'):
            raise ValueError('UNREGISTERED_DG_NETWORK')
        self.dg_name = learner
        base = {'dg_cnn1d': 'torch_cnn1d', 'dg_gru': 'torch_gru',
                'dg_transformer': 'torch_transformer', 'dg_tcn': 'torch_tcn'}[learner]
        self.policy = policy if policy is not None else DGTrainingPolicy()
        if not isinstance(self.policy, DGTrainingPolicy):
            raise ValueError('INVALID_DG_POLICY_TYPE')
        super().__init__(base, seed, epochs=self.policy.epochs,
                         batch_size=self.policy.batch_per_environment, max_parameters=max_parameters)
        self.domain_weights_ = None
        self.train_environments_ = None

    def fit(self, x, y, *, domain_groups=None):
        import torch
        x = self._check(x)
        labels = np.asarray(y, dtype=int)
        if labels.ndim != 1 or len(labels) != len(x) or len(np.unique(labels)) < 2:
            raise ValueError('INSUFFICIENT_DG_CLASSES')
        if domain_groups is None:
            raise ValueError('DG_REQUIRES_TRAINING_ENVIRONMENT_METADATA')
        domains = np.asarray(domain_groups)
        if domains.ndim != 1 or len(domains) != len(x):
            raise ValueError('DG_TRAINING_ENVIRONMENT_LENGTH_MISMATCH')
        if any(not isinstance(v, str) or not v.strip() for v in domains.tolist()):
            raise ValueError('DG_INVALID_ENVIRONMENT_ID')
        names = sorted(set(domains.tolist()))
        if not self.policy.min_training_environments <= len(names) <= self.policy.max_training_environments:
            raise ValueError('DG_INSUFFICIENT_OR_EXCESSIVE_TRAINING_ENVIRONMENTS')
        # If one environment contains only a single class, domain loss can be
        # dominated by label mixture rather than learnable sensor invariants.
        for g in names:
            if set(labels[domains == g]) != set(np.unique(labels)):
                raise ValueError('DG_ENVIRONMENT_CLASS_SUPPORT_MISSING')
        self.classes_ = np.unique(labels)
        encoded = np.searchsorted(self.classes_, labels)
        self.mean_ = x.mean(axis=(0, 1), keepdims=True)
        self.scale_ = np.maximum(x.std(axis=(0, 1), keepdims=True), 1e-5)
        normalized = np.clip((x - self.mean_) / self.scale_, -20., 20.)
        tx = torch.from_numpy(np.asarray(normalized, dtype=np.float32))
        ty = torch.from_numpy(encoded.astype(np.int64))
        group_indices = [[np.flatnonzero((domains == g) & (encoded == label))
                          for label in range(len(self.classes_))] for g in names]
        with _cpu_training_context(self.seed):
            model = self._network(x.shape[1], x.shape[2], len(self.classes_)).cpu()
            nparameters = sum(p.numel() for p in model.parameters())
            if nparameters > self.max_parameters:
                raise ValueError('PARAMETER_BUDGET_EXCEEDED')
            opt = torch.optim.AdamW(model.parameters(), lr=self.policy.learning_rate,
                                    weight_decay=self.policy.weight_decay)
            rng = np.random.default_rng(self.seed)
            log_weights = torch.zeros(len(names), dtype=torch.float32)
            history = []
            for _ in range(self.policy.epochs):
                model.train()
                losses = []
                for class_buckets in group_indices:
                    # Draw balanced source-environment AND class samples. This
                    # prevents abundant nominal trajectories drowning out weak
                    # fault gradients. All draws come only from this train fold.
                    count = self.policy.batch_per_environment // len(class_buckets)
                    if count < 1:
                        raise ValueError('DG_BATCH_TOO_SMALL_FOR_CLASSES')
                    selected = [rng.choice(bucket, size=count, replace=len(bucket) < count)
                                for bucket in class_buckets]
                    chosen = rng.permutation(np.concatenate(selected))
                    clean_x = tx[chosen]
                    aug_x = _channelsafe_perturb(clean_x, self.policy)
                    logits = model(aug_x)
                    group_loss = torch.nn.functional.cross_entropy(logits, ty[chosen])
                    if self.policy.consistency_weight:
                        with torch.no_grad():
                            clean_probs = torch.softmax(model(clean_x), dim=1)
                        consistency = torch.nn.functional.kl_div(
                            torch.log_softmax(logits, dim=1), clean_probs, reduction='batchmean')
                        group_loss += self.policy.consistency_weight * consistency
                    losses.append(group_loss)
                stacked = torch.stack(losses)
                if not torch.isfinite(stacked).all():
                    raise ValueError('NONFINITE_DG_ENVIRONMENT_LOSS')
                with torch.no_grad():
                    log_weights += self.policy.group_step_size * stacked.detach()
                    log_weights -= torch.logsumexp(log_weights, dim=0)
                    weights = torch.softmax(log_weights, dim=0)
                loss = (weights * stacked).sum()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
                opt.step()
                history.append(float(max(v.item() for v in stacked)))
            model.eval()
        self.model_ = model
        self.train_environments_ = tuple(names)
        self.domain_weights_ = {g: float(v) for g, v in zip(names, weights.tolist())}
        self.training_ = {'backend': 'torch_cpu', 'objective': 'group_dro',
                          'model': self.dg_name, 'epochs': self.policy.epochs,
                          'parameter_count': nparameters, 'normalization': 'training_partition_only',
                          'augmentation': 'training_partition_only',
                          'sampling': 'environment_and_class_balanced',
                          'consistency_weight': self.policy.consistency_weight,
                          'domain_source': 'TRAIN_LINEAGE_ONLY',
                          'train_environments': len(names),
                          'last_worst_train_domain_loss': history[-1],
                          'seed': self.seed, 'cuda_used': False,
                          'independent_qualification': False}
        return self
