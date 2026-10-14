"""Compact residual dilated temporal CNN for multichannel PHM telemetry.

Architecture is designed for representation learning from deployable time
series, not simulator truth or an engineered fault-label shortcut.
"""
from __future__ import annotations

import torch
from torch import nn


class _TemporalBlock(nn.Module):
    def __init__(self, width: int, dilation: int):
        super().__init__()
        self.depthwise = nn.Conv1d(width, width, 3, padding=dilation,
                                   dilation=dilation, groups=width)
        self.pointwise = nn.Conv1d(width, width, 1)
        self.normalizer = nn.GroupNorm(1, width)

    def forward(self, x):
        return torch.nn.functional.gelu(x + self.normalizer(self.pointwise(
            torch.nn.functional.gelu(self.depthwise(x)))))


class ResidualDilatedTCN(nn.Module):
    """Multiscale receptive fields with mean, max and std temporal pooling."""
    def __init__(self, channels: int, classes: int, width: int = 24):
        super().__init__()
        self.stem = nn.Conv1d(channels, width, 1)
        self.encoder = nn.Sequential(*[_TemporalBlock(width, d) for d in (1, 2, 4)])
        self.output_norm = nn.LayerNorm(3 * width)
        self.head = nn.Linear(3 * width, classes)

    def forward(self, observations):
        if observations.ndim != 3:
            raise ValueError('DG_TCN_EXPECTS_BATCH_TIME_CHANNELS')
        hidden = self.encoder(self.stem(observations.transpose(1, 2)))
        summary = torch.cat((hidden.mean(-1), hidden.amax(-1),
                             hidden.std(-1, correction=0)), dim=1)
        return self.head(self.output_norm(summary))
