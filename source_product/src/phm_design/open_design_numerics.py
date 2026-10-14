"""Finite, explicit numerical support descriptors; never fault labels.

A singular regression design has an infinite exact condition number. Encode
finite machine-resolution conditioning and rank deficiency separately. Do not
replace arbitrary NaNs, infinities, or malformed input with useful-looking data.
"""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class MatrixSupport:
    capped_log_condition: float
    rank_fraction: float
    rank_deficient: bool
    effective_rank: int

def matrix_support(values) -> MatrixSupport:
    a = np.asarray(values, dtype=float)
    if a.ndim != 2 or min(a.shape, default=0) < 1 or not np.isfinite(a).all():
        raise ValueError('INVALID_DESIGN_MATRIX')
    scale = float(np.max(np.abs(a)))
    eps = np.finfo(float).eps * max(a.shape)
    if scale == 0.0:
        return MatrixSupport(float(np.log1p(1.0 / eps)), 0.0, True, 0)
    singular = np.linalg.svd(a / scale, compute_uv=False)
    relative = singular / singular[0]
    rank = int(np.sum(relative > eps))
    ratio = float(relative[-1]) if a.shape[0] >= a.shape[1] else 0.0
    return MatrixSupport(float(np.log1p(1.0 / max(ratio, eps))),
                         rank / a.shape[1], rank < a.shape[1], rank)
