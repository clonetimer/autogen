"""R87 analytic Gaussian fuzzy-premise optimization on SOURCE TRAIN only.

The gradient differentiates softmax cross-entropy through normalized Gaussian
rule firing with affine TSK consequents held fixed. Coefficients are then
refitted using the predeclared ridge rule. A bounded backtracking step is
accepted only when the re-fitted TRAIN loss improves. No heldout labels are
used for gradients, stopping, or hyperparameter tuning.
"""
from __future__ import annotations

import numpy as np


def gaussian_firing(x, centers, log_widths):
    x, centers, log_widths = (
        np.asarray(a, dtype=np.float64) for a in (x, centers, log_widths))
    if (x.ndim != 2 or centers.ndim != 2 or
            centers.shape != log_widths.shape or
            centers.shape[1] != x.shape[1] or
            len(x) == 0 or len(centers) == 0 or
            not all(np.isfinite(a).all() for a in (x, centers, log_widths)) or
            np.any(np.abs(log_widths) > 3)):
        raise ValueError("R87_INVALID_GAUSSIAN_PREMISE")
    diff = x[:, None, :] - centers[None, :, :]
    invvar = np.exp(-2 * log_widths)
    log_strength = -0.5 * np.mean(diff * diff * invvar[None, :, :], axis=2)
    log_strength -= log_strength.max(axis=1, keepdims=True)
    strength = np.exp(np.maximum(log_strength, -700.0))
    return strength / strength.sum(axis=1, keepdims=True)


def fuzzy_design(x, centers, log_widths):
    f = gaussian_firing(x, centers, log_widths)
    bias = np.column_stack((np.ones(len(x)), x))
    return (f[:, :, None] * bias[:, None, :]).reshape(len(x), -1)


def _softmax(a):
    z = a - a.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def ridge_consequents(x, target, centers, log_widths, nclasses, ridge):
    z = fuzzy_design(x, centers, log_widths)
    y = np.eye(nclasses, dtype=np.float64)[target]
    penalty = np.eye(z.shape[1], dtype=np.float64) * ridge
    coef = np.linalg.solve(z.T @ z + penalty, z.T @ y)
    probabilities = _softmax(z @ coef)
    loss = float(-np.mean(np.log(np.maximum(
        probabilities[np.arange(len(target)), target], 1e-12))))
    return coef, probabilities, loss


def premise_gradient(x, target, centers, log_widths, coef):
    """Analytic d(softmax cross-entropy)/d(centers, log_sigma).

    Consequents are held fixed during the gradient, then re-fitted between
    steps; training labels can enter this function ONLY from a TRAIN fold.
    """
    x = np.asarray(x, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)
    log_widths = np.asarray(log_widths, dtype=np.float64)
    coef = np.asarray(coef, dtype=np.float64)
    nrules, ndim = centers.shape
    if coef.ndim != 2 or coef.shape[0] != nrules * (ndim + 1):
        raise ValueError("R87_INVALID_CONSEQUENT_DIMENSIONS")
    target = np.asarray(target)
    if (target.ndim != 1 or len(target) != len(x) or
            target.dtype.kind not in "iu" or
            (target < 0).any() or (target >= coef.shape[1]).any()):
        raise ValueError("R87_INVALID_TRAIN_LABELS")
    firing = gaussian_firing(x, centers, log_widths)
    bias = np.column_stack((np.ones(len(x)), x))
    q = np.einsum(
        "nf,rfk->nrk", bias, coef.reshape(nrules, ndim + 1, -1))
    logits = np.einsum("nr,nrk->nk", firing, q)
    p = _softmax(logits)
    dlogits = (p - np.eye(coef.shape[1])[target]) / len(x)
    da = np.einsum("nk,nrk->nr", dlogits, q)
    dlog_firing = firing * (da - np.sum(
        firing * da, axis=1, keepdims=True))
    diff = x[:, None, :] - centers[None, :, :]
    invvar = np.exp(-2 * log_widths)[None, :, :]
    dcenters = np.einsum("nr,nrd->rd", dlog_firing,
                         diff * invvar / ndim)
    dlogwidths = np.einsum("nr,nrd->rd", dlog_firing,
                           diff ** 2 * invvar / ndim)
    if not (np.isfinite(dcenters).all() and np.isfinite(dlogwidths).all()):
        raise ValueError("R87_NONFINITE_PREMISE_GRADIENT")
    return dcenters, dlogwidths


def refine_premises(x, target, centers, *, ridge, steps, learning_rate):
    """Bounded alternating fuzzy-premise and TSK-consequent training."""
    x = np.asarray(x, dtype=np.float64)
    c = np.asarray(centers, dtype=np.float64).copy()
    widths = np.zeros_like(c)
    classes = int(np.max(target)) + 1
    coef, probs, loss = ridge_consequents(
        x, target, c, widths, classes, ridge)
    initial_loss = loss
    accepted = 0
    consequent_refits = 1
    for _ in range(steps):
        grad_c, grad_w = premise_gradient(x, target, c, widths, coef)
        grad_norm = np.sqrt(np.sum(grad_c * grad_c) +
                            np.sum(grad_w * grad_w))
        if grad_norm <= 1e-12:
            break
        scale = max(1., float(grad_norm) / 2.)
        improved = False
        for backtrack in range(5):
            lr = learning_rate / (2 ** backtrack) / scale
            proposal_c = np.clip(c - lr * grad_c, -22., 22.)
            proposal_w = np.clip(widths - lr * grad_w, -1.5, 1.5)
            new_coef, new_probs, new_loss = ridge_consequents(
                x, target, proposal_c, proposal_w, classes, ridge)
            consequent_refits += 1
            if new_loss < loss - 1e-12:
                c, widths = proposal_c, proposal_w
                coef, probs, loss = new_coef, new_probs, new_loss
                accepted += 1
                improved = True
                break
        if not improved:
            break
    return {
        "centers": c, "log_widths": widths,
        "coef": coef, "probabilities": probs,
        "training_loss": loss, "initial_loss": initial_loss,
        "premise_steps_accepted": accepted,
        "consequent_refits": consequent_refits,
    }
