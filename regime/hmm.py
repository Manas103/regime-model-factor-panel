"""Hand-rolled scaled forward and forward-backward recursions (Rabiner,
1989), the fast path for filtered and smoothed regime probabilities.

Filtered: `forward_filter` returns P(state_t | obs_1..t), using only
information available up to and including week t, the point-in-time-
correct quantity.

Smoothed: `forward_backward_smoother` returns P(state_t | obs_1..T), which
uses every week in the sample including weeks after t. It is the quantity
a backtester gets "for free" if it is not careful, and the whole point of
measuring it separately here is to quantify how much that hindsight
inflates the regime-conditional volatility forecast's apparent skill (see
measurements.py and the README Findings).

`oracle.py` recomputes both with a slower, unscaled, more literal
translation of the same recursions and checks they agree.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import multivariate_normal


def emission_probs(X: np.ndarray, means: np.ndarray, covariances: np.ndarray) -> np.ndarray:
    """probs[t, s] = density of obs_t under state s's fitted Gaussian."""
    T = X.shape[0]
    n_states = means.shape[0]
    probs = np.empty((T, n_states))
    for s in range(n_states):
        probs[:, s] = multivariate_normal.pdf(X, mean=means[s], cov=covariances[s])
    return probs


def forward_filter(
    emission: np.ndarray, transition: np.ndarray, initial: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (alpha, scaling): alpha[t, :] is the normalized filtered
    distribution P(state_t | obs_1..t); scaling[t] is c_t, kept so the
    backward pass can reuse the same normalization (the standard scaled
    forward-backward construction)."""
    T, n_states = emission.shape
    alpha = np.empty((T, n_states))
    scaling = np.empty(T)

    unnorm = initial * emission[0]
    scaling[0] = unnorm.sum()
    alpha[0] = unnorm / scaling[0]

    for t in range(1, T):
        unnorm = emission[t] * (alpha[t - 1] @ transition)
        scaling[t] = unnorm.sum()
        alpha[t] = unnorm / scaling[t]

    return alpha, scaling


def forward_backward_smoother(
    emission: np.ndarray, transition: np.ndarray, initial: np.ndarray
) -> np.ndarray:
    """Returns gamma[t, :] = P(state_t | obs_1..T), the full-sample
    smoothed distribution, via the scaled forward-backward algorithm."""
    T, n_states = emission.shape
    alpha, scaling = forward_filter(emission, transition, initial)

    beta = np.empty((T, n_states))
    beta[T - 1] = 1.0
    for t in range(T - 2, -1, -1):
        beta[t] = (transition @ (emission[t + 1] * beta[t + 1])) / scaling[t + 1]

    gamma = alpha * beta
    gamma = gamma / gamma.sum(axis=1, keepdims=True)
    return gamma


def one_step_predictive(alpha: np.ndarray, transition: np.ndarray, initial: np.ndarray) -> np.ndarray:
    """pred[t, :] = P(state_t | obs_1..t-1): the distribution over week t's
    regime using only information through week t-1, before week t's own
    return is observed. pred[0] is the initial distribution itself, since
    there is no prior week. This, not the filtered alpha (which already
    conditions on week t's own return), is what "a live model would see"
    before forecasting week t's variance."""
    T, n_states = alpha.shape
    pred = np.empty((T, n_states))
    pred[0] = initial
    pred[1:] = alpha[:-1] @ transition
    return pred
