"""Independent, unscaled, more literal reimplementation of the forward and
forward-backward recursions in hmm.py, sharing no helper function with it.

Unscaled probabilities underflow over a long sequence, so this oracle is
only ever run over a short prefix of the real panel (the first 30 weeks);
`scripts/run_regime_model.py` diffs it against `hmm.py`'s scaled fast path
restricted to the same prefix, not over the full 1,566-week sample.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import multivariate_normal


def forward_filter_oracle(
    X: np.ndarray, transition: np.ndarray, initial: np.ndarray, means: np.ndarray, covariances: np.ndarray
) -> np.ndarray:
    T = X.shape[0]
    n_states = means.shape[0]

    def emission(t, s):
        return float(multivariate_normal.pdf(X[t], mean=means[s], cov=covariances[s]))

    alpha = [[0.0] * n_states for _ in range(T)]
    for s in range(n_states):
        alpha[0][s] = initial[s] * emission(0, s)
    total = sum(alpha[0])
    alpha[0] = [v / total for v in alpha[0]]

    for t in range(1, T):
        raw = []
        for s in range(n_states):
            acc = 0.0
            for sp in range(n_states):
                acc += alpha[t - 1][sp] * transition[sp, s]
            raw.append(acc * emission(t, s))
        total = sum(raw)
        alpha[t] = [v / total for v in raw]

    return np.array(alpha)


def forward_backward_smoother_oracle(
    X: np.ndarray, transition: np.ndarray, initial: np.ndarray, means: np.ndarray, covariances: np.ndarray
) -> np.ndarray:
    T = X.shape[0]
    n_states = means.shape[0]
    alpha = forward_filter_oracle(X, transition, initial, means, covariances)

    def emission(t, s):
        return float(multivariate_normal.pdf(X[t], mean=means[s], cov=covariances[s]))

    beta = [[0.0] * n_states for _ in range(T)]
    beta[T - 1] = [1.0] * n_states
    for t in range(T - 2, -1, -1):
        raw = []
        for s in range(n_states):
            acc = 0.0
            for sp in range(n_states):
                acc += transition[s, sp] * emission(t + 1, sp) * beta[t + 1][sp]
            raw.append(acc)
        total = sum(raw) if sum(raw) > 0 else 1.0
        beta[t] = [v / total for v in raw]

    gamma = []
    for t in range(T):
        row = [alpha[t][s] * beta[t][s] for s in range(n_states)]
        total = sum(row)
        gamma.append([v / total for v in row])
    return np.array(gamma)
