"""Fits the 3-state Gaussian mixture (the emission model) with
scikit-learn, and derives a Markov transition matrix from the mixture's
own hard state assignments.

The mixture itself is i.i.d. across weeks (`sklearn.mixture.GaussianMixture`
has no time dependence); the transition matrix is estimated afterward from
the mixture's own posterior-argmax path. This is a declared simplification
of jointly estimating emissions and transitions with Baum-Welch, stated
here and in the README: it lets the regime's own cluster shapes come from
a well-tested library fit, while the Markov structure needed for filtered
vs. smoothed probabilities is added on top by this project's own
hand-rolled forward-backward algorithm (`hmm.py`).
"""

from __future__ import annotations

import numpy as np
from sklearn.mixture import GaussianMixture

from regime import config


def fit_gmm(X: np.ndarray, n_states: int = config.N_STATES, seed: int = config.SEED) -> GaussianMixture:
    gmm = GaussianMixture(
        n_components=n_states,
        covariance_type="full",
        random_state=seed,
        n_init=10,
    )
    gmm.fit(X)
    return gmm


def hard_state_path(gmm: GaussianMixture, X: np.ndarray) -> np.ndarray:
    return gmm.predict(X)


def estimate_transition_matrix(state_path: np.ndarray, n_states: int) -> np.ndarray:
    """Row-normalized empirical transition counts from the hard state path.
    A state with no observed outgoing transition falls back to staying in
    itself with probability 1 (an absorbing-state default, disclosed), so
    the matrix is always a valid row-stochastic matrix."""
    counts = np.zeros((n_states, n_states))
    for t in range(len(state_path) - 1):
        counts[state_path[t], state_path[t + 1]] += 1
    row_sums = counts.sum(axis=1, keepdims=True)
    transition = np.divide(counts, row_sums, out=np.zeros_like(counts), where=row_sums > 0)
    for s in range(n_states):
        if row_sums[s, 0] == 0:
            transition[s, s] = 1.0
    return transition


def stationary_distribution(transition: np.ndarray) -> np.ndarray:
    """Solves pi A = pi, sum(pi) = 1 via eigen-decomposition; falls back to
    uniform if the solve is degenerate."""
    n = transition.shape[0]
    eigvals, eigvecs = np.linalg.eig(transition.T)
    idx = np.argmin(np.abs(eigvals - 1.0))
    vec = np.real(eigvecs[:, idx])
    if vec.sum() == 0 or np.any(np.isnan(vec)):
        return np.full(n, 1.0 / n)
    vec = np.abs(vec)
    return vec / vec.sum()
