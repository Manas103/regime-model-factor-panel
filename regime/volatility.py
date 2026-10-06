"""EWMA baseline volatility, regime-conditional volatility forecasts (from
either filtered or smoothed state probabilities), and the QLIKE loss used
to score both against realized squared return.
"""

from __future__ import annotations

import numpy as np

from regime import config


def ewma_variance(returns: np.ndarray, lam: float = config.EWMA_LAMBDA) -> np.ndarray:
    """RiskMetrics-style EWMA variance forecast for week t, using only
    returns strictly before t (so sigma2[0] falls back to the first
    squared return itself, the only information available)."""
    n = len(returns)
    sigma2 = np.empty(n)
    sigma2[0] = returns[0] ** 2
    for t in range(1, n):
        sigma2[t] = lam * sigma2[t - 1] + (1 - lam) * returns[t - 1] ** 2
    return sigma2


def regime_conditional_variance(state_probs: np.ndarray, state_variances: np.ndarray) -> np.ndarray:
    """Probability-weighted variance forecast: sum_s P(state_t=s) * var_s,
    using whichever state_probs are passed in (filtered or smoothed)."""
    return state_probs @ state_variances


def qlike(actual_returns: np.ndarray, forecast_variance: np.ndarray) -> np.ndarray:
    """QLIKE loss per week: log(sigma2_hat) + r^2 / sigma2_hat. Lower is
    better; this is the standard loss for comparing variance forecasts
    because, unlike MSE on variance, its expectation is minimized by the
    true conditional variance even when volatility itself is stochastic."""
    sigma2 = np.maximum(forecast_variance, 1e-10)
    return np.log(sigma2) + (actual_returns ** 2) / sigma2


def mean_qlike(actual_returns: np.ndarray, forecast_variance: np.ndarray) -> float:
    return float(np.mean(qlike(actual_returns, forecast_variance)))
