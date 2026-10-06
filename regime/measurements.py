"""Assembles every claim measurement: the mixture fit, filtered vs smoothed
regime probabilities, the EWMA-relative QLIKE improvement under each, the
smoothed-vs-filtered overstatement ratio, and the out-of-sample
return-predictability check.
"""

from __future__ import annotations

import numpy as np

from regime import config, hmm, mixture, predictability, volatility


def run_all_measurements(panel: np.ndarray) -> dict:
    """panel: (T, 3) array of weekly [Mkt-RF, SMB, RF] percent returns, in
    that column order (config.FACTOR_COLUMNS)."""
    target_idx = config.FACTOR_COLUMNS.index(config.VOL_TARGET_COLUMN)
    target_returns = panel[:, target_idx]

    gmm = mixture.fit_gmm(panel)
    state_path = mixture.hard_state_path(gmm, panel)
    transition = mixture.estimate_transition_matrix(state_path, config.N_STATES)
    initial = mixture.stationary_distribution(transition)

    means = gmm.means_
    covariances = gmm.covariances_

    emission = hmm.emission_probs(panel, means, covariances)
    alpha, _ = hmm.forward_filter(emission, transition, initial)
    smoothed = hmm.forward_backward_smoother(emission, transition, initial)
    predictive = hmm.one_step_predictive(alpha, transition, initial)

    # per-state variance of the volatility target, from the mixture's own
    # fitted covariance (the diagonal entry for the target factor).
    state_variances = covariances[:, target_idx, target_idx]

    ewma_var = volatility.ewma_variance(target_returns)
    filtered_var = volatility.regime_conditional_variance(predictive, state_variances)
    smoothed_var = volatility.regime_conditional_variance(smoothed, state_variances)

    qlike_ewma = volatility.mean_qlike(target_returns, ewma_var)
    qlike_filtered = volatility.mean_qlike(target_returns, filtered_var)
    qlike_smoothed = volatility.mean_qlike(target_returns, smoothed_var)

    # QLIKE is a loss (lower is better), so "improvement" is how much lower
    # the regime model's mean QLIKE is than EWMA's, as a fraction of EWMA's.
    filtered_improvement_pct = (qlike_ewma - qlike_filtered) / abs(qlike_ewma) * 100.0
    smoothed_improvement_pct = (qlike_ewma - qlike_smoothed) / abs(qlike_ewma) * 100.0
    overstatement_ratio = (
        smoothed_improvement_pct / filtered_improvement_pct
        if filtered_improvement_pct > 0
        else None  # undefined/not meaningful when the live baseline shows no genuine gain at all
    )
    overstatement_pct_points = smoothed_improvement_pct - filtered_improvement_pct

    actual, predicted = predictability.walk_forward_return_predictability(target_returns, predictive)
    oos_r2_returns = predictability.r_squared(actual, predicted)

    return {
        "n_weeks": panel.shape[0],
        "n_states": config.N_STATES,
        "transition_matrix": transition.tolist(),
        "state_variances_annualized_pct": (np.sqrt(state_variances) * np.sqrt(config.WEEKS_PER_YEAR)).tolist(),
        "qlike_ewma": qlike_ewma,
        "qlike_filtered": qlike_filtered,
        "qlike_smoothed": qlike_smoothed,
        "filtered_improvement_pct_over_ewma": filtered_improvement_pct,
        "smoothed_improvement_pct_over_ewma": smoothed_improvement_pct,
        "smoothed_overstatement_ratio": overstatement_ratio,
        "smoothed_overstatement_pct_points": overstatement_pct_points,
        "oos_r2_return_predictability": oos_r2_returns,
        "n_oos_weeks_return_predictability": len(actual),
    }
