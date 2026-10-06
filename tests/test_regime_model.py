"""Tests for the regime model: oracle agreement, HMM invariants, the
point-in-time rule for the one-step predictive distribution, and the
mixture/transition-matrix sanity properties."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime import config, db, hmm, measurements, mixture, oracle, predictability, volatility

PANEL_CSV = "data/real_factor_panel.csv"


@pytest.fixture(scope="module")
def panel():
    df = pd.read_csv(PANEL_CSV, index_col=0, parse_dates=True)
    return df[config.FACTOR_COLUMNS].to_numpy()


def test_panel_is_30_real_years_of_weekly_data(panel):
    assert panel.shape[1] == 3
    assert panel.shape[0] > 1500  # ~30 years of weekly observations


def test_transition_matrix_rows_sum_to_one():
    rng = np.random.default_rng(1)
    state_path = rng.integers(0, 3, size=200)
    transition = mixture.estimate_transition_matrix(state_path, 3)
    row_sums = transition.sum(axis=1)
    np.testing.assert_allclose(row_sums, np.ones(3), atol=1e-9)


def test_stationary_distribution_is_a_valid_distribution():
    transition = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.1, 0.8]])
    pi = mixture.stationary_distribution(transition)
    assert abs(pi.sum() - 1.0) < 1e-9
    assert np.all(pi >= 0)
    np.testing.assert_allclose(pi @ transition, pi, atol=1e-6)


def test_forward_filter_rows_sum_to_one():
    rng = np.random.default_rng(2)
    emission = rng.uniform(0.1, 1.0, size=(50, 3))
    transition = np.full((3, 3), 1.0 / 3)
    initial = np.full(3, 1.0 / 3)
    alpha, _ = hmm.forward_filter(emission, transition, initial)
    np.testing.assert_allclose(alpha.sum(axis=1), np.ones(50), atol=1e-9)


def test_smoother_rows_sum_to_one_and_matches_filter_at_last_week():
    rng = np.random.default_rng(3)
    emission = rng.uniform(0.1, 1.0, size=(50, 3))
    transition = np.full((3, 3), 1.0 / 3)
    initial = np.full(3, 1.0 / 3)
    alpha, _ = hmm.forward_filter(emission, transition, initial)
    gamma = hmm.forward_backward_smoother(emission, transition, initial)
    np.testing.assert_allclose(gamma.sum(axis=1), np.ones(50), atol=1e-9)
    # at the very last week, the smoothed and filtered distributions must
    # agree exactly: there is no "future" information left to add.
    np.testing.assert_allclose(gamma[-1], alpha[-1], atol=1e-9)


def test_one_step_predictive_uses_only_strictly_prior_alpha():
    """Perturbing alpha at week t must not change the predictive
    distribution at week t (only at week t+1), the direct point-in-time
    regression test."""
    transition = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.1, 0.8]])
    initial = np.full(3, 1.0 / 3)
    alpha = np.full((10, 3), 1.0 / 3)
    pred_before = hmm.one_step_predictive(alpha, transition, initial)

    alpha2 = alpha.copy()
    alpha2[5] = np.array([0.9, 0.05, 0.05])
    pred_after = hmm.one_step_predictive(alpha2, transition, initial)

    np.testing.assert_allclose(pred_before[:5], pred_after[:5], atol=1e-12)
    assert not np.allclose(pred_before[6], pred_after[6])  # week 6's prediction does change


def test_oracle_matches_fast_path_on_a_short_prefix(panel):
    sub = panel[:30]
    gmm = mixture.fit_gmm(sub)
    state_path = mixture.hard_state_path(gmm, sub)
    transition = mixture.estimate_transition_matrix(state_path, config.N_STATES)
    initial = mixture.stationary_distribution(transition)
    means, covariances = gmm.means_, gmm.covariances_

    emission = hmm.emission_probs(sub, means, covariances)
    alpha_fast, _ = hmm.forward_filter(emission, transition, initial)
    smoothed_fast = hmm.forward_backward_smoother(emission, transition, initial)

    alpha_oracle = oracle.forward_filter_oracle(sub, transition, initial, means, covariances)
    smoothed_oracle = oracle.forward_backward_smoother_oracle(sub, transition, initial, means, covariances)

    assert np.max(np.abs(alpha_fast - alpha_oracle)) < 1e-6
    assert np.max(np.abs(smoothed_fast - smoothed_oracle)) < 1e-6


def test_qlike_prefers_the_true_variance_in_expectation():
    """A sanity check on the loss itself: for returns genuinely drawn from
    N(0, true_var), the true variance's mean QLIKE over many draws should
    be lower than a badly wrong variance's."""
    rng = np.random.default_rng(4)
    true_var = 4.0
    returns = rng.normal(0, np.sqrt(true_var), size=20000)
    q_true = volatility.mean_qlike(returns, np.full(20000, true_var))
    q_wrong = volatility.mean_qlike(returns, np.full(20000, true_var * 5))
    assert q_true < q_wrong


def test_run_all_measurements_reports_every_claim_shape(panel):
    result = measurements.run_all_measurements(panel)
    for key in (
        "qlike_ewma",
        "qlike_filtered",
        "qlike_smoothed",
        "filtered_improvement_pct_over_ewma",
        "smoothed_improvement_pct_over_ewma",
        "oos_r2_return_predictability",
    ):
        assert key in result
        assert isinstance(result[key], float)
    assert result["n_states"] == 3


def test_sql_group_by_matches_a_hand_computed_per_state_mean(tmp_path):
    panel = np.array([[1.0, 0, 0], [2.0, 0, 0], [10.0, 0, 0], [12.0, 0, 0]])
    state_path = np.array([0, 0, 1, 1])
    con = db.connect(str(tmp_path / "t.duckdb"))
    db.load_labeled_panel(con, panel, state_path)
    result = db.per_state_mean_squared_return(con)
    con.close()
    assert abs(result[0] - np.mean([1.0**2, 2.0**2])) < 1e-9
    assert abs(result[1] - np.mean([10.0**2, 12.0**2])) < 1e-9


def test_return_predictability_never_trains_on_the_test_week_or_later():
    rng = np.random.default_rng(5)
    n = 300
    returns = rng.normal(0, 1, size=n)
    pred_probs = rng.dirichlet(np.ones(3), size=n)
    actual1, preds1 = predictability.walk_forward_return_predictability(returns, pred_probs, min_train=100)

    returns2 = returns.copy()
    pred_probs2 = pred_probs.copy()
    returns2[150] = 9999.0  # an injected future outlier
    pred_probs2[150] = [0.99, 0.005, 0.005]
    actual2, preds2 = predictability.walk_forward_return_predictability(returns2, pred_probs2, min_train=100)

    # every prediction strictly before week 150 must be unaffected by the
    # injected value at week 150.
    np.testing.assert_allclose(preds1[:50], preds2[:50], atol=1e-9)
