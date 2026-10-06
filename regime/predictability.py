"""Out-of-sample return-predictability test: does the one-step-ahead
predicted regime distribution (known before a week's own return is
observed) carry any genuine out-of-sample information about that week's
Mkt-RF return itself, as opposed to its variance.

An expanding-window walk-forward (the same point-in-time discipline as
`point-in-time-activity-index`'s `energy_nowcast/`) fits a linear
regression of next week's return on the current one-step-ahead state
probabilities, using only weeks strictly before the test week to fit.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression

from regime import config


def r_squared(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    ss_tot = np.sum((actual - actual.mean()) ** 2)
    ss_res = np.sum((actual - predicted) ** 2)
    if ss_tot == 0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def walk_forward_return_predictability(
    returns: np.ndarray, pred_state_probs: np.ndarray, min_train: int = config.MIN_TRAIN_WEEKS
) -> tuple[np.ndarray, np.ndarray]:
    """Features are the first (n_states - 1) columns of pred_state_probs
    (the last is redundant since the row sums to 1). Returns (actuals,
    predictions) over every week a prediction was made."""
    n_states = pred_state_probs.shape[1]
    X_full = pred_state_probs[:, : n_states - 1]
    n = len(returns)

    actuals = []
    preds = []
    for t in range(min_train, n):
        X_train = X_full[:t]
        y_train = returns[:t]
        model = LinearRegression()
        model.fit(X_train, y_train)
        pred = model.predict(X_full[t : t + 1])[0]
        actuals.append(returns[t])
        preds.append(pred)
    return np.array(actuals), np.array(preds)
