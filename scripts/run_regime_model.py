"""Runs the full regime model (mixture fit, filtered/smoothed probabilities,
QLIKE comparison, return-predictability check) from the committed
data/real_factor_panel.csv, diffs the oracle against the fast path over a
30-week prefix, and prints the measurement JSON.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from regime import config, db, hmm, measurements, mixture, oracle


def sql_vs_gmm_state_variance_check(panel: np.ndarray, db_path: str) -> dict:
    gmm = mixture.fit_gmm(panel)
    state_path = mixture.hard_state_path(gmm, panel)
    target_idx = config.FACTOR_COLUMNS.index(config.VOL_TARGET_COLUMN)
    gmm_variance = gmm.covariances_[:, target_idx, target_idx]

    if os.path.exists(db_path):
        os.remove(db_path)
    con = db.connect(db_path)
    db.load_labeled_panel(con, panel, state_path)
    sql_mean_sq = db.per_state_mean_squared_return(con)
    con.close()

    return {
        "gmm_fitted_variance": gmm_variance.tolist(),
        "sql_mean_squared_return_by_state": sql_mean_sq,
    }


def oracle_check(panel: np.ndarray, prefix: int = 30) -> dict:
    sub = panel[:prefix]
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

    return {
        "prefix_weeks": prefix,
        "filtered_max_abs_diff": float(np.max(np.abs(alpha_fast - alpha_oracle))),
        "smoothed_max_abs_diff": float(np.max(np.abs(smoothed_fast - smoothed_oracle))),
    }


def main() -> None:
    panel_path = sys.argv[1] if len(sys.argv) > 1 else "data/real_factor_panel.csv"
    df = pd.read_csv(panel_path, index_col=0, parse_dates=True)
    panel = df[config.FACTOR_COLUMNS].to_numpy()

    result = measurements.run_all_measurements(panel)
    result["oracle_check"] = oracle_check(panel)
    result["sql_vs_gmm_state_variance_check"] = sql_vs_gmm_state_variance_check(
        panel, "data/regime_panel.duckdb"
    )

    json.dump(result, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
