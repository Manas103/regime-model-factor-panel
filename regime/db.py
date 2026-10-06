"""DuckDB storage for the labeled weekly panel, and a SQL sanity check:
each state's empirical average squared return, computed by a plain SQL
GROUP BY, compared against that state's variance as fit by the Gaussian
mixture (a separate estimation path: EM over the full 3-factor covariance
structure, not a single-column average). The two are not expected to
match exactly (the mixture's per-state variance also reflects its jointly
fit mean and the other two factors' covariance structure), only to be in
the same neighborhood; a large divergence would flag a mixture fit that is
not actually describing the labeled weeks it claims to.
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS labeled_panel (
    week_index   INTEGER NOT NULL,
    mkt_rf       DOUBLE NOT NULL,
    smb          DOUBLE NOT NULL,
    rf           DOUBLE NOT NULL,
    state        INTEGER NOT NULL
)
"""


def connect(db_path: str) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(db_path)
    con.execute(SCHEMA)
    return con


def load_labeled_panel(con: duckdb.DuckDBPyConnection, panel: np.ndarray, state_path: np.ndarray) -> None:
    df = pd.DataFrame(
        {
            "week_index": np.arange(len(panel)),
            "mkt_rf": panel[:, 0],
            "smb": panel[:, 1],
            "rf": panel[:, 2],
            "state": state_path,
        }
    )
    con.register("_labeled", df)
    con.execute("INSERT INTO labeled_panel SELECT * FROM _labeled")
    con.unregister("_labeled")


def per_state_mean_squared_return(con: duckdb.DuckDBPyConnection) -> dict[int, float]:
    rows = con.execute(
        "SELECT state, AVG(mkt_rf * mkt_rf) AS mean_sq FROM labeled_panel GROUP BY state ORDER BY state"
    ).fetchall()
    return {int(state): float(mean_sq) for state, mean_sq in rows}
