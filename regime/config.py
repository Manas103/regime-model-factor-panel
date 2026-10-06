"""Declared constants for the regime model.

The panel is real: Kenneth French's weekly Fama-French 3-factor file
(`mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_weekly_CSV.zip`),
restricted to the three columns the breakdown names, equity (Mkt-RF), size
(SMB) and rate (RF), over the most recent 30 real years available in the
file, not the full 1926-present history.
"""

from __future__ import annotations

FACTOR_COLUMNS = ["Mkt-RF", "SMB", "RF"]
VOL_TARGET_COLUMN = "Mkt-RF"

N_STATES = 3
N_YEARS = 30
WEEKS_PER_YEAR = 52.0

EWMA_LAMBDA = 0.94  # RiskMetrics convention

# EM initialization and refit seed for sklearn.mixture.GaussianMixture.
SEED = 20260926

# Minimum weeks of history before the first out-of-sample return-predictability
# prediction is made (same expanding-window convention as the energy nowcast
# in point-in-time-activity-index).
MIN_TRAIN_WEEKS = 260  # 5 years
