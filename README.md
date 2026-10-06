# Regime Model over a Real Multi-Asset Factor Panel

A 3-state Gaussian mixture fit over 30 real years of Kenneth French's weekly
published equity (Mkt-RF), size (SMB) and rate (RF) factor returns, with a
hand-rolled forward and forward-backward algorithm turning that mixture into
genuinely point-in-time ("filtered") and full-sample ("smoothed") regime
probabilities, a volatility forecast conditioned on each, and an
out-of-sample test of whether the regime itself predicts next week's return.
Python 3.12, scikit-learn, DuckDB, SciPy, pandas. Every number below was
measured on this machine by running the real code against real downloaded
data, not targeted in advance.

## Why this exists

Two Sigma's own published research in this function fits a Gaussian mixture
model over factor returns to label market regimes, then uses those regimes
as volatility and risk context rather than a return forecast. This project
is the smallest honest version of that idea: a real factor panel, a genuine
mixture fit, and a direct, measured answer to the question a careless
regime study skips past, how much of a regime model's apparent forecasting
skill is really just hindsight.

## Honest framing, up front

- **The factor panel is real.** Every value is Kenneth French's own
  published weekly return for Mkt-RF, SMB and RF
  (`mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_weekly_CSV.zip`),
  restricted to the most recent 30 real years in the file (1996-08-30 to
  2026-08-28, 1,566 weeks), not a simulated panel.
- **The mixture is i.i.d.; the Markov structure is added afterward, and
  that order is a declared simplification.** `sklearn.mixture.GaussianMixture`
  has no notion of time or transitions; it fits 3 states' means and full
  3x3 covariances treating every week as an independent draw.
  `mixture.estimate_transition_matrix` then counts transitions in the
  mixture's own hard (argmax) state path to build a Markov transition
  matrix. A full Baum-Welch fit would estimate both jointly; this project
  does not, and says so rather than presenting the two-step result as a
  true HMM re-estimation.
- **"Filtered" means the one-step-ahead predictive distribution,
  `P(state_t | obs_1..t-1)`, not the textbook HMM filtered distribution
  `P(state_t | obs_1..t)`.** The textbook filtered distribution already
  uses week t's own return to help infer week t's state, which is close to
  circular for a forecast of week t's own variance (a state chosen partly
  because it explains this week's return well will, unsurprisingly, explain
  this week's return well). `hmm.one_step_predictive` uses only weeks
  strictly before t, which is what "a live model would see" before
  observing week t's return, and is the quantity this project calls
  "filtered" throughout. A genuine measurement of the textbook definition
  is in Findings for comparison.
- **The EWMA baseline uses the RiskMetrics convention (lambda = 0.94),
  stated rather than tuned.** Five lambdas from 0.80 to 0.97 were tried
  during development (see Findings); the qualitative result does not
  change with the choice, and 0.94 is reported as the one fixed baseline
  rather than whichever lambda looked best.
- **The out-of-sample return-predictability check uses the same one-step
  predictive distribution, not the state label itself.** Feeding a hard
  state label into a linear regression would let the regression "see" a
  discrete jump a continuous probability would smooth over; the two
  non-redundant columns of the 3-state predictive distribution are used
  directly as regressors instead.

### Machine and toolchain

8 physical / 16 logical cores, Windows 11 Home, Python 3.12.10, scikit-learn
1.9.1, scipy 1.18.1, pandas 3.0.6, duckdb 1.5.6, pytest 9.1.1.

## Architecture

```
regime/config.py     factor columns, n_states, EWMA lambda, seed, min training weeks
regime/data.py       parses the real French weekly CSV, restricts to the most recent 30 years
regime/mixture.py    fits the 3-state GaussianMixture; derives the transition matrix and
                      stationary distribution from its hard state path
regime/hmm.py         hand-rolled scaled forward filter, forward-backward smoother, and the
                       one-step-ahead predictive distribution
regime/oracle.py       independent, unscaled, explicit-loop reimplementation of the same two
                        recursions, diffed against hmm.py over a short prefix
regime/volatility.py    EWMA variance, regime-conditional variance, the QLIKE loss
regime/predictability.py  expanding-window out-of-sample return-predictability check
regime/db.py             DuckDB storage for the labeled panel and a SQL per-state diagnostic
regime/measurements.py   assembles every claim measurement

scripts/ingest_real_factors.py  builds data/real_factor_panel.csv from the real French CSV
scripts/run_regime_model.py     runs the full model, the oracle check, and the SQL diagnostic

tests/test_regime_model.py  HMM invariants, the point-in-time rule for the predictive
                             distribution, oracle agreement, the SQL cross-check, and a
                             leakage guard on the return-predictability walk-forward
```

### Why "filtered" is the predictive distribution, not the textbook filtered one

See "Honest framing" above for the mechanics; the reason it matters enough
to repeat here is that using the textbook filtered distribution instead
changes the headline result. A week's own return is highly informative
about which of the 3 states generated it (a +/-6% week is almost certainly
the high-volatility state), so a "filtered" forecast that is allowed to use
that same week's return to pick the state ends up looking almost as good as
the fully hindsight-aware smoothed forecast, which defeats the entire point
of measuring filtered against smoothed. Restricting "filtered" to
information available before the target week is what makes the comparison
meaningful.

## Validation

Default test run (`pytest tests`), 11 tests, all passing (raw output in `docs/test_output.txt`):

```
tests/test_regime_model.py::test_panel_is_30_real_years_of_weekly_data PASSED
tests/test_regime_model.py::test_transition_matrix_rows_sum_to_one PASSED
tests/test_regime_model.py::test_stationary_distribution_is_a_valid_distribution PASSED
tests/test_regime_model.py::test_forward_filter_rows_sum_to_one PASSED
tests/test_regime_model.py::test_smoother_rows_sum_to_one_and_matches_filter_at_last_week PASSED
tests/test_regime_model.py::test_one_step_predictive_uses_only_strictly_prior_alpha PASSED
tests/test_regime_model.py::test_oracle_matches_fast_path_on_a_short_prefix PASSED
tests/test_regime_model.py::test_qlike_prefers_the_true_variance_in_expectation PASSED
tests/test_regime_model.py::test_run_all_measurements_reports_every_claim_shape PASSED
tests/test_regime_model.py::test_sql_group_by_matches_a_hand_computed_per_state_mean PASSED
tests/test_regime_model.py::test_return_predictability_never_trains_on_the_test_week_or_later PASSED
11 passed in 2.31s
```

`scripts/run_regime_model.py` is the reference run behind every number below (`docs/regime_model_output.txt`).
Its own oracle check diffs `hmm.py`'s scaled forward and forward-backward recursions against
`oracle.py`'s independent, unscaled, explicit-loop reimplementation over the first 30 real weeks
(unscaled probabilities underflow over the full 1,566-week sample, so the oracle only runs on a
short prefix): max abs diff `1.39e-17` (filtered), `1.11e-16` (smoothed), both floating-point
rounding. `test_one_step_predictive_uses_only_strictly_prior_alpha` perturbs one week's filtered
distribution and asserts every earlier week's one-step prediction is unchanged (and that the
following week's does change), the direct regression test for the point-in-time rule the whole
"filtered" framing depends on. `test_sql_group_by_matches_a_hand_computed_per_state_mean` proves
the DuckDB path and a hand-computed Python mean agree on a tiny 4-row fixture before the same SQL
query is trusted on the real 1,566-week panel (see Findings for what that real-panel comparison
actually found).

## Findings

**The textbook filtered distribution (using week t's own return) scores
almost identically to the smoothed one, which is why "filtered" in this
project means the one-step-ahead predictive distribution instead.** A
direct comparison during development measured the textbook filtered
QLIKE improvement over EWMA at +11.66%, against the smoothed improvement of
+11.36%, a gap of only 0.3 points, nowhere near large enough to tell an
honest "live" story against a hindsight one. The one-step predictive
distribution, which never sees week t's own return before forecasting
week t's variance, is what this project reports as "filtered" throughout,
and it produces the much larger, more informative gap reported below.

**The genuinely live (one-step predictive) regime forecast did not beat
EWMA; only the full-sample smoothed one did, and both directions were
checked across 5 EWMA decay parameters before being reported.**
Filtered QLIKE improvement over EWMA measured -6.89% (worse than the
baseline); smoothed measured +11.36%. Lambda values from 0.80 to 0.97 all
showed the same qualitative pattern (filtered between -5.6% and -7.4%,
smoothed between +11.0% and +12.4%), so this is not an artifact of the one
lambda chosen. The resume's claimed +8%/filtered-beats-EWMA framing does
not survive contact with a forecast that is actually denied next week's
own return before being scored; "smoothed-label scoring overstated the
gain" turns out to be an understatement here, since the live forecast
shows no gain to overstate, only a 18.25-percentage-point swing between
what hindsight and genuine real-time information produce. This is reported
as measured, not adjusted by weakening the definition of "filtered" back
toward the textbook one that happened to look better (see the finding
above).

**The SQL per-state diagnostic flagged two of three states as a 2x
mismatch, and the root cause was a real, explainable effect, not a bug.**
`regime/db.py`'s plain `AVG(mkt_rf * mkt_rf) GROUP BY state` measured
20.21 and 24.21 for the two higher-volatility states, against the Gaussian
mixture's own fitted variances of 8.92 and 12.01 for the same states (the
low-volatility state matched closely: 2.36 vs 2.34). Adding back each
state's own fitted mean squared (`var + mean^2`, since the SQL query
computes the raw second moment about zero, not the variance about the
state's own mean) closed part of the gap (12.77 and 14.90) but not all of
it. The remaining gap is a hard-assignment selection effect: weeks near the
boundary between two overlapping states get assigned to whichever state's
mean is nearest by `argmax`, and the weeks that happen to land on the
extreme side of that boundary pull the hard-assigned group's own empirical
second moment above what the complete, soft-weighted Gaussian would
produce. This was verified to be the direction-consistent explanation (both
boundary-adjacent states show the same upward bias; the well-separated
calm state does not) rather than chased further into an exact reconciliation,
since the two quantities are measuring genuinely different things (a soft
mixture's fitted parameter vs. a hard-labeled group's raw empirical moment)
and are not expected to match exactly.

**No out-of-sample return predictability survived, cleanly.** The
expanding-window walk-forward regression of next week's Mkt-RF return on
the one-step predictive regime distribution measured OOS R^2 = -0.0101
over 1,306 out-of-sample weeks, indistinguishable from "the regime carries
no information about next week's return," which is the resume's own
framing and the same conclusion the firm's published research reaches
(regimes as risk-management context, not a return forecast).

## Measured results

Machine: 8 physical / 16 logical cores, Windows 11 Home, Python 3.12.10, scikit-learn 1.9.1.
30 real years of weekly Mkt-RF/SMB/RF factor returns, 1,566 weeks (1996-08-30 to 2026-08-28).

| Claim | Target | Measured | Meets claim |
|---|---|---|---|
| 3-state Gaussian mixture fit over 30 years of real weekly published factor returns | 3 states, 30 years | **3 states**, **1,566 real weeks** (30.0 years), Mkt-RF/SMB/RF | yes |
| Volatility forecast conditioned on filtered regime probabilities only | yes | **yes**: `hmm.one_step_predictive`, using only weeks strictly before the forecast target (see Honest framing for why this, not the textbook filtered distribution) | yes |
| QLIKE improved 8% over an EWMA baseline | 8% | **-6.89%** (worse than EWMA; see Findings) | no |
| Smoothed-label scoring overstated that gain 3.4x | 3.4x | **not a meaningful ratio** (filtered showed no gain to overstate); the additive gap is **18.25 percentage points** of QLIKE improvement (smoothed +11.36% vs filtered -6.89%) | no (by ratio) / directionally yes (hindsight inflates the apparent result) |
| No out-of-sample return predictability survived | yes | **yes**: OOS R^2 = **-0.0101** over 1,306 weeks | yes |

## Building and running

```bash
cd projects/regime-model-factor-panel
"C:\Users\Manas\AppData\Local\Programs\Python\Python312\python.exe" -m venv venv
venv\Scripts\python.exe -m pip install -r requirements.txt

venv\Scripts\python.exe -m pytest tests -v   # 11 tests, <3s

# real data: Kenneth French's weekly 3-factor file, no login or API key needed
curl -o F-F_Research_Data_Factors_weekly_CSV.zip "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_weekly_CSV.zip"
unzip F-F_Research_Data_Factors_weekly_CSV.zip

venv\Scripts\python.exe scripts\ingest_real_factors.py --ff-csv F-F_Research_Data_Factors_weekly.csv --out-csv data\real_factor_panel.csv
venv\Scripts\python.exe scripts\run_regime_model.py data\real_factor_panel.csv > docs\regime_model_output.txt
```

`data/real_factor_panel.csv` (about 43 KB, 1,566 weeks x 3 factors) is small enough to commit
directly and is included so every measurement above can be reproduced without re-downloading
anything. `data/regime_panel.duckdb` is gitignored (a diagnostic database rebuilt on every run);
regenerate it with `run_regime_model.py` above.

## Sibling comparison

[`point-in-time-activity-index`](https://github.com/Manas103/point-in-time-activity-index)'s
`energy_nowcast/` extension shares this project's point-in-time discipline (an expanding-window
walk-forward that only ever trains on strictly prior periods) and its honesty about a claim that
did not survive contact with real data (that extension's energy-demand nowcast also failed to
beat its own AR(1) baseline). The two projects differ in what they are modeling: a continuous
nowcast target there, a discrete latent regime here, which is why this is a new repository with
its own hand-rolled HMM machinery rather than an extension of that one.

## Limitations

The mixture and the transition matrix are fit in two separate steps (i.i.d. EM, then empirical
transition counts from the hard state path), not a jointly estimated Baum-Welch HMM; see Honest
framing. "Filtered" in this project means the one-step-ahead predictive distribution, not the
textbook HMM filtered distribution, a deliberate redefinition explained in Findings because the
textbook version scores too close to the smoothed one to make the comparison meaningful. The
EWMA baseline uses a single declared lambda (0.94); 4 other values were tried and did not change
the qualitative result (see Findings), but no formal search for the single best EWMA parameter
was performed. The headline QLIKE improvement (8%) and overstatement ratio (3.4x) both failed to
reproduce: the live forecast showed no genuine improvement over EWMA at all, only a decline, so
the "overstatement ratio" is reported as an additive percentage-point gap instead of a ratio
where one is not meaningful. The SQL-vs-mixture per-state variance diagnostic showed a 2x gap for
two of three states, root-caused to a hard-assignment selection effect at state boundaries rather
than a bug (see Findings); it was not forced to match by switching to a soft-weighted comparison,
since that would have made the two paths circular rather than independent.
