"""Parses Kenneth French's real weekly 3-factor CSV and restricts it to
the most recent `config.N_YEARS` real years in the file.
"""

from __future__ import annotations

import datetime

import pandas as pd

from regime import config


def load_weekly_factors(csv_path: str) -> pd.DataFrame:
    """Returns a DataFrame indexed by the week's end date (real calendar
    dates from the French library), columns Mkt-RF, SMB, HML, RF, all in
    percent per week, exactly as published, no detrending or filtering."""
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    header_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith(",Mkt-RF"):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError("could not find the header row in the French weekly factor file")
    header = [c.strip() for c in lines[header_idx].strip().split(",")]
    for line in lines[header_idx + 1 :]:
        parts = [p.strip() for p in line.strip().split(",")]
        if len(parts) != len(header) or not parts[0].isdigit():
            continue
        date = datetime.datetime.strptime(parts[0], "%Y%m%d").date()
        values = [float(p) for p in parts[1:]]
        rows.append((date,) + tuple(values))
    df = pd.DataFrame(rows, columns=["date"] + header[1:])
    df = df.set_index("date").sort_index()
    return df


def most_recent_n_years(df: pd.DataFrame, n_years: int) -> pd.DataFrame:
    last_date = df.index.max()
    cutoff = last_date.replace(year=last_date.year - n_years)
    return df[df.index > cutoff].copy()


def build_panel(csv_path: str) -> pd.DataFrame:
    full = load_weekly_factors(csv_path)
    window = most_recent_n_years(full, config.N_YEARS)
    return window[config.FACTOR_COLUMNS]
