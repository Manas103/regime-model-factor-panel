"""Builds data/real_factor_panel.csv (small, committed) from Kenneth
French's real weekly 3-factor CSV (not committed; see README for the exact
download URL).

Usage:
    python scripts/ingest_real_factors.py --ff-csv <F-F_Research_Data_Factors_weekly.csv> --out-csv data/real_factor_panel.csv
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from regime import data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ff-csv", required=True)
    parser.add_argument("--out-csv", default="data/real_factor_panel.csv")
    args = parser.parse_args()

    panel = data.build_panel(args.ff_csv)
    panel.to_csv(args.out_csv)
    print(f"n_weeks={len(panel)}")
    print(f"first_week={panel.index.min()}")
    print(f"last_week={panel.index.max()}")
    print(f"wrote {args.out_csv}")


if __name__ == "__main__":
    main()
