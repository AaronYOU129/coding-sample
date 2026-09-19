#!/usr/bin/env python3
"""
  Purpose:    Extend the Part A analysis to the US banking sector. Instead of
              asking how much of each stock's variance is explained by the broad
              market, this script asks how much is explained by the banking sector
              itself. Replacing Mkt-RF with a value-weighted banking sector return
              isolates within-sector co-movement, which is a direct measure of
              systemic risk concentration in banking. All other choices (rolling
              window, value-weighting, year-end evaluation) mirror Part A so the
              two figures are directly comparable.

  Inputs:     data/ff_factors_daily.csv  — Ken French daily factors (RF only,
                                           used to compute stock excess returns)
              data/crsp_daily.csv.gz     — CRSP daily returns and market caps,
                                           filtered to banking SICs 6020–6029
                                           and 6035–6036

  Outputs:    figures/figure4_banking.png  — banking-sector analogue of Figure 4

  Key Steps:  Load FF factors and filter CRSP to banking stocks
              → Construct a daily value-weighted banking sector excess return
              → Merge each banking stock with the sector return
              → Estimate rolling 5-year CAPM R² using the sector return as
                the benchmark (reusing Part A machinery)
              → Value-weight R² across banking stocks and plot

  How to Run: python3 code/main_partb.py
              (run from the project root directory)
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import shared code from Part A
sys.path.insert(0, os.path.dirname(__file__))
from main_parta import (
    load_ff_factors,
    build_all_stocks_rolling_r_squared_dataframe,
    compute_value_weighted_variance_shares,
    add_variance_share_lines,
    format_variance_share_figure,
    save_figure,
    elapsed,
    DATA_DIR,
    FIGURES_DIR,
    VALID_EXCH,
)

# ── Configuration ─────────────────────────────────────────────────────────────

# SIC codes for US commercial banking (assignment definition).
# 6020–6029: state and national commercial banks and trust companies.
# 6035–6036: savings institutions (thrifts), included because they have
# similar balance-sheet structures and face the same interest-rate risks.
BANKING_SICS = set(range(6020, 6030)) | {6035, 6036}


# ── Step 2: Load CRSP filtered to banking stocks ─────────────────────────────

def load_banking_crsp(path: str, ff: pd.DataFrame) -> pd.DataFrame:
    """
    Load CRSP daily data filtered to the US banking sector.

    Why needed: Part B asks about within-sector co-movement, so only banking
    stocks (by SIC code) should enter both the sector-return construction and
    the cross-sectional R² aggregation.  Applying the SIC filter at load time
    keeps memory usage low relative to loading the full CRSP universe.

    All quality filters from Part A (EQTY, NS, major exchanges, non-null return
    and cap) are preserved so the banking sample is a clean subset of the Part A
    sample.

    Inputs:
        path — path to the CRSP daily gzipped CSV
        ff   — Fama-French factors DataFrame (output of load_ff_factors),
               used to attach the risk-free rate for excess-return calculation

    Outputs:
        DataFrame with columns:
            PERMNO, SICCD, date, DlyCap, DlyRet, excess_ret, rf
        DlyRet and rf are kept (in addition to excess_ret) so that the banking
        sector return can be constructed in the next step.
    """
    KEEP = ["PERMNO", "PrimaryExch", "SecurityType", "ShareType",
            "SICCD", "DlyCalDt", "DlyCap", "DlyRet"]
    CHUNK = 5_000_000

    t0    = time.time()
    parts = []
    total_in = total_kept = 0

    print(f"Loading CRSP (banking stocks only) in {CHUNK:,}-row chunks …")
    reader = pd.read_csv(path, chunksize=CHUNK, usecols=KEEP, low_memory=False)

    for i, chunk in enumerate(reader, 1):
        total_in += len(chunk)

        mask = (
            (chunk["SecurityType"] == "EQTY") &
            (chunk["ShareType"]    == "NS")   &
            (chunk["PrimaryExch"].isin(VALID_EXCH)) &
            chunk["DlyRet"].notna() &
            chunk["DlyCap"].notna() &
            # Restrict to banking SIC codes; this is the key addition over Part A.
            chunk["SICCD"].isin(BANKING_SICS)
        )
        chunk = chunk.loc[mask].copy()

        if len(chunk) == 0:
            continue

        chunk["date"] = pd.to_datetime(chunk["DlyCalDt"])

        # Attach the risk-free rate from Fama-French so excess returns can be
        # computed; only days present in the FF sample are retained (inner join).
        chunk = chunk.join(ff[["rf"]], on="date", how="inner")

        chunk["excess_ret"] = chunk["DlyRet"] - chunk["rf"]

        chunk = chunk[["PERMNO", "SICCD", "date", "DlyCap", "DlyRet", "excess_ret", "rf"]]
        parts.append(chunk)
        total_kept += len(chunk)

        print(f"  Chunk {i:2d}: {len(chunk):>7,} kept  |  cumulative {total_kept:>9,}  [{elapsed(t0)}]")

    df = pd.concat(parts, ignore_index=True)
    print(
        f"Banking CRSP loaded: {total_in:,} rows in → {total_kept:,} rows kept  [{elapsed(t0)}]"
        f"\n  Unique banking stocks: {df['PERMNO'].nunique():,}"
        f"  |  Date range: {df['date'].min().date()} – {df['date'].max().date()}\n"
    )
    return df


# ── Step 3: Construct value-weighted banking sector return ────────────────────

def compute_banking_sector_return(banking_df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct the daily value-weighted banking sector excess return.

    Why needed: Part B replaces the broad Mkt-RF factor with a banking-sector
    portfolio return as the CAPM benchmark.  Regressing each bank's excess
    return on this sector return isolates within-sector co-movement — a direct
    measure of how much of any one bank's risk is driven by common shocks to
    the banking industry rather than idiosyncratic firm-level events.

    Construction follows standard portfolio-return methodology: weights are
    each stock's lagged (previous-day) market capitalisation, which avoids
    look-ahead bias.  Using contemporaneous caps would require knowing the
    end-of-day price to set the weight, creating a circular dependency.

    On each date t:
        banking_excess_ret_t = Σ_i [ cap_{i,t-1} × excess_ret_{i,t} ]
                               ─────────────────────────────────────
                                        Σ_i cap_{i,t-1}

    Inputs:
        banking_df — panel of banking stocks (output of load_banking_crsp)

    Outputs:
        DataFrame indexed by date with a single column:
            banking_mkt_rf — daily value-weighted banking sector excess return
    """
    # Sort so shift() produces the previous calendar-day observation per stock,
    # not the previous row in the DataFrame (which could belong to a different stock).
    df = banking_df.sort_values(["PERMNO", "date"]).copy()
    df["lagged_cap"] = df.groupby("PERMNO")["DlyCap"].shift(1)

    # Drop the first observation of each stock (lagged cap unavailable) and
    # any rows where the lagged cap is zero or missing, as these cannot be weighted.
    df = df.dropna(subset=["lagged_cap", "excess_ret"])
    df = df[df["lagged_cap"] > 0]

    # Pre-multiply each excess return by its lagged weight before grouping;
    # this avoids a weighted-mean aggregation and keeps the arithmetic transparent.
    df["weighted_ret"] = df["lagged_cap"] * df["excess_ret"]

    daily = df.groupby("date").agg(
        weighted_ret_sum=("weighted_ret", "sum"),
        cap_sum=("lagged_cap", "sum"),
    )
    daily["banking_mkt_rf"] = daily["weighted_ret_sum"] / daily["cap_sum"]
    daily = daily[["banking_mkt_rf"]].sort_index()

    print(
        f"Banking sector return: {len(daily):,} trading days  "
        f"({daily.index.min().date()} – {daily.index.max().date()})\n"
    )
    return daily


# ── Step 4: Merge banking stocks with banking sector return ───────────────────

def merge_banking_data(
    banking_df: pd.DataFrame,
    banking_sector_ret: pd.DataFrame,
) -> pd.DataFrame:
    """
    Attach the banking sector return to each banking stock observation,
    renaming it to 'mkt_rf' so that Part A's rolling-regression machinery
    can be called without modification.

    Why needed: The Part A function build_all_stocks_rolling_r_squared_dataframe
    expects a column named 'mkt_rf' as the CAPM benchmark.  By renaming the
    banking sector return to 'mkt_rf' here, we reuse the entire estimation
    pipeline without any changes — the only difference is what 'mkt_rf' represents
    (broad market vs. banking sector).

    Note: we do NOT use a leave-one-out sector return (the stock's own return is
    not excluded from the sector return).  This is consistent with Part A, where
    each stock is included in the FF market return.  For large stocks, inclusion
    slightly inflates R², but this bias is negligible given that no single bank
    dominates the sector return.

    Inputs:
        banking_df         — panel of banking stocks (output of load_banking_crsp)
        banking_sector_ret — daily sector return (output of compute_banking_sector_return)

    Outputs:
        Merged DataFrame with columns:
            PERMNO, SICCD, date, DlyCap, excess_ret, mkt_rf
        Ready to pass directly to build_all_stocks_rolling_r_squared_dataframe.
    """
    merged = banking_df.join(banking_sector_ret, on="date", how="inner")
    merged = merged.rename(columns={"banking_mkt_rf": "mkt_rf"})
    merged = merged[["PERMNO", "SICCD", "date", "DlyCap", "excess_ret", "mkt_rf"]]
    merged = merged.dropna(subset=["mkt_rf"])

    print(
        f"Banking merged: {len(merged):,} rows  |  "
        f"{merged['PERMNO'].nunique():,} unique stocks\n"
    )
    return merged


# ── Step 7: Plot ──────────────────────────────────────────────────────────────


def plot_figure4_banking(vw: pd.DataFrame, outpath: str) -> None:
    """
    Produce and save the banking-sector analogue of Figure 4.

    Why needed: Placing the banking figure on the same scale and with the same
    style as the Part A figure allows direct visual comparison of within-sector
    co-movement in banking against the broader market co-movement in Part A.

    Reuses Part A's add_variance_share_lines and format_variance_share_figure
    to guarantee identical styling across both figures.

    Inputs:
        vw      — output of compute_value_weighted_variance_shares for banking
        outpath — file path where the PNG will be saved
    """
    figure, axis = plt.subplots(figsize=(10, 6))
    add_variance_share_lines(axis, vw)
    format_variance_share_figure(axis)
    axis.set_title("The share of systematic vs idiosyncratic risk (US Banking Sector)")
    save_figure(figure, outpath)
    print(f"Figure saved → {outpath}")
    plt.close(figure)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    Full research pipeline for Part B (banking sector extension).

    Pipeline:
        Raw Data (FF factors + CRSP banking stocks)
          → Sector Return Construction     (Steps 2–3)
          → Merge & Rolling R² Estimation  (Steps 4–5)
          → Value-Weighted Aggregation     (Step 6)
          → Figure Output                  (Step 7)
    """
    t_total = time.time()

    # Use absolute paths derived from this file's location so the script can
    # be run from any working directory, not just the project root.
    _root     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ff_path   = os.path.join(_root, "data", "ff_factors_daily.csv")
    crsp_path = os.path.join(_root, "data", "crsp_daily.csv.gz")
    fig_path  = os.path.join(_root, "figures", "figure4_banking.png")

    year_ends = pd.date_range("1960-12-31", "2025-12-31", freq="YE")

    # ── 1. Fama-French (risk-free rate only) ────────────────────────────────
    # In Part B the market excess return (Mkt-RF) is not used directly; only
    # the risk-free rate is needed to convert raw CRSP returns to excess returns.
    print("=" * 60)
    print("Step 1: Load Fama-French daily factors (for RF)")
    print("=" * 60)
    ff = load_ff_factors(ff_path)
    print(f"  {len(ff):,} days  ({ff.index.min().date()} – {ff.index.max().date()})\n")

    # Verify date coverage and that values are in decimal (not percentage) form.
    print("FF date range:", ff.index.min(), "to", ff.index.max())
    print(ff.head())

    # ── 2. Load banking stocks ───────────────────────────────────────────────
    # Apply the SIC filter at load time to keep only banks and thrifts;
    # this reduces the working dataset from ~14M to ~3.7M rows.
    print("=" * 60)
    print("Step 2: Load banking stocks (SIC 6020–6029, 6035–6036)")
    print("=" * 60)
    banking_df = load_banking_crsp(crsp_path, ff)

    # Verify the banking filter produced a sensible number of stocks and
    # covers the expected date range (1962 onwards, when CRSP SIC codes begin).
    print("Banking shape:", banking_df.shape)
    print("Unique banking stocks:", banking_df["PERMNO"].nunique())
    print("Banking date range:", banking_df["date"].min(), "to", banking_df["date"].max())

    # ── 3. Construct VW banking sector return ────────────────────────────────
    # Build the daily value-weighted portfolio of all banking stocks; this
    # becomes the CAPM benchmark regressor replacing Mkt-RF from Part A.
    print("=" * 60)
    print("Step 3: Construct value-weighted banking sector excess return")
    print("=" * 60)
    banking_sector_ret = compute_banking_sector_return(banking_df)

    # ── 4. Merge: replace mkt_rf with banking sector return ─────────────────
    # Rename banking_mkt_rf → mkt_rf so Part A's regression function receives
    # exactly the column name it expects with no code modification.
    print("=" * 60)
    print("Step 4: Merge banking stocks with banking sector return")
    print("=" * 60)
    merged = merge_banking_data(banking_df, banking_sector_ret)

    # ── 5. Rolling R² — reuse Part A machinery ──────────────────────────────
    # Run the same rolling CAPM estimation as Part A, but with the banking
    # sector return as the benchmark.  R² now measures the fraction of each
    # bank's variance explained by common banking-sector shocks.
    print("=" * 60)
    print("Step 5: Rolling 5-year regression R² at each year-end")
    print("=" * 60)
    r2_df = build_all_stocks_rolling_r_squared_dataframe(merged, year_ends)

    # R² is a squared correlation and must lie in [0, 1] by construction.
    # A violation would indicate a numerical error in the correlation step.
    print("R² summary:")
    print(r2_df["R_squared"].describe())
    assert ((r2_df["R_squared"] >= 0) & (r2_df["R_squared"] <= 1)).all(), \
        "R_squared outside [0,1]"

    # ── 6. Value-weighted aggregation ────────────────────────────────────────
    # Aggregate bank-level R² into annual value-weighted sector shares.
    # Renaming R_squared → r2 matches the column name expected by the shared
    # aggregation function imported from Part A.
    print("=" * 60)
    print("Step 6: Value-weighted aggregation")
    print("=" * 60)
    vw = compute_value_weighted_variance_shares(r2_df.rename(columns={"R_squared": "r2"}))

    # Systematic + idiosyncratic shares must sum to 1 for every year because
    # idiosyncratic_share is defined as 1 − systematic_share.  A violation
    # would indicate an error in the aggregation function.
    print("VW head:")
    print(vw.head())
    share_sum = vw["systematic_share"] + vw["idiosyncratic_share"]
    assert np.allclose(share_sum, 1.0, atol=1e-8), \
        "Shares do not sum to 1"

    print(vw.to_string(index=False))

    # ── 7. Plot ──────────────────────────────────────────────────────────────
    # Produce the banking-sector figure using the same style as Part A,
    # enabling direct visual comparison of the two decompositions.
    print("\n" + "=" * 60)
    print("Step 7: Plot")
    print("=" * 60)
    plot_figure4_banking(vw, fig_path)

    print(f"\nAll done. Total elapsed: {elapsed(t_total)}")


if __name__ == "__main__":
    main()
