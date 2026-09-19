"""
  Purpose:    Replicate Figure 4 from Lochstoer and Muir (2025), which shows
              the long-run rise in the systematic share of US stock return
              variance. A rolling 5-year CAPM window is used because the paper's
              result is about slow structural change, not short-run volatility.
              Value-weighting by beginning-of-year market cap (not equal-weighting)
              prevents the long tail of small illiquid stocks from dominating.

  Inputs:     data/ff_factors_daily.csv  — Ken French daily Fama-French factors
                                           (Mkt-RF and RF, in decimal form)
              data/crsp_daily.csv.gz     — CRSP daily returns and market caps
                                           (common equity on NYSE/AMEX/NASDAQ)

  Outputs:    figures/figure4_replication.png  — replicated Figure 4

  Key Steps:  Load FF factors and CRSP data
              → Filter to common equity and merge to get excess returns
              → For each stock at each year-end, estimate CAPM R² over
                the preceding 5 years
              → Value-weight R² across stocks to get systematic share
              → Plot systematic and idiosyncratic shares with trend lines

  How to Run: python3 code/main_parta.py
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
import matplotlib.ticker as mtick

# ── Configuration ─────────────────────────────────────────────────────────────

# 5-year window follows Lochstoer and Muir (2025) and is long enough to
# estimate a reliable beta while remaining short enough to capture structural
# shifts over decades.
WINDOW_YEARS = 5       # rolling regression window in calendar years

# ~750 trading days ≈ 3 years of data within the 5-year window.  Requiring
# this floor prevents noisy R² estimates for thinly-traded or newly-listed
# stocks that happen to survive a year-end.
MIN_OBS      = 750     # minimum daily observations required in window

# Restrict to the three main US equity exchanges to exclude pink-sheet and
# OTC stocks, which have unreliable prices and market caps in CRSP.
VALID_EXCH   = {"N", "A", "Q"}   # NYSE, AMEX, NASDAQ

_ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(_ROOT, "data")
FIGURES_DIR = os.path.join(_ROOT, "figures")


# ── Helpers ───────────────────────────────────────────────────────────────────

def elapsed(t0: float) -> str:
    """Return a human-readable elapsed-time string since t0."""
    s = time.time() - t0
    return f"{s / 60:.1f}m" if s >= 60 else f"{s:.1f}s"


# ── Step 1: Fama-French factors ───────────────────────────────────────────────

def load_ff_factors(path: str) -> pd.DataFrame:
    """
    Load Ken French daily Fama-French factors and return market excess return
    and risk-free rate in decimal form.

    Why needed: The risk-free rate is required to convert raw CRSP returns into
    excess returns (r_i - rf) before running CAPM regressions.  Mkt-RF is the
    benchmark regressor in the CAPM.

    Inputs:
        path — path to the Ken French daily CSV file.  The file has a 4-row
               text header and a trailing copyright line; values are in
               percentage points.

    Outputs:
        DataFrame indexed by date with columns:
            mkt_rf  — daily market excess return (decimal)
            rf      — daily risk-free rate (decimal)
    """
    ff = pd.read_csv(path, skiprows=4, header=0)

    # The French file ends with a copyright line that cannot be parsed as a
    # date integer; dropping it avoids downstream type errors.
    ff = ff[pd.to_numeric(ff.iloc[:, 0], errors="coerce").notna()].copy()
    ff.columns = ["date_int", "mkt_rf", "smb", "hml", "rf"]

    ff["date"]   = pd.to_datetime(ff["date_int"].astype(int).astype(str), format="%Y%m%d")
    # French publishes factors as percentages (e.g. 0.15 means 0.15%).
    # Divide by 100 so all arithmetic with CRSP returns stays in decimal form.
    ff["mkt_rf"] = ff["mkt_rf"].astype(float) / 100
    ff["rf"]     = ff["rf"].astype(float) / 100

    ff = ff.set_index("date").sort_index()
    return ff[["mkt_rf", "rf"]]


# ── Step 2: CRSP daily returns ────────────────────────────────────────────────
def load_crsp_data(crsp_path: str, ff_data: pd.DataFrame) -> pd.DataFrame:
    """
    Load the full CRSP daily file, filter to investable common stocks, merge
    with Fama-French factors, and return a lean panel of excess returns.

    Why needed: CRSP contains all securities (bonds, ETFs, ADRs, preferred
    shares, etc.).  Restricting to EQTY/NS/major-exchange observations keeps
    only the common stocks that belong in a market-portfolio calculation,
    consistent with standard empirical asset-pricing practice.

    The file is ~103M rows and cannot be loaded into memory at once; chunked
    reading trades latency for a manageable memory footprint.

    Inputs:
        crsp_path — path to CRSP daily gzipped CSV (~103M rows)
        ff_data   — Fama-French factors DataFrame (output of load_ff_factors),
                    used to attach rf and mkt_rf to each stock-day observation

    Outputs:
        DataFrame with columns:
            PERMNO      — CRSP permanent security identifier
            SICCD       — SIC industry code (used in Part B for sector filters)
            date        — calendar date
            DlyCap      — market capitalisation (price × shares, in dollars)
            excess_ret  — daily excess return: DlyRet − rf
            mkt_rf      — daily market excess return (from Fama-French)
    """
    columns_to_keep = [
        "PERMNO",
        "PrimaryExch",
        "SecurityType",
        "ShareType",
        "SICCD",
        "DlyCalDt",
        "DlyCap",
        "DlyRet",
    ]
    chunk_size = 5_000_000

    start_time = time.time()
    processed_chunks = []
    total_input_rows = 0
    total_kept_rows = 0

    print(f"Loading CRSP in {chunk_size:,}-row chunks …")
    chunk_reader = pd.read_csv(
        crsp_path,
        chunksize=chunk_size,
        usecols=columns_to_keep,
        low_memory=False,
    )

    for chunk_number, crsp_chunk in enumerate(chunk_reader, start=1):
        total_input_rows += len(crsp_chunk)

        processed_chunk = process_crsp_chunk(crsp_chunk, ff_data)
        processed_chunks.append(processed_chunk)
        total_kept_rows += len(processed_chunk)

        report_crsp_chunk_progress(
            chunk_number=chunk_number,
            kept_rows=len(processed_chunk),
            total_kept_rows=total_kept_rows,
            start_time=start_time,
        )

    crsp_data = pd.concat(processed_chunks, ignore_index=True)
    print(
        f"CRSP loaded: {total_input_rows:,} rows in → "
        f"{total_kept_rows:,} rows kept  [{elapsed(start_time)}]\n"
    )
    return crsp_data


def process_crsp_chunk(crsp_chunk: pd.DataFrame, ff_data: pd.DataFrame) -> pd.DataFrame:
    """
    Apply quality filters to one CRSP chunk, attach Fama-French factors, and
    compute stock-level excess returns.

    Why needed: Separating per-chunk logic from the outer loop keeps the loading
    function readable and makes it easy to unit-test filtering in isolation.

    Inputs:
        crsp_chunk — one 5M-row slice of the raw CRSP file
        ff_data    — Fama-French factors (index = date) for the rf merge

    Outputs:
        Filtered DataFrame with columns matching the final CRSP schema, or an
        empty DataFrame with the same schema if no rows survive filtering.
    """
    filtered_chunk = filter_common_stocks(crsp_chunk)
    if filtered_chunk.empty:
        return build_empty_crsp_output()

    filtered_chunk = filtered_chunk.copy()
    filtered_chunk["date"] = pd.to_datetime(filtered_chunk["DlyCalDt"])

    # Inner join on date ensures we only keep stock-days for which we have a
    # risk-free rate; days outside the French factor sample are dropped.
    merged_chunk = filtered_chunk.join(ff_data[["mkt_rf", "rf"]], on="date", how="inner")
    if merged_chunk.empty:
        return build_empty_crsp_output()

    # Excess return = raw return minus the risk-free rate.  This is the
    # left-hand-side variable in the CAPM regression: r_i - rf = α + β(r_m - rf) + ε
    merged_chunk["excess_ret"] = merged_chunk["DlyRet"] - merged_chunk["rf"]

    return merged_chunk[["PERMNO", "SICCD", "date", "DlyCap", "excess_ret", "mkt_rf"]]


def filter_common_stocks(crsp_chunk: pd.DataFrame) -> pd.DataFrame:
    """
    Retain only investable common equity observations from one CRSP chunk.

    Why needed: CRSP contains many security types (bonds, preferred shares,
    ETFs, ADRs) and exchange venues (OTC, pink sheets) that should not enter a
    market-portfolio calculation.  The three filters below mirror the standard
    selection used in empirical asset pricing (e.g. Fama and French 1993):
        - SecurityType == "EQTY"  : drop non-equity instruments
        - ShareType    == "NS"    : ordinary common shares only
        - PrimaryExch  in {N,A,Q} : NYSE, AMEX, NASDAQ only
    Missing returns or market caps are dropped because both are required for
    the CAPM regression and value-weighting.

    Inputs:
        crsp_chunk — raw chunk from the CRSP daily CSV

    Outputs:
        Subset of crsp_chunk passing all filters (may be empty).
    """
    valid_stock_mask = (
        (crsp_chunk["SecurityType"] == "EQTY")
        & (crsp_chunk["ShareType"] == "NS")
        & (crsp_chunk["PrimaryExch"].isin(VALID_EXCH))
        & crsp_chunk["DlyRet"].notna()
        & crsp_chunk["DlyCap"].notna()
    )
    return crsp_chunk.loc[valid_stock_mask]


def build_empty_crsp_output() -> pd.DataFrame:
    """
    Return an empty DataFrame with the canonical CRSP output schema.

    Why needed: Ensures pd.concat() in the outer loop always receives
    DataFrames with identical columns, avoiding column-alignment errors when
    an entire chunk is filtered out (e.g. chunks that cover non-equity dates).
    """
    return pd.DataFrame(
        columns=["PERMNO", "SICCD", "date", "DlyCap", "excess_ret", "mkt_rf"]
    )


def report_crsp_chunk_progress(
    chunk_number: int,
    kept_rows: int,
    total_kept_rows: int,
    start_time: float,
) -> None:
    """
    Print a one-line progress update after each CRSP chunk is processed.

    Why needed: Loading 103M rows takes ~2 minutes; live progress lets the
    user verify the job is running and estimate remaining time.
    """
    print(
        f"  Chunk {chunk_number:2d}: {kept_rows:>9,} kept  |  "
        f"cumulative {total_kept_rows:>12,}  [{elapsed(start_time)}]"
    )


# ── Step 3: Rolling 5-year CAPM R² at year-end dates ─────────────────────────

def build_all_stocks_rolling_r_squared_dataframe(
    merged: pd.DataFrame,
    year_ends: pd.DatetimeIndex,
    min_obs: int = MIN_OBS,
    window_years: int = WINDOW_YEARS,
) -> pd.DataFrame:
    """
    Compute the rolling CAPM R² for every stock at every year-end, returning
    a long panel of stock-year R² values and market caps.

    Why needed: R² from the CAPM regression measures the fraction of a stock's
    return variance explained by the market factor — i.e. its systematic share.
    Evaluating at year-ends with a fixed 5-year lookback produces a time series
    of systematic shares comparable to Figure 4 in the paper.

    For simple OLS with an intercept, R² equals the squared Pearson correlation
    between stock excess returns and market excess returns.  Using this identity
    avoids constructing design matrices and running full OLS for each of the
    ~500,000 stock-year pairs, reducing runtime by an order of magnitude.

    Inputs:
        merged       — panel of daily stock excess returns and market returns
                       (output of load_crsp_data)
        year_ends    — DatetimeIndex of year-end evaluation dates (Dec 31 each year)
        min_obs      — minimum number of valid daily observations required within
                       the window; windows below this threshold are skipped to
                       avoid unreliable R² estimates for thinly-traded stocks
        window_years — length of the lookback window in calendar years

    Outputs:
        DataFrame with columns:
            year        — calendar year of the evaluation date
            PERMNO      — CRSP permanent security identifier
            R_squared   — CAPM R² over the preceding window_years years
            market_cap  — market capitalisation at year-end (used for value-weighting)
    """
    year_end_dates, window_start_dates = prepare_rolling_window_dates(year_ends, window_years)
    stock_groups = merged.sort_values("date").groupby("PERMNO", sort=False)

    number_of_stocks = merged["PERMNO"].nunique()
    start_time = time.time()
    results = []

    print(f"Computing rolling R² for {number_of_stocks:,} stocks × {len(year_ends)} year-ends …")

    for stock_index, (stock_id, stock_data) in enumerate(stock_groups, start=1):
        stock_results = build_single_stock_rolling_r_squared_dataframe(
            stock_id=stock_id,
            stock_data=stock_data,
            year_ends=year_ends,
            year_end_dates=year_end_dates,
            window_start_dates=window_start_dates,
            min_obs=min_obs,
        )
        results.extend(stock_results)

        report_progress(stock_index, number_of_stocks, start_time)

    print(f"Done. {len(results):,} stock-year observations  [{elapsed(start_time)}]\n")
    return pd.DataFrame(results, columns=["year", "PERMNO", "R_squared", "market_cap"])


def prepare_rolling_window_dates(
    year_ends: pd.DatetimeIndex,
    window_years: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Precompute year-end dates and window-start dates as numpy datetime64[D]
    arrays for use in binary-search window lookups.

    Why needed: Converting to numpy datetime64[D] once — rather than inside
    the per-stock loop — eliminates repeated type-casting overhead and enables
    vectorised searchsorted comparisons that are orders of magnitude faster
    than Python-level date arithmetic across 10,000+ stock-year pairs.

    Inputs:
        year_ends    — pandas DatetimeIndex of year-end evaluation dates
        window_years — lookback window length in calendar years

    Outputs:
        (year_end_dates, window_start_dates) — paired numpy arrays of the
        same length; window_start_dates[i] is exactly window_years before
        year_end_dates[i].
    """
    year_end_dates = year_ends.to_numpy().astype("datetime64[D]")
    window_start_dates = np.array(
        [(year_end - pd.DateOffset(years=window_years)).date() for year_end in year_ends],
        dtype="datetime64[D]",
    )
    return year_end_dates, window_start_dates


def build_single_stock_rolling_r_squared_dataframe(
    stock_id: int,
    stock_data: pd.DataFrame,
    year_ends: pd.DatetimeIndex,
    year_end_dates: np.ndarray,
    window_start_dates: np.ndarray,
    min_obs: int,
) -> list[tuple[int, int, float, float]]:
    """
    Compute the rolling CAPM R² for one stock across all year-end windows.

    Why needed: Isolating per-stock logic makes the outer loop over all stocks
    easier to read, and keeps the numpy array extraction (a one-time cost per
    stock) separate from the per-window estimation loop.

    Inputs:
        stock_id           — PERMNO identifier for this stock
        stock_data         — daily observations for this stock (date-sorted)
        year_ends          — full set of year-end evaluation dates
        year_end_dates     — same dates as numpy datetime64[D] for searchsorted
        window_start_dates — window start dates as numpy datetime64[D]
        min_obs            — minimum valid observations required per window

    Outputs:
        List of (year, PERMNO, R_squared, market_cap) tuples, one per valid
        year-end window.  Windows with insufficient data are omitted.
    """
    stock_data = stock_data.sort_values("date")

    # Extract as numpy arrays once per stock; avoids repeated pandas overhead
    # inside the inner year-end loop, which runs up to 66 times per stock.
    stock_dates = stock_data["date"].to_numpy().astype("datetime64[D]")
    stock_excess_returns = stock_data["excess_ret"].to_numpy(dtype=np.float64)
    stock_market_returns = stock_data["mkt_rf"].to_numpy(dtype=np.float64)
    stock_market_caps = stock_data["DlyCap"].to_numpy(dtype=np.float64)

    stock_results = []

    for year_end, window_start, year_end_date in zip(year_ends, window_start_dates, year_end_dates):
        start_index, end_index = find_window_bounds(stock_dates, window_start, year_end_date)

        r_squared = estimate_single_window_r_squared(
            stock_excess_returns=stock_excess_returns,
            stock_market_returns=stock_market_returns,
            start_index=start_index,
            end_index=end_index,
            min_obs=min_obs,
        )
        if r_squared is None:
            continue

        year_end_market_cap = get_year_end_market_cap(stock_market_caps, end_index)
        if year_end_market_cap is None:
            continue

        stock_results.append((year_end.year, int(stock_id), r_squared, year_end_market_cap))

    return stock_results


def find_window_bounds(
    stock_dates: np.ndarray,
    window_start: np.datetime64,
    year_end_date: np.datetime64,
) -> tuple[int, int]:
    """
    Return the half-open index range (start_index, end_index] that covers the
    rolling window (window_start, year_end_date] in the stock's date array.

    Why needed: Binary search on a sorted date array is O(log n) and avoids
    boolean masking, which would allocate a new array for every stock-year pair.

    Inputs:
        stock_dates    — sorted numpy datetime64[D] array of trading dates
        window_start   — first date excluded from the window (window_start < t ≤ year_end_date)
        year_end_date  — last date included in the window

    Outputs:
        (start_index, end_index) — integer indices such that
        stock_dates[start_index:end_index] covers the window.
    """
    # side="right" on window_start excludes the start date itself, giving
    # a half-open interval consistent with (window_start, year_end_date].
    start_index = int(np.searchsorted(stock_dates, window_start, side="right"))
    end_index = int(np.searchsorted(stock_dates, year_end_date, side="right"))
    return start_index, end_index


def estimate_single_window_r_squared(
    stock_excess_returns: np.ndarray,
    stock_market_returns: np.ndarray,
    start_index: int,
    end_index: int,
    min_obs: int,
) -> float | None:
    """
    Estimate CAPM R² for one rolling window using the Pearson correlation identity.

    Why needed: For a simple OLS regression with an intercept, R² equals the
    squared Pearson correlation between the dependent and independent variables.
    Computing np.corrcoef avoids constructing (X'X)^{-1} and the full OLS
    solution, cutting per-window cost from O(n) matrix ops to a single pass.

    Inputs:
        stock_excess_returns — full array of daily excess returns for this stock
        stock_market_returns — full array of daily market excess returns
        start_index          — first index of the window (inclusive)
        end_index            — last index of the window (exclusive)
        min_obs              — minimum number of finite observations required

    Outputs:
        R² in [0, 1], or None if the window is empty, too short, or produces
        a non-finite result (e.g. if the stock had zero return variance).
    """
    if end_index <= start_index:
        return None

    stock_excess_returns_window = stock_excess_returns[start_index:end_index]
    market_excess_returns_window = stock_market_returns[start_index:end_index]

    # Remove days where either series has a missing or infinite value; these
    # can arise from trading halts, data errors, or delisting returns.
    valid_mask = np.isfinite(stock_excess_returns_window) & np.isfinite(market_excess_returns_window)
    if valid_mask.sum() < min_obs:
        return None

    stock_excess_returns_window = stock_excess_returns_window[valid_mask]
    market_excess_returns_window = market_excess_returns_window[valid_mask]

    # R² = ρ(r_i - rf, r_m - rf)²  for a CAPM regression with intercept.
    # Squaring the off-diagonal element of the 2×2 correlation matrix gives
    # the proportion of the stock's excess-return variance explained by the
    # market factor — i.e. the systematic share.
    r_squared = float(np.corrcoef(stock_excess_returns_window, market_excess_returns_window)[0, 1]) ** 2
    if not np.isfinite(r_squared):
        # Can occur for stocks with exactly zero return variance in the window
        # (e.g. a price-frozen stock); treat as missing rather than R² = 0.
        return None

    return r_squared


def get_year_end_market_cap(
    stock_market_caps: np.ndarray,
    end_index: int,
) -> float | None:
    """
    Return the market capitalisation of the stock on the last trading day of
    the window, to be used as the value-weighting denominator.

    Why needed: Value-weighting by end-of-year market cap — rather than
    equal-weighting — ensures that large, liquid stocks drive the aggregate
    variance decomposition, consistent with the paper and with standard
    portfolio construction practice.

    Inputs:
        stock_market_caps — full array of daily market caps for this stock
        end_index         — exclusive upper bound of the window; the last
                            observation in the window is at end_index - 1

    Outputs:
        Market cap (positive finite float), or None if unavailable or invalid.
    """
    if end_index <= 0:
        return None

    year_end_market_cap = float(stock_market_caps[end_index - 1])
    if not np.isfinite(year_end_market_cap) or year_end_market_cap <= 0:
        return None

    return year_end_market_cap


def report_progress(
    stock_index: int,
    n_stocks: int,
    start_time: float,
) -> None:
    """
    Print a progress line every 2,000 stocks and at the final stock.

    Why needed: The outer loop over ~10,000 stocks runs for about 1–2 minutes;
    periodic updates let the user monitor throughput without flooding stdout.
    """
    if stock_index % 2_000 != 0 and stock_index != n_stocks:
        return

    print(
        f"  {stock_index:>6,}/{n_stocks:,}  "
        f"({100 * stock_index / n_stocks:4.1f}%)  "
        f"[{elapsed(start_time)}]"
    )

# ── Step 4: Value-weighted aggregation ────────────────────────────────────────

def compute_value_weighted_variance_shares(r2_data: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate stock-level CAPM R² values into yearly value-weighted systematic
    and idiosyncratic variance shares for the full CRSP universe.

    Why needed: The paper's Figure 4 shows aggregate (not average-firm)
    variance decomposition.  Value-weighting by market cap gives each stock
    a weight proportional to its share of total market capitalisation, so the
    aggregate reflects the experience of a value-weighted investor rather than
    the average small stock.

    Inputs:
        r2_data — long panel with columns: year, PERMNO, r2, market_cap
                  (output of build_all_stocks_rolling_r_squared_dataframe,
                   after renaming R_squared → r2)

    Outputs:
        DataFrame with one row per year and columns:
            year                — calendar year
            systematic_share    — Σ(cap_i × R²_i) / Σ(cap_i); fraction of
                                  total return variance explained by the market
            idiosyncratic_share — 1 − systematic_share
            n_stocks            — number of stocks contributing to that year
    """
    yearly_share_rows = []

    for year, yearly_stock_data in r2_data.groupby("year"):
        yearly_share_row = compute_yearly_value_weighted_shares(year, yearly_stock_data)
        if yearly_share_row is None:
            continue
        yearly_share_rows.append(yearly_share_row)

    return build_yearly_share_dataframe(yearly_share_rows)


def compute_yearly_value_weighted_shares(
    year: int,
    yearly_stock_data: pd.DataFrame,
) -> tuple[int, float, float, int] | None:
    """
    Compute the value-weighted systematic and idiosyncratic variance shares
    for a single year.

    Why needed: Isolating the per-year calculation makes it easy to skip
    degenerate years (e.g. years with no stocks or zero total market cap)
    without cluttering the outer aggregation loop.

    Inputs:
        year              — calendar year being aggregated
        yearly_stock_data — rows of r2_data belonging to this year, with
                            columns market_cap and r2

    Outputs:
        (year, systematic_share, idiosyncratic_share, n_stocks) tuple,
        or None if total market cap is zero or negative (cannot value-weight).
    """
    total_market_cap = yearly_stock_data["market_cap"].sum()
    if total_market_cap <= 0:
        return None

    # Value-weighted R² = Σ(cap_i × R²_i) / Σ(cap_i).
    # This is the share of aggregate return variance attributable to the
    # common market factor under the CAPM.
    weighted_r2_sum = (yearly_stock_data["market_cap"] * yearly_stock_data["r2"]).sum()
    systematic_share = float(weighted_r2_sum / total_market_cap)
    idiosyncratic_share = 1.0 - systematic_share
    number_of_stocks = len(yearly_stock_data)

    return year, systematic_share, idiosyncratic_share, number_of_stocks


def build_yearly_share_dataframe(
    yearly_share_rows: list[tuple[int, float, float, int]],
) -> pd.DataFrame:
    """
    Convert a list of per-year share tuples into a clean, year-sorted DataFrame.

    Why needed: Wrapping the DataFrame constructor here keeps the aggregation
    loop free of formatting concerns and ensures consistent column names and
    sort order for downstream plotting.
    """
    return (
        pd.DataFrame(
            yearly_share_rows,
            columns=["year", "systematic_share", "idiosyncratic_share", "n_stocks"],
        )
        .sort_values("year")
        .reset_index(drop=True)
    )

# ── Step 5: Plot ──────────────────────────────────────────────────────────────
def plot_value_weighted_variance_shares(
    yearly_share_data: pd.DataFrame,
    output_path: str,
) -> None:
    """
    Produce and save the replication of Figure 4: value-weighted systematic
    and idiosyncratic variance shares over time for the full CRSP universe.

    Why needed: The figure is the primary output of Part A.  Separating figure
    creation from data computation keeps the pipeline modular — the same
    plotting helpers are reused in Part B without modification.

    Inputs:
        yearly_share_data — output of compute_value_weighted_variance_shares
        output_path       — file path where the PNG will be saved
    """
    figure, axis = plt.subplots(figsize=(10, 5))

    add_variance_share_lines(axis, yearly_share_data)
    format_variance_share_figure(axis)
    save_figure(figure, output_path)

    print(f"Figure saved → {output_path}")
    plt.close(figure)


def add_variance_share_lines(
    axis: plt.Axes,
    yearly_share_data: pd.DataFrame,
) -> None:
    """
    Draw the systematic-share and idiosyncratic-share time series on the axis.

    Why needed: Separating data-series rendering from axis formatting allows
    Part B to reuse the same line-drawing logic with a different dataset
    (banking sector) without duplicating style choices.

    Inputs:
        axis             — matplotlib Axes object to draw on
        yearly_share_data — DataFrame with columns year, systematic_share,
                            idiosyncratic_share (output of aggregation step)
    """
    axis.plot(
        yearly_share_data["year"],
        yearly_share_data["systematic_share"],
        color="#2166ac",
        linewidth=1.8,
        label="Systematic Share (R²)",
    )
    axis.plot(
        yearly_share_data["year"],
        yearly_share_data["idiosyncratic_share"],
        color="#d6604d",
        linewidth=1.8,
        linestyle="--",
        label="Idiosyncratic Share (1 − R²)",
    )


def format_variance_share_figure(axis: plt.Axes) -> None:
    """
    Apply publication-style formatting to the variance-share axes.

    Why needed: Consistent styling across Part A and Part B figures makes the
    two panels directly comparable when placed side by side in a paper.
    The y-axis is formatted as a percentage to match the paper's presentation,
    even though the underlying values are stored as decimals in [0, 1].

    Inputs:
        axis — matplotlib Axes object to format (modified in place)
    """
    axis.set_xlabel("Year", fontsize=12)
    axis.set_ylabel("Share of Return Variance (%)", fontsize=12)
    axis.set_title(
        "Value-Weighted Systematic and Idiosyncratic Shares\n"
        "of Individual Stock Return Variance (CRSP Universe)",
        fontsize=13,
    )
    axis.set_xlim(1960, 2025)
    axis.set_ylim(0, 1)
    # PercentFormatter with xmax=1 converts decimal values to display percentages
    # (e.g. 0.40 → "40%") without altering the underlying data.
    axis.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))

    axis.legend(fontsize=11, framealpha=0.9)
    axis.grid(True, axis="y", alpha=0.3, linestyle=":")
    axis.spines[["top", "right"]].set_visible(False)


def save_figure(
    figure: plt.Figure,
    output_path: str,
) -> None:
    """
    Save a matplotlib figure to disk, creating the output directory if needed.

    Why needed: Centralising directory creation and save parameters ensures
    all figures are written at the same DPI and bounding-box settings,
    preventing layout differences between Part A and Part B outputs.

    Inputs:
        figure      — matplotlib Figure object to save
        output_path — destination file path (PNG)
    """
    output_directory = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_directory, exist_ok=True)

    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    Full research pipeline for Part A.

    Pipeline:
        Raw Data (FF factors + CRSP)
          → Cleaning & Merging        (Steps 1–2)
          → Rolling R² Estimation     (Step 3)
          → Value-Weighted Aggregation(Step 4)
          → Figure Output             (Step 5)
    """
    t_total = time.time()

    ff_path   = os.path.join(DATA_DIR, "ff_factors_daily.csv")
    crsp_path = os.path.join(DATA_DIR, "crsp_daily.csv.gz")
    fig_path  = os.path.join(FIGURES_DIR, "figure4_replication.png")

    # Year-ends define when each 5-year rolling window is evaluated.
    # Using Dec 31 of each year ensures the window always covers the same
    # calendar span regardless of holiday-adjusted trading calendars.
    year_ends = pd.date_range("1960-12-31", "2025-12-31", freq="YE")

    # ── 1. Fama-French ──────────────────────────────────────────────────────
    # Load the risk-free rate and market excess return used to demean returns
    # and as the CAPM benchmark regressor.
    print("=" * 60)
    print("Step 1: Load Fama-French daily factors")
    print("=" * 60)
    ff = load_ff_factors(ff_path)
    print(f"  {len(ff):,} days  ({ff.index.min().date()} – {ff.index.max().date()})\n")

    # Verify that the FF file covers the full sample and has sensible values.
    print("FF date range:", ff.index.min(), "to", ff.index.max())
    print(ff.head())

    # ── 2. CRSP ─────────────────────────────────────────────────────────────
    # Filter to investable common equity and compute daily excess returns.
    # The merge with FF also attaches mkt_rf as the CAPM regressor.
    print("=" * 60)
    print("Step 2: Load and filter CRSP daily data")
    print("=" * 60)
    merged = load_crsp_data(crsp_path, ff)

    # Verify row counts and date coverage after loading; a sharp drop in
    # unique stocks before 1963 or after 2023 would signal a merge problem.
    print("Merged shape:", merged.shape)
    print("Unique stocks:", merged["PERMNO"].nunique())
    print("CRSP date range:", merged["date"].min(), "to", merged["date"].max())

    # ── 3. Rolling R² ───────────────────────────────────────────────────────
    # For each stock × year-end, estimate the CAPM R² over the preceding
    # 5 years.  R² is the systematic variance share for that stock-year.
    print("=" * 60)
    print("Step 3: Rolling 5-year CAPM R² at each year-end")
    print("=" * 60)
    r_squared_df = build_all_stocks_rolling_r_squared_dataframe(merged, year_ends)

    # R² is a squared correlation and must lie in [0, 1] by construction.
    # A violation would indicate a numerical error in the correlation calculation.
    print("R² summary:")
    print(r_squared_df["R_squared"].describe())
    assert ((r_squared_df["R_squared"] >= 0) & (r_squared_df["R_squared"] <= 1)).all(), \
        "R_squared outside [0,1]"

    # ── 4. Value-weight ──────────────────────────────────────────────────────
    # Aggregate stock-level R² into annual value-weighted shares.
    # Renaming R_squared → r2 matches the column name expected by the shared
    # aggregation function (compute_value_weighted_variance_shares).
    print("=" * 60)
    print("Step 4: Value-weighted aggregation")
    print("=" * 60)
    r_squared_df = r_squared_df.rename(columns={"R_squared": "r2"})
    vw = compute_value_weighted_variance_shares(r_squared_df)

    # Systematic + idiosyncratic shares must sum to exactly 1 for every year
    # because idiosyncratic_share is defined as 1 − systematic_share.
    # A violation would indicate a construction error in the aggregation step.
    print("VW head:")
    print(vw.head())
    share_sum = vw["systematic_share"] + vw["idiosyncratic_share"]
    assert np.allclose(share_sum, 1.0, atol=1e-8), \
        "Shares do not sum to 1"

    print(vw.to_string(index=False))

    # ── 5. Plot ──────────────────────────────────────────────────────────────
    # Produce the replication of Figure 4 and write to disk.
    print("\n" + "=" * 60)
    print("Step 5: Plot")
    print("=" * 60)
    plot_value_weighted_variance_shares(vw, fig_path)

    print(f"\nAll done. Total elapsed: {elapsed(t_total)}")


if __name__ == "__main__":
    main()
