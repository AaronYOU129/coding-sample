# Purpose:    Explore the mixture sample before estimation so visible modes,
#             spread, skewness, and tail behavior inform starting values and
#             provide evidence about whether a two-component model is plausible.
# Inputs:     ../data/mixture_data.csv (column `x`).
# Outputs:    output/eda_summary.csv, output/eda_summary.tex, and
#             figures/eda.png.
# Key Steps:  Load the raw analysis column -> calculate explicit summary
#             statistics -> format CSV and LaTeX tables -> plot a histogram
#             and empirical CDF -> export.
# How to Run: `python eda.py` from task2/, or `python run_all.py`.

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

import paths

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main() -> None:
    """Create the EDA table and figure before any model is estimated.

    Saving both artifacts makes starting-value choices and model concerns auditable.
    """
    paths.ensure_directories()
    x = load_data()
    summary = summary_statistics(x)
    summary.to_csv(paths.EDA_SUMMARY, index=False)
    paths.EDA_SUMMARY_TEX.write_text(summary_statistics_latex(summary), encoding="utf-8")
    plot_distribution(x)

    print(summary.to_string(index=False))
    print(f"EDA tables written to {paths.EDA_SUMMARY} and {paths.EDA_SUMMARY_TEX}")
    print(f"EDA figure written to {paths.EDA_FIGURE}")


def load_data() -> pd.Series:
    """Load the raw x column without dropping missing values.

    Retaining missing values lets the EDA summary report data quality before estimation.
    """
    data = pd.read_csv(paths.DATA_FILE)
    if "x" not in data.columns:
        raise ValueError("Expected column 'x' in the mixture data.")
    return data["x"]


def summary_statistics(x: pd.Series) -> pd.DataFrame:
    """Calculate sample size, location, spread, quantiles, skewness, and kurtosis.

    Explicit statistics complement the plots and make the EDA reproducible in a CSV.
    """
    clean_x = x.dropna().to_numpy(dtype=float)
    if len(clean_x) == 0:
        raise ValueError("No non-missing observations are available for EDA.")

    quantiles = np.quantile(clean_x, [0.01, 0.25, 0.50, 0.75, 0.99])
    return pd.DataFrame([{
        "n_total": len(x),
        "n_non_missing": len(clean_x),
        "n_missing": int(x.isna().sum()),
        "mean": clean_x.mean(),
        "std": clean_x.std(ddof=1),
        "min": clean_x.min(),
        "p01": quantiles[0],
        "q25": quantiles[1],
        "median": quantiles[2],
        "q75": quantiles[3],
        "p99": quantiles[4],
        "max": clean_x.max(),
        "skewness": stats.skew(clean_x, bias=False),
        "excess_kurtosis": stats.kurtosis(clean_x, bias=False),
    }])


def summary_statistics_latex(summary: pd.DataFrame) -> str:
    """Format the one-row EDA summary as a report-ready vertical LaTeX table.

    A two-column layout stays readable on the page, unlike the wide CSV representation.
    """
    row = summary.iloc[0]
    formatted_statistics = [
        ("Total observations", f"{int(row['n_total']):,}"),
        ("Non-missing observations", f"{int(row['n_non_missing']):,}"),
        ("Missing observations", f"{int(row['n_missing']):,}"),
        ("Mean", f"{row['mean']:.3f}"),
        ("Standard deviation", f"{row['std']:.3f}"),
        ("Minimum", f"{row['min']:.3f}"),
        ("1st percentile", f"{row['p01']:.3f}"),
        ("25th percentile", f"{row['q25']:.3f}"),
        ("Median", f"{row['median']:.3f}"),
        ("75th percentile", f"{row['q75']:.3f}"),
        ("99th percentile", f"{row['p99']:.3f}"),
        ("Maximum", f"{row['max']:.3f}"),
        ("Skewness", f"{row['skewness']:.3f}"),
        ("Excess kurtosis", f"{row['excess_kurtosis']:.3f}"),
    ]
    table_rows = [f"{label} & {value} \\\\" for label, value in formatted_statistics]
    return "\n".join([
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Summary statistics for the Task 2 mixture sample}",
        r"\label{tab:task2-summary}",
        r"\begin{tabular}{lr}",
        r"\toprule",
        r"Statistic & Value \\",
        r"\midrule",
        *table_rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ])


def plot_distribution(x: pd.Series) -> None:
    """Plot the histogram and empirical CDF of all non-missing observations.

    The histogram exposes modal structure, while the ECDF summarizes the full distribution.
    """
    clean_x = np.sort(x.dropna().to_numpy(dtype=float))
    empirical_cdf = np.arange(1, len(clean_x) + 1) / len(clean_x)

    fig, (histogram_axis, cdf_axis) = plt.subplots(1, 2, figsize=(12, 4.5))
    histogram_axis.hist(clean_x, bins=80, density=True, alpha=0.65)
    histogram_axis.axvline(clean_x.mean(), color="crimson", label="mean")
    histogram_axis.axvline(np.median(clean_x), color="black", linestyle="--", label="median")
    histogram_axis.set(
        xlabel="x",
        ylabel="density",
        title="Distribution and visible modes",
    )
    histogram_axis.legend()

    cdf_axis.plot(clean_x, empirical_cdf)
    cdf_axis.set(
        xlabel="x",
        ylabel="empirical cumulative probability",
        title="Empirical CDF",
    )

    fig.suptitle("Task 2 exploratory data analysis")
    fig.tight_layout()
    fig.savefig(paths.EDA_FIGURE, dpi=150)
    plt.close(fig)


if __name__ == "__main__":  # pragma: no cover
    main()
