# Purpose:    Summarize the merged Task 3 sample before estimation because the
#             heavy right tails, large zero masses, and date-level instrument
#             patterns determine the transformations and inference choices.
# Inputs:     work/analysis_data.csv produced by prepare_data.py.
# Outputs:    output/eda_summary.csv, output/eda_summary.tex, and
#             figures/eda.png.
# Key Steps:  Load the analysis sample -> summarize outcomes, clicks, controls,
#             and instruments -> export CSV and LaTeX -> plot distributions,
#             zero shares, class composition, and instrument time paths.
# How to Run: `python eda.py` from task3/, or `python run_all.py`.

import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parent / "work" / "matplotlib"),
)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import paths  # noqa: E402

SUMMARY_VARIABLES = {
    "views": "Views",
    "duration_hours": "Duration (hours)",
    "followup_count": "Follow-up articles",
    "word_count": "Word count",
    "totshort": "Electricity shortage",
    "rain_1": "Rainfall, city 1",
    "rain_2": "Rainfall, city 2",
}


def main() -> None:
    """Create reproducible EDA tables and the exploratory figure."""
    paths.ensure_directories()
    analysis_data = pd.read_csv(paths.ANALYSIS_DATA)
    summary = build_summary_statistics(analysis_data)
    summary.to_csv(paths.EDA_SUMMARY, index=False)
    paths.EDA_SUMMARY_TEX.write_text(
        format_summary_statistics_latex(summary),
        encoding="utf-8",
    )
    plot_exploratory_summary(analysis_data)

    print(summary.round(3).to_string(index=False))
    print(
        "EDA outputs written to "
        f"{paths.EDA_SUMMARY}, {paths.EDA_SUMMARY_TEX}, "
        f"and {paths.EDA_FIGURE}"
    )


def build_summary_statistics(data: pd.DataFrame) -> pd.DataFrame:
    """Summarize distributions and missingness that affect the model specification."""
    rows = []
    for column_name, display_name in SUMMARY_VARIABLES.items():
        values = data[column_name].dropna()
        rows.append(
            {
                "variable": display_name,
                "n": int(values.count()),
                "missing": int(data[column_name].isna().sum()),
                "mean": values.mean(),
                "standard_deviation": values.std(),
                "minimum": values.min(),
                "p25": values.quantile(0.25),
                "median": values.median(),
                "p75": values.quantile(0.75),
                "maximum": values.max(),
                "zero_share": float(values.eq(0).mean()),
            }
        )
    return pd.DataFrame(rows)


def format_summary_statistics_latex(summary: pd.DataFrame) -> str:
    """Format the exported statistics as a compact report-ready LaTeX table."""
    table_rows = []
    for row in summary.itertuples(index=False):
        table_rows.append(
            f"{row.variable} & {row.n:,} & {row.mean:.2f} & "
            f"{row.standard_deviation:.2f} & {row.minimum:.2f} & "
            f"{row.median:.2f} & {row.maximum:.2f} & "
            f"{100 * row.zero_share:.1f}\\% \\\\"
        )

    return "\n".join(
        [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Task 3 summary statistics for the merged analysis sample.}",
            r"\label{tab:task3-summary}",
            r"\small",
            r"\begin{tabular}{lrrrrrrr}",
            r"\toprule",
            r"Variable & $N$ & Mean & SD & Min & Median & Max & Zero share \\",
            r"\midrule",
            *table_rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )


def plot_exploratory_summary(data: pd.DataFrame) -> None:
    """Plot the features that motivate logs, fixed effects, and weak-IV scrutiny."""
    figure, axes = plt.subplots(2, 3, figsize=(15, 8))

    axes[0, 0].hist(np.log1p(data["views"]), bins=50)
    axes[0, 0].set_title("log(1+views)")

    duration_zero_share = float(data["duration_hours"].eq(0).mean())
    axes[0, 1].hist(np.log1p(data["duration_hours"]), bins=50)
    axes[0, 1].set_title(
        f"log(1+duration hrs), {duration_zero_share:.0%} zero"
    )

    followup_zero_share = float(data["followup_count"].eq(0).mean())
    clipped_followups = data["followup_count"].clip(upper=10)
    axes[0, 2].hist(clipped_followups, bins=np.arange(-0.5, 11.5, 1))
    axes[0, 2].set_title(
        f"follow-ups (clipped at 10), {followup_zero_share:.0%} zero"
    )

    class_counts = data["story_class"].value_counts().sort_values()
    axes[1, 0].barh(class_counts.index, class_counts.values)
    axes[1, 0].set_title("stories per class")

    daily_instruments = (
        data.groupby("date")[["totshort", "rain_1", "rain_2"]].first().sort_index()
    )
    daily_instruments.index = pd.to_datetime(daily_instruments.index)
    axes[1, 1].plot(daily_instruments.index, daily_instruments["totshort"])
    axes[1, 1].set_title("totshort (note the ~March break + gaps)")

    for instrument in ["rain_1", "rain_2"]:
        axes[1, 2].plot(
            daily_instruments.index,
            daily_instruments[instrument],
            label=instrument,
        )
    axes[1, 2].legend()
    axes[1, 2].set_title("rainfall (monsoon-concentrated)")

    figure.tight_layout()
    figure.savefig(paths.EDA_FIGURE, dpi=150)
    plt.close(figure)


if __name__ == "__main__":  # pragma: no cover
    main()
