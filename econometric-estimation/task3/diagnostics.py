# Purpose:    Display 2SLS uncertainty because the weak first stage is easier to
#             understand from wide confidence intervals than from coefficients
#             and standard errors alone.
# Inputs:     output/estimates.csv produced by estimate.py.
# Outputs:    figures/results.png.
# Key Steps:  Load model estimates -> select 2SLS rows for each outcome -> plot
#             coefficients with date-clustered 95% confidence intervals.
# How to Run: `python diagnostics.py` from task3/, or `python run_all.py`.

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

OUTCOME_LABELS = {
    "log1p_duration": "Story duration",
    "log1p_followups": "Follow-up articles",
}


# Draw both figures, so data issues and weak-IV uncertainty are visible, not just tabulated.
def main() -> None:
    paths.ensure_directories()
    estimates = pd.read_csv(paths.ESTIMATES)
    plot_model_results(estimates)


# Plot 2SLS confidence intervals to make weak-IV imprecision visible.
def plot_model_results(estimates: pd.DataFrame) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))

    for axis, (outcome, label) in zip(axes, OUTCOME_LABELS.items(), strict=True):
        plotted_results = estimates[
            (estimates["outcome"] == outcome) & (estimates["model"] == "2SLS")
        ]
        positions = np.arange(len(plotted_results))
        axis.errorbar(
            plotted_results["coefficient"],
            positions,
            xerr=1.96 * plotted_results["standard_error"],
            fmt="o",
            capsize=3,
        )
        axis.axvline(0, color="black", linestyle="--", linewidth=0.8)
        axis.set_yticks(positions)
        axis.set_yticklabels(plotted_results["model"])
        axis.set_xlabel("Coefficient on log(views)")
        axis.set_title(label)

    figure.suptitle("2SLS is uninformative with weak instruments")
    figure.tight_layout()
    figure.savefig(paths.RESULTS_FIGURE, dpi=150)
    plt.close(figure)


if __name__ == "__main__":
    main()
