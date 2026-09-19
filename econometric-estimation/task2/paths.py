# Purpose:    Define every Task 2 input and output path in one place so the
#             simplified pipeline runs from any working directory without
#             duplicating path strings across scripts.
# Inputs:     None.
# Outputs:    None; imported by the other Task 2 scripts.
# Key Steps:  Resolve the project directory -> define data, work, output, and
#             figure paths -> create output directories when requested.
# How to Run: Not run directly; imported with `import paths`.

from pathlib import Path

TASK_DIR = Path(__file__).resolve().parent
DATA_DIR = TASK_DIR.parent / "data"
WORK_DIR = TASK_DIR / "work"
OUTPUT_DIR = TASK_DIR / "output"
FIGURES_DIR = TASK_DIR / "figures"

DATA_FILE = DATA_DIR / "mixture_data.csv"
MULTISTART_RESULTS = WORK_DIR / "multistart_results.csv"
K3_MULTISTART_RESULTS = WORK_DIR / "k3_multistart_results.csv"
EDA_SUMMARY = OUTPUT_DIR / "eda_summary.csv"
EDA_SUMMARY_TEX = OUTPUT_DIR / "eda_summary.tex"
ESTIMATES = OUTPUT_DIR / "estimates.csv"
K3_ESTIMATES = OUTPUT_DIR / "k3_estimates.csv"
MODEL_COMPARISON = OUTPUT_DIR / "model_comparison.csv"
MODEL_COMPARISON_TEX = OUTPUT_DIR / "model_comparison.tex"
DIAGNOSTICS_SUMMARY = OUTPUT_DIR / "diagnostics_summary.csv"
EDA_FIGURE = FIGURES_DIR / "eda.png"
FIT_FIGURE = FIGURES_DIR / "density_fit.png"
QQ_FIGURE = FIGURES_DIR / "qq_plot.png"


def ensure_directories() -> None:
    """Create every output directory before the simplified pipeline writes files.

    Shared setup prevents estimation and diagnostics from assuming hidden filesystem state.
    """
    for directory in (WORK_DIR, OUTPUT_DIR, FIGURES_DIR):
        directory.mkdir(parents=True, exist_ok=True)
