# Purpose:    Define every Task 3 input and output path in one place so scripts
#             remain portable and never rely on the shell's current directory.
# Inputs:     Repository layout rooted one directory above task3/.
# Outputs:    Path constants used by the data, estimation, and figure scripts.
# Key Steps:  Locate the project root -> name raw inputs -> name generated artifacts.
# How to Run: Imported by other scripts; not run directly.

from pathlib import Path

TASK_DIR = Path(__file__).resolve().parent
DATA_DIR = TASK_DIR.parent / "data"
WORK_DIR = TASK_DIR / "work"
OUTPUT_DIR = TASK_DIR / "output"
FIGURES_DIR = TASK_DIR / "figures"

NEWSPAPER_DATA = DATA_DIR / "newspaper.csv"
POWER_DATA = DATA_DIR / "power.csv"
RAIN_DATA = DATA_DIR / "raindata.csv"

ANALYSIS_DATA = WORK_DIR / "analysis_data.csv"
MERGE_LOG = WORK_DIR / "merge_log.txt"

FIRST_STAGE = OUTPUT_DIR / "first_stage.csv"
ESTIMATES = OUTPUT_DIR / "estimates.csv"
RESULTS_TABLE_TEX = OUTPUT_DIR / "results_table.tex"
EDA_SUMMARY = OUTPUT_DIR / "eda_summary.csv"
EDA_SUMMARY_TEX = OUTPUT_DIR / "eda_summary.tex"

EDA_FIGURE = FIGURES_DIR / "eda.png"
RESULTS_FIGURE = FIGURES_DIR / "results.png"


# Create output directories up front, so downstream writes never fail on a missing folder.
def ensure_directories() -> None:
    for directory in (WORK_DIR, OUTPUT_DIR, FIGURES_DIR):
        directory.mkdir(parents=True, exist_ok=True)
