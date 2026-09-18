# Purpose:    Single source of truth for all input/output paths in Task 1, so
#             every script reads and writes the same locations (no hard-coded
#             duplicates that can drift apart).
# Inputs:     None (constants only).
# Outputs:    None (imported by the other scripts).
# Key Steps:  Define directories relative to this file's location so the
#             pipeline runs unchanged on any machine.
# How to Run: Not run directly; imported via `import paths`.

from pathlib import Path

TASK_DIR = Path(__file__).resolve().parent
DATA_DIR = TASK_DIR.parent / "data"
WORK_DIR = TASK_DIR / "work"       # intermediate files passed between scripts
OUTPUT_DIR = TASK_DIR / "output"   # final deliverable datasets
FIGURES_DIR = TASK_DIR / "figures"

RAW_ORDERS = DATA_DIR / "orders_sample.csv"
EDA_FIGURE = FIGURES_DIR / "eda.png"
EDA_SUMMARY = OUTPUT_DIR / "eda_summary.csv"
EDA_SUMMARY_TEX = OUTPUT_DIR / "eda_summary.tex"

CLEAN_ORDERS = WORK_DIR / "orders_clean.csv"
CLEANING_LOG = WORK_DIR / "cleaning_log.txt"
RESIDUALS = WORK_DIR / "residuals.csv"
FIT_STATS = WORK_DIR / "fit_stats.csv"

PRODUCT_SIZES = OUTPUT_DIR / "product_sizes.csv"
PRICE_INDICES = OUTPUT_DIR / "price_indices.csv"
DIAGNOSTICS_SUMMARY = OUTPUT_DIR / "diagnostics_summary.csv"


def ensure_directories() -> None:
    for directory in (WORK_DIR, OUTPUT_DIR, FIGURES_DIR):
        directory.mkdir(parents=True, exist_ok=True)
