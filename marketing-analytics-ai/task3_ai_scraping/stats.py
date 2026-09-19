"""
Purpose:    Summarize the collected experiment. Produces stats.csv (one row per
            response file) and prints headline numbers: total files and, per
            prompt, the average word/brand/feature counts across its runs. Stats
            are derived purely from the saved JSON files so they can be recomputed
            any time without re-calling the API.
Inputs:     outputs/promptNN_runNN_*.json files (post-extraction).
Outputs:    outputs/stats.csv; a per-prompt summary printed to the console.
Key Steps:  Load each record -> build one stats row -> write CSV -> aggregate per
            prompt and print averages.
How to Run: python main.py stats
"""

import csv
import json
import logging
from collections import defaultdict

from config import OUTPUTS_DIR, STATS_FILE

logger = logging.getLogger(__name__)

_CSV_COLUMNS = [
    "filename", "prompt_index", "run_index", "timestamp",
    "word_count", "brand_count", "feature_count",
]


def build_stats(outputs_dir=OUTPUTS_DIR, stats_file=STATS_FILE):
    """Write stats.csv and print the per-prompt summary. Returns the rows."""
    rows = _collect_rows(outputs_dir)
    if not rows:
        logger.warning("No response files found in %s", outputs_dir)
        return rows

    _write_csv(rows, stats_file)
    _print_summary(rows)
    return rows


def _collect_rows(outputs_dir):
    """Build one stats row per response file, sorted by (prompt, run)."""
    rows = []
    for path in outputs_dir.glob("prompt*_run*.json"):
        record = json.loads(path.read_text())
        rows.append(
            {
                "filename": path.name,
                "prompt_index": record["prompt_index"],
                "run_index": record["run_index"],
                "timestamp": record["timestamp_iso"],
                "word_count": _word_count(record["raw_response_text"]),
                # .get(): a row still computes even if extraction hasn't run yet.
                "brand_count": record.get("brand_count", 0),
                "feature_count": record.get("feature_count", 0),
            }
        )
    rows.sort(key=lambda row: (row["prompt_index"], row["run_index"]))
    return rows


def _write_csv(rows, stats_file):
    with open(stats_file, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote %d rows to %s", len(rows), stats_file)


def _print_summary(rows):
    """Print total file count and per-prompt averages across runs."""
    by_prompt = defaultdict(list)
    for row in rows:
        by_prompt[row["prompt_index"]].append(row)

    print(f"\nTotal files: {len(rows)}")
    print(f"{'prompt':>6} {'runs':>5} {'avg_words':>10} {'avg_brands':>11} {'avg_features':>13}")
    for prompt_index in sorted(by_prompt):
        group = by_prompt[prompt_index]
        print(
            f"{prompt_index:>6} {len(group):>5} "
            f"{_mean(group, 'word_count'):>10.1f} "
            f"{_mean(group, 'brand_count'):>11.2f} "
            f"{_mean(group, 'feature_count'):>13.2f}"
        )


def _word_count(text):
    return len(text.split())


def _mean(rows, field):
    return sum(row[field] for row in rows) / len(rows)
