"""
Purpose:    Single source of truth for all paths, model IDs, and run constants
            so every stage (collect / extract / stats) agrees on where files live
            and which models to call. Centralizing these prevents the silent
            drift that breaks checkpoint/resume when two modules disagree on the
            outputs directory or filename scheme.
Inputs:     None (pure constants).
Outputs:    None.
Key Steps:  Define directories, model IDs, sampling size, and token limits.
How to Run: Imported by the other modules; not run directly.
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs"
LOG_FILE = OUTPUTS_DIR / "run.log"
STATS_FILE = OUTPUTS_DIR / "stats.csv"

# Both stages use Haiku 4.5 — the cheapest model — by explicit user choice to
# minimize cost across 200 calls. NOTE: the original assignment named
# claude-opus-4-8 for responses; the user overrode that to keep the run cheap.
# To restore the assignment's model, set RESPONSE_MODEL = "claude-opus-4-8".
RESPONSE_MODEL = "claude-haiku-4-5"
EXTRACTION_MODEL = "claude-haiku-4-5"

# 10 prompts x 10 runs = 100 target files.
TOTAL_RUNS = 10

# Shopping comparisons can be long (3+ options across several dimensions); give
# the response room so it is never truncated mid-comparison. Extraction returns
# a tiny JSON object, so it needs very little.
RESPONSE_MAX_TOKENS = 4096
EXTRACTION_MAX_TOKENS = 1024
