"""
Purpose:    Run the sampling experiment: retrieve each of the 10 prompts' answers
            on 10 separate occasions and save every answer as its own JSON file.
            Designed for a real 10-day run, so it is checkpoint/resume-safe — on
            startup it scans outputs/ and skips any (prompt, run) pair already on
            disk, meaning an interrupted or cron-driven process never duplicates
            work. Cadence is configurable; --test removes all waiting.
Inputs:     prompts.PROMPTS; ANTHROPIC_API_KEY (via client).
Outputs:    outputs/promptNN_runNN_YYYYMMDD-HHMMSS.json, one per response, each
            with prompt_index, prompt_text, run_index, timestamp_iso, model,
            raw_response_text, usage (and brand/feature fields after extraction).
Key Steps:  Find completed pairs -> for each run round, call the model for each
            missing prompt, save + extract immediately -> sleep between rounds.
How to Run: python main.py collect [--interval-hours 24] [--test] [--once]
"""

import json
import logging
import re
import time
from datetime import datetime

from client import generate
from config import OUTPUTS_DIR, RESPONSE_MAX_TOKENS, RESPONSE_MODEL, TOTAL_RUNS
from extract import extract_brands_and_features
from prompts import numbered_prompts

logger = logging.getLogger(__name__)

_FILENAME_RE = re.compile(r"prompt(\d{2})_run(\d{2})_")


def collect(client, interval_hours=24, test=False, once=False, runs=TOTAL_RUNS, max_prompts=None):
    """Run the sampling schedule round by round.

    A "round" is one occasion: every prompt sampled once. We sleep interval_hours
    between rounds in a real run; --test sets the wait to zero, and --once does a
    single round then exits (the natural unit for a daily cron invocation, which
    relies on resume to pick up where the last invocation stopped).
    """
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    done = _completed_pairs()
    wait_seconds = 0 if test else interval_hours * 3600

    for run_index in range(1, runs + 1):
        worked = _run_round(client, run_index, done, max_prompts)
        if not worked:
            continue  # this round was already fully collected — resume past it
        if once:
            logger.info("--once: completed one round, exiting")
            return
        if run_index < runs and wait_seconds:
            logger.info("Round %d done; sleeping %.1fh until next round", run_index, wait_seconds / 3600)
            time.sleep(wait_seconds)

    logger.info("Collection complete: %d files in %s", len(_completed_pairs()), OUTPUTS_DIR)


def _run_round(client, run_index, done, max_prompts):
    """Sample every still-missing prompt for one run_index. Returns True if any
    new response was actually retrieved this round."""
    worked = False
    for prompt_index, prompt_text in numbered_prompts():
        if max_prompts is not None and prompt_index > max_prompts:
            break
        if (prompt_index, run_index) in done:
            continue
        _sample_one(client, prompt_index, prompt_text, run_index)
        done.add((prompt_index, run_index))
        worked = True
    return worked


def _sample_one(client, prompt_index, prompt_text, run_index):
    """Retrieve one response, enrich it, and persist it as a single JSON file."""
    logger.info("Sampling prompt %d, run %d", prompt_index, run_index)
    called_at = datetime.now().astimezone()
    text, usage = generate(client, RESPONSE_MODEL, prompt_text, RESPONSE_MAX_TOKENS)

    brands, features = extract_brands_and_features(client, text)
    record = {
        "prompt_index": prompt_index,
        "prompt_text": prompt_text,
        "run_index": run_index,
        "timestamp_iso": called_at.isoformat(),
        "model": RESPONSE_MODEL,
        "raw_response_text": text,
        "usage": usage,
        "brands": brands,
        "features": features,
        "brand_count": len(brands),
        "feature_count": len(features),
    }
    _write_record(record, called_at)


def _write_record(record, called_at):
    """Write one response record to promptNN_runNN_YYYYMMDD-HHMMSS.json."""
    filename = "prompt{:02d}_run{:02d}_{}.json".format(
        record["prompt_index"], record["run_index"], called_at.strftime("%Y%m%d-%H%M%S")
    )
    path = OUTPUTS_DIR / filename
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False))
    logger.info("Saved %s", filename)


def _completed_pairs():
    """Scan outputs/ and return the set of (prompt_index, run_index) already
    saved. This is the checkpoint that makes the run resumable."""
    pairs = set()
    for path in OUTPUTS_DIR.glob("prompt*_run*.json"):
        match = _FILENAME_RE.match(path.name)
        if match:
            pairs.add((int(match.group(1)), int(match.group(2))))
    return pairs
