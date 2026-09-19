"""
Purpose:    Command-line entry point that wires the three stages of the
            experiment together (collect -> extract -> stats) behind one CLI, and
            configures timestamped logging to both the console and a log file so a
            10-day run leaves an auditable trail. Splitting the stages into
            subcommands lets the long collection run, the repair-any-gaps
            extraction pass, and the cheap stats recompute each be invoked alone.
Inputs:     CLI arguments; ANTHROPIC_API_KEY (read by the client).
Outputs:    Delegates to collect/extract/stats; writes outputs/run.log.
Key Steps:  Parse args -> set up logging -> dispatch to the chosen subcommand.
How to Run: python main.py collect --test          # full run, no waiting
            python main.py collect                  # real once-per-day run
            python main.py collect --once           # one round (for cron)
            python main.py extract                   # fill any missing extractions
            python main.py stats                     # write stats.csv + summary
            python main.py all --test                # collect -> extract -> stats
"""

import argparse
import logging

from config import LOG_FILE, OUTPUTS_DIR, TOTAL_RUNS


def main():
    args = _parse_args()
    _configure_logging()

    if args.command == "stats":
        _run_stats()
        return
    if args.command == "extract":
        _run_extract()
        return
    if args.command == "collect":
        _run_collect(args)
        return
    # "all": collect, then fill any extraction gaps, then summarize.
    _run_collect(args)
    _run_extract()
    _run_stats()


def _run_collect(args):
    from client import build_client
    from collect import collect

    collect(
        build_client(),
        interval_hours=args.interval_hours,
        test=args.test,
        once=args.once,
        runs=args.runs,
        max_prompts=args.max_prompts,
    )


def _run_extract():
    from client import build_client
    from extract import enrich_all

    enrich_all(build_client())


def _run_stats():
    from stats import build_stats

    build_stats()


def _parse_args():
    parser = argparse.ArgumentParser(description="Shopping-prompt AI response experiment.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name in ("collect", "all"):
        sub = subparsers.add_parser(name, help=f"{name} the experiment")
        sub.add_argument("--interval-hours", type=float, default=24,
                         help="Hours to wait between sampling rounds (default 24).")
        sub.add_argument("--test", action="store_true",
                         help="Run all rounds back-to-back with no waiting.")
        sub.add_argument("--once", action="store_true",
                         help="Run a single round then exit (for cron-style daily runs).")
        sub.add_argument("--runs", type=int, default=TOTAL_RUNS,
                         help="Number of sampling rounds (default 10).")
        sub.add_argument("--max-prompts", type=int, default=None,
                         help="Limit prompts per round (for cheap smoke tests).")

    subparsers.add_parser("extract", help="Fill brand/feature data for any files missing it.")
    subparsers.add_parser("stats", help="Write stats.csv and print the summary.")
    return parser.parse_args()


def _configure_logging():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
    )


if __name__ == "__main__":
    main()
