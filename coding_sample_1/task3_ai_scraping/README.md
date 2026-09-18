# Task 3 — AI Shopping-Prompt Response Experiment

Runs 10 shopping-related prompts against the Anthropic API, retrieves each one's
answer on 10 separate occasions, saves every answer as its own JSON file, then
extracts the brands/products and product features mentioned and reports summary
statistics.

- **Responses:** `claude-haiku-4-5` (cheapest model, chosen to minimize cost; the
  original assignment named `claude-opus-4-8` — switch it back in `config.py` if needed)
- **Brand/feature extraction:** `claude-haiku-4-5` (cheap, fast — extraction is an easy JSON task)
- **API key:** read from the `ANTHROPIC_API_KEY` environment variable (never hardcoded)

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
```

## Run modes

```bash
# Development: run all 10 rounds back-to-back, no waiting (≈100 + 100 API calls)
python main.py all --test

# Real experiment: one round per day for 10 days (process stays alive)
python main.py collect                 # then, after it finishes:
python main.py extract                 # (collect already extracts inline; this fills any gaps)
python main.py stats

# Cheap smoke test: 1 prompt x 1 run end-to-end (≈2 API calls)
python main.py all --test --runs 1 --max-prompts 1
```

### Real 10-day mode — two ways

The schedule is configurable via `--interval-hours` (default 24).

1. **Long-running process** — `python main.py collect` samples a round, sleeps
   24h, samples the next, for 10 rounds.
2. **Cron-style (more robust over 10 days)** — schedule a daily invocation of
   `python main.py collect --once`. Each invocation does exactly one round; the
   checkpoint/resume logic (below) means it always picks up the next pending
   round. Example crontab line:
   ```
   0 9 * * *  cd /path/to/task3_ai_scraping && ANTHROPIC_API_KEY=sk-... python main.py collect --once >> outputs/cron.log 2>&1
   ```

## How it works

- **Checkpoint / resume.** On startup the collector scans `outputs/` and skips
  any `(prompt, run)` pair already saved, so an interrupted or cron-driven run
  never duplicates work and resumes exactly where it stopped.
- **Retry / backoff.** Every API call is wrapped in explicit exponential backoff
  with jitter (`client.py`); transient errors (429, 5xx, connection/timeout)
  retry, real client errors surface immediately.
- **Strict-JSON extraction.** The second call asks for `{"brands": [...],
  "features": [...]}` only; parsing strips accidental code fences, slices to the
  outermost braces, and falls back to empty lists on any malformed reply.
- **Logging.** Timestamped progress to the console and `outputs/run.log`.

## Output

```
outputs/
├── promptNN_runNN_YYYYMMDD-HHMMSS.json   # one file per response (target: 100)
├── run.log                               # timestamped run log
└── stats.csv                             # one row per response file
```

Each response JSON contains:
`prompt_index, prompt_text, run_index, timestamp_iso, model, raw_response_text,
usage` plus, after extraction, `brands, features, brand_count, feature_count`.

`stats.csv` columns:
`filename, prompt_index, run_index, timestamp, word_count, brand_count,
feature_count`. The console also prints the total file count and, per prompt, the
average word/brand/feature counts across its 10 runs.

## Files

```
task3_ai_scraping/
├── config.py        # paths, model IDs, run constants (single source of truth)
├── prompts.py       # the 10 prompts, frozen in order
├── client.py        # Anthropic client + exponential-backoff API calls
├── collect.py       # sampling schedule + checkpoint/resume + save
├── extract.py       # brand/feature extraction (strict JSON, safe parse)
├── stats.py         # stats.csv + per-prompt summary
├── main.py          # CLI: collect | extract | stats | all
├── requirements.txt
├── README.md
└── SOURCES.md       # where the 10 prompts come from + source-quality argument
```

See `SOURCES.md` for how the 10 representative shopping prompts were chosen and
why the source is credible.
