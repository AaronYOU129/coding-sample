# Task 3 — Do Readers' Clicks Affect Story Length?

Task 3 asks whether reader clicks cause editors to keep a story active longer
or publish more follow-up articles.

## Inputs

- `../data/newspaper.csv` — story-level outcomes and first-article attributes.
- `../data/power.csv` — daily electricity shortage.
- `../data/raindata.csv` — daily rainfall in two cities.

## Data construction

The three date formats are parsed explicitly and the daily datasets are merged
onto each story's start date. Rainfall covers all story dates. Stories on dates
without electricity-shortage data are dropped rather than assigning an
artificial instrument value, leaving 33,094 observations.

- `duration_hours` is the difference between the end and start timestamps.
- `followup_count = nnarticles - 1`.
- Outcomes use `log(1 + y)` to retain zero values.
- Positive, right-skewed views use `log(views)`.
- Month, day-of-week, story-class, and first-article controls are constructed
  for estimation.

## Identification and inference

Clicks may be endogenous because story importance can affect both readership
and editorial investment, while continued coverage may generate additional
clicks. The analysis instruments `log_views` with daily electricity shortage
and rainfall in two cities. The specification includes story controls and
fixed effects for class, month, and day of week. Standard errors are clustered
by start date, the level at which the instruments vary.

The cluster-robust first-stage statistic is 0.59, so the instruments are weak.
The 2SLS estimates are therefore imprecise. Anderson–Rubin inference does not
reject a zero effect for either outcome, and both confidence sets are
unbounded over the evaluated range. The evidence does not establish that
clicks have no effect; it shows that these instruments do not identify the
effect precisely.

## Pipeline

1. `prepare_data.py` parses, validates, constructs variables, and merges the
   three datasets.
2. `eda.py` exports summary statistics and the exploratory figure.
3. `estimate.py` estimates the first stage, 2SLS models, and Anderson–Rubin
   inference and exports a LaTeX results table.
4. `diagnostics.py` plots the two 2SLS estimates and confidence intervals.
5. `run_all.py` runs these stages in order.

`paths.py` contains all Task 3 input and output paths.

## Run

From `task3/`:

```bash
python run_all.py
```

## Generated files

Intermediate files:

- `work/analysis_data.csv`
- `work/merge_log.txt`

Final outputs:

- `output/eda_summary.csv`
- `output/eda_summary.tex`
- `output/first_stage.csv`
- `output/estimates.csv`
- `output/results_table.tex`

Figures:

- `figures/eda.png`
- `figures/results.png`

## Main results

- First-stage joint statistic: 0.590.
- Duration 2SLS coefficient: -0.300 (SE 0.496; p-value 0.545).
- Follow-up 2SLS coefficient: -0.026 (SE 0.185; p-value 0.890).
- Anderson–Rubin p-values at a zero effect: 0.842 and 0.658.

## Checks

The three tests verify the merged analysis sample and constructed outcomes,
recover a known effect from simulated endogenous data, and check exported
result schemas.

```bash
ruff check .
mypy paths.py prepare_data.py eda.py estimate.py diagnostics.py run_all.py
pytest -q
```

