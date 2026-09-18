# Coding Sample 3 — Experimental Economics Replication

Replication and inference exercises based on Cappelen, List, Samek, and
Tungodden (2020), “The Effect of Early-Childhood Education on Social Preferences.”

## Technical contents

- `code_q1.py`: balance tables and OLS regressions with fixed effects; exports
  LaTeX tables and compares results against reference values.
- `code_q2.py`: permutation inference, Romano–Wolf stepdown adjustments, and
  BKY sharpened q-values; uses Frisch–Waugh–Lovell residualization and checks
  observed regression statistics against statsmodels.
- `code_q3.py`: coefficient plot with confidence intervals.
- `report.pdf`: technical analysis and existing results.
- `tables/`: existing balance and regression tables.

## Reproduction

Install dependencies with `python -m pip install -r requirements.txt`.
Obtain an authorized copy of `data.csv` and place it beside the scripts, then run:

```bash
python code_q1.py
python code_q2.py
python code_q3.py
```

Raw data, the supplied codebook, and the third-party paper appendix are not
redistributed. The original analysis logic and results are preserved. This
publication cleanup does not constitute an independent methodological audit.
