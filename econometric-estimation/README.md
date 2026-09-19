# Econometric Estimation & Diagnostics

Python solutions to the three-part predoc evaluation. Each task has an
independent pipeline, tests, intermediate files, final outputs, and figures.

## Tasks

- **`task1/` — Restaurant Price Index:** decompose order-line prices into
  product sizes and restaurant-date price indices using two-way fixed effects.
- **`task2/` — Normal Mixture MLE:** estimate the required two-component normal
  mixture and compare its fit with a three-component diagnostic model.
- **`task3/` — Clicks and Story Length:** merge story, electricity-shortage,
  and rainfall data and estimate the effect of clicks using instrumental
  variables.

## Repository layout

```text
data/                 raw inputs (not distributed)
prompt_log.md         AI prompt log
requirements.txt      Python package versions
task1/, task2/, task3/
  paths.py            task-specific input and output paths
  *.py                analysis modules
  run_all.py          complete pipeline in execution order
  tests/              pytest tests
  work/               generated intermediate files
  output/             generated tables and datasets
  figures/            generated figures
  README.md           task-specific documentation
```

## Environment setup

The analysis was developed with Python 3.12. Create a fresh virtual
environment and install the required packages from the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Confirm that the virtual-environment interpreter is active:

```bash
which python
python --version
```

## Run a task

Run each pipeline from its own directory:

```bash
cd task1  # or task2 / task3
python run_all.py
```

The task-specific READMEs document the modules, execution order, and outputs.
Tests target the main data transformations and estimators, including simulated
examples with known parameters; they do not claim 100% line coverage.

## Data availability

The original recruiter-provided input files and original assignment are not
distributed in this repository. Analysis code, tests, reports, figures,
summary outputs, processed datasets, and the AI prompt log are included.

To rerun the complete pipelines, obtain authorized copies of
`orders_sample.csv`, `mixture_data.csv`, `newspaper.csv`, `power.csv`, and
`raindata.csv` and place them in `data/`. See [report.pdf](report.pdf) for the
analysis report.
