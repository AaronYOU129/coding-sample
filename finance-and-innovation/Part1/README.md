# Part 1: Replicating Figure 4 from Lochstoer and Muir (2025)

This project replicates Figure 4 from Lochstoer and Muir (2025), which decomposes US stock return variance into systematic (market-driven) and idiosyncratic shares over time. The analysis documents a long-run structural shift: the systematic share of variance has risen while the idiosyncratic share has fallen.

## Project Structure

```
Part1/
├── code/
│   ├── main_parta.py          # Replicates Figure 4 (full CRSP universe)
│   └── main_partb.py          # Extends analysis to the US banking sector
├── data/
│   ├── ff_factors_daily.csv   # Ken French daily Fama-French factors
│   └── crsp_daily.csv.gz     # CRSP daily stock file (~103M rows)
├── figures/
│   ├── figure4_replication.png  # Output: replicated Figure 4
│   └── figure4_banking.png     # Output: banking sector analogue
├── CLAUDE.md                  # Project and coding conventions
├── procedure.md               # Workflow for using Claude Code
├── writeup.pdf
```

## Methodology

- **Rolling 5-year CAPM window** at each year-end to capture slow structural change.
- **Value-weighted by beginning-of-year market cap** so small illiquid stocks do not dominate the aggregate.
- **R² via squared Pearson correlation** (equivalent to OLS R² for simple CAPM, but much faster across 10,000+ stock-year observations).
- **Part A**: Benchmark is the broad market excess return (Mkt-RF).
- **Part B**: Benchmark is a value-weighted banking sector return (SIC 6020-6029, 6035-6036), measuring within-sector systemic risk concentration.

## Data Sources

| File | Description |
|------|-------------|
| `data/ff_factors_daily.csv` | Ken French daily factors (Mkt-RF, RF). 4-row text header; values in percentage points. |
| `data/crsp_daily.csv.gz` | CRSP daily stock file. Filtered to common equity (SecurityType=EQTY, ShareType=NS) on NYSE/AMEX/NASDAQ with non-missing returns and market cap. |

## Requirements

- Python 3
- NumPy
- pandas
- Matplotlib

## How to Run

Run from the project root directory:

```bash
# Part A: Full CRSP universe (replicates Figure 4)
python3 code/main_parta.py

# Part B: US banking sector extension
python3 code/main_partb.py
```

Output figures are saved to `figures/`.

## Results

**Part A** — The systematic share of individual stock return variance has trended upward over time, with a notable spike around the 2008 financial crisis.

**Part B** — Within the banking sector, co-movement with the sector benchmark is substantially higher and shows distinct dynamics, reflecting systemic risk concentration among US banks.
