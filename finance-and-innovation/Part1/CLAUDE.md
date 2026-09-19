## Project Background

This project replicates Figure 4 from Lochstoer and Muir (2025), which decomposes
US stock return variance into systematic and idiosyncratic shares over time.

Key methodological choices:
- Rolling 5-year CAPM window at each year-end: captures slow structural change,
  not short-run volatility.
- Value-weighted by beginning-of-year market cap: prevents small illiquid stocks
  from dominating the aggregate.
- R² computed as squared Pearson correlation (not full OLS): equivalent for simple
  CAPM, but orders of magnitude faster across 10,000+ stock-year observations.
- Part B replaces Mkt-RF with a value-weighted banking sector return (SIC 6020–6029,
  6035–6036) to measure within-sector systemic risk concentration.

## Data Sources

- `data/ff_factors_daily.csv`: Ken French daily Fama-French factors. 4-row text
  header; values in percentage points (divide by 100). Columns used: Mkt-RF, RF.
- `data/crsp_daily.csv.gz`: CRSP daily stock file, ~103M rows. Load in 5M-row
  chunks. Filter to SecurityType=EQTY, ShareType=NS, PrimaryExch in {N, A, Q},
  non-missing DlyRet and DlyCap.

# Coding Principles

Good code is not just correct — it is readable, modular, and reproducible.

0. File Structure & Reproducibility

Every script should be understandable without reading the full code.
Each file must begin with a header comment in this exact format:

  Purpose:    What the script does and why the approach was chosen.
  Inputs:     Exact file paths and descriptions of each input.
  Outputs:    Generated files and their locations.
  Key Steps:  High-level logical workflow (not repeating function names).
  How to Run: Command to execute the script.

Purpose must explain why key methodological choices were made,
not just restate what the code does or what function names already say.
A reader should understand the script within 30 seconds by reading the header.
Code should be reproducible from scratch with no hidden dependencies.
Input and output paths must be explicit and consistent.
Scripts should reflect the research pipeline:
Data → Processing → Estimation → Output
Avoid code duplication by extracting shared logic into reusable modules.

## 1. Function Design
- Keep functions small whenever possible.
- Each function should do one thing and have one clear responsibility.
- If a function mixes multiple responsibilities, split it into smaller helper functions.
- High-level functions should describe the workflow; low-level functions should handle implementation details.
- Code should read like a newspaper: the top should show the main logic, and the details should appear below.

## 2. Control Flow
- Use early returns to reduce nesting.
- Handle invalid, edge, or exceptional cases first, then keep the main path clear.
- Prefer flat, readable control flow over deeply nested conditionals.
- Keep the "happy path" visually obvious.

## 3. Abstraction
- Keep one level of abstraction per function.
- Do not mix business logic, numerical details, I/O, logging, and formatting in the same function unless the function is very small and linear.
- Separate loading, cleaning, computation, plotting, and exporting when they are meaningfully distinct.

## 4. Naming
- Names should reflect the purpose of the code.
- Names should describe what something is, not how it is implemented.
- Names should be pronounceable and searchable.
- Use consistent terminology across the project.
- Prefer clear, specific names over short but ambiguous abbreviations.

## 5. Readability
- Write code for humans first, not just for the interpreter.
- Prefer explicit, readable code over clever shortcuts.
- The main flow should be understandable without reading every implementation detail.
- Use comments to explain why, not to compensate for poor naming.

## 6. Refactoring Guardrails
- Split large functions, but do not create tiny wrappers with no semantic value.
- Do not over-engineer short, linear code that is already easy to read.
- Refactor when it improves clarity, responsibility boundaries, or testability.
