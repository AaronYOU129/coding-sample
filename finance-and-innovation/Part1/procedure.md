# Procedure: Working with Claude Code

## Step 0. Prepare `claude.md`
- Write the project-specific `claude.md` before starting any coding.
- Must include:
  - **Coding principles**: header format, function design, naming, readability standards
    (can reuse a shared `coding_principles.md` across projects).
  - **Project background**: what paper/task this replicates, key methodological choices and why.
  - **Data sources**: where the data comes from, file formats, known quirks.
- This file is Claude Code's single source of truth for the project.
- Keep it concise — principles, not paragraphs.

## Step 1. Project Overview
- Let Claude Code scan the full project folder to understand the existing structure.
- Command: ask Claude Code to get an overview of the directory, list files, and summarize what exists.
- This ensures Claude Code does not duplicate or conflict with existing work.

## Step 2. Define the Task
Every task given to Claude Code must follow this structure:

- **Role**: What role Claude Code should assume (e.g., "You are a research assistant replicating a published figure").
- **Objective**: What the output should be — one clear deliverable per task.
- **Context**: All necessary background so Claude Code understands why, not just what (e.g., paper reference, methodology rationale, data quirks).
- **Constraints**: What to avoid, what tools/packages to use, style requirements, performance expectations.
- **Evaluation Criteria**: How you will judge the output (e.g., "The figure should match Figure 4 in the paper", "Coefficients should be within 5% of published estimates").
- **Output Format**: Exact file type, naming convention, and location (e.g., `output/figure4.png`, a `.py` script with header per `claude.md`).
- **Sanity Checks**: Specify upfront what assertions or diagnostics the generated code should include (e.g., "print row counts after each merge", "assert systematic + idiosyncratic shares sum to 1"). This is more effective than adding checks after the fact.

### Example Prompt

> **Role**: You are a research assistant replicating a published empirical figure.
>
> **Objective**: Replicate Figure 4 from Lochstoer and Muir (2025), which decomposes
> US stock return variance into systematic and idiosyncratic shares over time.
> Save the figure to `figures/figure4_replication.png`.
>
> **Context**: The paper documents a long-run structural shift — the systematic
> (market-driven) share of variance has risen while the idiosyncratic share has
> fallen. Use a rolling 5-year CAPM window evaluated at each year-end, because the
> result is about slow-moving structural change, not short-run volatility.
> Value-weight by beginning-of-year market capitalization so that large-cap stocks
> are not drowned out by the long tail of small, illiquid CRSP names. For
> computational efficiency, use the identity that OLS R² equals the squared Pearson
> correlation between stock excess returns and market excess returns — this avoids
> running full regressions stock by stock.
>
> **Constraints**:
> - Use Python (pandas, numpy, matplotlib). No statsmodels needed.
> - Data: `data/ff_factors_daily.csv` (Ken French daily factors, 4-row header,
>   values in percentage points) and `data/crsp_daily.csv.gz` (CRSP daily returns
>   and market caps, ~103M rows — load in 5M-row chunks).
> - Filter CRSP to: SecurityType == "EQTY", ShareType == "NS", PrimaryExch in
>   {"N", "A", "Q"}, non-missing DlyRet and DlyCap.
> - Require at least 750 daily observations in each 5-year window.
> - Follow `claude.md` for header format, function design, and naming conventions.
>
> **Evaluation Criteria**: The output figure should visually match Figure 4 in the
> paper — two lines (systematic and idiosyncratic share) that sum to 1 at every
> year, with systematic share rising over time and an acceleration visible around
> 2008. Print summary statistics at key years (1970, 1990, 2010, 2020) as a sanity
> check.
>
> **Output Format**: Script at `code/main_parta.py`, figure saved to
> `figures/figure4_replication.png`.
>
> **Sanity Checks** (must be included in the generated code):
> - Print row count, unique stock count, and date range after loading CRSP.
> - Assert all R² values lie in [0, 1].
> - Assert systematic share + idiosyncratic share = 1 at every year-end.
> - Print the full value-weighted results table before plotting.

## Step 2.5. Ask for an implementation plan before coding.
- Require Claude Code to first outline:
  - the workflow,
  - the proposed functions,
  - the key assumptions,
  - the expected inputs and outputs,
  - and the sanity checks it will include.
- Review this plan before asking it to write code.

## Step 3. Generate the Code
- Let Claude Code write the full script in one pass.
- The script must comply with `claude.md` (header format, function design, naming, etc.).
- Review the generated code before running — check not only structure and syntax, but also whether the implementation matches the intended methodology, sample definition, weighting scheme, time window, and output specification.
## Step 4. Debug and Iterate
- Run the code yourself.
- Before asking Claude Code to fix an error, first identify the layer of the problem:
file path, data parsing, filtering, merge logic, estimation logic, plotting, or conceptual mismatch.
Debug the root cause, not just the traceback.
- Fix errors interactively with Claude Code — describe the error, share the traceback, and let it propose a fix.
- Keep each debugging prompt focused on one issue at a time.
- **When to regenerate instead of debug**: If the same error persists after 2–3 rounds of fixes, or if the code structure itself is wrong (not just a bug), ask Claude Code to rewrite the relevant function or script from scratch rather than patching.

## Step 5. Pass the Sanity Checks
- Run the script and confirm all sanity checks defined in Step 2 pass.
- If any assertion fails or printed diagnostics look wrong, return to Step 4.

## Step 6. Finalize
- Confirm the output matches expectations (visual comparison, numerical checks).
- Clean up any temporary debugging code.
- Ensure the final script runs end-to-end from scratch with no manual intervention.