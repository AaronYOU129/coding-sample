# Project Background

Empirical research on patent novelty and market valuation.

Research Question: Does the stock market correctly value novel innovation?
Hypothesis: Investors systematically undervalue novel patents because novel technologies carry greater information uncertainty, making it harder for the market to assess their true economic value.

Sample: US patents granted 2021-2022.

## Data Sources

| File | Source | Description |
|------|--------|-------------|
| `data/KPSS_2024.csv` | Kogan, Papanikolaou, Seru, Stoffman (2017, QJE) | Patent-level market value estimates (xi_real) derived from stock returns around grant dates. Key fields: patent_num, permno, issue_date, xi_real, cites |
| `data/PatentSimilarityImportanceBreakthrough_forPost2022.csv` | Kelly, Papanikolaou, Seru, Taddy (2021, AER: Insights) | Text-based patent similarity measures. bsim5 = backward similarity to prior 5 years of patents (higher = less novel). fcitALL = forward citations |
| `data/Match_patent_cpc_2024.csv` | KPSS extended data | Patent-to-CPC classification mapping. Semicolon-delimited; first code is primary |
| `data/g_patent.tsv` | PatentsView (USPTO) | Patent bibliographic data: type, grant date, title, num_claims |



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
