# Does the Market Correctly Value Novel Innovation?

Empirical research on patent novelty and market valuation.

## Research Question

Do investors systematically undervalue novel patents? This project tests whether patents with low backward similarity to prior art receive lower market valuations, and whether technology-level information uncertainty is the mechanism behind this discount.

## Interpretation of the existing results

The data do not support the hypothesis. Novelty has no statistically significant effect on patent market value. Technology uncertainty is strongly associated with lower valuations, but this discount does not fall disproportionately on novel patents. An auxiliary cell-level regression finds a negative association between novelty and the CV proxy. Later diagnostics show that cell size and the choice of dispersion measure affect the interpretation; this does not establish that novel technologies face less investor uncertainty.

## Project Structure

```
Part2/
|-- code/
|   |-- main.py                  # Full analysis pipeline (single entry point)
|
|-- data/
|   |-- KPSS_2024.csv            # Patent market values (Kogan et al. 2017, QJE)
|   |-- PatentSimilarity...csv   # Backward similarity scores (Kelly et al. 2021, AER:I)
|   |-- Match_patent_cpc_2024.csv# Patent-to-CPC classification mapping
|   |-- g_patent.tsv             # PatentsView bibliographic data (USPTO)
|   |-- merged_patent_data.csv   # [generated] Merged analysis dataset
|
|-- output/
|   |-- regression_table.csv     # [generated] Regression results
|   |-- regression_table.tex     # [generated] LaTeX regression table
|
|-- Research_memo.pdf            # Written research memo
|-- Documentation_log.pdf        # Process documentation and tool usage log
|-- CLAUDE.md                    # Coding principles and project context
|-- README.md                    # This file
```

## How to Reproduce

**Requirements:** Python 3 with pandas, numpy, statsmodels.

```bash
pip install pandas numpy statsmodels
```

**Run the full pipeline:**

```bash
cd Part2
python code/main.py
```

This executes all steps sequentially:

1. **Data merge** -- Inner-join KPSS, Kelly et al., CPC, and PatentsView on `patent_num` for 2021-2022 utility patents (196,428 patents, 99.5% match rate)
2. **Variable construction** -- Novelty (negated standardized backward similarity), log market value, log backward citations
3. **Technology uncertainty** -- Coefficient of variation of `xi_real` by CPC subclass x grant year (min 10 patents per group)
4. **Regressions** -- Main hypothesis test, mechanism interaction test, and first-stage cell-level regression, all with standard errors clustered at CPC subclass level
5. **Sanity checks** -- Automated checks on merge integrity, variable distributions, and regression outputs
6. **Export** -- Regression tables saved as CSV and LaTeX

## Data Sources

| Dataset | Source | Reference |
|---------|--------|-----------|
| KPSS patent values | [KPSS GitHub](https://github.com/KPSS2017/Technological-Innovation-Resource-Allocation-and-Growth-Extended-Data) | Kogan, Papanikolaou, Seru, Stoffman (2017). "Technological Innovation, Resource Allocation, and Growth." *QJE* 132(2). |
| Backward similarity | Kelly et al. extended data | Kelly, Papanikolaou, Seru, Taddy (2021). "Measuring Technological Innovation over the Long Run." *AER: Insights* 3(3). |
| CPC classification | KPSS extended data | Matched patent-to-CPC codes |
| Patent bibliographic | [PatentsView](https://patentsview.org) | USPTO patent grant data (type, date, claims) |

## Regression Results

| Variable | (1) Main | (2) Mechanism |
|----------|----------|---------------|
| Novelty | 0.0771 | 0.0329 |
| | (0.0739) | (0.2612) |
| Tech Uncertainty (CV) | | -0.7505*** |
| | | (0.2022) |
| Novelty x Uncertainty | | 0.0116 |
| | | (0.1279) |
| R-squared | 0.1010 | 0.1308 |
| N | 194,937 | 194,937 |

Clustered standard errors (CPC subclass) in parentheses. \* p<0.10, \*\* p<0.05, \*\*\* p<0.01.

**First stage** (CPC subclass x year, N=680): mean_novelty -> uncertainty = -0.3937*** (SE=0.0637).

## Sample Period

Patents granted between January 2021 and December 2022.

## Reading the diagnostics

Read `output/stress_test.md` and `output/mechanism_test.md` alongside the original memo and exploratory decomposition. Earlier claims of greater homogeneity are qualified by cell-size controls and alternative dispersion measures. The implementation uses log(x+1), which is not identical to log(x). Existing estimates have not been rerun for this publication cleanup.
