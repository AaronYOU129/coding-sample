> **Exploratory result, qualified by later checks:** The homogeneity interpretation below is not the final conclusion. See `stress_test.md`: controlling for cell size attenuates the log-dispersion association, and MAD/IQR do not provide consistent support.

# First-stage decomposition: why does mean_novelty NEGATIVELY predict CV?

_Specification: cell = (cpc_subclass, grant_year). All regressions include year FE; SE clustered at cpc_subclass._

## Summary

The first-stage estimate reproduces at α₁ = -0.394 (p=0.0000). Decomposing CV = SD/mean with parallel year-FE regressions on log(mean) and log(SD), **SD channel dominates**: novel cells are *more homogeneous* in patent value (SD falls), which directly reduces CV. Adding mean_novelty² is significant (p=0.037). After trimming the top/bottom 1% of CV the first-stage coefficient is -0.361 (p=0.000), so the sign is robust to outliers.

**Implications for the four memo explanations:**

(a) **Mechanical artifact of CV.** Not the main driver — the mean channel does not move enough with novelty to explain the negative sign mechanically.

(b) **Novel areas more homogeneous (lower SD).** Supported — SD of patent value is significantly lower in novel cells, consistent with herding/coordination in emerging tech areas.

(c) **CV does not capture investor uncertainty (conceptual).** Neither corroborated nor refuted by these data; needs a non-CV dispersion measure (e.g., MAD, IQR, or analyst-disagreement style metrics) to test.

(d) **Text novelty ≠ market-perceived novelty (conceptual).** Cannot be diagnosed from the first stage alone; the main regression on individual ln(xi_real) is the right place to test this.


## 1. First-stage reproduction (full sample)
### CV / ln(mean) / ln(sd) on mean_novelty + year FE

_N = 680 cells._  

| Dependent variable | mean_novelty | R² | N |
|---|---|---|---|
| CV | -0.3937*** (SE=0.0769, p=0.0000) | 0.0875 | 680 |
| ln(mean_value + 1) | +0.0515 (SE=0.0800, p=0.5195) | 0.0592 | 680 |
| ln(sd_value + 1) | -0.2453*** (SE=0.0757, p=0.0012) | 0.0320 | 680 |

## 2. Robustness — CV outliers trimmed (top/bottom 1%)
_Dropped 14 cells with CV outside [0.402, 3.708]._

### Same specs, CV-trimmed sample

_N = 666 cells._  

| Dependent variable | mean_novelty | R² | N |
|---|---|---|---|
| CV | -0.3605*** (SE=0.0655, p=0.0000) | 0.1052 | 666 |
| ln(mean_value + 1) | +0.0302 (SE=0.0792, p=0.7029) | 0.0624 | 666 |
| ln(sd_value + 1) | -0.2434*** (SE=0.0757, p=0.0013) | 0.0379 | 666 |

## 3. Robustness — bottom-5% mean_value trimmed
_Dropped 34 cells with mean_value < 5.2298._

### Same specs, low-mean-value-trimmed sample

_N = 646 cells._  

| Dependent variable | mean_novelty | R² | N |
|---|---|---|---|
| CV | -0.4049*** (SE=0.0763, p=0.0000) | 0.0929 | 646 |
| ln(mean_value + 1) | +0.1005 (SE=0.0737, p=0.1726) | 0.0450 | 646 |
| ln(sd_value + 1) | -0.2020*** (SE=0.0711, p=0.0045) | 0.0190 | 646 |

## 4. Nonlinearity
### Nonlinearity: CV ~ mean_novelty + mean_novelty² + year FE

_N = 680 cells._  

| Term | Coef (SE, p) |
|---|---|
| mean_novelty | -0.1795 (SE=0.1397, p=0.1988) |
| mean_novelty² | -0.3139** (SE=0.1507, p=0.0372) |
| R² | 0.0949 |

## 5. Sample size and balance

- Cells used: **680** (non-missing on all required variables).
- Cells per year:
  - 2021: 338
  - 2022: 342

- Distribution of mean_novelty:
count    680.000000
mean       0.521934
std        0.399322
min       -0.907114
25%        0.284546
50%        0.622307
75%        0.812311
max        1.195312

- Distribution of CV:
count    680.000000
mean       1.599530
std        0.699802
min        0.311382
25%        1.179193
50%        1.530528
75%        1.929119
max        8.248595

- Pairwise correlations (cell-level):
              mean_novelty      CV  mean_value  sd_value
mean_novelty        1.0000 -0.2217      0.1076   -0.0249
CV                 -0.2217  1.0000     -0.2001    0.3458
mean_value          0.1076 -0.2001      1.0000    0.7531
sd_value           -0.0249  0.3458      0.7531    1.0000

## Notes
- Novelty is the standardized negated bsim5 (Kelly et al. 2021), constructed   patent-level then averaged within (cpc_subclass, grant_year) cells. A higher   value means a more novel technology.
- CV = sd(xi_real) / mean(xi_real) within cell, restricted to cells with   patent_count >= 10 (matches main.py).
- log(x + 1) used on mean_value and sd_value to guard against tiny denominators;   positive values do not make log(x+1) equal to log(x).
- SE clustered at cpc_subclass when the regression sample contains > 1 cluster,   HC1 otherwise.