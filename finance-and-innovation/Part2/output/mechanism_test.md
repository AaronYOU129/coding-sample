# Mechanism test: A vs B vs C for the cell-size confound

_Cell = (cpc_subclass, grant_year), n_patents ≥ 10. 680 cells. HC1 robust SE._

## VERDICT MATRIX

| Mechanism | Verdict | Evidence |
|---|---|---|
| **A — pure statistical bias (c4)** | NOT SUPPORTED | c4 correction shifts the coefficient by only 2.44% — the small-sample SD bias defined by c4(n) explains essentially none of the −0.245 result. (Bootstrap median = -0.1471, but this conflates A with C — see Mechanism C verdict for separation.) |
| **B — substantive heterogeneity** | NOT SUPPORTED | At least one non-value feature shows HIGHER within-cell SD in novel cells (with log(n) control), against the homogeneity story. |
| **C — outlier-probability mechanics** | SUPPORTED | max/median grows significantly with log(n) (+64.603, p=0.0201) AND log(median_value) ~ novelty is null/positive (+0.383, p=0.000). Outlier probability mechanically inflates SD in larger cells; the location measure (median) doesn't share log(SD)'s pattern. |

_Note on bootstrap: it draws iid from the pooled empirical xi_real distribution, preserving the empirical novelty/n correlation. The simulated coefficient distribution captures **both** A and C jointly. To separate them, A's narrow form (c4) is tested in isolation by Test 1; what's left in the bootstrap distribution after subtracting A is attributable to C._

## Data summary

- Cells: 680
- n_patents per cell: min=10, median=59, mean=287, max=16432
- c4(n) correction range: 0.9727 to 1.0000 (mean 0.9928). With this n distribution the narrow Mechanism A correction is by construction tiny.

## Test 1 — c4 small-sample correction (Mechanism A, narrow)

| Spec | mean_novelty coef |
|---|---|
| log(SD+1) ~ novelty + year FE        | -0.2453*** (SE=0.0596, p=0.0000) |
| log(SD/c4 + 1) ~ novelty + year FE   | -0.2393*** (SE=0.0595, p=0.0001) |

Δ in coefficient = 0.0060 (2.4% of baseline).

## Test 2 — bootstrap from pooled xi_real (Mechanisms A + C jointly)

_500 reps. For each cell, n_patents drawn iid with replacement from the pooled empirical xi_real distribution; fake SD computed; log(fake_SD+1) regressed on mean_novelty + year FE. Under the null of no real novelty/dispersion relationship, this captures all the n-driven mechanical bias (A + C)._

| Statistic | Value |
|---|---|
| n | 500 |
| mean | -0.1476 |
| median | -0.1471 |
| std | +0.0371 |
| p05 | -0.2071 |
| p25 | -0.1734 |
| p75 | -0.1221 |
| p95 | -0.0875 |
| min | -0.2539 |
| max | -0.0451 |

_Observed coefficient on real data: -0.2453._

**Mechanical share:** the bootstrap median (-0.1471) covers 60.0% of the observed −0.245. That fraction is attributable to A+C combined; the rest must come from B or unobserved factors.

## Test 3 — within-cell SDs of non-value features (Mechanism B)

| Within-cell feature SD | No log(n) | With log(n) | log(n) coef |
|---|---|---|---|
| ln(SD log(num_claims) + 1) | +0.0362*** (SE=0.0072, p=0.0000) | +0.0468*** (SE=0.0080, p=0.0000) | +0.0063** (SE=0.0025, p=0.0132) |
| ln(SD log(backward_cites+1) + 1) | -0.0462*** (SE=0.0106, p=0.0000) | -0.0221 (SE=0.0138, p=0.1090) | +0.0143*** (SE=0.0040, p=0.0004) |
| ln(SD patent novelty + 1) | -0.3287*** (SE=0.0082, p=0.0000) | -0.3398*** (SE=0.0097, p=0.0000) | -0.0066** (SE=0.0028, p=0.0191) |

## Test 4 — outlier probability (Mechanism C)

| Spec | Coef on focal regressor |
|---|---|
| max_to_median ~ log(n) + year FE | log(n): +64.6032** (SE=27.7955, p=0.0201) |
| max_to_median ~ novelty + year FE | novelty: +137.7469 (SE=92.9899, p=0.1385) |
| log(median+1) ~ novelty + year FE | novelty: +0.3834*** (SE=0.0852, p=0.0000) |

## Most likely mechanism

- Mechanism A in its narrow c4 form is essentially ruled out — the correction barely moves the coefficient because cells have n ≥ 10 (and most are much larger), so the bias term is negligible.
- Mechanism C is corroborated: max/median grows strongly with log(n) (coef +64.603, p=0.0201), confirming that bigger cells mechanically pick up larger maxima from the right tail.
- Mechanism B is not supported: within-cell SDs of non-value features do not consistently fall in novel cells once log(n) is controlled.
- The bootstrap (A+C) median of -0.1471 explains only ~60% of the observed −0.245. A meaningful residual remains, which is where Mechanism B (or other unmodeled factors) could matter.
