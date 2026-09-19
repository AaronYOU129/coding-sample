# Right-skew validation: stress-testing the +0.38 log(median) finding

_Cell = (cpc_subclass, grant_year), n_patents ≥ 10 (Tests 2-5). 680 cells. HC1 robust SE._

## FINDING ROBUSTNESS VERDICT

- **+0.38 log(median) finding:** **QUALIFIED**
- **60% mechanical share:** **FRAGILE**
- **Overall confidence in the right-skew substantive interpretation:** **MEDIUM**

_Permno is the publicly-traded parent firm linked by KPSS, not the raw assignee text. Subsidiaries with the same parent permno collapse to one. This is the closest firm-composition proxy available in the working dataset._

## Test 1 — cutoff sensitivity for log(median)

| Cutoff | N cells | mean_novelty coef on log(median+1) |
|---|---|---|
| n_patents >=   5 | 680 | +0.3834*** (SE=0.0852, p=0.0000) |
| n_patents >=  10 | 680 | +0.3834*** (SE=0.0852, p=0.0000) |
| n_patents >=  20 | 542 | +0.4011*** (SE=0.0872, p=0.0000) |
| n_patents >=  50 | 366 | +0.3188*** (SE=0.1044, p=0.0023) |
| n_patents >= 100 | 252 | +0.2572** (SE=0.1201, p=0.0322) |

**Verdict (QUALIFIED):** Coefficients vary by 37.5% across cutoffs but not monotonically.

## Test 2 — log(n) control on log(median)

| Spec | mean_novelty coef | log(n) coef |
|---|---|---|
| Original                | +0.3834*** (SE=0.0852, p=0.0000) | — |
| + log(n_patents)        | +0.3991*** (SE=0.1016, p=0.0001) | +0.0093 (SE=0.0278, p=0.7381) |

Survival: 104.1% of original coefficient.

**Verdict (ROBUST):** 104.1% of the original coefficient survives the log(n) control.

## Test 3 — alternative skewness measures

If novel cells genuinely have less right-skew, all three skewness measures should be **negative** (less Pearson skewness, shorter relative right tail).

| Skewness measure | No log(n) | With log(n) | log(n) coef |
|---|---|---|---|
| skew_within = (mean - median) / sd | -0.0492*** (SE=0.0145, p=0.0007) | -0.0565*** (SE=0.0190, p=0.0029) | -0.0043 (SE=0.0051, p=0.4049) |
| log(p90 / median) | -0.5136*** (SE=0.1062, p=0.0000) | -0.4426*** (SE=0.1297, p=0.0006) | +0.0421 (SE=0.0362, p=0.2448) |
| log(p99 / median) | -0.6443*** (SE=0.1239, p=0.0000) | -0.3329** (SE=0.1502, p=0.0267) | +0.1844*** (SE=0.0422, p=0.0000) |

**Verdict (ROBUST):** All 3 alternative skewness measures show significantly less right-skew in novel cells (with log(n) control).

## Test 4 — firm-composition confound (permno proxy)

- Corr(mean_novelty, n_unique_firms)        = -0.4403
- Corr(mean_novelty, log(n_unique_firms))   = -0.5100

| Spec | mean_novelty coef | log(n_firms) coef | log(n) coef |
|---|---|---|---|
| Original                | +0.3834*** (SE=0.0852, p=0.0000) | — | — |
| + log(n_firms)          | +0.5063*** (SE=0.1046, p=0.0000) | +0.0971** (SE=0.0428, p=0.0234) | — |
| + log(n) + log(n_firms) | +0.4754*** (SE=0.1042, p=0.0000) | +0.2158*** (SE=0.0770, p=0.0051) | -0.1073** (SE=0.0495, p=0.0300) |

**Verdict (ROBUST):** Firm-count control preserves 132.1% of coef; with both controls, 124.0% survives. Firm composition is not the main driver.

## Test 5 — bootstrap log(SD) ~ novelty under two distributions

_Pooled xi_real fitted to log-normal: mu_log = 1.5314, sigma_log = 2.4749. Bootstrap samples drawn iid per cell from each distribution, novelty/n correlation preserved._

| Statistic | Empirical | Log-normal |
|---|---|---|
| n | 500 | 500 |
| mean | -0.1476 | -0.7289 |
| median | -0.1471 | -0.7285 |
| std | +0.0371 | +0.0893 |
| p05 | -0.2071 | -0.8785 |
| p95 | -0.0875 | -0.5862 |

**Verdict (FRAGILE):** Empirical -0.147 vs log-normal -0.728 (395.2% apart). Mechanical share is highly distribution-dependent.

## Caveats

- **permno ≠ assignee.** permno covers the publicly-traded parent firm linked by KPSS. Subsidiaries / private subcontractors / non-public co-assignees collapse to a single permno or are omitted. A finer firm-composition test would use raw assignee text.
- **log-normal vs empirical.** The log-normal bootstrap uses moments matched to the pooled log(xi_real); KPSS values may have heavier (Pareto-like) right tails than log-normal predicts, in which case the empirical bootstrap is the more conservative number.
- **log(median+1) offset.** Median values in this sample are large enough that the +1 offset is essentially a no-op.