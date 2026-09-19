# Stress test of the log(SD) ~ mean_novelty result

_Specification: cell = (cpc_subclass, grant_year), HC1 robust SE._  
_Baseline: log(SD+1) ~ mean_novelty + year FE coef = -0.245 (cluster SE in earlier diagnostic)._  

_Sanity: 0 cells with sd=0 / mad=0 / iqr=0; the +1 offset on log(x+1) remains different from log(x), even when all values are positive._

## BOTTOM LINE

| Claim | Verdict | Note |
|---|---|---|
| **S1** Year FE irrelevant (already verified) | HOLDS UNDER STRESS | 2021 vs 2022 cell-mean novelty 0.516 vs 0.528, mean SD 34.9 vs 32.0 — between-year contrast is small. |
| **S2** Raw SD ~ novelty is null | QUALIFIED | Raw SD relationship is **negative but outlier-masked**. Full-sample coef -1.63 (p=0.49) is null, but 4/4 outlier-handling treatments produce a more negative coef and 3/4 reach p<0.10. |
| **S3** Log SD ~ novelty is real bulk pattern | DOES NOT HOLD | Compound verdict: tied to A1 (DOES NOT HOLD, only 31% of coef survives log(n) control) and R1 (DOES NOT HOLD, scale-robust measures disagree with log SD). |
| **S4** Top-SD cells systematically high novelty (blockbuster pattern) | QUALIFIED | **U-shaped pattern**: both lowest and highest SD deciles are above-average novelty. Consistent with 'novel = tighter bulk + occasional blockbusters' story but only weakly. |
| **A1** Cell size (n_patents) confound | DOES NOT HOLD | Only 30.9% of the original coefficient survives. Cell size is a real confound. |
| **A2** Heteroskedasticity bias | HOLDS WITH QUALIFICATION | BP test rejects homoskedasticity (p=0.000), but the HC3 correction does not flip the qualitative significance (HC1 p=0.0000, HC3 p=0.0000). |
| **A3** n_patents cutoff sensitivity | HOLDS UNDER STRESS | Coefficients span -0.245 to -0.214, range/baseline = 12.7% (< 30%). |
| **R1** MAD / IQR robustness (substantive) | DOES NOT HOLD | MAD goes the wrong direction (MAD: +0.151, p=0.054 | IQR: -0.090, p=0.254); the only same-direction measure IQR is insignificant and only 37% of the log(SD) baseline. The log(SD) result is largely a right-tail compression artifact — scale-robust measures do NOT show novel cells as more homogeneous. |

## 1. S1 — year FE irrelevance (already established)

Year-by-year cell averages (from previous diagnostic):
- 2021: mean novelty = +0.516, mean SD = 34.88, mean ln(SD) = 3.375
- 2022: mean novelty = +0.528, mean SD = 32.04, mean ln(SD) = 3.206

Adding year FE shifts the slope by 1-2% in both raw and log specs. No reason to revisit.

## 2. S2 — raw SD outlier sensitivity

| Outlier treatment | mean_novelty coef | N |
|---|---|---|
| (a) Full sample | -1.6265 (SE=2.3320, p=0.4855) | 680 |
| (b) Trim top 1% (drop SD > 131.31) | -4.5645** (SE=2.0221, p=0.0240) | 673 |
| (c) Trim top 5% (drop SD > 87.39) | -7.1018*** (SE=1.6523, p=0.0000) | 646 |
| (d) Drop SD > 100 (hard threshold) | -7.8474*** (SE=1.8284, p=0.0000) | 654 |
| (e) Winsorize top 1% (cap at 131.31) | -2.1804 (SE=2.2417, p=0.3307) | 680 |

**Verdict (QUALIFIED):** Raw SD relationship is **negative but outlier-masked**. Full-sample coef -1.63 (p=0.49) is null, but 4/4 outlier-handling treatments produce a more negative coef and 3/4 reach p<0.10.

## 3. A1 — cell size confound

- Corr(mean_novelty, n_patents)        = -0.3361  
- Corr(mean_novelty, log(n_patents))   = -0.4835  

| Spec | mean_novelty coef | log(n) coef |
|---|---|---|
| Original (no log(n)) | -0.2453*** (SE=0.0596, p=0.0000) | — |
| With log(n) control  | -0.0757 (SE=0.0674, p=0.2607) | +0.1004*** (SE=0.0221, p=0.0000) |

Survival ratio: 30.9% of |-0.245| survives the log(n) control.  
**Verdict (DOES NOT HOLD):** Only 30.9% of the original coefficient survives. Cell size is a real confound.

## 4. A2 — heteroskedasticity

- log(SD+1) ~ novelty + year FE under HC1: -0.2453*** (SE=0.0596, p=0.0000)
- log(SD+1) ~ novelty + year FE under HC3: -0.2453*** (SE=0.0598, p=0.0000)
- Breusch-Pagan (residuals on mean_novelty): chi² = 27.506, p = 0.0000

Diagnostic plot saved to `output/het_check.png`.

**Verdict (HOLDS WITH QUALIFICATION):** BP test rejects homoskedasticity (p=0.000), but the HC3 correction does not flip the qualitative significance (HC1 p=0.0000, HC3 p=0.0000).

## 5. A3 — n_patents cutoff sensitivity

| Cutoff | N cells | mean_novelty coef on log(SD+1) |
|---|---|---|
| n_patents >= 5 | 680 | -0.2453*** (SE=0.0596, p=0.0000) |
| n_patents >= 10 | 680 | -0.2453*** (SE=0.0596, p=0.0000) |
| n_patents >= 20 | 542 | -0.2213*** (SE=0.0656, p=0.0007) |
| n_patents >= 30 | 479 | -0.2177*** (SE=0.0670, p=0.0012) |
| n_patents >= 50 | 366 | -0.2141*** (SE=0.0742, p=0.0039) |

**Verdict (HOLDS UNDER STRESS):** Coefficients span -0.245 to -0.214, range/baseline = 12.7% (< 30%).

## 6. R1 — MAD and IQR (substantive)

| Dispersion measure | No log(n) | With log(n) control |
|---|---|---|
| log(SD+1) (baseline) | -0.2453*** (SE=0.0596, p=0.0000) | -0.0757 (SE=0.0674, p=0.2607) |
| log(MAD+1)           | +0.1514* (SE=0.0787, p=0.0543) | +0.2165** (SE=0.0897, p=0.0158) |
| log(IQR+1)           | -0.0903 (SE=0.0793, p=0.2543) | -0.0109 (SE=0.0869, p=0.9004) |

Magnitudes relative to log(SD) baseline (-0.245):  
- MAD / SD = -0.62  
- IQR / SD = 0.37  

**Verdict (DOES NOT HOLD):** MAD goes the wrong direction (MAD: +0.151, p=0.054 | IQR: -0.090, p=0.254); the only same-direction measure IQR is insignificant and only 37% of the log(SD) baseline. The log(SD) result is largely a right-tail compression artifact — scale-robust measures do NOT show novel cells as more homogeneous.

## 7. S4 — SD decile vs mean novelty

| SD decile | N | mean(mean_novelty) | SE | SD range |
|---|---|---|---|---|
| 1 | 68 | +0.687 | 0.030 | [0.29, 9.79] |
| 2 | 68 | +0.646 | 0.037 | [9.92, 14.83] |
| 3 | 68 | +0.612 | 0.036 | [14.90, 18.55] |
| 4 | 68 | +0.563 | 0.041 | [18.64, 21.96] |
| 5 | 68 | +0.440 | 0.045 | [22.02, 25.87] |
| 6 | 68 | +0.461 | 0.049 | [25.88, 31.73] |
| 7 | 68 | +0.399 | 0.053 | [31.78, 37.64] |
| 8 | 68 | +0.373 | 0.056 | [37.67, 47.43] |
| 9 | 68 | +0.459 | 0.060 | [47.47, 63.60] |
| 10 | 68 | +0.579 | 0.054 | [63.78, 259.41] |

Decile plot saved to `output/sd_decile_novelty.png`.

**Verdict (QUALIFIED):** **U-shaped pattern**: both lowest and highest SD deciles are above-average novelty. Consistent with 'novel = tighter bulk + occasional blockbusters' story but only weakly.

## Notes & framing concerns

- **HC1 vs cluster SE.** Earlier diagnostics used cluster-at-cpc_subclass SE; this stress test uses HC1 per your brief. Point estimates are identical; SEs may differ marginally.
- **"Top-5 cells all high-novelty" wording.** The five cells had mean_novelty 0.71-1.04, against an overall mean of +0.52 and 75th percentile +0.81. Three of the five sit above the 75th pctile, two are between the median and the 75th pctile. "Above-median" is more accurate than "high-novelty" — worth softening in the memo.
- **MAD/IQR vs SD scaling.** For a normal distribution MAD ~ 0.67 SD and IQR ~ 1.35 SD, but log-slopes wrt novelty are scale-invariant. So if MAD/IQR coefficients are materially different from log(SD) coef, that is a real signal about distributional shape, not a units mismatch.
