# Diagnosing the SD-divergence: log vs. year FE

## Verdict

- **Adding the log transformation alone** (no year FE): raw SD coef -1.6789 (SE=2.9431, p=0.5684) -> log(SD) coef -0.2484*** (SE=0.0758, p=0.0010).
- **Adding year FE alone** (raw SD): -1.6789 (SE=2.9431, p=0.5684) -> -1.6265 (SE=2.9406, p=0.5802).
- **Both** (log + year FE): -0.2453*** (SE=0.0757, p=0.0012) — the specification you reported as -0.245.
- **Drop the single largest-SD cell** (F16D 2021, SD=259): raw-SD/no-FE goes from -1.6789 (SE=2.9431, p=0.5684) to -2.0852 (SE=2.9227, p=0.4756).

_See per-step reasoning below._

_Specification: cell = (cpc_subclass, grant_year), N = 680, SE clustered at cpc_subclass._

## 1. The 2x2 grid

### Full sample (N = 680)

| Dependent | No year FE | With year FE |
|---|---|---|
| raw SD | -1.6789 (SE=2.9431, p=0.5684) | -1.6265 (SE=2.9406, p=0.5802) |
| ln(SD + 1) | -0.2484*** (SE=0.0758, p=0.0010) | -0.2453*** (SE=0.0757, p=0.0012) |

## 2. Year-by-year structure

### Year-by-year correlations (raw and log SD vs. mean_novelty)

| Year | N | corr(novelty, SD) | corr(novelty, ln(SD)) | corr(novelty, mean_value) |
|---|---|---|---|---|
| 2021 | 338 | -0.065 | -0.147 | +0.098 |
| 2022 | 342 | +0.011 | -0.129 | +0.126 |

_Pooled (no FE, raw):_  -0.025  
_Pooled (no FE, log):_  -0.137  

### Between-year shifts (cell averages by year)

| Year | mean(mean_novelty) | mean(SD) | mean(ln(SD)) | mean(mean_value) |
|---|---|---|---|---|
| 2021 | +0.516 | 34.88 | 3.375 | 25.33 |
| 2022 | +0.528 | 32.04 | 3.206 | 19.50 |

## 3. Outlier check

### Top 5 cells by raw SD (outlier check)

| cpc_subclass | year | mean_novelty | SD | mean_value | CV | n_patents |
|---|---|---|---|---|---|---|
| F16D | 2021 | +0.716 | 259.4 | 31.45 | 8.25 | 148 |
| C07B | 2022 | +0.846 | 195.1 | 85.81 | 2.27 | 40 |
| C10L | 2022 | +0.996 | 145.4 | 97.08 | 1.50 | 24 |
| C10G | 2022 | +1.041 | 140.4 | 135.41 | 1.04 | 60 |
| C09C | 2022 | +0.711 | 139.2 | 58.04 | 2.40 | 12 |

**Drop single largest-SD cell:**

### After dropping top-1 SD cell (N = 679)

| Dependent | No year FE | With year FE |
|---|---|---|
| raw SD | -2.0852 (SE=2.9227, p=0.4756) | -2.0433 (SE=2.9182, p=0.4838) |
| ln(SD + 1) | -0.2525*** (SE=0.0758, p=0.0009) | -0.2494*** (SE=0.0757, p=0.0010) |

**Trim top/bottom 1% of raw SD:**

### After 1%/99% raw-SD trim (N = 666)

| Dependent | No year FE | With year FE |
|---|---|---|
| raw SD | -4.4010* (SE=2.5894, p=0.0892) | -4.3737* (SE=2.5869, p=0.0909) |
| ln(SD + 1) | -0.2776*** (SE=0.0690, p=0.0001) | -0.2762*** (SE=0.0689, p=0.0001) |
