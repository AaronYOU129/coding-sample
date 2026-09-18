# Task 2 — Normal Mixture Maximum Likelihood

Task 2 estimates the required two-component normal mixture:

\[
f(x)=\pi N(\mu_1,\sigma_1)+(1-\pi)N(\mu_2,\sigma_2).
\]

The mixture likelihood and parameter transformations are implemented directly.
SciPy provides numerical functions and the general-purpose L-BFGS-B optimizer;
no pre-built mixture-model estimator is used.

## Input

- `../data/mixture_data.csv` — 10,000 observations in column `x`.

## Method

The likelihood is evaluated with log-sum-exp for numerical stability. Mixture
weights use a logit transformation, standard deviations are optimized on the
log scale, and a lower standard-deviation bound prevents degenerate solutions.
Because the likelihood is non-convex, the estimator runs 25 starting values
with a fixed seed and retains the highest-likelihood converged result.
Components are sorted by their means to resolve label switching.

The required K=2 model is the main estimate. A K=3 fit is used only as a
diagnostic comparison through fitted densities, QQ plots, AIC, and BIC.

## Pipeline

1. `eda.py` summarizes and plots the observed distribution.
2. `estimate.py` estimates the K=2 model.
3. `estimate_k3.py` estimates K=3 and constructs the model comparison.
4. `diagnostics.py` compares fitted densities, QQ behavior, and moments.
5. `run_all.py` runs these stages in order.

`loglik.py` contains the likelihood, CDF, and parameter transformations.
`paths.py` contains all Task 2 input and output paths.

## Run

From `task2/`:

```bash
python run_all.py
```

## Generated files

Intermediate files:

- `work/multistart_results.csv`
- `work/k3_multistart_results.csv`

Final outputs:

- `output/eda_summary.csv`
- `output/eda_summary.tex`
- `output/estimates.csv`
- `output/k3_estimates.csv`
- `output/model_comparison.csv`
- `output/model_comparison.tex`
- `output/diagnostics_summary.csv`

Figures:

- `figures/eda.png`
- `figures/density_fit.png`
- `figures/qq_plot.png`

## Main results

The best K=2 solution across 25 starts has
\(\hat\pi=0.912\), \(\hat\mu_1=3.859\),
\(\hat\sigma_1=1.865\), \(\hat\mu_2=8.517\), and
\(\hat\sigma_2=0.358\), with log-likelihood \(-21{,}521.40\).

The K=2 QQ RMSE is 0.251, compared with 0.012 for K=3. AIC and BIC are also
lower for K=3. These diagnostics indicate that the required K=2 model does not
fully reproduce the three visible modes, while K=2 remains the requested main
specification.

## Checks

The nine tests cover likelihood calculations, parameter transformations,
simulated-data recovery, EDA output, the K=3 comparison, and QQ diagnostics.

```bash
ruff check .
mypy paths.py eda.py loglik.py estimate.py estimate_k3.py diagnostics.py run_all.py
pytest -q
```

