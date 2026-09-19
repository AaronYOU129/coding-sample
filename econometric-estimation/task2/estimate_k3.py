# Purpose:    Fit a three-component normal mixture as a diagnostic comparison
#             for the visibly three-modal data. The same hand-coded likelihood,
#             transformed parameters, multi-start strategy, sigma floor, and
#             label convention as K=2 make AIC/BIC differences attributable to
#             component count rather than a different estimation method.
# Inputs:     ../data/mixture_data.csv, output/estimates.csv (K=2 result).
# Outputs:    output/k3_estimates.csv, work/k3_multistart_results.csv,
#             output/model_comparison.csv, and output/model_comparison.tex.
# Key Steps:  Generate K=3 starts -> run bounded L-BFGS-B from each -> select
#             the best converged endpoint -> calculate K=2/K=3 AIC and BIC ->
#             export estimates and comparison tables.
# How to Run: `python estimate.py` followed by `python estimate_k3.py`, or
#             `python run_all.py` from task2/.

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import estimate
import loglik
import paths

K = 3
SEED = 20260723
N_STARTS = 25
SIGMA_FLOOR_FRACTION = 0.01
BEST_LOG_LIKELIHOOD_TOLERANCE = 1e-4
EDA_WEIGHTS = np.array([1 / 3, 1 / 3, 1 / 3])
EDA_MEANS = np.array([2.0, 5.0, 8.5])
EDA_SIGMAS = np.array([0.8, 1.2, 0.4])


@dataclass
class K3FitResult:
    weights: np.ndarray
    means: np.ndarray
    sigmas: np.ndarray
    loglik: float
    converged: bool
    n_iterations: int
    floor_hit: bool


def main() -> None:
    """Fit K=3 and export its estimates plus penalized comparison with K=2.

    Reading the saved K=2 likelihood ensures the table compares the reported model runs.
    """
    paths.ensure_directories()
    x = estimate.load_data()
    rng = np.random.default_rng(SEED)
    best, starts_table = fit_multistart(x, N_STARTS, rng)
    k3_estimates = estimate_row(best, starts_table)
    k3_estimates.to_csv(paths.K3_ESTIMATES, index=False)
    starts_table.to_csv(paths.K3_MULTISTART_RESULTS, index=False)

    k2_loglik = float(pd.read_csv(paths.ESTIMATES).iloc[0]["loglik"])
    comparison = model_comparison(k2_loglik, best.loglik, len(x))
    comparison.to_csv(paths.MODEL_COMPARISON, index=False)
    paths.MODEL_COMPARISON_TEX.write_text(
        comparison_latex(comparison),
        encoding="utf-8",
    )

    n_at_best = int(starts_table["reached_best"].sum())
    print(
        f"Best K=3 log-likelihood: {best.loglik:.2f}; "
        f"{n_at_best}/{N_STARTS} starts reached it."
    )
    print(comparison.to_string(index=False))


def draw_starts(
    rng: np.random.Generator,
    x: np.ndarray,
    n_starts: int,
) -> list[np.ndarray]:
    """Create one three-mode EDA start and additional broad random starts.

    Diverse starts are needed because adding a component creates more likelihood basins.
    """
    starts = [
        loglik.components_to_unconstrained(EDA_WEIGHTS, EDA_MEANS, EDA_SIGMAS)
    ]
    low, high = np.quantile(x, [0.01, 0.99])
    for _ in range(n_starts - 1):
        weights = rng.dirichlet(np.ones(K))
        means = rng.uniform(low, high, size=K)
        sigmas = rng.uniform(0.1, 1.0, size=K) * x.std()
        starts.append(loglik.components_to_unconstrained(weights, means, sigmas))
    return starts


def sort_components(
    parameters: loglik.ComponentParameters,
) -> loglik.ComponentParameters:
    """Order all K=3 parameters by increasing mean.

    Moving weights and sigmas with the means fixes label switching without changing density.
    """
    weights, means, sigmas = parameters
    order = np.argsort(means)
    return weights[order], means[order], sigmas[order]


def fit_once(theta0: np.ndarray, x: np.ndarray) -> K3FitResult:
    """Run one bounded K=3 L-BFGS-B optimization from a supplied start.

    The same data-scaled sigma floor used for K=2 prevents degenerate point spikes.
    """
    sigma_floor = SIGMA_FLOOR_FRACTION * x.std()
    log_floor = np.log(sigma_floor)
    bounds = (
        [(None, None)] * K
        + [(log_floor, None)] * K
        + [(None, None)] * (K - 1)
    )
    result = minimize(
        loglik.components_negative_loglik,
        theta0,
        args=(x, K),
        method="L-BFGS-B",
        bounds=bounds,
    )
    weights, means, sigmas = sort_components(
        loglik.components_from_unconstrained(result.x, K)
    )
    return K3FitResult(
        weights=weights,
        means=means,
        sigmas=sigmas,
        loglik=-float(result.fun),
        converged=bool(result.success),
        n_iterations=int(result.nit),
        floor_hit=bool(sigmas.min() <= sigma_floor * (1.0 + 1e-6)),
    )


def fit_multistart(
    x: np.ndarray,
    n_starts: int,
    rng: np.random.Generator,
) -> tuple[K3FitResult, pd.DataFrame]:
    """Fit every K=3 start and retain the highest-likelihood converged endpoint.

    Recording every endpoint shows whether model comparison depends on a rare basin.
    """
    fits = [fit_once(theta0, x) for theta0 in draw_starts(rng, x, n_starts)]
    successful_fits = [fit_result for fit_result in fits if fit_result.converged]
    if not successful_fits:
        raise RuntimeError("No K=3 optimizer start converged.")

    best = max(successful_fits, key=lambda fit_result: fit_result.loglik)
    rows = []
    for start_id, fit_result in enumerate(fits):
        row: dict[str, object] = {
            "start_id": start_id,
            "loglik": fit_result.loglik,
            "converged": fit_result.converged,
            "n_iterations": fit_result.n_iterations,
            "floor_hit": fit_result.floor_hit,
            "reached_best": bool(
                np.isclose(
                    fit_result.loglik,
                    best.loglik,
                    atol=BEST_LOG_LIKELIHOOD_TOLERANCE,
                )
            ),
        }
        for component in range(K):
            row[f"weight_{component + 1}"] = fit_result.weights[component]
            row[f"mu_{component + 1}"] = fit_result.means[component]
            row[f"sigma_{component + 1}"] = fit_result.sigmas[component]
        rows.append(row)
    return best, pd.DataFrame(rows)


def model_comparison(
    k2_loglik: float,
    k3_loglik: float,
    n_obs: int,
) -> pd.DataFrame:
    """Calculate AIC and BIC using 3K-1 free parameters for each model.

    Both criteria reward fit improvement but penalize K=3 for three extra parameters.
    """
    rows = []
    for k, model_loglik in ((2, k2_loglik), (3, k3_loglik)):
        n_parameters = 3 * k - 1
        rows.append({
            "k": k,
            "n_parameters": n_parameters,
            "loglik": model_loglik,
            "aic": 2 * n_parameters - 2 * model_loglik,
            "bic": n_parameters * np.log(n_obs) - 2 * model_loglik,
        })
    return pd.DataFrame(rows)


def comparison_latex(comparison: pd.DataFrame) -> str:
    """Format AIC/BIC results as a compact report-ready LaTeX table.

    A standalone table makes the penalized model comparison easy to cite in the report.
    """
    table_rows = []
    for row in comparison.itertuples(index=False):
        table_rows.append(
            f"{row.k} & {row.n_parameters} & {row.loglik:.2f} "
            f"& {row.aic:.2f} & {row.bic:.2f} \\\\"
        )
    return "\n".join([
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Comparison of two- and three-component normal mixtures}",
        r"\label{tab:mixture-model-comparison}",
        r"\begin{tabular}{rrrrr}",
        r"\toprule",
        r"$K$ & Parameters & Log-likelihood & AIC & BIC \\",
        r"\midrule",
        *table_rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ])


def estimate_row(best: K3FitResult, starts_table: pd.DataFrame) -> pd.DataFrame:
    """Build one wide K=3 estimate row for diagnostics and reporting.

    Explicit component columns keep the optional comparison separate from K=2 output.
    """
    row: dict[str, object] = {
        "loglik": best.loglik,
        "converged": best.converged,
        "n_iterations": best.n_iterations,
        "floor_hit": best.floor_hit,
        "n_starts": N_STARTS,
        "n_starts_at_best": int(starts_table["reached_best"].sum()),
    }
    for component in range(K):
        row[f"weight_{component + 1}"] = best.weights[component]
        row[f"mu_{component + 1}"] = best.means[component]
        row[f"sigma_{component + 1}"] = best.sigmas[component]
    return pd.DataFrame([row])


if __name__ == "__main__":  # pragma: no cover
    main()
