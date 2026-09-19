# Purpose:    Estimate the required two-component normal mixture by maximum
#             likelihood with a transparent optimizer driver. Multiple starting
#             values are used because mixture likelihoods have local optima; a
#             small sigma floor prevents degenerate point-mass solutions, and
#             sorting by mean removes arbitrary component-label switching.
# Inputs:     ../data/mixture_data.csv (10,000 observations in column `x`).
# Outputs:    output/estimates.csv (best K=2 estimate) and
#             work/multistart_results.csv (one row per optimizer start).
# Key Steps:  Load data -> generate one EDA and several random starts -> run
#             L-BFGS-B from every start -> sort component labels -> retain the
#             successful result with the largest log-likelihood -> export.
# How to Run: `python estimate.py` from task2/, or `python run_all.py`.

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import loglik
import paths

SEED = 20260722
N_STARTS = 25
SIGMA_FLOOR_FRACTION = 0.01
BEST_LOG_LIKELIHOOD_TOLERANCE = 1e-4
EDA_START = (0.5, 2.0, 1.0, 5.5, 1.0)


@dataclass
class FitResult:
    pi: float
    mu1: float
    sigma1: float
    mu2: float
    sigma2: float
    loglik: float
    converged: bool
    n_iterations: int
    floor_hit: bool


def main() -> None:
    """Estimate the required K=2 model and export its best and per-start results.

    A fixed seed makes the multi-start evidence reproducible for review and modification.
    """
    paths.ensure_directories()
    x = load_data()
    rng = np.random.default_rng(SEED)
    best, starts_table = fit_multistart(x, N_STARTS, rng)

    estimate = vars(best).copy()
    estimate["n_starts"] = N_STARTS
    estimate["n_starts_at_best"] = int(starts_table["reached_best"].sum())
    pd.DataFrame([estimate]).to_csv(paths.ESTIMATES, index=False)
    starts_table.to_csv(paths.MULTISTART_RESULTS, index=False)

    print(
        f"Best K=2 log-likelihood: {best.loglik:.2f}; "
        f"{estimate['n_starts_at_best']}/{N_STARTS} starts reached it."
    )
    print(f"Estimates written to {paths.ESTIMATES}")


def load_data() -> np.ndarray:
    """Load the required analysis column and validate the expected sample.

    Failing early prevents optimization from silently using missing or incomplete input data.
    """
    x = pd.read_csv(paths.DATA_FILE)["x"].to_numpy(dtype=float)
    if len(x) != 10_000 or not np.isfinite(x).all():
        raise ValueError("Expected 10,000 finite observations in column 'x'.")
    return x


def draw_starts(
    rng: np.random.Generator,
    x: np.ndarray,
    n_starts: int,
) -> list[np.ndarray]:
    """Create one EDA-guided start and additional data-informed random starts.

    Diverse starts reduce the risk that the non-convex likelihood selects one poor local optimum.
    """
    starts = [loglik.to_unconstrained(*EDA_START)]
    low, high = np.quantile(x, [0.01, 0.99])

    for _ in range(n_starts - 1):
        means = rng.uniform(low, high, size=2)
        sigmas = rng.uniform(0.1, 1.0, size=2) * x.std()
        pi = float(rng.uniform(0.05, 0.95))
        starts.append(
            loglik.to_unconstrained(pi, means[0], sigmas[0], means[1], sigmas[1])
        )
    return starts


def sort_components(parameters: loglik.ModelParameters) -> loglik.ModelParameters:
    """Place the smaller-mean component first and move its weight and sigma with it.

    This fixed convention removes arbitrary label switching without changing model fit.
    """
    pi, mu1, sigma1, mu2, sigma2 = parameters
    if mu1 <= mu2:
        return parameters
    return 1.0 - pi, mu2, sigma2, mu1, sigma1


def fit_once(theta0: np.ndarray, x: np.ndarray) -> FitResult:
    """Run one bounded L-BFGS-B optimization from a supplied starting vector.

    The sigma floor blocks degenerate spikes, while metadata records convergence reliability.
    """
    sigma_floor = SIGMA_FLOOR_FRACTION * x.std()
    log_floor = np.log(sigma_floor)
    bounds = [
        (None, None),
        (None, None),
        (log_floor, None),
        (log_floor, None),
        (None, None),
    ]
    result = minimize(
        loglik.negative_loglik,
        theta0,
        args=(x,),
        method="L-BFGS-B",
        bounds=bounds,
    )
    pi, mu1, sigma1, mu2, sigma2 = sort_components(
        loglik.from_unconstrained(result.x)
    )
    return FitResult(
        pi=pi,
        mu1=mu1,
        sigma1=sigma1,
        mu2=mu2,
        sigma2=sigma2,
        loglik=-float(result.fun),
        converged=bool(result.success),
        n_iterations=int(result.nit),
        floor_hit=bool(min(sigma1, sigma2) <= sigma_floor * (1.0 + 1e-6)),
    )


def fit_multistart(
    x: np.ndarray,
    n_starts: int,
    rng: np.random.Generator,
) -> tuple[FitResult, pd.DataFrame]:
    """Fit all starts, keep the best converged result, and record every endpoint.

    Filtering failed runs avoids selecting an unreliable objective value as the reported optimum.
    """
    fits = [fit_once(theta0, x) for theta0 in draw_starts(rng, x, n_starts)]
    successful_fits = [fit_result for fit_result in fits if fit_result.converged]
    if not successful_fits:
        raise RuntimeError("No optimizer start converged.")

    best = max(successful_fits, key=lambda fit_result: fit_result.loglik)
    rows = []
    for start_id, fit_result in enumerate(fits):
        row = vars(fit_result).copy()
        row["start_id"] = start_id
        row["reached_best"] = bool(
            np.isclose(
                fit_result.loglik,
                best.loglik,
                atol=BEST_LOG_LIKELIHOOD_TOLERANCE,
            )
        )
        rows.append(row)
    return best, pd.DataFrame(rows)


if __name__ == "__main__":  # pragma: no cover
    main()
