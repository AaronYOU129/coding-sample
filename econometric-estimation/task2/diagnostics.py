# Purpose:    Assess whether the required K=2 mixture adequately describes the
#             data and whether optional K=3 improves the distributional fit.
#             Shared density and QQ comparisons make the improvement visible.
# Inputs:     ../data/mixture_data.csv, output/estimates.csv, and
#             output/k3_estimates.csv.
# Outputs:    figures/density_fit.png, figures/qq_plot.png, and
#             output/diagnostics_summary.csv.
# Key Steps:  Read both estimates -> compare their fitted densities -> invert
#             both fitted CDFs for matched QQ plots -> measure distance from
#             the 45-degree line -> export figures and summary.
# How to Run: `python diagnostics.py` after `python estimate.py`, or `python run_all.py`.

import matplotlib
import numpy as np
import pandas as pd

import estimate
import loglik
import paths

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

Parameters = loglik.ModelParameters
K3Parameters = loglik.ComponentParameters


def main() -> None:
    """Generate the simplified density, QQ, and moment diagnostics.

    Reading saved estimates keeps model checking separate from the estimation procedure.
    """
    paths.ensure_directories()
    x = estimate.load_data()
    parameters = read_estimate()
    k3_parameters = read_k3_estimate()

    plot_density(x, parameters, k3_parameters)
    k2_qq_rmse, k3_qq_rmse = plot_qq(x, parameters, k3_parameters)
    model_mean, model_std = model_mean_and_std(parameters)
    summary = pd.DataFrame([{
        "k2_qq_rmse": k2_qq_rmse,
        "k3_qq_rmse": k3_qq_rmse,
        "qq_preferred_k": 2 if k2_qq_rmse < k3_qq_rmse else 3,
        "data_mean": x.mean(),
        "model_mean": model_mean,
        "data_std": x.std(ddof=1),
        "model_std": model_std,
    }])
    summary.to_csv(paths.DIAGNOSTICS_SUMMARY, index=False)

    print(summary.to_string(index=False))
    print(f"Figures written to {paths.FIGURES_DIR}")


def read_estimate() -> Parameters:
    """Read the single fitted K=2 parameter row into the model's tuple layout.

    Central parsing keeps every diagnostic consistent with the exported estimate schema.
    """
    row = pd.read_csv(paths.ESTIMATES).iloc[0]
    return (
        float(row["pi"]),
        float(row["mu1"]),
        float(row["sigma1"]),
        float(row["mu2"]),
        float(row["sigma2"]),
    )


def read_k3_estimate() -> K3Parameters:
    """Read the diagnostic K=3 estimate into component arrays.

    Keeping K=3 separate prevents the optional comparison from obscuring the required K=2 API.
    """
    row = pd.read_csv(paths.K3_ESTIMATES).iloc[0]
    weights = np.array([row[f"weight_{component}"] for component in range(1, 4)])
    means = np.array([row[f"mu_{component}"] for component in range(1, 4)])
    sigmas = np.array([row[f"sigma_{component}"] for component in range(1, 4)])
    return weights, means, sigmas


def theoretical_quantiles(x: np.ndarray, parameters: Parameters) -> np.ndarray:
    """Approximate fitted K=2 quantiles by interpolating the mixture CDF grid.

    Grid inversion avoids a slower root-finding problem for every observed quantile.
    """
    probabilities = (np.arange(1, len(x) + 1) - 0.5) / len(x)
    grid = np.linspace(x.min() - 3.0, x.max() + 3.0, 4001)
    fitted_cdf = loglik.mixture_cdf(grid, *parameters)
    return np.asarray(np.interp(probabilities, fitted_cdf, grid))


def theoretical_quantiles_components(
    x: np.ndarray,
    parameters: K3Parameters,
) -> np.ndarray:
    """Approximate fitted K=3 quantiles by interpolating its mixture CDF grid.

    Matching the K=2 probabilities and grid makes the two QQ plots directly comparable.
    """
    probabilities = (np.arange(1, len(x) + 1) - 0.5) / len(x)
    grid = np.linspace(x.min() - 3.0, x.max() + 3.0, 4001)
    fitted_cdf = loglik.mixture_cdf_components(grid, *parameters)
    return np.asarray(np.interp(probabilities, fitted_cdf, grid))


def quantile_rmse(observed: np.ndarray, theoretical: np.ndarray) -> float:
    """Measure the typical vertical distance from the QQ plot's 45-degree line.

    Because both models use the same data scale, a smaller value supports a fair comparison.
    """
    return float(np.sqrt(np.mean((observed - theoretical) ** 2)))


def plot_density(
    x: np.ndarray,
    parameters: Parameters,
    k3_parameters: K3Parameters,
) -> None:
    """Overlay fitted K=2 and K=3 densities on the observed histogram.

    A shared plot shows whether K=3's fit improvement corresponds to the visible third mode.
    """
    grid = np.linspace(x.min() - 0.5, x.max() + 0.5, 1000)
    fitted_density = np.exp(loglik.mixture_logpdf(grid, *parameters))
    k3_density = np.exp(loglik.mixture_logpdf_components(grid, *k3_parameters))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(x, bins=80, density=True, alpha=0.45, label="data")
    ax.plot(grid, fitted_density, linewidth=2, label="fitted K=2 mixture")
    ax.plot(grid, k3_density, linewidth=2, label="fitted K=3 mixture")
    ax.set(xlabel="x", ylabel="density", title="Data and fitted normal mixtures")
    ax.legend()
    fig.tight_layout()
    fig.savefig(paths.FIT_FIGURE, dpi=150)
    plt.close(fig)


def plot_qq(
    x: np.ndarray,
    parameters: Parameters,
    k3_parameters: K3Parameters,
) -> tuple[float, float]:
    """Compare K=2 and K=3 QQ plots against one shared 45-degree reference.

    Shared axes plus RMSE prevent visual scaling from making one model look artificially better.
    """
    observed = np.sort(x)
    k2_theoretical = theoretical_quantiles(x, parameters)
    k3_theoretical = theoretical_quantiles_components(x, k3_parameters)
    k2_rmse = quantile_rmse(observed, k2_theoretical)
    k3_rmse = quantile_rmse(observed, k3_theoretical)
    limits = [
        min(observed.min(), k2_theoretical.min(), k3_theoretical.min()),
        max(observed.max(), k2_theoretical.max(), k3_theoretical.max()),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), sharex=True, sharey=True)
    plot_details = (
        (axes[0], k2_theoretical, 2, k2_rmse),
        (axes[1], k3_theoretical, 3, k3_rmse),
    )
    for ax, theoretical, n_components, rmse in plot_details:
        ax.scatter(theoretical, observed, s=4, alpha=0.3)
        ax.plot(limits, limits, linewidth=1, color="black")
        ax.set(
            xlabel=f"fitted K={n_components} quantiles",
            ylabel="observed quantiles",
            title=f"K={n_components}: QQ RMSE = {rmse:.3f}",
            xlim=limits,
            ylim=limits,
        )
    fig.suptitle("QQ comparison: closer to the 45-degree line is better")
    fig.tight_layout()
    fig.savefig(paths.QQ_FIGURE, dpi=150)
    plt.close(fig)
    return k2_rmse, k3_rmse


def model_mean_and_std(parameters: Parameters) -> tuple[float, float]:
    """Calculate the exact mean and standard deviation implied by the fitted mixture.

    Closed-form moments provide a simple fit check without adding simulation noise.
    """
    pi, mu1, sigma1, mu2, sigma2 = parameters
    mean = pi * mu1 + (1.0 - pi) * mu2
    variance = (
        pi * (sigma1**2 + (mu1 - mean) ** 2)
        + (1.0 - pi) * (sigma2**2 + (mu2 - mean) ** 2)
    )
    return float(mean), float(np.sqrt(variance))


if __name__ == "__main__":  # pragma: no cover
    main()
