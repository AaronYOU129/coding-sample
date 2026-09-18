# Purpose:    Verify the simplified Task 2 likelihood, parameter conversion,
#             estimation, and diagnostics against independent numerical
#             references so later modifications cannot silently break the core.
# Inputs:     Small arrays and simulated data created inside the tests.
# Outputs:    Pytest pass/fail results only.
# Key Steps:  Compare likelihood to direct density calculation -> round-trip
#             transformed parameters -> verify EDA summaries -> recover a
#             known mixture -> verify K=3 comparison -> check QQ distance.
# How to Run: `pytest` from the task2/ directory.

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import diagnostics
import eda
import estimate
import estimate_k3
import loglik
import paths

PARAMETERS = (0.4, 0.0, 0.7, 5.0, 1.0)


def test_loglik_matches_direct_density() -> None:
    """Verify the explicit K=2 likelihood matches direct density arithmetic."""
    x = np.array([-1.0, 0.0, 4.0, 6.0])
    pi, mu1, sigma1, mu2, sigma2 = PARAMETERS
    direct = (
        pi * stats.norm.pdf(x, mu1, sigma1)
        + (1.0 - pi) * stats.norm.pdf(x, mu2, sigma2)
    )
    assert loglik.loglik(x, *PARAMETERS) == pytest.approx(float(np.log(direct).sum()))


def test_parameter_transform_round_trip() -> None:
    """Ensure optimizer transformations preserve all five reported parameters."""
    recovered = loglik.from_unconstrained(loglik.to_unconstrained(*PARAMETERS))
    assert np.allclose(recovered, PARAMETERS)


def test_generic_component_transform_round_trip() -> None:
    """Ensure the shared K=3 optimizer transformation preserves all components."""
    weights = np.array([0.2, 0.5, 0.3])
    means = np.array([2.0, 5.0, 8.5])
    sigmas = np.array([0.8, 1.2, 0.4])
    theta = loglik.components_to_unconstrained(weights, means, sigmas)
    recovered = loglik.components_from_unconstrained(theta, 3)
    for recovered_values, expected_values in zip(
        recovered,
        (weights, means, sigmas),
        strict=True,
    ):
        assert np.allclose(recovered_values, expected_values)


def test_eda_summary_reports_missingness_and_quantiles() -> None:
    """Check the exported EDA fields distinguish total and usable observations."""
    summary = eda.summary_statistics(pd.Series([1.0, 2.0, 3.0, np.nan])).iloc[0]
    assert summary["n_total"] == 4
    assert summary["n_non_missing"] == 3
    assert summary["n_missing"] == 1
    assert summary["mean"] == pytest.approx(2.0)
    assert summary["median"] == pytest.approx(2.0)


def test_eda_summary_latex_is_report_ready() -> None:
    """Verify the LaTeX export contains a complete table and formatted values."""
    summary = eda.summary_statistics(pd.Series([1.0, 2.0, 3.0]))
    latex = eda.summary_statistics_latex(summary)
    assert r"\begin{table}" in latex
    assert "Mean & 2.000" in latex
    assert r"\label{tab:task2-summary}" in latex


def test_eda_figure_is_written(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Smoke-test the EDA plot because numeric tests cannot verify file output."""
    figure_path = tmp_path / "eda_distribution.png"
    monkeypatch.setattr(paths, "EDA_FIGURE", figure_path)
    eda.plot_distribution(pd.Series([1.0, 2.0, 2.5, 3.0, 5.0]))
    assert figure_path.exists()


def test_fit_recovers_known_mixture() -> None:
    """Check the simplified optimizer recovers known simulated parameters."""
    rng = np.random.default_rng(11)
    components = rng.random(4000) >= PARAMETERS[0]
    x = rng.normal(
        np.where(components, PARAMETERS[3], PARAMETERS[1]),
        np.where(components, PARAMETERS[4], PARAMETERS[2]),
    )
    theta0 = loglik.to_unconstrained(0.5, 0.2, 1.0, 4.8, 1.2)
    result = estimate.fit_once(theta0, x)
    assert result.converged
    assert result.pi == pytest.approx(PARAMETERS[0], abs=0.03)
    assert result.mu1 == pytest.approx(PARAMETERS[1], abs=0.08)
    assert result.mu2 == pytest.approx(PARAMETERS[3], abs=0.08)


def test_model_comparison_uses_correct_parameter_penalties() -> None:
    """Check AIC/BIC count five K=2 and eight K=3 free parameters."""
    comparison = estimate_k3.model_comparison(
        k2_loglik=-100.0,
        k3_loglik=-95.0,
        n_obs=1000,
    )
    k2_row, k3_row = comparison.iloc[0], comparison.iloc[1]
    assert k2_row["n_parameters"] == 5
    assert k3_row["n_parameters"] == 8
    assert k2_row["aic"] == pytest.approx(210.0)
    assert k3_row["aic"] == pytest.approx(206.0)
    assert k2_row["bic"] == pytest.approx(5 * np.log(1000) + 200)
    assert r"\begin{table}" in estimate_k3.comparison_latex(comparison)


def test_quantile_rmse_measures_distance_from_equality_line() -> None:
    """Check that QQ RMSE is zero on and positive away from the 45-degree line."""
    observed = np.array([1.0, 2.0, 3.0])
    assert diagnostics.quantile_rmse(observed, observed) == pytest.approx(0.0)
    assert diagnostics.quantile_rmse(observed, observed + 1.0) == pytest.approx(1.0)
