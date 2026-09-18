# Purpose:    Verify that the package specifications implement the intended causal
#             design. Simulated data have a known effect, so successful 2SLS
#             recovery checks variable placement, formulas, clustering, and
#             instrument use without retesting package internals.
# Inputs:     simulated_analysis_data fixture from tests/conftest.py.
# Outputs:    None; pytest assertions check first-stage strength, causal recovery,
#             weak-IV-robust inference, and exported output schemas.
# Key Steps:  Fit first stage -> compare 2SLS with known truth -> invert AR tests
#             -> verify CSV and LaTeX outputs from the file-writing entry point.
# How to Run: `pytest tests/test_estimate.py` from task3/.

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import estimate
import paths
from tests.conftest import TRUE_CAUSAL_EFFECT


# Strong instruments should make 2SLS recover the known causal effect.
def test_strong_instruments_recover_known_causal_effect(
    simulated_analysis_data: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, first_stage_statistic = estimate.estimate_first_stage(simulated_analysis_data)
    instrumental_variables_result = estimate.fit_outcome_instrumental_variables(
        simulated_analysis_data,
        "log1p_duration",
    )

    instrumental_variables_coefficient = float(
        instrumental_variables_result.params[estimate.ENDOGENOUS_REGRESSOR]
    )

    assert first_stage_statistic > 100
    assert abs(instrumental_variables_coefficient - TRUE_CAUSAL_EFFECT) < 0.15

    monkeypatch.setattr(
        estimate,
        "ANDERSON_RUBIN_GRID",
        np.linspace(0.0, 4.0, 21),
    )
    confidence_set = estimate.calculate_anderson_rubin_confidence_set(
        simulated_analysis_data,
        "log1p_duration",
    )
    assert float(confidence_set["p_value_at_zero"]) < 0.05
    assert float(confidence_set["lower_bound"]) <= TRUE_CAUSAL_EFFECT
    assert float(confidence_set["upper_bound"]) >= TRUE_CAUSAL_EFFECT
    assert confidence_set["unbounded"] is False


# Smoke-test main writes CSV and report-ready LaTeX with the expected models.
def test_main_writes_expected_tables(
    simulated_analysis_data: pd.DataFrame,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    analysis_file = tmp_path / "analysis_data.csv"
    first_stage_file = tmp_path / "first_stage.csv"
    estimates_file = tmp_path / "estimates.csv"
    results_table_tex_file = tmp_path / "results_table.tex"
    simulated_analysis_data.to_csv(analysis_file, index=False)

    monkeypatch.setattr(paths, "ANALYSIS_DATA", analysis_file)
    monkeypatch.setattr(paths, "FIRST_STAGE", first_stage_file)
    monkeypatch.setattr(paths, "ESTIMATES", estimates_file)
    monkeypatch.setattr(paths, "RESULTS_TABLE_TEX", results_table_tex_file)
    monkeypatch.setattr(paths, "ensure_directories", lambda: None)
    monkeypatch.setattr(estimate, "ANDERSON_RUBIN_GRID", np.linspace(0.0, 4.0, 11))

    estimate.main()

    first_stage_table = pd.read_csv(first_stage_file)
    estimates = pd.read_csv(estimates_file)
    results_table = results_table_tex_file.read_text(encoding="utf-8")
    assert set(first_stage_table["term"]) == set(estimate.EXCLUDED_INSTRUMENTS)
    assert set(estimates["model"]) == {"2SLS", "Anderson-Rubin"}
    assert set(estimates["outcome"]) == set(estimate.OUTCOMES)
    assert r"\label{tab:t3main}" in results_table
    assert "2SLS" in results_table
    assert "OLS &" not in results_table
