# Purpose:    Estimate the causal effect of clicks with standard IV packages so
#             the specification stays readable, while retaining Anderson-Rubin
#             inference because rainfall and shortage are empirically weak.
# Inputs:     work/analysis_data.csv produced by prepare_data.py.
# Outputs:    output/first_stage.csv, output/estimates.csv, and
#             output/results_table.tex.
# Key Steps:  Fit a clustered first stage -> estimate 2SLS for each outcome
#             -> invert weak-IV-robust tests -> export CSV and LaTeX results.
# How to Run: `python estimate.py` from task3/, or `python run_all.py`.

from collections.abc import Sequence

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
import statsmodels.formula.api as smf
from linearmodels.iv import IV2SLS
from linearmodels.iv.results import OLSResults as InstrumentalVariablesResults
from statsmodels.regression.linear_model import RegressionResultsWrapper

import paths

EXCLUDED_INSTRUMENTS = ["totshort", "rain_1", "rain_2"]
ENDOGENOUS_REGRESSOR = "log_views"
OUTCOMES = ["log1p_duration", "log1p_followups"]
CONTROLS = (
    "PTI + home_page + front_page + editor_pick + video + image + log_word_count "
    "+ C(story_class) + C(month) + C(day_of_week)"
)
INSTRUMENT_FORMULA = " + ".join(EXCLUDED_INSTRUMENTS)
ANDERSON_RUBIN_GRID = np.linspace(-5.0, 5.0, 201)
SIGNIFICANCE_LEVEL = 0.05


# Run the first stage and causal models, then export machine- and report-ready results.
def main() -> None:
    paths.ensure_directories()
    analysis_data = pd.read_csv(paths.ANALYSIS_DATA)

    first_stage_table, first_stage_statistic = estimate_first_stage(analysis_data)
    first_stage_table.to_csv(paths.FIRST_STAGE, index=False)

    estimates = estimate_all_outcomes(analysis_data)
    estimates.to_csv(paths.ESTIMATES, index=False)
    paths.RESULTS_TABLE_TEX.write_text(
        format_results_table_latex(estimates, first_stage_statistic),
        encoding="utf-8",
    )

    print(f"First-stage joint statistic: {first_stage_statistic:.2f}")
    print(estimates.to_string(index=False))


# Fit the first stage and its joint F, the number that flags whether instruments are weak.
def estimate_first_stage(data: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    result = fit_first_stage(data)
    first_stage_statistic = calculate_first_stage_statistic(result)
    first_stage_table = build_first_stage_table(result, first_stage_statistic)
    return first_stage_table, first_stage_statistic


# Estimate every outcome and stack the rows, so both outcomes tabulate together.
def estimate_all_outcomes(data: pd.DataFrame) -> pd.DataFrame:
    estimate_rows = []
    for outcome in OUTCOMES:
        estimate_rows.extend(estimate_outcome(data, outcome))
    return pd.DataFrame(estimate_rows)


# Run 2SLS and weak-IV-robust inference for one outcome.
def estimate_outcome(
    data: pd.DataFrame,
    outcome: str,
) -> list[dict[str, object]]:
    instrumental_variables_result = fit_outcome_instrumental_variables(data, outcome)
    anderson_rubin_result = calculate_anderson_rubin_confidence_set(data, outcome)

    return build_outcome_result_rows(
        data,
        outcome,
        instrumental_variables_result,
        anderson_rubin_result,
    )


# Regress endogenous views on instruments plus controls, the relevance regression 2SLS relies on.
def fit_first_stage(data: pd.DataFrame) -> RegressionResultsWrapper:
    formula = f"{ENDOGENOUS_REGRESSOR} ~ {INSTRUMENT_FORMULA} + {CONTROLS}"
    return fit_clustered_ordinary_least_squares(formula, data)


# Joint Wald F on the instruments; quantifies first-stage strength for the weak-IV check.
def calculate_first_stage_statistic(result: RegressionResultsWrapper) -> float:
    joint_test = result.wald_test(
        joint_zero_hypothesis(EXCLUDED_INSTRUMENTS),
        use_f=True,
        scalar=True,
    )
    return float(joint_test.statistic)


# Tabulate each instrument's coefficient and the shared F, so first-stage evidence is exportable.
def build_first_stage_table(
    result: RegressionResultsWrapper,
    first_stage_statistic: float,
) -> pd.DataFrame:
    rows = [
        {
            "term": instrument,
            "coefficient": float(result.params[instrument]),
            "standard_error": float(result.bse[instrument]),
            "first_stage_statistic": first_stage_statistic,
            "p_value": float(result.pvalues[instrument]),
        }
        for instrument in EXCLUDED_INSTRUMENTS
    ]
    return pd.DataFrame(rows)


# Fit 2SLS with the excluded instruments to isolate exogenous variation in views.
def fit_outcome_instrumental_variables(
    data: pd.DataFrame,
    outcome: str,
) -> InstrumentalVariablesResults:
    formula = (
        f"{outcome} ~ 1 + {CONTROLS} "
        f"+ [{ENDOGENOUS_REGRESSOR} ~ {INSTRUMENT_FORMULA}]"
    )
    return IV2SLS.from_formula(formula, data).fit(
        cov_type="clustered",
        clusters=data["date"],
        debiased=False,
    )


# Shared clustered-OLS fitter; SEs cluster by date since instruments vary at the date level.
def fit_clustered_ordinary_least_squares(
    formula: str,
    data: pd.DataFrame,
) -> RegressionResultsWrapper:
    return smf.ols(formula, data=data).fit(
        cov_type="cluster",
        cov_kwds={"groups": data["date"], "use_correction": False},
        use_t=False,
    )


# Build the AR confidence set for one outcome; valid inference when instruments are weak.
def calculate_anderson_rubin_confidence_set(
    data: pd.DataFrame,
    outcome: str,
) -> dict[str, float | bool]:
    instrument_and_control_matrix = patsy.dmatrix(
        f"{INSTRUMENT_FORMULA} + {CONTROLS}",
        data,
        return_type="dataframe",
    )
    p_values = calculate_anderson_rubin_grid_p_values(
        data[outcome].to_numpy(dtype=float),
        data[ENDOGENOUS_REGRESSOR].to_numpy(dtype=float),
        instrument_and_control_matrix,
        data["date"].to_numpy(),
    )
    return summarize_anderson_rubin_confidence_set(p_values)


# Compute an AR p-value at each candidate beta, the raw material for inverting the test.
def calculate_anderson_rubin_grid_p_values(
    outcome_values: np.ndarray,
    endogenous_values: np.ndarray,
    instrument_and_control_matrix: pd.DataFrame,
    date_clusters: np.ndarray,
) -> np.ndarray:
    return np.array(
        [
            anderson_rubin_p_value(
                outcome_values,
                endogenous_values,
                instrument_and_control_matrix,
                date_clusters,
                candidate_coefficient,
            )
            for candidate_coefficient in ANDERSON_RUBIN_GRID
        ]
    )


# Turn accepted nulls into bounds and an unbounded flag; unbounded means the effect is unidentified.
def summarize_anderson_rubin_confidence_set(
    p_values: np.ndarray,
) -> dict[str, float | bool]:
    accepted_hypotheses = p_values > SIGNIFICANCE_LEVEL
    zero_index = int(np.argmin(np.abs(ANDERSON_RUBIN_GRID)))

    if not accepted_hypotheses.any():
        return {
            "p_value_at_zero": float(p_values[zero_index]),
            "lower_bound": float("nan"),
            "upper_bound": float("nan"),
            "unbounded": False,
        }

    accepted_coefficients = ANDERSON_RUBIN_GRID[accepted_hypotheses]
    return {
        "p_value_at_zero": float(p_values[zero_index]),
        "lower_bound": float(accepted_coefficients[0]),
        "upper_bound": float(accepted_coefficients[-1]),
        "unbounded": bool(accepted_hypotheses[0] or accepted_hypotheses[-1]),
    }


# AR test at one null: instruments' joint significance stays valid under weak identification.
def anderson_rubin_p_value(
    outcome_values: np.ndarray,
    endogenous_values: np.ndarray,
    instrument_and_control_matrix: pd.DataFrame,
    date_clusters: np.ndarray,
    candidate_coefficient: float,
) -> float:
    transformed_outcome = outcome_values - candidate_coefficient * endogenous_values
    result = sm.OLS(transformed_outcome, instrument_and_control_matrix).fit(
        cov_type="cluster",
        cov_kwds={"groups": date_clusters, "use_correction": False},
        use_t=False,
    )
    joint_test = result.wald_test(
        joint_zero_hypothesis(EXCLUDED_INSTRUMENTS),
        use_f=False,
        scalar=True,
    )
    return float(joint_test.pvalue)


# Build the "a = 0, b = 0" string that statsmodels' wald_test expects for joint tests.
def joint_zero_hypothesis(variable_names: Sequence[str]) -> str:
    return ", ".join(f"{variable_name} = 0" for variable_name in variable_names)


# Assemble 2SLS and AR rows for one outcome into a uniform, exportable shape.
def build_outcome_result_rows(
    data: pd.DataFrame,
    outcome: str,
    instrumental_variables_result: InstrumentalVariablesResults,
    anderson_rubin_result: dict[str, float | bool],
) -> list[dict[str, object]]:
    return [
        model_result_row(
            outcome,
            "2SLS",
            instrumental_variables_result.params,
            instrumental_variables_result.std_errors,
            instrumental_variables_result.pvalues,
            data,
        ),
        anderson_rubin_result_row(outcome, anderson_rubin_result, data),
    ]


# Format the report table from the same estimates exported to CSV.
def format_results_table_latex(
    estimates: pd.DataFrame,
    first_stage_statistic: float,
) -> str:
    indexed_results = estimates.set_index(["outcome", "model"])
    duration_2sls = indexed_results.loc[("log1p_duration", "2SLS")]
    followups_2sls = indexed_results.loc[("log1p_followups", "2SLS")]
    duration_ar = indexed_results.loc[("log1p_duration", "Anderson-Rubin")]
    followups_ar = indexed_results.loc[("log1p_followups", "Anderson-Rubin")]
    duration_confidence_set = format_anderson_rubin_confidence_set(duration_ar)
    followups_confidence_set = format_anderson_rubin_confidence_set(followups_ar)

    return "\n".join(
        [
            r"\begin{table}[H]",
            r"\centering",
            r"\caption{IV estimates of the effect of log views.}",
            r"\label{tab:t3main}",
            r"\begin{tabular}{lcc}",
            r"\toprule",
            r" & $\log(1+\text{duration})$ & $\log(1+\text{follow-ups})$ \\",
            r"\midrule",
            (
                f"2SLS & ${duration_2sls['coefficient']:.3f}$ "
                f"({duration_2sls['standard_error']:.3f}) & "
                f"${followups_2sls['coefficient']:.3f}$ "
                f"({followups_2sls['standard_error']:.3f}) \\\\"
            ),
            (
                f"AR $p$-value at $\\beta=0$ & "
                f"{duration_ar['p_value']:.2f} & "
                f"{followups_ar['p_value']:.2f} \\\\"
            ),
            (
                "AR 95\\% confidence set & "
                f"{duration_confidence_set} & {followups_confidence_set} \\\\"
            ),
            (
                "First-stage statistic & "
                f"\\multicolumn{{2}}{{c}}{{{first_stage_statistic:.2f}}} \\\\"
            ),
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )


# Display unbounded identification honestly while retaining finite intervals when available.
def format_anderson_rubin_confidence_set(result: pd.Series) -> str:
    if bool(result["confidence_set_unbounded"]):
        return "unbounded"
    return (
        f"[{result['confidence_interval_low']:.2f}, "
        f"{result['confidence_interval_high']:.2f}]"
    )


# Format the AR result as a row with CI bounds, since it has no point estimate/SE.
def anderson_rubin_result_row(
    outcome: str,
    result: dict[str, float | bool],
    data: pd.DataFrame,
) -> dict[str, object]:
    return {
        "outcome": outcome,
        "model": "Anderson-Rubin",
        "coefficient": float("nan"),
        "standard_error": float("nan"),
        "p_value": result["p_value_at_zero"],
        "confidence_interval_low": result["lower_bound"],
        "confidence_interval_high": result["upper_bound"],
        "confidence_set_unbounded": result["unbounded"],
        "observations": len(data),
        "date_clusters": data["date"].nunique(),
    }


# Extract the instrumented-views coefficient into a tidy result row.
def model_result_row(
    outcome: str,
    model: str,
    coefficients: pd.Series,
    standard_errors: pd.Series,
    p_values: pd.Series,
    data: pd.DataFrame,
) -> dict[str, object]:
    return {
        "outcome": outcome,
        "model": model,
        "coefficient": float(coefficients[ENDOGENOUS_REGRESSOR]),
        "standard_error": float(standard_errors[ENDOGENOUS_REGRESSOR]),
        "p_value": float(p_values[ENDOGENOUS_REGRESSOR]),
        "confidence_interval_low": float("nan"),
        "confidence_interval_high": float("nan"),
        "confidence_set_unbounded": float("nan"),
        "observations": len(data),
        "date_clusters": data["date"].nunique(),
    }


if __name__ == "__main__":
    main()
