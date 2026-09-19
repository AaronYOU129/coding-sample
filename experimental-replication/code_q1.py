"""
Purpose:    Replicate Tables 1 (Balance) and 3 (Main Regressions) from
            Cappelen, List, Samek & Tungodden (2020),
            "The Effect of Early-Childhood Education on Social Preferences,"
            Journal of Political Economy 128(7): 2739-2758.

            Methodological choices (explained where non-obvious):
              - Main sample = rows with in_experiment == 1 (N = 302 total;
                Table 3's per-column N = 301/301/298/301/297 arises naturally
                from item-level non-response on each outcome, per paper
                footnote to Table 3).
              - Inequality outcomes use the five author-constructed variables
                inequalitydictator / inequalityefficiency / inequalitylucky /
                inequalitymerit / inequalitymeritlucky. Their non-missing
                counts in N=302 already equal the paper's reported N per
                regression, confirming the author used exactly these.
                Efficiency inequality is (6-1)/(6+1) = 5/7 when the child
                picks (1,6); 0 when (2,2) — the Gini-for-two definition the
                paper describes in Section III.
              - Table 3 always includes time-of-day fixed effects and
                experimenter fixed effects. Time-of-day uses the seven
                author-provided hour dummies time_dummy1..7 (not the
                continuous time_hours2) — this discrete specification is
                what reproduces the paper's R^2 to three decimals across
                all five outcomes. Experimenter FE uses expdummy1..26 plus
                an omitted baseline experimenter. Standard errors are
                classical OLS; robust/cluster variants are the appendix
                Table A4 robustness check.
              - Table 1 F-test is a joint significance test of the two
                treatment dummies in a linear regression of each covariate
                on preschool + parent_academy.

Inputs:     data.csv           Experiment data (823 rows, 119 cols)
            data_codebook.csv  Variable dictionary (for reference, not loaded)

Outputs:    tables/table1.tex  Balance table (booktabs LaTeX fragment)
            tables/table3.tex  Main regressions (booktabs LaTeX fragment)

Key Steps:
    Data        -> load raw CSV, filter to experimental sample (in_experiment==1)
    Processing  -> attach treatment dummies, identify covariate and FE columns
    Estimation  -> Table 1: group means + SEs + joint F-tests
                   Table 3: 10 OLS regressions with FE, classical SEs,
                            Wald and joint F-tests on the bottom rows
    Output      -> emit LaTeX fragments, print a diff vs. paper targets

How to Run: python3 code_q1.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data.csv"
OUTPUT_DIR = PROJECT_DIR / "tables"

TREATMENT_LEVELS = ["Control", "PA", "PK"]
TREATMENT_LABELS = {"Control": "Control", "PA": "Parent Academy", "PK": "Preschool"}

BALANCE_VARS = [
    ("age_at_test", "Age"),
    ("female", "Female"),
    ("black", "Black"),
    ("hispanic", "Hispanic"),
    ("white", "White"),
    ("time_hours2", "Time of experiment"),
]

OUTCOME_COLS = [
    ("inequalitydictator", "Dictator"),
    ("inequalityefficiency", "Efficiency"),
    ("inequalitylucky", "Luck"),
    ("inequalitymerit", "Merit"),
    ("inequalitymeritlucky", "Merit and Luck"),
]

DEMOGRAPHIC_CONTROLS = ["age_at_test", "female", "black", "hispanic"]
TIME_OF_DAY_DUMMIES = [f"time_dummy{i}" for i in range(1, 8)]
EXPERIMENTER_DUMMIES = [f"expdummy{i}" for i in range(1, 27)]


# ---------------------------------------------------------------------------
# High-level workflow
# ---------------------------------------------------------------------------

def main() -> None:
    sample = load_main_sample(DATA_PATH)
    print(f"[data] main sample N = {len(sample)}  "
          f"(treat: {sample['treat'].value_counts().to_dict()})")

    table1 = build_balance_table(sample, BALANCE_VARS)
    write_table1_latex(table1, OUTPUT_DIR / "table1.tex")

    table3 = build_main_regressions(sample, OUTCOME_COLS, DEMOGRAPHIC_CONTROLS)
    write_table3_latex(table3, OUTPUT_DIR / "table3.tex")

    print_validation_report(table1, table3)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_main_sample(path: Path) -> pd.DataFrame:
    """Return rows flagged as in the social-preference experiment with
    treatment dummies attached."""
    raw = pd.read_csv(path)
    sample = raw.loc[raw["in_experiment"] == 1].copy()
    sample["preschool"] = (sample["treat"] == "PK").astype(int)
    sample["parent_academy"] = (sample["treat"] == "PA").astype(int)
    return sample


# ---------------------------------------------------------------------------
# Table 1: balance across treatment arms
# ---------------------------------------------------------------------------

def build_balance_table(sample: pd.DataFrame,
                        variables: list[tuple[str, str]]) -> pd.DataFrame:
    """Return a long-form balance table with one row per (variable, arm).

    Columns: mean, sem, n for each arm plus 'Total', and a joint F-test
    p-value attached at the variable level.
    """
    rows = []
    for var, label in variables:
        for arm in TREATMENT_LEVELS:
            subset = sample.loc[sample["treat"] == arm, var].dropna()
            rows.append(_cell(label, arm, subset))
        total = sample[var].dropna()
        rows.append(_cell(label, "Total", total))
        rows.append({"variable": label, "arm": "F_pvalue",
                     "mean": joint_f_pvalue(sample, var), "sem": np.nan, "n": len(total)})
    return pd.DataFrame(rows)


def _cell(label: str, arm: str, values: pd.Series) -> dict:
    return {
        "variable": label,
        "arm": arm,
        "mean": values.mean(),
        "sem": values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else np.nan,
        "n": len(values),
    }


def joint_f_pvalue(sample: pd.DataFrame, outcome: str) -> float:
    """p-value from OLS: outcome ~ preschool + parent_academy (joint F)."""
    data = sample[[outcome, "preschool", "parent_academy"]].dropna()
    X = sm.add_constant(data[["preschool", "parent_academy"]])
    fit = sm.OLS(data[outcome], X).fit()
    return float(fit.f_test("preschool = 0, parent_academy = 0").pvalue)


# ---------------------------------------------------------------------------
# Table 3: 5 outcomes x 2 specs, with experimenter FE and time-of-day
# ---------------------------------------------------------------------------

def build_main_regressions(sample: pd.DataFrame,
                           outcomes: list[tuple[str, str]],
                           controls: list[str]) -> list[dict]:
    """Run 10 regressions (5 outcomes x {no controls, with controls}).

    Each result dict carries: coef/se tables for the variables we want to
    print, N, R-squared, and two auxiliary tests needed for the footer.
    """
    results = []
    for outcome_col, outcome_label in outcomes:
        for with_controls in (False, True):
            extra = controls if with_controls else []
            results.append(
                run_regression(sample, outcome_col, outcome_label, extra_controls=extra)
            )
    return results


def run_regression(sample: pd.DataFrame,
                   outcome: str,
                   outcome_label: str,
                   extra_controls: list[str]) -> dict:
    """Fit one column of Table 3.

    The always-included FE (time of day + 26 experimenter dummies) plus
    extra demographic controls when requested. Drops experimenter dummies
    that are identically zero in the current subsample to avoid statsmodels
    warnings (the omitted category absorbs them).
    """
    treatment_vars = ["preschool", "parent_academy"]
    fe_cols = ([c for c in TIME_OF_DAY_DUMMIES if sample[c].sum() > 0]
               + [c for c in EXPERIMENTER_DUMMIES if sample[c].sum() > 0])
    regressors = treatment_vars + extra_controls + fe_cols
    data = sample[[outcome] + regressors].dropna()

    X = sm.add_constant(data[regressors])
    fit = sm.OLS(data[outcome], X).fit()

    return {
        "outcome_label": outcome_label,
        "with_controls": bool(extra_controls),
        "fit": fit,
        "n": int(fit.nobs),
        "r2": float(fit.rsquared),
        "p_ps_eq_pa": float(fit.f_test("preschool = parent_academy").pvalue),
        "p_joint_zero": float(fit.f_test("preschool = 0, parent_academy = 0").pvalue),
    }


# ---------------------------------------------------------------------------
# LaTeX output
# ---------------------------------------------------------------------------

def write_table1_latex(table1: pd.DataFrame, path: Path) -> None:
    """Emit a booktabs-style balance table to `path`."""
    header = (
        r"\begin{tabular}{lccccc}" "\n"
        r"\toprule" "\n"
        r" & Control & Parent Academy & Preschool & Total & $F$-test \\" "\n"
        r"\midrule" "\n"
    )
    body_rows = []
    for variable in [lab for _, lab in BALANCE_VARS]:
        var_rows = table1[table1["variable"] == variable]
        means = var_rows.set_index("arm")["mean"]
        sems = var_rows.set_index("arm")["sem"]
        body_rows.append(
            f"{variable} & "
            f"{_fmt(means['Control'])} & {_fmt(means['PA'])} & "
            f"{_fmt(means['PK'])} & {_fmt(means['Total'])} & "
            f"{_fmt_p(means['F_pvalue'])} \\\\"
        )
        body_rows.append(
            f" & ({_fmt_se(sems['Control'])}) & ({_fmt_se(sems['PA'])}) & "
            f"({_fmt_se(sems['PK'])}) & ({_fmt_se(sems['Total'])}) & \\\\"
        )
    footer = r"\bottomrule" "\n" r"\end{tabular}" "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + "\n".join(body_rows) + "\n" + footer)


def write_table3_latex(results: list[dict], path: Path) -> None:
    """Emit Table 3 as a wide booktabs LaTeX fragment."""
    n_cols = len(results)
    col_spec = "l" + "c" * n_cols

    header = (
        r"\begin{tabular}{" + col_spec + "}\n"
        r"\toprule" "\n"
        + _table3_top_header(results) + "\n"
        r"\midrule" "\n"
    )

    body_lines = []
    for var_name, var_label in [("preschool", "Preschool"),
                                ("parent_academy", "Parent Academy"),
                                ("age_at_test", "Age"),
                                ("female", "Female"),
                                ("black", "Black"),
                                ("hispanic", "Hispanic")]:
        body_lines.append(_coefficient_row(results, var_name, var_label))
        body_lines.append(_se_row(results, var_name))

    body_lines.append(_constant_row(results))
    body_lines.append(_se_row(results, "const"))

    body_lines.append(r"\midrule")
    body_lines.append("Observations & " + " & ".join(f"{r['n']}" for r in results) + r" \\")
    body_lines.append("$R^2$ & " + " & ".join(f"{r['r2']:.3f}" for r in results) + r" \\")
    body_lines.append(r"$p$-value (PS $=$ PA) & "
                      + " & ".join(f"{r['p_ps_eq_pa']:.3f}" for r in results) + r" \\")
    body_lines.append(r"$p$-value (PS $=$ PA $= 0$) & "
                      + " & ".join(f"{r['p_joint_zero']:.3f}" for r in results) + r" \\")

    footer = r"\bottomrule" "\n" r"\end{tabular}" "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + "\n".join(body_lines) + "\n" + footer)


def _table3_top_header(results: list[dict]) -> str:
    """Two-level header: outcome group spans two columns each."""
    groups = []
    current_label = None
    span = 0
    for r in results:
        if r["outcome_label"] != current_label:
            if current_label is not None:
                groups.append((current_label, span))
            current_label = r["outcome_label"]
            span = 1
        else:
            span += 1
    groups.append((current_label, span))

    top_cells = [r"\multicolumn{" + str(span) + r"}{c}{" + label + "}"
                 for label, span in groups]
    col_numbers = [f"({i + 1})" for i in range(len(results))]
    return (" & " + " & ".join(top_cells) + r" \\" + "\n"
            " & " + " & ".join(col_numbers) + r" \\")


def _coefficient_row(results: list[dict], var: str, label: str) -> str:
    cells = [_coef_cell(r, var) for r in results]
    return f"{label} & " + " & ".join(cells) + r" \\"


def _se_row(results: list[dict], var: str) -> str:
    cells = [_se_cell(r, var) for r in results]
    return r" & " + " & ".join(cells) + r" \\"


def _constant_row(results: list[dict]) -> str:
    cells = [_coef_cell(r, "const") for r in results]
    return "Constant & " + " & ".join(cells) + r" \\"


def _coef_cell(result: dict, var: str) -> str:
    fit = result["fit"]
    if var not in fit.params.index:
        return ""
    coef = fit.params[var]
    pval = fit.pvalues[var]
    return f"{coef:.2f}{_stars(pval)}"


def _se_cell(result: dict, var: str) -> str:
    fit = result["fit"]
    if var not in fit.bse.index:
        return ""
    return f"({fit.bse[var]:.2f})"


def _stars(p: float) -> str:
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

def _fmt(x: float) -> str:
    return "" if pd.isna(x) else f"{x:.3f}"


def _fmt_se(x: float) -> str:
    return "" if pd.isna(x) else f"{x:.4f}"


def _fmt_p(x: float) -> str:
    return "" if pd.isna(x) else f"{x:.3f}"


# ---------------------------------------------------------------------------
# Validation: print a diff-style comparison vs. paper targets
# ---------------------------------------------------------------------------

PAPER_TABLE1 = {
    # variable: (control, pa, pk, total, F-p)
    "Age":                (7.569, 7.582, 7.663, 7.602, 0.645),
    "Female":             (0.441, 0.481, 0.539, 0.484, 0.536),
    "Black":              (0.151, 0.182, 0.224, 0.183, 0.616),
    "Hispanic":           (0.785, 0.766, 0.697, 0.752, 0.391),
    "White":              (0.0645, 0.0519, 0.0526, 0.0569, 0.987),
    "Time of experiment": (9.828, 10.18, 9.829, 9.939, 0.677),
}

PAPER_TABLE3_COEFS = {  # per-column (preschool_coef, pa_coef)
    "col1":  (0.01, 0.03),
    "col2":  (0.01, 0.03),
    "col3":  (0.03, 0.12),
    "col4":  (0.02, 0.11),
    "col5":  (-0.11, -0.05),
    "col6":  (-0.10, -0.04),
    "col7":  (-0.06, 0.02),
    "col8":  (-0.06, 0.02),
    "col9":  (-0.09, -0.02),
    "col10": (-0.08, -0.01),
}
PAPER_TABLE3_N = [301, 301, 301, 301, 298, 298, 301, 301, 297, 297]


def print_validation_report(table1: pd.DataFrame, table3: list[dict]) -> None:
    print()
    print("=" * 78)
    print("Table 1: computed vs paper  (difference in parens)")
    print("=" * 78)
    header = f"{'Variable':<22}{'Control':>16}{'PA':>16}{'PK':>16}{'Total':>16}{'F_p':>8}"
    print(header)
    for variable in [lab for _, lab in BALANCE_VARS]:
        rows = table1[table1["variable"] == variable]
        means = rows.set_index("arm")["mean"]
        target = PAPER_TABLE1[variable]
        print(f"{variable:<22}"
              f"{_compare(means['Control'], target[0]):>16}"
              f"{_compare(means['PA'], target[1]):>16}"
              f"{_compare(means['PK'], target[2]):>16}"
              f"{_compare(means['Total'], target[3]):>16}"
              f"{_compare(means['F_pvalue'], target[4], decimals=3):>8}")

    print()
    print("=" * 78)
    print("Table 3: computed vs paper  (preschool coef, parent_academy coef, N)")
    print("=" * 78)
    for idx, res in enumerate(table3, start=1):
        ps_hat = res["fit"].params["preschool"]
        pa_hat = res["fit"].params["parent_academy"]
        ps_tgt, pa_tgt = PAPER_TABLE3_COEFS[f"col{idx}"]
        n_tgt = PAPER_TABLE3_N[idx - 1]
        print(f"col{idx:<2} {res['outcome_label']:<16} "
              f"PS={ps_hat:+.3f} (target {ps_tgt:+.2f}, diff {ps_hat - ps_tgt:+.3f})  "
              f"PA={pa_hat:+.3f} (target {pa_tgt:+.2f}, diff {pa_hat - pa_tgt:+.3f})  "
              f"N={res['n']} (target {n_tgt}, diff {res['n'] - n_tgt:+d})")


def _compare(value: float, target: float, decimals: int = 3) -> str:
    if pd.isna(value):
        return "n/a"
    diff = value - target
    return f"{value:.{decimals}f} ({diff:+.{decimals}f})"


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
