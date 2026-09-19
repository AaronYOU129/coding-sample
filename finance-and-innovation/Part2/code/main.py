"""
Purpose:    Complete empirical analysis for "Does the Market Correctly Value Novel Innovation?"
            Executes the full pipeline: data merging, uncertainty construction, and three OLS
            regressions (main hypothesis, mechanism interaction, first-stage cell-level test).
            Standard errors are clustered at the CPC subclass level throughout.

Inputs:     data/KPSS_2024.csv
            data/PatentSimilarityImportanceBreakthrough_forPost2022.csv
            data/Match_patent_cpc_2024.csv
            data/g_patent.tsv

Outputs:    data/merged_patent_data.csv        — Merged patent-level dataset
            output/regression_table.csv         — Main + mechanism regression table (CSV)
            output/regression_table.tex         — Main + mechanism regression table (LaTeX)

Key Steps:  1. Merge four datasets on patent_num for 2021-2022 utility patents
            2. Sanity-check merge (no duplicates, plausible sample size)
            3. Construct novelty (negated z-score of bsim5) and controls
            4. Sanity-check variables (standardization, no inf/NaN)
            5. Construct tech uncertainty (CV of xi_real by CPC subclass × year)
            6. Run main regression, mechanism test, and first-stage regression
            7. Sanity-check regression outputs (sample size, R², sign, magnitude)
            8. Export publication-style regression tables

How to Run: python code/main.py
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import os
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
DATA_DIR = "data"
OUTPUT_DIR = "output"
SAMPLE_YEARS = ["2021", "2022"]
MIN_GROUP_SIZE = 10

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ===========================================================================
# STEP 1: DATA MERGE
# ===========================================================================

def load_kpss_patent_values():
    """Load KPSS patent-level market value estimates, filter to sample period."""
    kpss_values = pd.read_csv(f"{DATA_DIR}/KPSS_2024.csv")
    kpss_values["issue_date_str"] = kpss_values["issue_date"].astype(str)
    kpss_values = kpss_values[
        kpss_values["issue_date_str"].str[:4].isin(SAMPLE_YEARS)
    ].copy()
    print(f"  KPSS 2021-2022: {len(kpss_values):,}")
    return kpss_values


def load_novelty_scores():
    """Load Kelly et al. backward similarity scores, filter to sample period."""
    novelty_scores = pd.read_csv(
        f"{DATA_DIR}/PatentSimilarityImportanceBreakthrough_forPost2022.csv"
    )
    novelty_scores = novelty_scores[
        novelty_scores["issue_year"].isin([2021, 2022])
    ][["patent_num", "bsim5", "fcitALL"]].copy()
    print(f"  PatentSimilarity 2021-2022: {len(novelty_scores):,}")
    return novelty_scores


def load_cpc_classifications():
    """Load CPC mapping, extract primary code / section / subclass."""
    classifications = pd.read_csv(f"{DATA_DIR}/Match_patent_cpc_2024.csv")
    classifications["primary_cpc"] = classifications["cpc"].str.split(";").str[0]
    classifications["cpc_section"] = classifications["primary_cpc"].str[0]
    classifications["cpc_subclass"] = classifications["primary_cpc"].str[:4]
    print(f"  CPC mapping: {len(classifications):,}")
    return classifications[["patent_num", "cpc_section", "cpc_subclass"]]


def load_patent_bibliographic():
    """Load PatentsView bibliographic data, filter to 2021-2022 utility patents."""
    bibliographic = pd.read_csv(f"{DATA_DIR}/g_patent.tsv", sep="\t", low_memory=False)
    bibliographic["patent_num"] = pd.to_numeric(bibliographic["patent_id"], errors="coerce")
    bibliographic = bibliographic[
        (bibliographic["patent_date"].str[:4].isin(SAMPLE_YEARS))
        & (bibliographic["patent_type"] == "utility")
    ].copy()
    bibliographic = bibliographic.dropna(subset=["patent_num"])
    bibliographic["patent_num"] = bibliographic["patent_num"].astype(int)
    bibliographic["grant_year"] = bibliographic["patent_date"].str[:4].astype(int)
    print(f"  g_patent 2021-2022 utility: {len(bibliographic):,}")
    return bibliographic[["patent_num", "num_claims", "grant_year"]]


def merge_datasets():
    print("=" * 60)
    print("STEP 1: Data Merge")
    print("=" * 60)

    kpss_values = load_kpss_patent_values()
    novelty_scores = load_novelty_scores()
    classifications = load_cpc_classifications()
    bibliographic = load_patent_bibliographic()

    # Sequential inner join on patent_num
    merged_patents = kpss_values.merge(novelty_scores, on="patent_num", how="inner")
    print(f"\n  + Similarity: {len(merged_patents):,} "
          f"({len(merged_patents)/len(kpss_values)*100:.1f}% of KPSS)")

    merged_patents = merged_patents.merge(classifications, on="patent_num", how="inner")
    print(f"  + CPC:        {len(merged_patents):,}")

    merged_patents = merged_patents.merge(bibliographic, on="patent_num", how="inner")
    print(f"  + g_patent:   {len(merged_patents):,}")

    # Merge sanity checks: catch duplicate keys or unexpected attrition
    duplicate_count = merged_patents["patent_num"].duplicated().sum()
    assert duplicate_count == 0, f"Merge produced {duplicate_count} duplicate patents"
    assert len(merged_patents) > 100_000, (
        f"Unexpectedly small sample after merge: {len(merged_patents):,}"
    )
    print(f"\n  [CHECK] No duplicate patent_num; sample size plausible")

    # Construct analysis variables
    merged_patents["novelty"] = -(
        (merged_patents["bsim5"] - merged_patents["bsim5"].mean())
        / merged_patents["bsim5"].std()
    )
    merged_patents["ln_xi_real"] = np.log(merged_patents["xi_real"])
    merged_patents["ln_backward_cites"] = np.log(merged_patents["cites"] + 1)

    # Variable construction sanity checks
    assert abs(merged_patents["novelty"].mean()) < 0.01, "Novelty mean not ≈ 0 after standardization"
    assert abs(merged_patents["novelty"].std() - 1.0) < 0.01, "Novelty std not ≈ 1 after standardization"
    assert np.isfinite(merged_patents["ln_xi_real"]).all(), "ln_xi_real contains inf or NaN (xi_real <= 0?)"
    assert merged_patents["ln_backward_cites"].isna().sum() == 0, "ln_backward_cites has NaN"
    print(f"  [CHECK] Novelty mean={merged_patents['novelty'].mean():.4f}, "
          f"std={merged_patents['novelty'].std():.4f}; no inf/NaN in log variables")

    print(f"\n  Final sample: {len(merged_patents):,} patents")
    return merged_patents


# ===========================================================================
# STEP 2: TECHNOLOGY UNCERTAINTY
# ===========================================================================

def add_tech_uncertainty(merged_patents):
    print("\n" + "=" * 60)
    print("STEP 2: Technology Uncertainty (CV of xi_real by CPC subclass x year)")
    print("=" * 60)

    uncertainty_cell_stats = merged_patents.groupby(
        ["cpc_subclass", "grant_year"]
    )["xi_real"].agg(
        mean_value="mean", std_value="std", patent_count="count"
    ).reset_index()

    # CV only for groups with enough patents to produce stable estimates
    uncertainty_cell_stats["tech_uncertainty"] = np.where(
        uncertainty_cell_stats["patent_count"] >= MIN_GROUP_SIZE,
        uncertainty_cell_stats["std_value"] / uncertainty_cell_stats["mean_value"],
        np.nan
    )
    # Save cell-level diagnostic data
    uncertainty_cell_stats.to_csv(
        f"{DATA_DIR}/uncertainty_cell_stats.csv",
        index=False, encoding="utf-8"
    )
    print(f"  Saved: {DATA_DIR}/uncertainty_cell_stats.csv")

    valid_group_count = (uncertainty_cell_stats["patent_count"] >= MIN_GROUP_SIZE).sum()
    print(f"  Valid groups (>={MIN_GROUP_SIZE} patents): "
          f"{valid_group_count} / {len(uncertainty_cell_stats)}")

    merged_patents = merged_patents.merge(
        uncertainty_cell_stats[["cpc_subclass", "grant_year", "tech_uncertainty"]],
        on=["cpc_subclass", "grant_year"],
        how="left"
    )
    merged_patents["novelty_x_uncertainty"] = (
        merged_patents["novelty"] * merged_patents["tech_uncertainty"]
    )

    patent_count_with_uncertainty = merged_patents["tech_uncertainty"].notna().sum()
    print(f"  Patents with uncertainty: {patent_count_with_uncertainty:,} "
          f"({patent_count_with_uncertainty/len(merged_patents)*100:.1f}%)")
    return merged_patents


# ===========================================================================
# STEP 3: REGRESSIONS
# ===========================================================================

def significance_stars(pvalue):
    if pvalue < 0.01:
        return "***"
    elif pvalue < 0.05:
        return "**"
    elif pvalue < 0.10:
        return "*"
    return ""


def build_regression_sample(merged_patents):
    """Drop patents missing tech_uncertainty and prepare cluster IDs."""
    regression_sample = merged_patents.dropna(subset=["tech_uncertainty"]).copy()
    cluster_ids = pd.Categorical(regression_sample["cpc_subclass"]).codes
    return regression_sample, cluster_ids


def build_fixed_effect_dummies(regression_sample):
    """Create year and CPC section fixed-effect dummies."""
    year_fixed_effects = pd.get_dummies(
        regression_sample["grant_year"].astype(str), prefix="yr", drop_first=True, dtype=float
    )
    section_fixed_effects = pd.get_dummies(
        regression_sample["cpc_section"], prefix="sec", drop_first=True, dtype=float
    )
    return year_fixed_effects, section_fixed_effects


def run_main_regression(regression_sample, log_market_value, year_fixed_effects,
                        section_fixed_effects, cluster_ids):
    """Test whether the market discounts novel patents."""
    main_design_matrix = sm.add_constant(pd.concat([
        regression_sample[["novelty", "num_claims", "ln_backward_cites"]],
        year_fixed_effects, section_fixed_effects
    ], axis=1))
    main_result = sm.OLS(log_market_value, main_design_matrix).fit(
        cov_type="cluster", cov_kwds={"groups": cluster_ids}
    )
    print(f"\n  Reg 1 (Main): novelty = {main_result.params['novelty']:.4f} "
          f"(p={main_result.pvalues['novelty']:.4f}), R²={main_result.rsquared:.4f}")
    return main_result


def run_mechanism_regression(regression_sample, log_market_value, year_fixed_effects,
                             section_fixed_effects, cluster_ids):
    """Test whether uncertainty amplifies the novelty discount."""
    mechanism_design_matrix = sm.add_constant(pd.concat([
        regression_sample[["novelty", "tech_uncertainty", "novelty_x_uncertainty",
                           "num_claims", "ln_backward_cites"]],
        year_fixed_effects, section_fixed_effects
    ], axis=1))
    mechanism_result = sm.OLS(log_market_value, mechanism_design_matrix).fit(
        cov_type="cluster", cov_kwds={"groups": cluster_ids}
    )
    print(f"  Reg 2 (Mechanism): uncertainty = "
          f"{mechanism_result.params['tech_uncertainty']:.4f}"
          f"{significance_stars(mechanism_result.pvalues['tech_uncertainty'])}, "
          f"interaction = {mechanism_result.params['novelty_x_uncertainty']:.4f} "
          f"(p={mechanism_result.pvalues['novelty_x_uncertainty']:.4f}), "
          f"R²={mechanism_result.rsquared:.4f}")
    return mechanism_result


def run_first_stage_regression(regression_sample):
    """Test whether more-novel technology areas have higher uncertainty."""
    subclass_year_cells = regression_sample.groupby(
        ["cpc_subclass", "grant_year"]
    ).agg(
        mean_novelty=("novelty", "mean"),
        tech_uncertainty=("tech_uncertainty", "first"),
    ).reset_index()

    first_stage_year_fe = pd.get_dummies(
        subclass_year_cells["grant_year"].astype(str), prefix="yr", drop_first=True, dtype=float
    )
    first_stage_design_matrix = sm.add_constant(
        pd.concat([subclass_year_cells[["mean_novelty"]], first_stage_year_fe], axis=1)
    )
    first_stage_result = sm.OLS(
        subclass_year_cells["tech_uncertainty"], first_stage_design_matrix
    ).fit(cov_type="HC1")

    first_stage_coef = first_stage_result.params["mean_novelty"]
    first_stage_se = first_stage_result.bse["mean_novelty"]
    first_stage_pvalue = first_stage_result.pvalues["mean_novelty"]
    print(f"\n  --- First Stage (CPC subclass x year level) ---")
    print(f"  Uncertainty_ct = a * mean_novelty_ct + year FE + e")
    print(f"  mean_novelty:  {first_stage_coef:.4f}"
          f"{significance_stars(first_stage_pvalue)} "
          f"(SE={first_stage_se:.4f}, p={first_stage_pvalue:.4f})")
    print(f"  R²={first_stage_result.rsquared:.4f}, N={int(first_stage_result.nobs)}")
    return first_stage_result


def sanity_check_regressions(main_result, mechanism_result, regression_sample):
    """
    Verify regression outputs before exporting.

    Catches common issues: dropped observations from collinearity or NaN leaking
    through, negative or >1 R² from misspecification, wrong signs on well-established
    controls (would indicate a merge or variable construction error), and implausibly
    large coefficients that suggest a scaling mistake in the novelty measure.
    """
    print(f"\n  --- Sanity Checks ---")
    checks_passed = 0
    checks_total = 0

    # Both models should use the full regression sample — any mismatch means
    # observations were silently dropped (e.g., NaN in a regressor)
    checks_total += 1
    expected_n = len(regression_sample)
    if int(main_result.nobs) == expected_n and int(mechanism_result.nobs) == expected_n:
        checks_passed += 1
        print(f"  [PASS] Sample size: both models use {expected_n:,} obs")
    else:
        print(f"  [FAIL] Sample size mismatch: expected {expected_n:,}, "
              f"got {int(main_result.nobs):,} / {int(mechanism_result.nobs):,}")

    # R² should be in (0, 1); values outside suggest a coding error
    checks_total += 1
    if 0 < main_result.rsquared < 1 and 0 < mechanism_result.rsquared < 1:
        checks_passed += 1
        print(f"  [PASS] R² in valid range: {main_result.rsquared:.4f}, {mechanism_result.rsquared:.4f}")
    else:
        print(f"  [FAIL] R² out of range")

    # Mechanism model nests the main model, so its R² must be weakly higher
    checks_total += 1
    if mechanism_result.rsquared >= main_result.rsquared:
        checks_passed += 1
        print(f"  [PASS] Mechanism R² >= Main R² (adding variables improves fit)")
    else:
        print(f"  [WARN] Mechanism R² < Main R² — unexpected")

    # More claims and more citations should predict higher patent value;
    # wrong signs would indicate a variable construction or merge error
    checks_total += 1
    claims_positive = main_result.params["num_claims"] > 0
    cites_positive = main_result.params["ln_backward_cites"] > 0
    if claims_positive and cites_positive:
        checks_passed += 1
        print(f"  [PASS] Controls have expected signs (claims +, cites +)")
    else:
        print(f"  [WARN] Unexpected control signs: claims={main_result.params['num_claims']:.4f}, "
              f"cites={main_result.params['ln_backward_cites']:.4f}")

    # Novelty is standardized (mean 0, sd 1), so a coefficient above 10
    # would imply a 1-SD novelty change moves log value by 10 — likely a scaling bug
    checks_total += 1
    max_coef = max(abs(main_result.params["novelty"]),
                   abs(mechanism_result.params["novelty"]))
    if max_coef < 10:
        checks_passed += 1
        print(f"  [PASS] Novelty coefficients in plausible range (max |coef| = {max_coef:.4f})")
    else:
        print(f"  [WARN] Novelty coefficient implausibly large: {max_coef:.4f}")

    print(f"  Result: {checks_passed}/{checks_total} checks passed")


def run_regressions(merged_patents):
    """Orchestrate all three regressions, sanity checks, and table export."""
    print("\n" + "=" * 60)
    print("STEP 3: Regressions")
    print("=" * 60)

    regression_sample, cluster_ids = build_regression_sample(merged_patents)
    print(f"  Regression sample: {len(regression_sample):,}")

    regression_sample.to_csv(
        f"{DATA_DIR}/regression_sample.csv",
        index=False
    )
    print(f"  Saved: {DATA_DIR}/regression_sample.csv")

    year_fixed_effects, section_fixed_effects = build_fixed_effect_dummies(regression_sample)
    print(f"  Clusters (CPC subclass): {regression_sample['cpc_subclass'].nunique()}")

    log_market_value = regression_sample["ln_xi_real"]

    main_result = run_main_regression(
        regression_sample, log_market_value,
        year_fixed_effects, section_fixed_effects, cluster_ids
    )
    mechanism_result = run_mechanism_regression(
        regression_sample, log_market_value,
        year_fixed_effects, section_fixed_effects, cluster_ids
    )
    first_stage_result = run_first_stage_regression(regression_sample)

    sanity_check_regressions(main_result, mechanism_result, regression_sample)
    export_regression_table(main_result, mechanism_result)

    return main_result, mechanism_result, first_stage_result


# ===========================================================================
# TABLE EXPORT
# ===========================================================================

def export_regression_table(main_result, mechanism_result):
    """Build and save publication-style regression table as CSV and LaTeX."""
    display_vars = [
        ("novelty", "Novelty"),
        ("tech_uncertainty", "Tech Uncertainty (CV)"),
        ("novelty_x_uncertainty", "Novelty x Uncertainty"),
        ("num_claims", "Num Claims"),
        ("ln_backward_cites", "ln(Backward Cites + 1)"),
        ("const", "Constant"),
    ]

    rows = []
    for column_name, display_name in display_vars:
        coef_row = [display_name]
        se_row = [""]
        for model in [main_result, mechanism_result]:
            if column_name in model.params.index:
                coef = model.params[column_name]
                std_err = model.bse[column_name]
                pvalue = model.pvalues[column_name]
                coef_row.append(f"{coef:.4f}{significance_stars(pvalue)}")
                se_row.append(f"({std_err:.4f})")
            else:
                coef_row.append("")
                se_row.append("")
        rows.append(coef_row)
        rows.append(se_row)

    rows.append(["R-squared"] + [f"{m.rsquared:.4f}" for m in [main_result, mechanism_result]])
    rows.append(["Adj R-squared"] + [f"{m.rsquared_adj:.4f}" for m in [main_result, mechanism_result]])
    rows.append(["N"] + [f"{int(m.nobs):,}" for m in [main_result, mechanism_result]])
    rows.append(["Year FE", "Yes", "Yes"])
    rows.append(["CPC Section FE", "Yes", "Yes"])
    rows.append(["Cluster SE", "CPC subclass", "CPC subclass"])

    table = pd.DataFrame(rows, columns=["Variable", "(1) Main", "(2) Mechanism"])
    table.to_csv(f"{OUTPUT_DIR}/regression_table.csv", index=False)

    # LaTeX
    lines = [
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{Does the Market Discount Novel Patents?}",
        r"\label{tab:main_regression}", r"\begin{tabular}{lcc}",
        r"\hline\hline", r" & (1) Main & (2) Mechanism \\", r"\hline",
    ]
    summary_vars = {"R-squared", "Adj R-squared", "N", "Year FE", "CPC Section FE", "Cluster SE"}
    added_hline = False
    for _, row in table.iterrows():
        variable, col_main, col_mechanism = (
            row["Variable"], row["(1) Main"], row["(2) Mechanism"]
        )
        if variable in summary_vars and not added_hline:
            lines.append(r"\hline")
            added_hline = True
        lines.append(f"{variable} & {col_main} & {col_mechanism} \\\\")
    lines += [
        r"\hline\hline",
        r"\multicolumn{3}{l}{\footnotesize Standard errors clustered at CPC subclass level in parentheses.} \\",
        r"\multicolumn{3}{l}{\footnotesize * p$<$0.10, ** p$<$0.05, *** p$<$0.01} \\",
        r"\end{tabular}", r"\end{table}",
    ]
    with open(f"{OUTPUT_DIR}/regression_table.tex", "w") as f:
        f.write("\n".join(lines))

    print(f"\n  Saved: {OUTPUT_DIR}/regression_table.csv")
    print(f"  Saved: {OUTPUT_DIR}/regression_table.tex")


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    merged_patents = merge_datasets()
    merged_patents = add_tech_uncertainty(merged_patents)

    merged_patents.to_csv(f"{DATA_DIR}/merged_patent_data.csv", index=False)
    print(f"\n  Saved: {DATA_DIR}/merged_patent_data.csv")

    run_regressions(merged_patents)

    print("\n" + "=" * 60)
    print("All steps complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
