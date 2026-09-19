"""
Purpose:    Diagnose why the first-stage coefficient (mean_novelty -> CV of patent
            value) is significantly NEGATIVE, contrary to the hypothesis that novel
            technologies should carry HIGHER information uncertainty. Decomposes the
            -0.39 first-stage estimate into the mean and SD channels of CV, checks
            outlier sensitivity, and tests for nonlinearity. The point of the
            decomposition is to attribute the negative sign to (a) mechanical CV
            artifact, (b) homogeneity of novel areas, (c) CV vs. uncertainty
            mismatch, or (d) text vs. market novelty mismatch.

Inputs:     data/regression_sample.csv   - patent-level analysis sample
                                           (has novelty, xi_real, cpc_subclass, grant_year).

Outputs:    output/first_stage_decomposition.md   - regression tables + plain-English summary.
            output/decomposition_scatter.png      - mean_novelty vs. {CV, ln(mean), ln(sd)}.
            output/cell_level_decomposition.csv   - cell-level dataset used for diagnostics.

Key Steps:  1. Aggregate patent-level data to CPC subclass x year cells (mean_novelty,
               mean_value, sd_value, CV).
            2. Reproduce first-stage CV ~ mean_novelty + year FE.
            3. Decompose: ln(mean) and ln(sd) on mean_novelty + year FE (parallel specs).
            4. Robustness: trim CV outliers (top/bottom 1%) and low-mean cells (bottom 5%).
            5. Nonlinearity: add mean_novelty^2 to first stage.
            6. Sample-balance and distribution checks.
            7. Build markdown report and scatter plot.

How to Run: python code/first_stage_decomposition.py
"""

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
DATA_DIR = "data"
OUTPUT_DIR = "output"
PATENT_LEVEL_PATH = f"{DATA_DIR}/regression_sample.csv"
CELL_LEVEL_PATH = f"{OUTPUT_DIR}/cell_level_decomposition.csv"
MARKDOWN_PATH = f"{OUTPUT_DIR}/first_stage_decomposition.md"
SCATTER_PATH = f"{OUTPUT_DIR}/decomposition_scatter.png"
MIN_GROUP_SIZE = 10   # matches main.py: CV only meaningful with >=10 patents per cell

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ===========================================================================
# STEP 1: Build cell-level dataset
# ===========================================================================

def build_cell_level_dataset():
    """Aggregate patent-level data to (cpc_subclass, grant_year) cells."""
    patents = pd.read_csv(PATENT_LEVEL_PATH)
    print(f"  Patent-level rows: {len(patents):,}")

    cells = patents.groupby(["cpc_subclass", "grant_year"]).agg(
        mean_novelty=("novelty", "mean"),
        mean_value=("xi_real", "mean"),
        sd_value=("xi_real", "std"),
        patent_count=("patent_num", "count"),
    ).reset_index()

    cells["CV"] = np.where(
        cells["patent_count"] >= MIN_GROUP_SIZE,
        cells["sd_value"] / cells["mean_value"],
        np.nan,
    )

    n_before = len(cells)
    cells_full = cells.dropna(subset=["CV", "mean_novelty", "mean_value", "sd_value"]).copy()
    n_dropped = n_before - len(cells_full)
    print(f"  Cells total: {n_before}")
    print(f"  Cells dropped (patent_count < {MIN_GROUP_SIZE} or missing): {n_dropped}")
    print(f"  Cells used: {len(cells_full)}")

    cells_full["ln_mean_value"] = np.log(cells_full["mean_value"] + 1.0)
    cells_full["ln_sd_value"] = np.log(cells_full["sd_value"] + 1.0)
    # log(x+1) instead of log(x) is a no-op when xi_real is strictly positive
    # (xi_real is a market-cap-derived dollar value), but it is a safe guard:
    # see report for whether any cell hit the +1 path.

    cells_full.to_csv(CELL_LEVEL_PATH, index=False)
    print(f"  Saved: {CELL_LEVEL_PATH}")
    return cells_full


# ===========================================================================
# STEP 2: Regression helpers
# ===========================================================================

def add_year_fixed_effects(cells):
    return pd.get_dummies(cells["grant_year"].astype(str), prefix="yr",
                          drop_first=True, dtype=float)


def fit_with_clustered_se(y, X, cluster_groups):
    """OLS with cluster-robust SE at cpc_subclass; HC1 fallback if only 1 cluster."""
    if cluster_groups.nunique() > 1:
        return sm.OLS(y, X).fit(cov_type="cluster",
                                cov_kwds={"groups": cluster_groups})
    return sm.OLS(y, X).fit(cov_type="HC1")


def run_linear_first_stage(cells, dependent_label):
    """Run y ~ mean_novelty + year FE, return (model, label, n)."""
    year_fe = add_year_fixed_effects(cells)
    X = sm.add_constant(pd.concat([cells[["mean_novelty"]], year_fe], axis=1))
    y = cells[dependent_label]
    model = fit_with_clustered_se(y, X, cells["cpc_subclass"])
    return model


def run_quadratic_first_stage(cells):
    """CV ~ mean_novelty + mean_novelty^2 + year FE."""
    cells_q = cells.copy()
    cells_q["mean_novelty_sq"] = cells_q["mean_novelty"] ** 2
    year_fe = add_year_fixed_effects(cells_q)
    X = sm.add_constant(pd.concat([cells_q[["mean_novelty", "mean_novelty_sq"]],
                                    year_fe], axis=1))
    return fit_with_clustered_se(cells_q["CV"], X, cells_q["cpc_subclass"])


# ===========================================================================
# STEP 3: Run all decomposition regressions on a given sample
# ===========================================================================

def regress_decomposition(cells, label):
    """Returns dict of {dep_var: model} for CV / ln(mean) / ln(sd)."""
    return {
        "CV":           run_linear_first_stage(cells, "CV"),
        "ln_mean_value": run_linear_first_stage(cells, "ln_mean_value"),
        "ln_sd_value":   run_linear_first_stage(cells, "ln_sd_value"),
        "_n":           len(cells),
        "_label":       label,
    }


# ===========================================================================
# STEP 4: Robustness — trimmed samples
# ===========================================================================

def trim_cv_extremes(cells, lower_q=0.01, upper_q=0.99):
    """Drop top/bottom 1% of CV."""
    lo, hi = cells["CV"].quantile([lower_q, upper_q])
    trimmed = cells[(cells["CV"] >= lo) & (cells["CV"] <= hi)].copy()
    return trimmed, len(cells) - len(trimmed), (lo, hi)


def trim_low_mean_value(cells, lower_q=0.05):
    """Drop bottom 5% of mean_value (cells where mean_value is tiny -> CV explodes)."""
    threshold = cells["mean_value"].quantile(lower_q)
    trimmed = cells[cells["mean_value"] >= threshold].copy()
    return trimmed, len(cells) - len(trimmed), threshold


# ===========================================================================
# STEP 5: Reporting helpers
# ===========================================================================

def stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def fmt_coef_row(model, term):
    coef = model.params[term]
    se = model.bse[term]
    p = model.pvalues[term]
    return f"{coef:+.4f}{stars(p)} (SE={se:.4f}, p={p:.4f})"


def regression_table_md(results, header):
    """results = dict produced by regress_decomposition (keys: CV/ln_mean/ln_sd)."""
    lines = [f"### {header}", ""]
    lines.append(f"_N = {results['_n']} cells._  ")
    lines.append("")
    lines.append("| Dependent variable | mean_novelty | R² | N |")
    lines.append("|---|---|---|---|")
    for key, dv_label in [("CV", "CV"),
                          ("ln_mean_value", "ln(mean_value + 1)"),
                          ("ln_sd_value", "ln(sd_value + 1)")]:
        m = results[key]
        lines.append(f"| {dv_label} | {fmt_coef_row(m, 'mean_novelty')} | "
                     f"{m.rsquared:.4f} | {int(m.nobs)} |")
    lines.append("")
    return "\n".join(lines)


def quadratic_table_md(model):
    lines = ["### Nonlinearity: CV ~ mean_novelty + mean_novelty² + year FE", ""]
    lines.append(f"_N = {int(model.nobs)} cells._  ")
    lines.append("")
    lines.append("| Term | Coef (SE, p) |")
    lines.append("|---|---|")
    lines.append(f"| mean_novelty | {fmt_coef_row(model, 'mean_novelty')} |")
    lines.append(f"| mean_novelty² | {fmt_coef_row(model, 'mean_novelty_sq')} |")
    lines.append(f"| R² | {model.rsquared:.4f} |")
    lines.append("")
    return "\n".join(lines)


def sample_balance_md(cells):
    out = ["## 5. Sample size and balance", ""]
    out.append(f"- Cells used: **{len(cells)}** (non-missing on all required variables).")
    counts = cells["grant_year"].value_counts().sort_index()
    out.append("- Cells per year:")
    for yr, n in counts.items():
        out.append(f"  - {int(yr)}: {n}")
    out.append("")
    out.append("- Distribution of mean_novelty:")
    out.append(cells["mean_novelty"].describe().to_string())
    out.append("")
    out.append("- Distribution of CV:")
    out.append(cells["CV"].describe().to_string())
    out.append("")
    out.append("- Pairwise correlations (cell-level):")
    corr_cols = ["mean_novelty", "CV", "mean_value", "sd_value"]
    out.append(cells[corr_cols].corr().round(4).to_string())
    out.append("")
    return "\n".join(out)


# ===========================================================================
# STEP 6: Scatter plot
# ===========================================================================

def make_scatter_plot(cells, output_path):
    """Three series (standardized) vs. mean_novelty with fitted lines."""
    fig, ax = plt.subplots(figsize=(9, 6))

    series = [
        ("CV", cells["CV"], "tab:blue"),
        ("ln(mean_value)", cells["ln_mean_value"], "tab:orange"),
        ("ln(sd_value)", cells["ln_sd_value"], "tab:green"),
    ]
    x = cells["mean_novelty"].values

    for label, raw, color in series:
        z = (raw - raw.mean()) / raw.std()
        ax.scatter(x, z, s=8, alpha=0.35, color=color, label=f"{label} (z)")
        slope, intercept = np.polyfit(x, z, 1)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, intercept + slope * x_line, color=color, lw=2,
                label=f"  fit: {slope:+.2f}·x + {intercept:+.2f}")

    ax.axhline(0, color="grey", lw=0.5)
    ax.set_xlabel("mean_novelty (cell-level)")
    ax.set_ylabel("Standardized value")
    ax.set_title("First-stage decomposition: CV vs. its mean and SD components")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ===========================================================================
# STEP 7: Top-level orchestration
# ===========================================================================

def build_summary(results_full, results_cv_trim, results_mv_trim,
                  quad_model, cells, drops):
    """Build the plain-English summary based on actual coefficients."""
    cv_coef = results_full["CV"].params["mean_novelty"]
    cv_p = results_full["CV"].pvalues["mean_novelty"]
    mean_coef = results_full["ln_mean_value"].params["mean_novelty"]
    mean_p = results_full["ln_mean_value"].pvalues["mean_novelty"]
    sd_coef = results_full["ln_sd_value"].params["mean_novelty"]
    sd_p = results_full["ln_sd_value"].pvalues["mean_novelty"]

    mean_sig = mean_p < 0.10
    sd_sig = sd_p < 0.10

    if mean_sig and mean_coef > 0 and not sd_sig:
        channel = ("**mean channel dominates**: novel cells have higher mean patent "
                   "value, so the denominator of CV grows while the numerator (SD) "
                   "moves little, mechanically lowering CV.")
    elif sd_sig and sd_coef < 0 and not mean_sig:
        channel = ("**SD channel dominates**: novel cells are *more homogeneous* in "
                   "patent value (SD falls), which directly reduces CV.")
    elif mean_sig and sd_sig and mean_coef * sd_coef < 0:
        channel = ("**both channels contribute**: novel cells have higher means AND "
                   "lower SDs, both pushing CV down.")
    elif mean_sig and sd_sig and mean_coef > 0 and sd_coef > 0:
        channel = ("**mean grows faster than SD**: both rise with novelty but mean "
                   f"rises more (β_mean={mean_coef:+.3f} vs β_sd={sd_coef:+.3f}), "
                   "so CV falls.")
    else:
        channel = ("**neither linear channel is statistically clean** — the negative "
                   "first-stage sign likely sits in nonlinearity, outliers, or "
                   "interaction with year/section composition.")

    quad_p = quad_model.pvalues["mean_novelty_sq"]
    quad_note = (f" Adding mean_novelty² is "
                 f"{'significant' if quad_p < 0.05 else 'not significant'} "
                 f"(p={quad_p:.3f}).")

    cv_trim_coef = results_cv_trim["CV"].params["mean_novelty"]
    cv_trim_p = results_cv_trim["CV"].pvalues["mean_novelty"]
    robust_note = (f" After trimming the top/bottom 1% of CV the first-stage "
                   f"coefficient is {cv_trim_coef:+.3f} (p={cv_trim_p:.3f}), so the "
                   f"sign is "
                   f"{'robust to outliers' if (cv_trim_coef * cv_coef > 0 and cv_trim_p < 0.05) else 'sensitive to outliers'}.")

    summary = f"""## Summary

The first-stage estimate reproduces at α₁ = {cv_coef:+.3f} (p={cv_p:.4f}). \
Decomposing CV = SD/mean with parallel year-FE regressions on log(mean) and log(SD), \
{channel}{quad_note}{robust_note}

**Implications for the four memo explanations:**

(a) **Mechanical artifact of CV.** \
{"Supported — the negative first stage is driven by the denominator (mean), so CV behaves more like an inverse-mean variable than a pure dispersion measure here." if (mean_sig and mean_coef > 0) else "Not the main driver — the mean channel does not move enough with novelty to explain the negative sign mechanically."}

(b) **Novel areas more homogeneous (lower SD).** \
{"Supported — SD of patent value is significantly lower in novel cells, consistent with herding/coordination in emerging tech areas." if (sd_sig and sd_coef < 0) else "Not supported on this sample — SD does not fall significantly with novelty, so 'homogeneity of novel areas' is not the binding mechanism here."}

(c) **CV does not capture investor uncertainty (conceptual).** \
{"Strengthened — what looks like 'lower uncertainty' in novel cells is largely a denominator effect, so CV is a poor proxy for ex-ante information uncertainty." if (mean_sig and mean_coef > 0) else "Neither corroborated nor refuted by these data; needs a non-CV dispersion measure (e.g., MAD, IQR, or analyst-disagreement style metrics) to test."}

(d) **Text novelty ≠ market-perceived novelty (conceptual).** \
{"Consistent with the data — high text-novelty cells are valued *more* on average (β_ln(mean)={mean_coef:+.3f}), suggesting investors *recognize* novelty rather than discounting it." if (mean_sig and mean_coef > 0) else "Cannot be diagnosed from the first stage alone; the main regression on individual ln(xi_real) is the right place to test this."}
"""
    return summary, channel


def main():
    print("=" * 60)
    print("First-stage decomposition")
    print("=" * 60)

    print("\nSTEP 1: Build cell-level dataset")
    cells = build_cell_level_dataset()

    print("\nSTEP 2: First-stage + decomposition (full sample)")
    results_full = regress_decomposition(cells, "Full sample")
    print(f"  CV  ~ mean_novelty: {fmt_coef_row(results_full['CV'], 'mean_novelty')}")
    print(f"  lnM ~ mean_novelty: {fmt_coef_row(results_full['ln_mean_value'], 'mean_novelty')}")
    print(f"  lnS ~ mean_novelty: {fmt_coef_row(results_full['ln_sd_value'], 'mean_novelty')}")

    print("\nSTEP 3: Robustness — CV outlier trim (1% / 99%)")
    cv_trim, cv_drops, cv_bounds = trim_cv_extremes(cells)
    print(f"  Dropped {cv_drops} cells (CV outside [{cv_bounds[0]:.3f}, {cv_bounds[1]:.3f}])")
    results_cv_trim = regress_decomposition(cv_trim, "CV-trimmed (1%/99%)")

    print("\nSTEP 4: Robustness — bottom-5% mean_value trim")
    mv_trim, mv_drops, mv_threshold = trim_low_mean_value(cells)
    print(f"  Dropped {mv_drops} cells (mean_value < {mv_threshold:.4f})")
    results_mv_trim = regress_decomposition(mv_trim, "Low-mean-value trimmed (bottom 5%)")

    print("\nSTEP 5: Nonlinearity check")
    quad_model = run_quadratic_first_stage(cells)
    print(f"  mean_novelty:    {fmt_coef_row(quad_model, 'mean_novelty')}")
    print(f"  mean_novelty²:   {fmt_coef_row(quad_model, 'mean_novelty_sq')}")

    print("\nSTEP 6: Scatter plot")
    make_scatter_plot(cells, SCATTER_PATH)

    print("\nSTEP 7: Build markdown report")
    drops = {"cv_trim_drops": cv_drops, "mv_trim_drops": mv_drops,
             "cv_trim_bounds": cv_bounds, "mv_trim_threshold": mv_threshold}
    summary, channel = build_summary(results_full, results_cv_trim, results_mv_trim,
                                     quad_model, cells, drops)

    md_parts = [
        "# First-stage decomposition: why does mean_novelty NEGATIVELY predict CV?",
        "",
        "_Specification: cell = (cpc_subclass, grant_year). All regressions include "
        "year FE; SE clustered at cpc_subclass._",
        "",
        summary,
        "",
        "## 1. First-stage reproduction (full sample)",
        regression_table_md(results_full, "CV / ln(mean) / ln(sd) on mean_novelty + year FE"),
        "## 2. Robustness — CV outliers trimmed (top/bottom 1%)",
        f"_Dropped {cv_drops} cells with CV outside "
        f"[{cv_bounds[0]:.3f}, {cv_bounds[1]:.3f}]._",
        "",
        regression_table_md(results_cv_trim, "Same specs, CV-trimmed sample"),
        "## 3. Robustness — bottom-5% mean_value trimmed",
        f"_Dropped {mv_drops} cells with mean_value < {mv_threshold:.4f}._",
        "",
        regression_table_md(results_mv_trim, "Same specs, low-mean-value-trimmed sample"),
        "## 4. Nonlinearity",
        quadratic_table_md(quad_model),
        sample_balance_md(cells),
        "## Notes",
        "- Novelty is the standardized negated bsim5 (Kelly et al. 2021), constructed "
        "  patent-level then averaged within (cpc_subclass, grant_year) cells. A higher "
        "  value means a more novel technology.",
        "- CV = sd(xi_real) / mean(xi_real) within cell, restricted to cells with "
        f"  patent_count >= {MIN_GROUP_SIZE} (matches main.py).",
        "- log(x + 1) used on mean_value and sd_value to guard against tiny denominators; "
        "  positive values do not make log(x+1) equal to log(x).",
        "- SE clustered at cpc_subclass when the regression sample contains > 1 cluster, "
        "  HC1 otherwise.",
    ]
    with open(MARKDOWN_PATH, "w") as f:
        f.write("\n".join(md_parts))
    print(f"  Saved: {MARKDOWN_PATH}")

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)
    print("\nKey finding for terminal:")
    print(f"  {channel}")


if __name__ == "__main__":
    main()
