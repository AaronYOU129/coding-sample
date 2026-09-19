"""
Purpose:    Decompose the divergence between two earlier results:
              (A) Corr(mean_novelty, raw SD)             = -0.025  (null)
              (B) log(sd) ~ mean_novelty + year FE       = -0.245  (p=0.001)
            Cross {raw vs. log SD} x {with vs. without year FE} to attribute
            the flip to the log transformation, year FE, both, or an interaction.
            Also report what happens if the single CV~8 / SD~250 outlier is dropped.

Inputs:     output/cell_level_decomposition.csv  - 680 cell rows from the previous
            decomposition (cpc_subclass, grant_year, mean_novelty, mean_value,
            sd_value, CV, ln_mean_value, ln_sd_value).

Outputs:    output/sd_divergence_diagnostic.md   - 2x2 grid + outlier check.

How to Run: python code/sd_divergence_diagnostic.py
"""

import os
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore")

CELL_PATH = "output/cell_level_decomposition.csv"
MD_PATH = "output/sd_divergence_diagnostic.md"


def fit(y, X, cluster):
    if cluster.nunique() > 1:
        return sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": cluster})
    return sm.OLS(y, X).fit(cov_type="HC1")


def stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def fmt(model, term):
    c = model.params[term]
    se = model.bse[term]
    p = model.pvalues[term]
    return f"{c:+.4f}{stars(p)} (SE={se:.4f}, p={p:.4f})"


def run_spec(cells, dep, with_year_fe):
    X_cols = [cells[["mean_novelty"]]]
    if with_year_fe:
        X_cols.append(pd.get_dummies(cells["grant_year"].astype(str),
                                     prefix="yr", drop_first=True, dtype=float))
    X = sm.add_constant(pd.concat(X_cols, axis=1))
    return fit(cells[dep], X, cells["cpc_subclass"])


def grid_table(cells, label, n):
    out = [f"### {label} (N = {n})", ""]
    out.append("| Dependent | No year FE | With year FE |")
    out.append("|---|---|---|")
    for dep, dep_label in [("sd_value", "raw SD"),
                           ("ln_sd_value", "ln(SD + 1)")]:
        no_fe = run_spec(cells, dep, False)
        with_fe = run_spec(cells, dep, True)
        out.append(f"| {dep_label} | {fmt(no_fe, 'mean_novelty')} | "
                   f"{fmt(with_fe, 'mean_novelty')} |")
    out.append("")
    return "\n".join(out)


def correlations_by_year(cells):
    out = ["### Year-by-year correlations (raw and log SD vs. mean_novelty)", ""]
    out.append("| Year | N | corr(novelty, SD) | corr(novelty, ln(SD)) | "
               "corr(novelty, mean_value) |")
    out.append("|---|---|---|---|---|")
    for year, group in cells.groupby("grant_year"):
        n = len(group)
        c_raw = group["mean_novelty"].corr(group["sd_value"])
        c_log = group["mean_novelty"].corr(group["ln_sd_value"])
        c_mean = group["mean_novelty"].corr(group["mean_value"])
        out.append(f"| {int(year)} | {n} | {c_raw:+.3f} | {c_log:+.3f} | {c_mean:+.3f} |")
    out.append("")
    out.append("_Pooled (no FE, raw):_  "
               f"{cells['mean_novelty'].corr(cells['sd_value']):+.3f}  ")
    out.append("_Pooled (no FE, log):_  "
               f"{cells['mean_novelty'].corr(cells['ln_sd_value']):+.3f}  ")
    out.append("")
    return "\n".join(out)


def year_means_table(cells):
    out = ["### Between-year shifts (cell averages by year)", ""]
    out.append("| Year | mean(mean_novelty) | mean(SD) | mean(ln(SD)) | mean(mean_value) |")
    out.append("|---|---|---|---|---|")
    for year, g in cells.groupby("grant_year"):
        out.append(f"| {int(year)} | {g['mean_novelty'].mean():+.3f} | "
                   f"{g['sd_value'].mean():.2f} | {g['ln_sd_value'].mean():.3f} | "
                   f"{g['mean_value'].mean():.2f} |")
    out.append("")
    return "\n".join(out)


def top_sd_outliers(cells, k=5):
    top = cells.sort_values("sd_value", ascending=False).head(k)
    out = [f"### Top {k} cells by raw SD (outlier check)", ""]
    out.append("| cpc_subclass | year | mean_novelty | SD | mean_value | CV | n_patents |")
    out.append("|---|---|---|---|---|---|---|")
    for _, r in top.iterrows():
        out.append(f"| {r['cpc_subclass']} | {int(r['grant_year'])} | "
                   f"{r['mean_novelty']:+.3f} | {r['sd_value']:.1f} | "
                   f"{r['mean_value']:.2f} | {r['CV']:.2f} | {int(r['patent_count'])} |")
    out.append("")
    return "\n".join(out)


def main():
    cells = pd.read_csv(CELL_PATH)
    print(f"Loaded {len(cells)} cells")

    print("\n--- 2x2 grid: {raw vs ln(SD)} x {no FE vs year FE} ---")
    for dep in ["sd_value", "ln_sd_value"]:
        for fe in [False, True]:
            m = run_spec(cells, dep, fe)
            tag = f"{dep:>12s}, year_FE={fe}"
            print(f"  {tag}: {fmt(m, 'mean_novelty')}, R²={m.rsquared:.4f}")

    # Outlier sensitivity: drop the single highest-SD cell
    print("\n--- Top SD cells ---")
    top = cells.nlargest(5, "sd_value")
    print(top[["cpc_subclass", "grant_year", "mean_novelty", "sd_value",
               "mean_value", "CV", "patent_count"]].to_string(index=False))

    cells_no_top = cells.sort_values("sd_value", ascending=False).iloc[1:].copy()
    print(f"\n--- Drop single largest SD ({len(cells)} -> {len(cells_no_top)}) ---")
    for dep in ["sd_value", "ln_sd_value"]:
        for fe in [False, True]:
            m = run_spec(cells_no_top, dep, fe)
            print(f"  {dep:>12s}, year_FE={fe}: {fmt(m, 'mean_novelty')}")

    cells_top1 = cells[(cells["sd_value"] >= cells["sd_value"].quantile(0.01)) &
                       (cells["sd_value"] <= cells["sd_value"].quantile(0.99))].copy()
    print(f"\n--- Trim top/bottom 1% of raw SD ({len(cells)} -> {len(cells_top1)}) ---")
    for dep in ["sd_value", "ln_sd_value"]:
        for fe in [False, True]:
            m = run_spec(cells_top1, dep, fe)
            print(f"  {dep:>12s}, year_FE={fe}: {fmt(m, 'mean_novelty')}")

    # Compose markdown
    md = []
    md.append("# Diagnosing the SD-divergence: log vs. year FE\n")
    md.append("_Specification: cell = (cpc_subclass, grant_year), N = 680, "
              "SE clustered at cpc_subclass._\n")

    md.append("## 1. The 2x2 grid\n")
    md.append(grid_table(cells, "Full sample", len(cells)))

    md.append("## 2. Year-by-year structure\n")
    md.append(correlations_by_year(cells))
    md.append(year_means_table(cells))

    md.append("## 3. Outlier check\n")
    md.append(top_sd_outliers(cells, 5))
    md.append("**Drop single largest-SD cell:**\n")
    md.append(grid_table(cells_no_top, "After dropping top-1 SD cell",
                         len(cells_no_top)))
    md.append("**Trim top/bottom 1% of raw SD:**\n")
    md.append(grid_table(cells_top1, "After 1%/99% raw-SD trim",
                         len(cells_top1)))

    # Build the verdict from the actual numbers
    raw_no_fe = run_spec(cells, "sd_value", False)
    raw_fe = run_spec(cells, "sd_value", True)
    log_no_fe = run_spec(cells, "ln_sd_value", False)
    log_fe = run_spec(cells, "ln_sd_value", True)

    raw_no_fe_drop = run_spec(cells_no_top, "sd_value", False)
    log_no_fe_drop = run_spec(cells_no_top, "ln_sd_value", False)

    md.insert(1,
        "## Verdict\n\n"
        f"- **Adding the log transformation alone** (no year FE): raw SD coef "
        f"{fmt(raw_no_fe, 'mean_novelty')} -> log(SD) coef "
        f"{fmt(log_no_fe, 'mean_novelty')}.\n"
        f"- **Adding year FE alone** (raw SD): "
        f"{fmt(raw_no_fe, 'mean_novelty')} -> {fmt(raw_fe, 'mean_novelty')}.\n"
        f"- **Both** (log + year FE): {fmt(log_fe, 'mean_novelty')} — the "
        f"specification you reported as -0.245.\n"
        f"- **Drop the single largest-SD cell** "
        f"({cells.nlargest(1, 'sd_value').iloc[0]['cpc_subclass']} "
        f"{int(cells.nlargest(1, 'sd_value').iloc[0]['grant_year'])}, "
        f"SD={cells['sd_value'].max():.0f}): "
        f"raw-SD/no-FE goes from {fmt(raw_no_fe, 'mean_novelty')} to "
        f"{fmt(raw_no_fe_drop, 'mean_novelty')}.\n\n"
        "_See per-step reasoning below._\n"
    )

    with open(MD_PATH, "w") as f:
        f.write("\n".join(md))
    print(f"\nSaved: {MD_PATH}")


if __name__ == "__main__":
    main()
