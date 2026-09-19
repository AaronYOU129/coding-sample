"""
Purpose:    Distinguish among three mechanisms for the cell-size confound on the
            log(SD) ~ mean_novelty first-stage result:
              A. Pure statistical bias (c4 small-sample SD bias)
              B. Substantive heterogeneity in patent composition
              C. Outlier-probability mechanics (max grows with n by extreme-value
                 statistics, inflating SD in large cells)
            The test design is interpretive: c4 isolates A; the iid-pooled
            bootstrap captures A+C jointly; within-cell SDs of non-value features
            (claims, cites, novelty itself) test B; max/median ratio + log(median)
            specifically test C.

Inputs:     data/regression_sample.csv   - 195k patent-level rows.

Outputs:    output/mechanism_test.md     - regression tables, bootstrap distribution,
                                            verdict matrix, recommended interpretation.

Key Steps:  1. Build cells with all needed dispersion + within-feature stats.
            2. Test 1: c4(n) correction on raw SD.
            3. Test 2: 500-rep bootstrap from pooled empirical xi_real distribution.
            4. Test 3: within-cell SDs of log(claims), log(cites+1), patent novelty.
            5. Test 4: max/median ratio regressions + log(median_value).
            6. Render verdicts with explicit treatment of A vs C in the bootstrap.

How to Run: python code/mechanism_test.py
"""

import os
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import gammaln

warnings.filterwarnings("ignore")

PATENT_PATH = "data/regression_sample.csv"
MD_PATH = "output/mechanism_test.md"

N_BOOTSTRAP = 500
SEED = 42
LOG_SD_BASELINE = -0.2453   # log(SD+1) ~ mean_novelty + year FE coefficient


# ===========================================================================
# Helpers
# ===========================================================================

def c4(n):
    """Sample-SD bias correction: c4(n) = sqrt(2/(n-1)) * Gamma(n/2) / Gamma((n-1)/2)."""
    n = np.asarray(n, dtype=float)
    return np.exp(0.5 * np.log(2 / (n - 1)) + gammaln(n / 2) - gammaln((n - 1) / 2))


def stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def fmt(model, term):
    c = model.params[term]
    se = model.bse[term]
    p = model.pvalues[term]
    return f"{c:+.4f}{stars(p)} (SE={se:.4f}, p={p:.4f})"


def design_matrix(cells, *extra_col_dfs):
    cols = [cells[["mean_novelty"]]]
    for col_set in extra_col_dfs:
        cols.append(col_set)
    cols.append(pd.get_dummies(cells["grant_year"].astype(str),
                               prefix="yr", drop_first=True, dtype=float))
    return sm.add_constant(pd.concat(cols, axis=1))


def fit_ols(y, X):
    return sm.OLS(y, X).fit(cov_type="HC1")


# ===========================================================================
# Cell aggregation
# ===========================================================================

def aggregate_cells(patents, n_min=10):
    """Cell-level aggregation including within-cell feature SDs."""
    p = patents.copy()
    p["ln_num_claims"] = np.log(p["num_claims"].clip(lower=0) + 1.0)

    base = p.groupby(["cpc_subclass", "grant_year"]).agg(
        mean_novelty=("novelty", "mean"),
        sd_value=("xi_real", "std"),
        median_value=("xi_real", "median"),
        max_value=("xi_real", "max"),
        n_patents=("patent_num", "count"),
        sd_log_claims=("ln_num_claims", "std"),
        sd_log_cites=("ln_backward_cites", "std"),
        sd_novelty_within=("novelty", "std"),
    ).reset_index()

    cells = base[base["n_patents"] >= n_min].dropna(subset=["sd_value"]).copy()
    cells["c4"]              = c4(cells["n_patents"])
    cells["unbiased_sd"]     = cells["sd_value"] / cells["c4"]
    cells["max_to_median"]   = cells["max_value"] / cells["median_value"]
    cells["ln_n"]            = np.log(cells["n_patents"])
    cells["ln_sd"]           = np.log(cells["sd_value"] + 1.0)
    cells["ln_unbiased_sd"]  = np.log(cells["unbiased_sd"] + 1.0)
    cells["ln_median_value"] = np.log(cells["median_value"] + 1.0)
    cells["ln_sd_log_claims"]      = np.log(cells["sd_log_claims"].clip(lower=1e-9) + 1.0)
    cells["ln_sd_log_cites"]       = np.log(cells["sd_log_cites"].clip(lower=1e-9) + 1.0)
    cells["ln_sd_novelty_within"]  = np.log(cells["sd_novelty_within"].clip(lower=1e-9) + 1.0)
    return cells, p


# ===========================================================================
# TEST 1: c4 correction
# ===========================================================================

def test_1_c4(cells):
    X = design_matrix(cells)
    m_orig = fit_ols(cells["ln_sd"],          X)
    m_corr = fit_ols(cells["ln_unbiased_sd"], X)
    return m_orig, m_corr


# ===========================================================================
# TEST 2: bootstrap
# ===========================================================================

def test_2_bootstrap(patents, cells, n_boot=500, seed=42):
    """For each cell, draw n_patents iid from pooled xi_real, compute fake SD,
    regress log(fakeSD+1) ~ mean_novelty + year FE. 500 reps."""
    rng = np.random.default_rng(seed)
    pooled = patents["xi_real"].values

    cells_idx = cells.reset_index(drop=True)
    n_arr = cells_idx["n_patents"].values.astype(int)
    splits = np.cumsum(n_arr)[:-1]
    n_total = int(n_arr.sum())

    year_fe = pd.get_dummies(cells_idx["grant_year"].astype(str),
                             prefix="yr", drop_first=True, dtype=float).reset_index(drop=True)
    X = sm.add_constant(pd.concat([cells_idx[["mean_novelty"]], year_fe],
                                   axis=1)).values
    nov_idx = 1   # const at 0, mean_novelty at 1

    coefs = np.empty(n_boot)
    for b in range(n_boot):
        big = rng.choice(pooled, size=n_total, replace=True)
        chunks = np.split(big, splits)
        fake_sd = np.array([c.std(ddof=1) for c in chunks])
        y = np.log(fake_sd + 1.0)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        coefs[b] = beta[nov_idx]
    return coefs


# ===========================================================================
# TEST 3: within-cell SDs of non-value features
# ===========================================================================

def test_3_within_cell_sds(cells):
    X = design_matrix(cells)
    X_n = design_matrix(cells, cells[["ln_n"]])

    deps = [
        ("ln(SD log(num_claims) + 1)",      "ln_sd_log_claims"),
        ("ln(SD log(backward_cites+1) + 1)", "ln_sd_log_cites"),
        ("ln(SD patent novelty + 1)",        "ln_sd_novelty_within"),
    ]
    rows = []
    for label, dep in deps:
        m1 = fit_ols(cells[dep], X)
        m2 = fit_ols(cells[dep], X_n)
        rows.append((label, m1, m2))
    return rows


# ===========================================================================
# TEST 4: outlier probability
# ===========================================================================

def test_4_outlier(cells):
    year_fe = pd.get_dummies(cells["grant_year"].astype(str),
                             prefix="yr", drop_first=True, dtype=float)
    X_n = sm.add_constant(pd.concat([cells[["ln_n"]], year_fe], axis=1))
    X_v = design_matrix(cells)

    m_max_n     = fit_ols(cells["max_to_median"],   X_n)
    m_max_nov   = fit_ols(cells["max_to_median"],   X_v)
    m_median    = fit_ols(cells["ln_median_value"], X_v)
    return m_max_n, m_max_nov, m_median


# ===========================================================================
# Verdicts
# ===========================================================================

def verdict_a(m_orig, m_corr, boot_coefs, c4_min, c4_mean):
    """Mechanism A in the narrow sense (c4 only). Bootstrap conflates A+C."""
    delta = abs(m_corr.params["mean_novelty"] - m_orig.params["mean_novelty"])
    rel = delta / abs(m_orig.params["mean_novelty"])
    boot_med = float(np.median(boot_coefs))

    if rel < 0.05:
        narrow = ("c4 correction shifts the coefficient by only "
                  f"{rel*100:.2f}% — the small-sample SD bias defined by c4(n) "
                  f"explains essentially none of the −0.245 result.")
    else:
        narrow = (f"c4 correction shifts the coefficient by {rel*100:.1f}% — "
                  "the small-sample bias is doing real work.")

    return ("NOT SUPPORTED" if rel < 0.05 else "PARTIALLY SUPPORTED",
            narrow + f" (Bootstrap median = {boot_med:+.4f}, but this conflates "
                     "A with C — see Mechanism C verdict for separation.)")


def verdict_b(test3_rows, baseline=LOG_SD_BASELINE):
    """Negative & significant on multiple non-value features after log(n) control."""
    results = []
    for label, m1, m2 in test3_rows:
        c1 = m1.params["mean_novelty"]
        p1 = m1.pvalues["mean_novelty"]
        c2 = m2.params["mean_novelty"]
        p2 = m2.pvalues["mean_novelty"]
        results.append((label, c1, p1, c2, p2))

    n_neg_sig_with_n = sum(1 for _, _, _, c, p in results
                           if c < 0 and p < 0.10)
    n_pos_sig_with_n = sum(1 for _, _, _, c, p in results
                           if c > 0 and p < 0.10)

    if n_neg_sig_with_n >= 2 and n_pos_sig_with_n == 0:
        return ("SUPPORTED",
                f"{n_neg_sig_with_n}/3 non-value features show significantly "
                "lower within-cell SD in novel cells AFTER log(n) control. "
                "Substantive homogeneity holds beyond patent value.")
    if n_neg_sig_with_n == 1 and n_pos_sig_with_n == 0:
        return ("INCONCLUSIVE",
                "Only 1/3 non-value features supports B; could be feature-specific.")
    if n_pos_sig_with_n >= 1:
        return ("NOT SUPPORTED",
                f"At least one non-value feature shows HIGHER within-cell SD "
                f"in novel cells (with log(n) control), against the homogeneity story.")
    return ("NOT SUPPORTED",
            "No non-value feature shows significantly lower within-cell SD in "
            "novel cells after log(n) control.")


def verdict_c(m_max_n, m_max_nov, m_median, baseline=LOG_SD_BASELINE):
    n_coef = m_max_n.params["ln_n"]
    n_p = m_max_n.pvalues["ln_n"]
    median_coef = m_median.params["mean_novelty"]
    median_p = m_median.pvalues["mean_novelty"]

    n_grows_max = n_coef > 0 and n_p < 0.05
    median_null_or_pos = median_coef > -0.10   # weaker than baseline -0.245

    if n_grows_max and median_null_or_pos:
        return ("SUPPORTED",
                f"max/median grows significantly with log(n) "
                f"({n_coef:+.3f}, p={n_p:.4f}) AND log(median_value) ~ novelty "
                f"is null/positive ({median_coef:+.3f}, p={median_p:.3f}). "
                "Outlier probability mechanically inflates SD in larger cells; "
                "the location measure (median) doesn't share log(SD)'s pattern.")
    if n_grows_max and not median_null_or_pos:
        return ("PARTIALLY SUPPORTED",
                f"max/median grows with log(n) ({n_coef:+.3f}, p={n_p:.4f}) — "
                "outlier mechanics are real — but log(median) also moves with "
                f"novelty ({median_coef:+.3f}), so C is not the full story.")
    return ("NOT SUPPORTED",
            f"max/median does not grow with log(n) ({n_coef:+.3f}, p={n_p:.4f}); "
            "outlier-probability mechanism is not visible.")


# ===========================================================================
# Markdown helpers
# ===========================================================================

def t3_table(test3_rows):
    out = ["| Within-cell feature SD | No log(n) | With log(n) | log(n) coef |",
           "|---|---|---|---|"]
    for label, m1, m2 in test3_rows:
        out.append(f"| {label} | {fmt(m1, 'mean_novelty')} | "
                   f"{fmt(m2, 'mean_novelty')} | {fmt(m2, 'ln_n')} |")
    return "\n".join(out)


def boot_summary(boot_coefs):
    return {
        "n":      len(boot_coefs),
        "mean":   float(np.mean(boot_coefs)),
        "median": float(np.median(boot_coefs)),
        "std":    float(np.std(boot_coefs)),
        "p05":    float(np.percentile(boot_coefs, 5)),
        "p25":    float(np.percentile(boot_coefs, 25)),
        "p75":    float(np.percentile(boot_coefs, 75)),
        "p95":    float(np.percentile(boot_coefs, 95)),
        "min":    float(np.min(boot_coefs)),
        "max":    float(np.max(boot_coefs)),
    }


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("=" * 60)
    print("Mechanism test")
    print("=" * 60)

    patents = pd.read_csv(PATENT_PATH)
    cells, patents = aggregate_cells(patents, n_min=10)
    print(f"Patents: {len(patents):,}; cells: {len(cells)}")
    print(f"n_patents distribution: min={cells['n_patents'].min()}, "
          f"med={cells['n_patents'].median():.0f}, "
          f"max={cells['n_patents'].max()}, "
          f"mean={cells['n_patents'].mean():.0f}")
    print(f"c4 range: min={cells['c4'].min():.4f}, "
          f"mean={cells['c4'].mean():.4f}, max={cells['c4'].max():.4f}")

    print("\n--- TEST 1: c4 small-sample correction ---")
    m_orig, m_corr = test_1_c4(cells)
    print(f"  log(SD+1) ~ novelty + year FE:           {fmt(m_orig, 'mean_novelty')}")
    print(f"  log(SD/c4 + 1) ~ novelty + year FE:      {fmt(m_corr, 'mean_novelty')}")

    print("\n--- TEST 2: bootstrap with iid pooled draws ---")
    print(f"  Drawing {N_BOOTSTRAP} reps × {len(cells)} cells × ~"
          f"{cells['n_patents'].mean():.0f} patents per cell...")
    boot_coefs = test_2_bootstrap(patents, cells,
                                   n_boot=N_BOOTSTRAP, seed=SEED)
    boot = boot_summary(boot_coefs)
    for k, v in boot.items():
        if k == "n":
            print(f"  {k}      = {v}")
        else:
            print(f"  {k}    = {v:+.4f}")

    print("\n--- TEST 3: within-cell SDs of non-value features ---")
    test3_rows = test_3_within_cell_sds(cells)
    for label, m1, m2 in test3_rows:
        print(f"  {label}:")
        print(f"    no log(n):   {fmt(m1, 'mean_novelty')}")
        print(f"    with log(n): {fmt(m2, 'mean_novelty')}, "
              f"log(n) coef {fmt(m2, 'ln_n')}")

    print("\n--- TEST 4: outlier probability ---")
    m_max_n, m_max_nov, m_median = test_4_outlier(cells)
    print(f"  max/median ~ log(n) + year FE: log(n)  {fmt(m_max_n, 'ln_n')}")
    print(f"  max/median ~ novelty + year FE: novelty {fmt(m_max_nov, 'mean_novelty')}")
    print(f"  log(median+1) ~ novelty + year FE: novelty {fmt(m_median, 'mean_novelty')}")

    # ----- verdicts -----
    a_tag, a_msg = verdict_a(m_orig, m_corr, boot_coefs,
                              cells["c4"].min(), cells["c4"].mean())
    b_tag, b_msg = verdict_b(test3_rows)
    c_tag, c_msg = verdict_c(m_max_n, m_max_nov, m_median)

    print("\n--- VERDICTS ---")
    print(f"  A: {a_tag} — {a_msg}")
    print(f"  B: {b_tag} — {b_msg}")
    print(f"  C: {c_tag} — {c_msg}")

    # ----- markdown -----
    md = []
    md.append("# Mechanism test: A vs B vs C for the cell-size confound")
    md.append("")
    md.append(f"_Cell = (cpc_subclass, grant_year), n_patents ≥ 10. {len(cells)} cells. "
              f"HC1 robust SE._")
    md.append("")

    md.append("## VERDICT MATRIX")
    md.append("")
    md.append("| Mechanism | Verdict | Evidence |")
    md.append("|---|---|---|")
    md.append(f"| **A — pure statistical bias (c4)** | {a_tag} | {a_msg} |")
    md.append(f"| **B — substantive heterogeneity** | {b_tag} | {b_msg} |")
    md.append(f"| **C — outlier-probability mechanics** | {c_tag} | {c_msg} |")
    md.append("")
    md.append("_Note on bootstrap: it draws iid from the pooled empirical xi_real "
              "distribution, preserving the empirical novelty/n correlation. "
              "The simulated coefficient distribution captures **both** A and C "
              "jointly. To separate them, A's narrow form (c4) is tested in "
              "isolation by Test 1; what's left in the bootstrap distribution "
              "after subtracting A is attributable to C._")
    md.append("")

    md.append("## Data summary")
    md.append("")
    md.append(f"- Cells: {len(cells)}")
    md.append(f"- n_patents per cell: min={cells['n_patents'].min()}, "
              f"median={cells['n_patents'].median():.0f}, "
              f"mean={cells['n_patents'].mean():.0f}, "
              f"max={cells['n_patents'].max()}")
    md.append(f"- c4(n) correction range: "
              f"{cells['c4'].min():.4f} to {cells['c4'].max():.4f} "
              f"(mean {cells['c4'].mean():.4f}). With this n distribution the "
              f"narrow Mechanism A correction is by construction tiny.")
    md.append("")

    md.append("## Test 1 — c4 small-sample correction (Mechanism A, narrow)")
    md.append("")
    md.append("| Spec | mean_novelty coef |")
    md.append("|---|---|")
    md.append(f"| log(SD+1) ~ novelty + year FE        | {fmt(m_orig, 'mean_novelty')} |")
    md.append(f"| log(SD/c4 + 1) ~ novelty + year FE   | {fmt(m_corr, 'mean_novelty')} |")
    md.append("")
    md.append(f"Δ in coefficient = "
              f"{abs(m_corr.params['mean_novelty'] - m_orig.params['mean_novelty']):.4f} "
              f"({100 * abs(m_corr.params['mean_novelty'] - m_orig.params['mean_novelty']) / abs(m_orig.params['mean_novelty']):.1f}% of baseline).")
    md.append("")

    md.append("## Test 2 — bootstrap from pooled xi_real (Mechanisms A + C jointly)")
    md.append("")
    md.append(f"_500 reps. For each cell, n_patents drawn iid with replacement from "
              "the pooled empirical xi_real distribution; fake SD computed; "
              "log(fake_SD+1) regressed on mean_novelty + year FE. Under the "
              "null of no real novelty/dispersion relationship, this captures all "
              "the n-driven mechanical bias (A + C)._")
    md.append("")
    md.append("| Statistic | Value |")
    md.append("|---|---|")
    for k in ["n", "mean", "median", "std", "p05", "p25", "p75", "p95",
              "min", "max"]:
        v = boot[k]
        if k == "n":
            md.append(f"| {k} | {v} |")
        else:
            md.append(f"| {k} | {v:+.4f} |")
    md.append("")
    md.append(f"_Observed coefficient on real data: {LOG_SD_BASELINE:+.4f}._")
    md.append("")
    boot_share = boot["median"] / LOG_SD_BASELINE
    md.append(f"**Mechanical share:** the bootstrap median ({boot['median']:+.4f}) "
              f"covers {boot_share*100:.1f}% of the observed −0.245. "
              f"That fraction is attributable to A+C combined; the rest must "
              f"come from B or unobserved factors.")
    md.append("")

    md.append("## Test 3 — within-cell SDs of non-value features (Mechanism B)")
    md.append("")
    md.append(t3_table(test3_rows))
    md.append("")

    md.append("## Test 4 — outlier probability (Mechanism C)")
    md.append("")
    md.append("| Spec | Coef on focal regressor |")
    md.append("|---|---|")
    md.append(f"| max_to_median ~ log(n) + year FE | "
              f"log(n): {fmt(m_max_n, 'ln_n')} |")
    md.append(f"| max_to_median ~ novelty + year FE | "
              f"novelty: {fmt(m_max_nov, 'mean_novelty')} |")
    md.append(f"| log(median+1) ~ novelty + year FE | "
              f"novelty: {fmt(m_median, 'mean_novelty')} |")
    md.append("")

    # Most-likely-mechanism narrative
    boot_med = boot["median"]
    boot_5 = boot["p05"]
    boot_95 = boot["p95"]
    n_grows_max_p = m_max_n.pvalues["ln_n"]
    likely = []
    if a_tag == "NOT SUPPORTED":
        likely.append("Mechanism A in its narrow c4 form is essentially ruled out — "
                      "the correction barely moves the coefficient because cells "
                      "have n ≥ 10 (and most are much larger), so the bias term is "
                      "negligible.")
    if c_tag in ("SUPPORTED", "PARTIALLY SUPPORTED"):
        likely.append("Mechanism C is corroborated: max/median grows strongly with "
                      f"log(n) (coef {m_max_n.params['ln_n']:+.3f}, "
                      f"p={n_grows_max_p:.4f}), confirming that bigger cells "
                      "mechanically pick up larger maxima from the right tail.")
    if b_tag == "SUPPORTED":
        likely.append("Mechanism B is corroborated by within-cell SDs of non-value "
                      "features tracking novelty in the same direction.")
    elif b_tag == "NOT SUPPORTED":
        likely.append("Mechanism B is not supported: within-cell SDs of "
                      "non-value features do not consistently fall in novel cells "
                      "once log(n) is controlled.")

    boot_explains = abs(boot_med) / abs(LOG_SD_BASELINE)
    if boot_explains > 0.7:
        likely.append(f"The bootstrap (A+C) median of {boot_med:+.4f} alone covers "
                      f"{boot_explains*100:.0f}% of the observed −0.245, leaving "
                      "very little to explain. Most of the observed effect is "
                      "n-driven mechanics.")
    elif boot_explains > 0.3:
        likely.append(f"The bootstrap (A+C) median of {boot_med:+.4f} explains "
                      f"only ~{boot_explains*100:.0f}% of the observed −0.245. "
                      "A meaningful residual remains, which is where Mechanism B "
                      "(or other unmodeled factors) could matter.")
    else:
        likely.append(f"The bootstrap (A+C) median of {boot_med:+.4f} explains "
                      f"only ~{boot_explains*100:.0f}% of the observed −0.245. "
                      "Mechanical n-driven artifacts are NOT enough to reproduce "
                      "the observed effect; the real driver is elsewhere.")

    md.append("## Most likely mechanism")
    md.append("")
    for s in likely:
        md.append(f"- {s}")
    md.append("")

    with open(MD_PATH, "w") as f:
        f.write("\n".join(md))
    print(f"\nSaved: {MD_PATH}")


if __name__ == "__main__":
    main()
