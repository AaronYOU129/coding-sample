"""
Purpose:    Stress-test the +0.38 log(median_value) ~ mean_novelty + year FE
            finding from the mechanism test, plus re-validate the bootstrap-
            quantified 60% mechanical share. Five tests:
              (1) n_patents cutoff sensitivity for log(median)
              (2) log(n) control on log(median)
              (3) alternative skewness measures (skew_within, log(p90/median),
                  log(p99/median))
              (4) firm-composition confound (using KPSS permno as firm proxy
                  — note: this is the publicly-traded parent firm, not the raw
                  assignee text)
              (5) bootstrap log(SD) ~ novelty under empirical vs log-normal
                  pooled distributions.

Inputs:     data/regression_sample.csv

Outputs:    output/right_skew_validation.md

How to Run: python code/right_skew_validation.py
"""

import os
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore")

PATENT_PATH = "data/regression_sample.csv"
MD_PATH = "output/right_skew_validation.md"

N_BOOTSTRAP = 500
SEED = 42

LOG_MEDIAN_BASELINE = 0.3834
LOG_SD_BASELINE = -0.2453
BOOTSTRAP_EMPIRICAL_PRIOR = -0.1471


# ===========================================================================
# Helpers
# ===========================================================================

def stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def fmt(model, term):
    c = model.params[term]
    se = model.bse[term]
    p = model.pvalues[term]
    return f"{c:+.4f}{stars(p)} (SE={se:.4f}, p={p:.4f})"


def design_matrix(cells, *extras):
    cols = [cells[["mean_novelty"]]]
    for col_set in extras:
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
    base = patents.groupby(["cpc_subclass", "grant_year"]).agg(
        mean_novelty=("novelty", "mean"),
        mean_value=("xi_real", "mean"),
        sd_value=("xi_real", "std"),
        median_value=("xi_real", "median"),
        n_patents=("patent_num", "count"),
        n_unique_firms=("permno", "nunique"),
    ).reset_index()
    pcts = (patents.groupby(["cpc_subclass", "grant_year"])["xi_real"]
            .agg(p90_value=lambda s: float(s.quantile(0.90)),
                 p99_value=lambda s: float(s.quantile(0.99)))
            .reset_index())
    cells = base.merge(pcts, on=["cpc_subclass", "grant_year"])

    cells = cells[cells["n_patents"] >= n_min].dropna(subset=["sd_value"]).copy()
    cells["ln_n"] = np.log(cells["n_patents"])
    cells["ln_n_firms"] = np.log(cells["n_unique_firms"].clip(lower=1))
    cells["ln_median"] = np.log(cells["median_value"] + 1.0)
    cells["skew_within"] = (cells["mean_value"] - cells["median_value"]) / cells["sd_value"]
    # p90 >= median and p99 >= median by construction, so ratios >= 1
    cells["log_p90_to_median"] = np.log(cells["p90_value"] / cells["median_value"])
    cells["log_p99_to_median"] = np.log(cells["p99_value"] / cells["median_value"])
    return cells


# ===========================================================================
# TEST 1: cutoff sensitivity
# ===========================================================================

def test_1_cutoff(patents):
    rows = []
    for n_min in [5, 10, 20, 50, 100]:
        cells = aggregate_cells(patents, n_min=n_min)
        m = fit_ols(cells["ln_median"], design_matrix(cells))
        rows.append((n_min, len(cells), m))
    return rows


# ===========================================================================
# TEST 2: log(n) control
# ===========================================================================

def test_2_log_n_control(cells):
    m_orig = fit_ols(cells["ln_median"], design_matrix(cells))
    m_with = fit_ols(cells["ln_median"], design_matrix(cells, cells[["ln_n"]]))
    return m_orig, m_with


# ===========================================================================
# TEST 3: alternative skewness measures
# ===========================================================================

def test_3_skew_measures(cells):
    deps = [
        ("skew_within = (mean - median) / sd", "skew_within"),
        ("log(p90 / median)",                  "log_p90_to_median"),
        ("log(p99 / median)",                  "log_p99_to_median"),
    ]
    rows = []
    for label, dep in deps:
        m_no = fit_ols(cells[dep], design_matrix(cells))
        m_n  = fit_ols(cells[dep], design_matrix(cells, cells[["ln_n"]]))
        rows.append((label, m_no, m_n))
    return rows


# ===========================================================================
# TEST 4: firm-composition confound
# ===========================================================================

def test_4_firm(cells):
    corr_n = cells["mean_novelty"].corr(cells["n_unique_firms"])
    corr_log = cells["mean_novelty"].corr(cells["ln_n_firms"])

    m_orig = fit_ols(cells["ln_median"], design_matrix(cells))
    m_firm = fit_ols(cells["ln_median"],
                     design_matrix(cells, cells[["ln_n_firms"]]))
    m_both = fit_ols(cells["ln_median"],
                     design_matrix(cells, cells[["ln_n", "ln_n_firms"]]))
    return corr_n, corr_log, m_orig, m_firm, m_both


# ===========================================================================
# TEST 5: bootstrap log(SD) ~ novelty under two distributions
# ===========================================================================

def _run_bootstrap(rng, sampler, n_total, splits, X, nov_idx, n_boot):
    coefs = np.empty(n_boot)
    for b in range(n_boot):
        big = sampler(rng, n_total)
        chunks = np.split(big, splits)
        fake_sd = np.array([c.std(ddof=1) for c in chunks])
        y = np.log(fake_sd + 1.0)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        coefs[b] = beta[nov_idx]
    return coefs


def test_5_bootstrap_two_distributions(patents, cells, n_boot=500, seed=42):
    pooled = patents["xi_real"].values
    pooled = pooled[pooled > 0]
    log_pooled = np.log(pooled)
    mu, sigma = float(log_pooled.mean()), float(log_pooled.std())

    cells_idx = cells.reset_index(drop=True)
    n_arr = cells_idx["n_patents"].values.astype(int)
    splits = np.cumsum(n_arr)[:-1]
    n_total = int(n_arr.sum())
    year_fe = pd.get_dummies(cells_idx["grant_year"].astype(str),
                              prefix="yr", drop_first=True,
                              dtype=float).reset_index(drop=True)
    X = sm.add_constant(pd.concat([cells_idx[["mean_novelty"]], year_fe],
                                   axis=1)).values
    nov_idx = 1   # const at 0, mean_novelty at 1

    rng_emp = np.random.default_rng(seed)
    rng_ln = np.random.default_rng(seed)

    def sampler_emp(rng, n):
        return rng.choice(pooled, size=n, replace=True)

    def sampler_ln(rng, n):
        return rng.lognormal(mean=mu, sigma=sigma, size=n)

    coefs_emp = _run_bootstrap(rng_emp, sampler_emp, n_total, splits, X,
                                nov_idx, n_boot)
    coefs_ln  = _run_bootstrap(rng_ln,  sampler_ln,  n_total, splits, X,
                                nov_idx, n_boot)
    return coefs_emp, coefs_ln, mu, sigma


# ===========================================================================
# Verdicts
# ===========================================================================

def verdict_t1(rows, baseline=LOG_MEDIAN_BASELINE):
    coefs = np.array([m.params["mean_novelty"] for _, _, m in rows])
    rng_pct = (coefs.max() - coefs.min()) / abs(baseline)
    monotonic = (all(coefs[i] >= coefs[i+1] for i in range(len(coefs)-1)) or
                 all(coefs[i] <= coefs[i+1] for i in range(len(coefs)-1)))
    if rng_pct < 0.30:
        return ("ROBUST",
                f"Coefficients span {coefs.min():+.3f} to {coefs.max():+.3f}, "
                f"range/baseline = {rng_pct*100:.1f}% (< 30%).")
    if monotonic:
        return ("FRAGILE",
                f"Coefficients vary monotonically with cutoff "
                f"({coefs[0]:+.3f} → {coefs[-1]:+.3f}); suggests cell-size dependence.")
    return ("QUALIFIED",
            f"Coefficients vary by {rng_pct*100:.1f}% across cutoffs but not monotonically.")


def verdict_t2(m_orig, m_with):
    orig = m_orig.params["mean_novelty"]
    new = m_with.params["mean_novelty"]
    survival = new / orig
    if survival > 0.85:
        return ("ROBUST",
                f"{survival*100:.1f}% of the original coefficient survives the "
                f"log(n) control.")
    if survival > 0.50:
        return ("QUALIFIED",
                f"{survival*100:.1f}% survives — partly cell-size driven.")
    return ("FRAGILE",
            f"Only {survival*100:.1f}% survives — finding is mostly cell-size driven.")


def verdict_t3(rows):
    """Less right-skew should manifest as: skew_within negative, p90/median
    negative, p99/median negative — all three same sign and significant."""
    expected_signs_match = []
    sigs = []
    for _, m_no, m_n in rows:
        c_n = m_n.params["mean_novelty"]
        p_n = m_n.pvalues["mean_novelty"]
        expected_signs_match.append(c_n < 0)   # less skew = negative
        sigs.append(p_n < 0.05)
    n_sig_neg = sum(1 for sm_, sg in zip(expected_signs_match, sigs)
                    if sm_ and sg)
    n_sig_pos = sum(1 for sm_, sg in zip(expected_signs_match, sigs)
                    if (not sm_) and sg)
    if n_sig_neg == 3:
        return ("ROBUST",
                "All 3 alternative skewness measures show significantly less "
                "right-skew in novel cells (with log(n) control).")
    if n_sig_neg == 2 and n_sig_pos == 0:
        return ("QUALIFIED",
                "2/3 alternative skewness measures support the right-skew "
                "interpretation; one is null or mixed.")
    if n_sig_pos > 0:
        return ("FRAGILE",
                f"At least {n_sig_pos} skewness measure(s) go the WRONG "
                "direction relative to the right-skew claim.")
    return ("FRAGILE",
            f"Only {n_sig_neg}/3 measures support the right-skew "
            "interpretation; rest are null.")


def verdict_t4(corr, m_orig, m_firm, m_both):
    orig = m_orig.params["mean_novelty"]
    firm = m_firm.params["mean_novelty"]
    both = m_both.params["mean_novelty"]
    s_firm = firm / orig
    s_both = both / orig
    # If firm count alone collapses the coefficient, firm composition is the driver
    if s_firm > 0.85 and s_both > 0.5:
        return ("ROBUST",
                f"Firm-count control preserves {s_firm*100:.1f}% of coef; "
                f"with both controls, {s_both*100:.1f}% survives. Firm composition "
                "is not the main driver.")
    if s_firm > 0.5:
        return ("QUALIFIED",
                f"Firm-count alone preserves {s_firm*100:.1f}%; with both controls "
                f"{s_both*100:.1f}%. Some firm-composition confounding.")
    return ("FRAGILE",
            f"Firm-count control reduces coef to {s_firm*100:.1f}% of baseline. "
            "Firm composition is a real confound.")


def verdict_t5(coefs_emp, coefs_ln):
    med_emp = float(np.median(coefs_emp))
    med_ln  = float(np.median(coefs_ln))
    diff_pct = abs(med_ln - med_emp) / abs(med_emp)
    if diff_pct < 0.20:
        return ("ROBUST",
                f"Empirical median {med_emp:+.3f}, log-normal median {med_ln:+.3f} "
                f"({diff_pct*100:.1f}% apart < 20%). Mechanical share is robust to "
                "distributional choice.")
    if diff_pct < 0.50:
        return ("QUALIFIED",
                f"Empirical {med_emp:+.3f} vs log-normal {med_ln:+.3f} "
                f"({diff_pct*100:.1f}% apart). Sensitive to distribution; "
                "soften the 60% headline number.")
    return ("FRAGILE",
            f"Empirical {med_emp:+.3f} vs log-normal {med_ln:+.3f} "
            f"({diff_pct*100:.1f}% apart). Mechanical share is highly distribution-dependent.")


# ===========================================================================
# Markdown helpers
# ===========================================================================

def t1_table(rows):
    out = ["| Cutoff | N cells | mean_novelty coef on log(median+1) |",
           "|---|---|---|"]
    for n_min, N, m in rows:
        out.append(f"| n_patents >= {n_min:>3} | {N} | {fmt(m, 'mean_novelty')} |")
    return "\n".join(out)


def t3_table(rows):
    out = ["| Skewness measure | No log(n) | With log(n) | log(n) coef |",
           "|---|---|---|---|"]
    for label, m_no, m_n in rows:
        out.append(f"| {label} | {fmt(m_no, 'mean_novelty')} | "
                   f"{fmt(m_n, 'mean_novelty')} | {fmt(m_n, 'ln_n')} |")
    return "\n".join(out)


def boot_summary(coefs):
    return {
        "n":      len(coefs),
        "mean":   float(np.mean(coefs)),
        "median": float(np.median(coefs)),
        "std":    float(np.std(coefs)),
        "p05":    float(np.percentile(coefs, 5)),
        "p95":    float(np.percentile(coefs, 95)),
    }


def boot_table(emp, ln):
    out = ["| Statistic | Empirical | Log-normal |", "|---|---|---|"]
    for k in ["n", "mean", "median", "std", "p05", "p95"]:
        v_e = emp[k]; v_l = ln[k]
        if k == "n":
            out.append(f"| {k} | {v_e} | {v_l} |")
        else:
            out.append(f"| {k} | {v_e:+.4f} | {v_l:+.4f} |")
    return "\n".join(out)


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("=" * 60)
    print("Right-skew validation")
    print("=" * 60)

    patents = pd.read_csv(PATENT_PATH)
    cells = aggregate_cells(patents, n_min=10)
    print(f"Patents: {len(patents):,}, Cells: {len(cells)}")
    print(f"median_value: min={cells['median_value'].min():.2f}, "
          f"med={cells['median_value'].median():.2f}, "
          f"max={cells['median_value'].max():.2f}")
    print(f"n_unique_firms per cell: min={cells['n_unique_firms'].min()}, "
          f"med={cells['n_unique_firms'].median():.0f}, "
          f"max={cells['n_unique_firms'].max()}")

    print("\n--- TEST 1: cutoff sensitivity ---")
    t1 = test_1_cutoff(patents)
    for n_min, N, m in t1:
        print(f"  n>={n_min}: N={N}, {fmt(m, 'mean_novelty')}")
    t1_tag, t1_msg = verdict_t1(t1)
    print(f"  VERDICT: {t1_tag} — {t1_msg}")

    print("\n--- TEST 2: log(n) control ---")
    m2_orig, m2_with = test_2_log_n_control(cells)
    print(f"  Original:    {fmt(m2_orig, 'mean_novelty')}")
    print(f"  With log(n): {fmt(m2_with, 'mean_novelty')}, "
          f"log(n) coef {fmt(m2_with, 'ln_n')}")
    t2_tag, t2_msg = verdict_t2(m2_orig, m2_with)
    print(f"  VERDICT: {t2_tag} — {t2_msg}")

    print("\n--- TEST 3: alternative skewness measures ---")
    t3 = test_3_skew_measures(cells)
    for label, m_no, m_n in t3:
        print(f"  {label}:")
        print(f"    no log(n):   {fmt(m_no, 'mean_novelty')}")
        print(f"    with log(n): {fmt(m_n, 'mean_novelty')}")
    t3_tag, t3_msg = verdict_t3(t3)
    print(f"  VERDICT: {t3_tag} — {t3_msg}")

    print("\n--- TEST 4: firm-composition confound (permno proxy) ---")
    corr_n, corr_logn, m4_orig, m4_firm, m4_both = test_4_firm(cells)
    print(f"  Corr(novelty, n_unique_firms) = {corr_n:+.4f}")
    print(f"  Corr(novelty, log(n_firms))   = {corr_logn:+.4f}")
    print(f"  Original                      : {fmt(m4_orig, 'mean_novelty')}")
    print(f"  + log(n_firms)                : {fmt(m4_firm, 'mean_novelty')}, "
          f"firms coef {fmt(m4_firm, 'ln_n_firms')}")
    print(f"  + log(n) + log(n_firms)       : {fmt(m4_both, 'mean_novelty')}, "
          f"n coef {fmt(m4_both, 'ln_n')}, firms coef {fmt(m4_both, 'ln_n_firms')}")
    t4_tag, t4_msg = verdict_t4(corr_n, m4_orig, m4_firm, m4_both)
    print(f"  VERDICT: {t4_tag} — {t4_msg}")

    print("\n--- TEST 5: bootstrap with two distributions ---")
    coefs_emp, coefs_ln, mu, sigma = test_5_bootstrap_two_distributions(
        patents, cells, n_boot=N_BOOTSTRAP, seed=SEED)
    emp_summary = boot_summary(coefs_emp)
    ln_summary = boot_summary(coefs_ln)
    print(f"  log-normal fit: mu_log={mu:.4f}, sigma_log={sigma:.4f}")
    print(f"  Empirical : median {emp_summary['median']:+.4f}, "
          f"5-95: [{emp_summary['p05']:+.4f}, {emp_summary['p95']:+.4f}]")
    print(f"  Log-normal: median {ln_summary['median']:+.4f}, "
          f"5-95: [{ln_summary['p05']:+.4f}, {ln_summary['p95']:+.4f}]")
    t5_tag, t5_msg = verdict_t5(coefs_emp, coefs_ln)
    print(f"  VERDICT: {t5_tag} — {t5_msg}")

    # Top-level overall verdicts
    finding_overall = ("ROBUST" if all(t in ("ROBUST",) for t in [t1_tag, t2_tag, t4_tag])
                       else "FRAGILE" if any(t == "FRAGILE" for t in [t1_tag, t2_tag, t4_tag])
                       else "QUALIFIED")
    bootstrap_overall = t5_tag
    interp_overall = ("HIGH" if t3_tag == "ROBUST" and finding_overall == "ROBUST"
                      else "LOW" if t3_tag == "FRAGILE" or finding_overall == "FRAGILE"
                      else "MEDIUM")

    print("\n--- OVERALL ---")
    print(f"  +0.38 log(median) finding: {finding_overall}")
    print(f"  60% mechanical share:      {bootstrap_overall}")
    print(f"  Right-skew interpretation: {interp_overall}")

    # ---- markdown ----
    md = []
    md.append("# Right-skew validation: stress-testing the +0.38 log(median) finding")
    md.append("")
    md.append(f"_Cell = (cpc_subclass, grant_year), n_patents ≥ 10 (Tests 2-5). "
              f"{len(cells)} cells. HC1 robust SE._")
    md.append("")
    md.append("## FINDING ROBUSTNESS VERDICT")
    md.append("")
    md.append(f"- **+0.38 log(median) finding:** **{finding_overall}**")
    md.append(f"- **60% mechanical share:** **{bootstrap_overall}**")
    md.append(f"- **Overall confidence in the right-skew substantive interpretation:** "
              f"**{interp_overall}**")
    md.append("")
    md.append(f"_Permno is the publicly-traded parent firm linked by KPSS, "
              "not the raw assignee text. Subsidiaries with the same parent "
              "permno collapse to one. This is the closest firm-composition "
              "proxy available in the working dataset._")
    md.append("")

    md.append("## Test 1 — cutoff sensitivity for log(median)")
    md.append("")
    md.append(t1_table(t1))
    md.append("")
    md.append(f"**Verdict ({t1_tag}):** {t1_msg}")
    md.append("")

    md.append("## Test 2 — log(n) control on log(median)")
    md.append("")
    md.append("| Spec | mean_novelty coef | log(n) coef |")
    md.append("|---|---|---|")
    md.append(f"| Original                | {fmt(m2_orig, 'mean_novelty')} | — |")
    md.append(f"| + log(n_patents)        | {fmt(m2_with, 'mean_novelty')} | "
              f"{fmt(m2_with, 'ln_n')} |")
    md.append("")
    md.append(f"Survival: {m2_with.params['mean_novelty'] / m2_orig.params['mean_novelty'] * 100:.1f}% "
              f"of original coefficient.")
    md.append("")
    md.append(f"**Verdict ({t2_tag}):** {t2_msg}")
    md.append("")

    md.append("## Test 3 — alternative skewness measures")
    md.append("")
    md.append("If novel cells genuinely have less right-skew, all three "
              "skewness measures should be **negative** (less Pearson skewness, "
              "shorter relative right tail).")
    md.append("")
    md.append(t3_table(t3))
    md.append("")
    md.append(f"**Verdict ({t3_tag}):** {t3_msg}")
    md.append("")

    md.append("## Test 4 — firm-composition confound (permno proxy)")
    md.append("")
    md.append(f"- Corr(mean_novelty, n_unique_firms)        = {corr_n:+.4f}")
    md.append(f"- Corr(mean_novelty, log(n_unique_firms))   = {corr_logn:+.4f}")
    md.append("")
    md.append("| Spec | mean_novelty coef | log(n_firms) coef | log(n) coef |")
    md.append("|---|---|---|---|")
    md.append(f"| Original                | {fmt(m4_orig, 'mean_novelty')} | — | — |")
    md.append(f"| + log(n_firms)          | {fmt(m4_firm, 'mean_novelty')} | "
              f"{fmt(m4_firm, 'ln_n_firms')} | — |")
    md.append(f"| + log(n) + log(n_firms) | {fmt(m4_both, 'mean_novelty')} | "
              f"{fmt(m4_both, 'ln_n_firms')} | {fmt(m4_both, 'ln_n')} |")
    md.append("")
    md.append(f"**Verdict ({t4_tag}):** {t4_msg}")
    md.append("")

    md.append("## Test 5 — bootstrap log(SD) ~ novelty under two distributions")
    md.append("")
    md.append(f"_Pooled xi_real fitted to log-normal: mu_log = {mu:.4f}, "
              f"sigma_log = {sigma:.4f}. Bootstrap samples drawn iid per cell "
              f"from each distribution, novelty/n correlation preserved._")
    md.append("")
    md.append(boot_table(emp_summary, ln_summary))
    md.append("")
    md.append(f"**Verdict ({t5_tag}):** {t5_msg}")
    md.append("")

    md.append("## Caveats")
    md.append("")
    md.append("- **permno ≠ assignee.** permno covers the publicly-traded parent firm "
              "linked by KPSS. Subsidiaries / private subcontractors / non-public "
              "co-assignees collapse to a single permno or are omitted. A finer "
              "firm-composition test would use raw assignee text.")
    md.append("- **log-normal vs empirical.** The log-normal bootstrap uses moments "
              "matched to the pooled log(xi_real); KPSS values may have heavier "
              "(Pareto-like) right tails than log-normal predicts, in which case "
              "the empirical bootstrap is the more conservative number.")
    md.append("- **log(median+1) offset.** Median values in this sample are large "
              "enough that the +1 offset is essentially a no-op.")

    with open(MD_PATH, "w") as f:
        f.write("\n".join(md))
    print(f"\nSaved: {MD_PATH}")


if __name__ == "__main__":
    main()
