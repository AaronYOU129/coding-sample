"""
Purpose:    Robustness check for Cappelen, List, Samek & Tungodden (2020).
            Three layers of inference on the Table 3 treatment effects:
              (i)   Randomization Inference (RI) p-values to validate that
                    asymptotic OLS inference is well calibrated at N ~= 300.
              (ii)  Romano-Wolf (2005, 2016) FWER-adjusted p-values over a
                    pre-registered family of 6 hypotheses.
              (iii) Anderson (2008 JASA, Sec III.B) sharpened q-values over
                    the same 6-hypothesis family, delivering FDR control as
                    a complement to FWER.

            Why this check does not overlap Appendix A1-A9:
              - A4 (cluster-on-school SE) addresses within-school dependence
                but keeps the asymptotic OLS distribution. RI drops that
                assumption entirely.
              - Nothing in A1-A9 controls multiplicity. Table 3 reports ten
                p-values with no adjustment; Romano-Wolf and sharpened
                q-values close that gap under FWER and FDR respectively.

            Why the family is 6 (not 10):
              The paper's utility function V(y_i) = y_i - b_i*(y_i - m_i)^2
              - a_i*(X_i - maxX)^2 has three behavioural parameters:
                b_i     (self-interest)       -> Dictator
                b_i/a_i (efficiency vs fair)  -> Efficiency
                m_i     (fairness view)       -> Merit+Luck combined
              The paper explicitly uses the Merit+Luck *combined* index as
              the primary measure of m_i. Luck and Merit alone are the
              two components of that combined index and are highly
              within-child correlated -- including all three in the family
              would multiply-count m_i and inflate the family size.
              The pre-registered family therefore spans the three
              theoretical dimensions x {Preschool, Parent Academy} = 6.

Inputs:     data.csv             (loaded via code_q1.py helpers)

Outputs:    tables/table_q2.tex  Booktabs LaTeX fragment comparing, for each
                                 treatment x outcome x specification:
                                 beta, OLS p, RI p, RW adj p (family=6),
                                 sharpened q-value (family=6).

Key Steps:
    1. For each of the 10 Table 3 regressions, precompute QR of
       [intercept + time FE + experimenter FE + demographic controls]
       and residualize the outcome (Frisch-Waugh-Lovell setup). Each
       permutation then costs only two matrix-vector multiplies per spec.
    2. Fit observed treatment coefficients and t-statistics.
    3. Draw B = 5000 permutations of the treat label across the 302-row
       experimental sample and record 20 permuted t-statistics per draw
       (Preschool and Parent Academy coefs across 10 specs).
    4. RI two-sided p:
            p_RI = (#{|t_perm| >= |t_obs|} + 1) / (B + 1)
    5. Romano-Wolf stepdown over the 6-hypothesis family: the max-|t|
       null is recomputed using only the 6 family members (not filtered
       from a wider max).
    6. Anderson (2008) sharpened q-values over the same 6-hypothesis
       family, using OLS p-values as input (the applied-micro convention
       following Anderson 2008 and List-Shaikh-Xu 2019). This keeps a
       clean narrative split: RI validates OLS inference; q-values
       adjust for multiple testing.

How to Run: python3 code_q2.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm

from code_q1 import (
    DATA_PATH,
    DEMOGRAPHIC_CONTROLS,
    EXPERIMENTER_DUMMIES,
    OUTCOME_COLS,
    TIME_OF_DAY_DUMMIES,
    load_main_sample,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_PERM = 5000
RNG_SEED = 20260423
OUTPUT_PATH = Path(__file__).resolve().parent / "tables" / "table_q2.tex"

TREATMENT_LABELS = [("preschool", "Preschool"), ("parent_academy", "Parent Academy")]


# ---------------------------------------------------------------------------
# High-level workflow
# ---------------------------------------------------------------------------

def main() -> None:
    sample = load_main_sample(DATA_PATH)
    treat_vec = sample["treat"].to_numpy()

    specs = build_specs()
    prepped = [precompute_spec(sample, spec) for spec in specs]
    print(f"[data]     N = {len(sample)}; prepared {len(specs)} regression specs")

    validate_fwl_matches_statsmodels(sample, specs, prepped)
    print("[validate] FWL t-stats match statsmodels t-stats to 1e-6")

    observed_beta, observed_t = fit_all(prepped, treat_vec)
    ols_p = statsmodels_pvalues(sample, specs)

    print(f"[RI]       running {N_PERM} permutations ...")
    perm_t = run_permutations(prepped, treat_vec, N_PERM, RNG_SEED)

    family = pre_registered_family_indices(specs)
    print(f"[family]   {len(family)} hypotheses: "
          f"{', '.join(_describe_family_members(specs, family))}")

    ri_p = two_sided_pvalues(observed_t, perm_t)
    rw_p = romano_wolf_in_family(observed_t, perm_t, family=family)
    q_value = sharpened_qvalues_in_family(ols_p, family=family,
                                          n_hypotheses=len(observed_t))

    emit_latex(specs, observed_beta, ols_p, ri_p, rw_p, q_value, OUTPUT_PATH)
    print_summary(specs, observed_beta, ols_p, ri_p, rw_p, q_value)


# ---------------------------------------------------------------------------
# Specifications
# ---------------------------------------------------------------------------

def build_specs() -> list[dict]:
    """Return the 10 regression specs of Table 3 in a fixed order."""
    return [
        {"outcome_col": col, "outcome_label": label, "with_controls": with_ctrl}
        for col, label in OUTCOME_COLS
        for with_ctrl in (False, True)
    ]


FAMILY_OUTCOMES = {"Dictator", "Efficiency", "Merit and Luck"}


def pre_registered_family_indices(specs: Sequence[dict]) -> np.ndarray:
    """Indices (into the flat 20-hypothesis vector) of the 6 pre-registered
    hypotheses: three theoretical dimensions of the paper's utility function
    (b_i, b_i/a_i, m_i) x {Preschool, Parent Academy}, with-controls spec.

    Luck and Merit standalone are excluded because they are components of
    the Merit+Luck combined index (the paper's primary measure of m_i);
    including them would multiply-count the fairness-view dimension."""
    out = []
    for i, spec in enumerate(specs):
        if spec["with_controls"] and spec["outcome_label"] in FAMILY_OUTCOMES:
            out.extend([2 * i, 2 * i + 1])
    return np.array(sorted(out))


def _describe_family_members(specs: Sequence[dict], family: np.ndarray) -> list[str]:
    """Human-readable labels for each index in the family (for logging)."""
    labels = []
    for idx in family:
        spec = specs[idx // 2]
        treat = "PK" if idx % 2 == 0 else "PA"
        labels.append(f"{treat}->{spec['outcome_label']}")
    return labels


# ---------------------------------------------------------------------------
# Frisch-Waugh-Lovell preparation (once per spec, re-used per permutation)
# ---------------------------------------------------------------------------

def precompute_spec(sample: pd.DataFrame, spec: dict) -> dict:
    """Precompute the objects needed to evaluate PK/PA t-stats cheaply
    under any treat permutation."""
    controls = list(DEMOGRAPHIC_CONTROLS) if spec["with_controls"] else []
    candidate_fe = TIME_OF_DAY_DUMMIES + EXPERIMENTER_DUMMIES

    relevant_cols = [spec["outcome_col"]] + controls + candidate_fe
    data = sample.dropna(subset=[c for c in relevant_cols if c != spec["outcome_col"]]
                         + [spec["outcome_col"]]).copy()

    # A dummy that is identically 0 after the outcome-NA drop is rank-deficient
    # and must be excluded (its "singleton" observation was in the dropped row).
    fe = [c for c in candidate_fe if data[c].sum() > 0]

    X = np.column_stack([
        np.ones(len(data)),
        data[controls + fe].to_numpy(dtype=float),
    ])
    y = data[spec["outcome_col"]].to_numpy(dtype=float)

    # QR provides an orthonormal basis Q whose column space equals col(X).
    # Residualizing any vector v on X is v - Q @ (Q' v).
    q_matrix, _ = np.linalg.qr(X)
    y_residual = y - q_matrix @ (q_matrix.T @ y)

    row_mask = sample.index.isin(data.index)

    return {
        "row_mask": row_mask,
        "q_matrix": q_matrix,
        "y_residual": y_residual,
        "n": len(data),
        "n_other_regressors": X.shape[1],
    }


def treatment_tstats(prepped: dict, treat_vec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute (beta, t) for [preschool, parent_academy] via FWL."""
    masked_treat = treat_vec[prepped["row_mask"]]
    treatment_matrix = np.column_stack([
        (masked_treat == "PK").astype(float),
        (masked_treat == "PA").astype(float),
    ])

    t_tilde = treatment_matrix - prepped["q_matrix"] @ (prepped["q_matrix"].T @ treatment_matrix)
    gram = t_tilde.T @ t_tilde
    cross = t_tilde.T @ prepped["y_residual"]
    beta = np.linalg.solve(gram, cross)

    residual = prepped["y_residual"] - t_tilde @ beta
    dof = prepped["n"] - prepped["n_other_regressors"] - 2
    sigma2 = residual @ residual / dof
    variance = sigma2 * np.linalg.inv(gram)
    se = np.sqrt(np.diag(variance))
    return beta, beta / se


# ---------------------------------------------------------------------------
# Observed fit + sanity check
# ---------------------------------------------------------------------------

def fit_all(prepped_list: Sequence[dict], treat_vec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return flat (20,) arrays of betas and t-stats for all 10 specs."""
    betas = np.empty(2 * len(prepped_list))
    tstats = np.empty(2 * len(prepped_list))
    for i, prepped in enumerate(prepped_list):
        beta, tstat = treatment_tstats(prepped, treat_vec)
        betas[2 * i : 2 * i + 2] = beta
        tstats[2 * i : 2 * i + 2] = tstat
    return betas, tstats


def validate_fwl_matches_statsmodels(sample: pd.DataFrame,
                                     specs: Sequence[dict],
                                     prepped_list: Sequence[dict]) -> None:
    """Ensure our FWL implementation agrees with statsmodels on observed data."""
    treat_vec = sample["treat"].to_numpy()
    for spec, prepped in zip(specs, prepped_list):
        _, ours = treatment_tstats(prepped, treat_vec)
        sm_fit = _fit_statsmodels(sample, spec)
        sm_t = np.array([
            sm_fit.tvalues["preschool"],
            sm_fit.tvalues["parent_academy"],
        ])
        diff = np.max(np.abs(ours - sm_t))
        assert diff < 1e-6, f"FWL/statsmodels disagree by {diff:.2e} on {spec}"


def _fit_statsmodels(sample: pd.DataFrame, spec: dict):
    controls = list(DEMOGRAPHIC_CONTROLS) if spec["with_controls"] else []
    candidate_fe = TIME_OF_DAY_DUMMIES + EXPERIMENTER_DUMMIES
    data = sample[[spec["outcome_col"], "preschool", "parent_academy"]
                  + controls + candidate_fe].dropna()
    fe = [c for c in candidate_fe if data[c].sum() > 0]
    regressors = ["preschool", "parent_academy"] + controls + fe
    return sm.OLS(data[spec["outcome_col"]], sm.add_constant(data[regressors])).fit()


def statsmodels_pvalues(sample: pd.DataFrame, specs: Sequence[dict]) -> np.ndarray:
    out = np.empty(2 * len(specs))
    for i, spec in enumerate(specs):
        fit = _fit_statsmodels(sample, spec)
        out[2 * i] = fit.pvalues["preschool"]
        out[2 * i + 1] = fit.pvalues["parent_academy"]
    return out


# ---------------------------------------------------------------------------
# Permutations
# ---------------------------------------------------------------------------

def run_permutations(prepped_list: Sequence[dict], treat_vec: np.ndarray,
                     n_perm: int, seed: int) -> np.ndarray:
    """Return (20, n_perm) array of permuted t-stats."""
    rng = np.random.default_rng(seed)
    result = np.empty((2 * len(prepped_list), n_perm))
    for b in range(n_perm):
        perm_treat = rng.permutation(treat_vec)
        for i, prepped in enumerate(prepped_list):
            _, tstat = treatment_tstats(prepped, perm_treat)
            result[2 * i : 2 * i + 2, b] = tstat
        if (b + 1) % 1000 == 0:
            print(f"             ... {b + 1}/{n_perm}")
    return result


# ---------------------------------------------------------------------------
# P-value computations
# ---------------------------------------------------------------------------

def two_sided_pvalues(obs_t: np.ndarray, perm_t: np.ndarray) -> np.ndarray:
    """p = (#{|t_perm| >= |t_obs|} + 1) / (B + 1).  +1 avoids a zero p-value
    when no permutation is as extreme as the observed stat (standard
    permutation-test convention)."""
    abs_obs = np.abs(obs_t)[:, None]
    abs_perm = np.abs(perm_t)
    numerator = (abs_perm >= abs_obs).sum(axis=1) + 1
    denominator = perm_t.shape[1] + 1
    return numerator / denominator


def romano_wolf_in_family(obs_t: np.ndarray, perm_t: np.ndarray,
                          family: np.ndarray) -> np.ndarray:
    """Romano-Wolf stepdown FWER-adjusted p-values for the pre-specified
    family only. Positions outside the family get np.nan."""
    family_t = obs_t[family]
    family_perm = perm_t[family, :]

    adj_in_family = romano_wolf_stepdown(family_t, family_perm)

    all_p = np.full_like(obs_t, np.nan, dtype=float)
    all_p[family] = adj_in_family
    return all_p


def romano_wolf_stepdown(obs_t: np.ndarray, perm_t: np.ndarray) -> np.ndarray:
    """Stepdown Romano-Wolf (2005, 2016). Orders hypotheses by |t_obs|
    descending; for position k computes raw_p = fraction of permutation draws
    in which max |t_perm_j| over remaining hypotheses meets or exceeds the
    observed |t_obs_k|; then enforces monotonicity along the ordering.

    Note on family recomputation: `perm_t` is expected to already be
    sliced to the family of interest. The `remaining` subset therefore
    refers to family members only, so the max-|t| null is recomputed on
    the family rather than inherited from a larger collection."""
    m, n_perm = perm_t.shape
    abs_obs = np.abs(obs_t)
    abs_perm = np.abs(perm_t)

    order = np.argsort(-abs_obs)
    raw = np.empty(m)
    for position, j in enumerate(order):
        remaining = order[position:]
        max_perm = abs_perm[remaining, :].max(axis=0)
        raw[j] = ((max_perm >= abs_obs[j]).sum() + 1) / (n_perm + 1)

    adj = np.empty(m)
    running_max = 0.0
    for j in order:
        running_max = max(running_max, raw[j])
        adj[j] = running_max
    return adj


# ---------------------------------------------------------------------------
# Anderson (2008) sharpened q-values (BKY two-stage FDR)
# ---------------------------------------------------------------------------

def sharpened_qvalues_in_family(pvals: np.ndarray, family: np.ndarray,
                                n_hypotheses: int,
                                alpha: float = 0.05) -> np.ndarray:
    """Apply sharpened q-values only to the pre-registered family.
    Positions outside the family get np.nan."""
    q_in_family = sharpened_qvalues(pvals[family], alpha=alpha)
    out = np.full(n_hypotheses, np.nan, dtype=float)
    out[family] = q_in_family
    return out


def sharpened_qvalues(p: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Anderson (2008 JASA, Sec III.B) sharpened q-values via the
    Benjamini-Krieger-Yekutieli (2006) two-stage FDR procedure.

    Stage 1. Apply Benjamini-Hochberg at level alpha / (1 + alpha) to
             estimate the number of true nulls. R1 is the number of
             stage-one rejections; the estimated true-null count is
             m* = m - R1 (Anderson uses the BKY plug-in, not Storey's
             (m + 1) - R1 correction).
    Stage 2. Sharpened q-value for the i-th smallest p is
                 q(i) = min_{k >= i} m* * p_(k) / k,
             with monotonicity enforced along the ranking and the final
             values capped at 1.

    The input `p` should be the raw p-values for the family only. We use
    OLS p-values as input (applied-micro convention per Anderson 2008 and
    List, Shaikh & Xu 2019) so that the q-value column reads as a
    multiple-testing correction *applied to* standard inference, while
    RI p-values separately attest to the validity of that inference.

    Edge cases:
    - If R1 == m (every hypothesis rejected at stage 1), m* = 0 and every
      sharpened q-value collapses to 0. Cap [0, 1] keeps that sensible.
    - If R1 == 0 (no stage-1 rejections), m* = m and the procedure reduces
      to the ordinary Benjamini-Hochberg q-value."""
    m = len(p)
    order = np.argsort(p)
    p_sorted = p[order]
    ranks = np.arange(1, m + 1, dtype=float)

    stage1_threshold = ranks * alpha / (m * (1.0 + alpha))
    stage1_rejected = np.where(p_sorted <= stage1_threshold)[0]
    r1 = int(stage1_rejected[-1]) + 1 if stage1_rejected.size else 0
    m_star = m - r1

    raw = m_star * p_sorted / ranks
    # Enforce monotonicity: q is non-decreasing in rank, so at rank i take
    # the min of raw[i..m-1]. Equivalent to a cummin from the right.
    monotone = np.minimum.accumulate(raw[::-1])[::-1]
    clipped = np.clip(monotone, 0.0, 1.0)

    out = np.empty(m)
    out[order] = clipped
    return out


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def emit_latex(specs: Sequence[dict], beta: np.ndarray, ols_p: np.ndarray,
               ri_p: np.ndarray, rw_p: np.ndarray, q_value: np.ndarray,
               path: Path) -> None:
    header = (
        r"\begin{tabular}{llcrrrrr}" "\n"
        r"\toprule" "\n"
        r"Outcome & Treatment & Spec & $\hat\beta$ & OLS $p$ & RI $p$ & "
        r"RW adj.\ $p$ & $q$-value \\" "\n"
        r"\midrule" "\n"
    )
    body = []
    for spec_i, spec in enumerate(specs):
        for j, (_, treat_label) in enumerate(TREATMENT_LABELS):
            k = 2 * spec_i + j
            spec_label = "w/ ctrls" if spec["with_controls"] else "no ctrls"
            body.append(
                f"{spec['outcome_label']} & {treat_label} & {spec_label} & "
                f"{beta[k]:+.3f} & {_fmt_p(ols_p[k])} & {_fmt_p(ri_p[k])} & "
                f"{_fmt_p(rw_p[k])} & {_fmt_p(q_value[k])} \\\\"
            )
    footer = r"\bottomrule" "\n" r"\end{tabular}" "\n"

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + "\n".join(body) + "\n" + footer)


def _fmt_p(p: float) -> str:
    if np.isnan(p):
        return "--"
    if p < 1 / (N_PERM + 1):
        return f"$<${1 / (N_PERM + 1):.4f}"
    return f"{p:.3f}"


def print_summary(specs: Sequence[dict], beta: np.ndarray, ols_p: np.ndarray,
                  ri_p: np.ndarray, rw_p: np.ndarray,
                  q_value: np.ndarray) -> None:
    print()
    print("=" * 107)
    print(f"{'Outcome':<18}{'Treatment':<18}{'Spec':<12}"
          f"{'β':>10}{'OLS p':>10}{'RI p':>10}{'RW adj. p':>12}{'q-value':>12}")
    print("-" * 107)
    for spec_i, spec in enumerate(specs):
        for j, (_, treat_label) in enumerate(TREATMENT_LABELS):
            k = 2 * spec_i + j
            rw_str = "--" if np.isnan(rw_p[k]) else f"{rw_p[k]:.3f}"
            q_str = "--" if np.isnan(q_value[k]) else f"{q_value[k]:.3f}"
            spec_str = "w/ ctrls" if spec["with_controls"] else "no ctrls"
            print(f"{spec['outcome_label']:<18}{treat_label:<18}{spec_str:<12}"
                  f"{beta[k]:>+10.3f}{ols_p[k]:>10.3f}{ri_p[k]:>10.3f}"
                  f"{rw_str:>12}{q_str:>12}")


if __name__ == "__main__":
    main()
