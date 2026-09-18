"""
Purpose:    Produce the Q3 deliverable -- a single publication-quality
            coefficient plot (forest plot) summarizing the 10 treatment
            effects in the with-controls (even-column) specifications of
            Table 3 from Cappelen, List, Samek & Tungodden (2020).

            Design rationale:
            - Five outcomes are unordered categorical (four parallel
              experiments plus one combined spec), so the five outcome
              groups are *not* connected by a line.
            - Each outcome group holds two points, Preschool on top and
              Parent Academy on bottom, with a visual gap between groups.
            - Horizontal error bars show 95% confidence intervals
              (coefficient +- 1.96 * SE); a vertical dashed reference
              line at x = 0 lets a conference audience read significance
              at a glance.
            - Marker style encodes significance in three tiers that mirror
              the paper's star convention:
                * p < 0.05         -> solid marker, full opacity
                * 0.05 <= p < 0.10 -> solid marker, alpha = 0.5 (faded)
                * p >= 0.10        -> hollow marker
              Error bars are always drawn at full opacity so that the
              faded (marginal) tier doesn't wash out the CI line; only
              the marker's fill and alpha encode the significance tier.
              This three-tier scheme lets a marginal headline finding
              (PS -> Luck, p ~ 0.05-0.10) be visually distinguished from
              both the strong results (PA -> Efficiency, PS -> Merit+Luck)
              and the nulls.
            - The X range is symmetric and wide enough to show the full
              CI for PA -> Efficiency (upper end ~ 0.21) without clipping.

Inputs:     data.csv                 (loaded via code_q1.py helpers)

Outputs:    figures/coefplot.pdf     Vector PDF to be embedded into the
                                     technical analysis report.

Key Steps:
    1. Re-fit the 5 with-controls Table 3 regressions by calling Q1's
       helper `run_regression` with the demographic-controls argument.
       (We do not pickle fits, but re-fitting is cheap and keeps a
       single source of truth in code_q1.py.)
    2. Extract the preschool and parent_academy coefficients and
       standard errors from each fit.
    3. Lay out 5 outcome groups x 2 treatments as 10 rows, with
       Dictator at the top of the plot and Merit+Luck at the bottom.
    4. Save the figure as a vector PDF.

How to Run: python3 code_q3.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from code_q1 import (
    DATA_PATH,
    DEMOGRAPHIC_CONTROLS,
    OUTCOME_COLS,
    load_main_sample,
    run_regression,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FIGURE_DIR = Path(__file__).resolve().parent / "figures"
PDF_PATH = FIGURE_DIR / "coefplot.pdf"

CI_MULTIPLIER = 1.96                 # 95% CI under normal approximation
P_STRONG = 0.05                      # p < P_STRONG -> solid full opacity
P_MARGINAL = 0.10                    # P_STRONG <= p < P_MARGINAL -> faded
MARGINAL_ALPHA = 0.5                 # opacity used for the marginal tier
X_LIMIT = (-0.25, 0.25)
FIG_SIZE = (10, 6)

GROUP_SPACING = 3.0                  # vertical distance between outcome groups
INTRA_GROUP_OFFSET = 0.4             # half-distance between PS and PA points

COLOR_PRESCHOOL = "C0"
COLOR_PARENT_ACADEMY = "C1"

# Labels as printed on the y-axis; matches the paper's Table 3 column header
# "Merit and Luck" for the combined index.
Y_AXIS_LABELS = {
    "Dictator": "Dictator",
    "Efficiency": "Efficiency",
    "Luck": "Luck",
    "Merit": "Merit",
    "Merit and Luck": "Merit and Luck",
}


# ---------------------------------------------------------------------------
# High-level workflow
# ---------------------------------------------------------------------------

def main() -> None:
    estimates = collect_with_controls_estimates()
    log_estimates(estimates)

    figure = build_coefficient_plot(estimates)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(PDF_PATH, bbox_inches="tight", pad_inches=0.1)
    plt.close(figure)

    print(f"[done] wrote {PDF_PATH}")


# ---------------------------------------------------------------------------
# Data extraction: pull the 10 coefficients from the even-column specs
# ---------------------------------------------------------------------------

def collect_with_controls_estimates() -> list[dict]:
    """Re-fit the five with-controls Table 3 regressions and extract the
    preschool and parent_academy coefficients, standard errors, and
    two-sided OLS p-values."""
    sample = load_main_sample(DATA_PATH)
    estimates = []
    for outcome_col, outcome_label in OUTCOME_COLS:
        result = run_regression(
            sample,
            outcome=outcome_col,
            outcome_label=outcome_label,
            extra_controls=list(DEMOGRAPHIC_CONTROLS),
        )
        fit = result["fit"]
        estimates.append({
            "label": outcome_label,
            "ps_coef": float(fit.params["preschool"]),
            "ps_se": float(fit.bse["preschool"]),
            "ps_p": float(fit.pvalues["preschool"]),
            "pa_coef": float(fit.params["parent_academy"]),
            "pa_se": float(fit.bse["parent_academy"]),
            "pa_p": float(fit.pvalues["parent_academy"]),
        })
    return estimates


def log_estimates(estimates: list[dict]) -> None:
    print(f"[estimates] with-controls Table 3 treatment effects "
          f"(solid/faded/hollow tiers at p<{P_STRONG}, <{P_MARGINAL}, >=): ")
    for e in estimates:
        ps_mark = _tier_for(e["ps_p"])
        pa_mark = _tier_for(e["pa_p"])
        print(f"  {e['label']:<18s} "
              f"PS = {e['ps_coef']:+.3f} ({e['ps_se']:.3f})  p={e['ps_p']:.4f}  [{ps_mark}]   "
              f"PA = {e['pa_coef']:+.3f} ({e['pa_se']:.3f})  p={e['pa_p']:.4f}  [{pa_mark}]")


def _tier_for(p_value: float) -> str:
    """Return 'strong', 'marginal', or 'null' for the three visual tiers."""
    if p_value < P_STRONG:
        return "strong"
    if p_value < P_MARGINAL:
        return "marginal"
    return "null"


# ---------------------------------------------------------------------------
# Plot construction
# ---------------------------------------------------------------------------

def build_coefficient_plot(estimates: list[dict]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    group_centers = _group_y_centers(len(estimates))

    for i, est in enumerate(estimates):
        _draw_group(ax, est, center_y=group_centers[i])

    _draw_reference_line(ax)
    _configure_axes(ax, estimates, group_centers)
    _add_legend(ax)
    _apply_spine_styling(ax)
    _apply_title(ax)

    fig.tight_layout()
    return fig


def _group_y_centers(n_groups: int) -> list[float]:
    """Place groups so Dictator sits at the top and Merit+Luck at the bottom.
    With matplotlib's default axes, larger y means higher on the plot."""
    return [GROUP_SPACING * (n_groups - 1 - i) for i in range(n_groups)]


def _draw_group(ax: plt.Axes, estimate: dict, center_y: float) -> None:
    """Draw the Preschool point above and the Parent Academy point below
    the group's centre. Error bars are 95% normal-approximation CIs."""
    ps_y = center_y + INTRA_GROUP_OFFSET
    pa_y = center_y - INTRA_GROUP_OFFSET

    _draw_point(
        ax, x=estimate["ps_coef"], y=ps_y,
        xerr=CI_MULTIPLIER * estimate["ps_se"],
        color=COLOR_PRESCHOOL, p_value=estimate["ps_p"],
    )
    _draw_point(
        ax, x=estimate["pa_coef"], y=pa_y,
        xerr=CI_MULTIPLIER * estimate["pa_se"],
        color=COLOR_PARENT_ACADEMY, p_value=estimate["pa_p"],
    )


def _draw_point(ax: plt.Axes, x: float, y: float, xerr: float,
                color: str, p_value: float) -> None:
    tier = _tier_for(p_value)

    # Step 1: error bar line, always full opacity.
    ax.errorbar(
        x, y, xerr=xerr,
        fmt="none",
        ecolor=color, capsize=3, linewidth=1.2,
        alpha=1.0,
        zorder=2,
    )

    # Step 2: marker. For the marginal tier, lay down an opaque white
    # disc first so the error-bar line doesn't show through the faded
    # coloured marker on top.
    if tier == "strong":
        ax.plot(x, y, marker="o", markersize=8,
                markerfacecolor=color, markeredgecolor=color,
                markeredgewidth=1.5, linestyle="", zorder=3)
    elif tier == "marginal":
        # White underlay to hide the line segment behind the marker.
        ax.plot(x, y, marker="o", markersize=8,
                markerfacecolor="white", markeredgecolor="none",
                linestyle="", zorder=3)
        # Faded coloured marker on top.
        ax.plot(x, y, marker="o", markersize=8,
                markerfacecolor=color, markeredgecolor=color,
                markeredgewidth=1.5, linestyle="",
                alpha=MARGINAL_ALPHA, zorder=4)
    else:  # null: hollow
        ax.plot(x, y, marker="o", markersize=8,
                markerfacecolor="white", markeredgecolor=color,
                markeredgewidth=1.5, linestyle="", zorder=3)

def _draw_reference_line(ax: plt.Axes) -> None:
    ax.axvline(0, color="gray", linestyle="--", alpha=0.6, linewidth=0.8)


def _configure_axes(ax: plt.Axes, estimates: list[dict],
                    group_centers: list[float]) -> None:
    ax.set_xlim(*X_LIMIT)
    ax.set_xlabel("Treatment coefficient on implemented inequality",
                  fontsize=12)
    ax.tick_params(axis="x", labelsize=10)

    ax.set_yticks(group_centers)
    ax.set_yticklabels(
        [Y_AXIS_LABELS[e["label"]] for e in estimates],
        fontsize=10,
    )

    top = max(group_centers) + 1.5
    bottom = min(group_centers) - 1.5
    ax.set_ylim(bottom, top)


def _add_legend(ax: plt.Axes) -> None:
    """Two coexisting legends: treatment colours at lower right (unchanged
    from the binary version), significance tiers at upper right (new)."""
    color_handles = [
        plt.Line2D([], [], marker="o", color=COLOR_PRESCHOOL, linestyle="",
                   markersize=8, label="Preschool"),
        plt.Line2D([], [], marker="o", color=COLOR_PARENT_ACADEMY, linestyle="",
                   markersize=8, label="Parent Academy"),
    ]
    color_legend = ax.legend(handles=color_handles, loc="lower right",
                             frameon=False, fontsize=10)
    # Keep the first legend on the axes before creating the second one --
    # otherwise the second legend replaces it entirely.
    ax.add_artist(color_legend)

    sig_marker_color = "black"
    sig_handles = [
        plt.Line2D([], [], marker="o", color=sig_marker_color, linestyle="",
                   markersize=8, label="p < 0.05"),
        plt.Line2D([], [], marker="o", color=sig_marker_color, linestyle="",
                   markersize=8, alpha=MARGINAL_ALPHA,
                   label="0.05 ≤ p < 0.10"),
        plt.Line2D([], [], marker="o", linestyle="",
                   markerfacecolor="white", markeredgecolor=sig_marker_color,
                   markeredgewidth=1.5, markersize=8,
                   label="p ≥ 0.10"),
    ]
    ax.legend(handles=sig_handles, loc="upper right",
              frameon=False, fontsize=10)


def _apply_spine_styling(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _apply_title(ax: plt.Axes) -> None:
    ax.set_title("Estimated treatment effects across five experiments",
                 fontsize=14)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
