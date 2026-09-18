# Purpose:    Validate the price decomposition without touching the estimates.
#             Each figure targets one way the method could fail: fig1 tests
#             the multiplicative structure itself (large residuals = prices do
#             not factor into size x index); fig2 is an external sanity check
#             (drinks/desserts should be smaller than entrees, which should
#             center on 1); fig3 checks that indices move smoothly rather than
#             with composition-driven spikes. The summary table flags data
#             issues (flat-price restaurants, disconnected product-date graphs,
#             volatile indices) that a user of these outputs should know about.
# Inputs:     work/orders_clean.csv, work/residuals.csv, work/fit_stats.csv,
#             output/product_sizes.csv, output/price_indices.csv
# Outputs:    figures/fig1_fit_quality.png, figures/fig2_sizes_by_category.png,
#             figures/fig3_price_indices.png, output/diagnostics_summary.csv
# Key Steps:  Load estimation outputs -> per-restaurant fit quality ->
#             category-level size sanity -> index time series for the busiest
#             restaurants -> assemble summary table with issue flags.
# How to Run: python diagnostics.py   (from the task1/ directory; run
#             prepare_data.py and estimate.py first), or python run_all.py.

import matplotlib

# Select the non-interactive backend BEFORE pyplot is imported, so a headless
# machine never tries to initialize a GUI backend.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import paths  # noqa: E402
import prepare_data  # noqa: E402

CATEGORY_ORDER = ["drink", "dessert", "side", "appetizer", "entree",
                  "uncertain", "unclassified"]
N_INDEX_PANELS = 6
VOLATILE_INDEX_CV = 0.25
SPARSE_DAY_MEDIAN_OBS = 5
SINGLE_OBS_SHARE_LIMIT = 0.25


def main() -> None:
    paths.ensure_directories()
    orders = prepare_data.read_clean_orders()
    residuals = pd.read_csv(paths.RESIDUALS)
    fit_stats = pd.read_csv(paths.FIT_STATS)
    # keep_default_na=False: the "nan" product_key must stay a string, not NaN.
    sizes = pd.read_csv(paths.PRODUCT_SIZES, keep_default_na=False, na_values=[""])
    indices = pd.read_csv(paths.PRICE_INDICES)

    plot_fit_quality(fit_stats, residuals)
    plot_sizes_by_category(sizes)
    plot_price_indices(indices, fit_stats)

    summary = build_summary(orders, fit_stats, indices)
    summary.to_csv(paths.DIAGNOSTICS_SUMMARY, index=False)

    print(summary[["flag_flat_index", "flag_disconnected", "flag_volatile_index",
                   "flag_sparse_days", "flag_many_single_obs"]].sum())
    print("\nMedian size by category:")
    print(sizes.groupby("category")["size"].median().sort_values().round(3))
    print(f"\nFigures written to {paths.FIGURES_DIR}")


def plot_fit_quality(fit_stats: pd.DataFrame, residuals: pd.DataFrame) -> None:
    fig, (ax_r2, ax_resid) = plt.subplots(1, 2, figsize=(11, 4))
    ax_r2.hist(fit_stats["r_squared"], bins=30)
    ax_r2.set_xlabel("R-squared")
    ax_r2.set_ylabel("restaurants")
    ax_r2.set_title("Fit of log p = product FE + date FE, by restaurant")

    nonzero = residuals.loc[residuals["residual"].abs() > 1e-12, "residual"]
    ax_resid.hist(nonzero, bins=60)
    ax_resid.set_xlabel("residual (log points)")
    ax_resid.set_title(f"Nonzero residuals ({len(nonzero):,} of {len(residuals):,} lines)")
    fig.tight_layout()
    fig.savefig(paths.FIGURES_DIR / "fig1_fit_quality.png", dpi=150)
    plt.close(fig)


def plot_sizes_by_category(sizes: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    # Only categories present in the data: boxplot rejects empty groups.
    present = [cat for cat in CATEGORY_ORDER if (sizes["category"] == cat).any()]
    groups = [np.log(sizes.loc[sizes["category"] == cat, "size"])
              for cat in present]
    # Labels set separately: boxplot's label argument was renamed between
    # Matplotlib versions (labels -> tick_labels); this works on all of them.
    ax.boxplot(groups)
    ax.set_xticklabels(present)
    ax.axhline(0.0, linestyle="--", linewidth=1)  # log(size) = 0 <=> one entree unit
    ax.set_ylabel("log(size)")
    ax.set_title("Product sizes by category (dashed line = average entree)")
    fig.tight_layout()
    fig.savefig(paths.FIGURES_DIR / "fig2_sizes_by_category.png", dpi=150)
    plt.close(fig)


def plot_price_indices(indices: pd.DataFrame, fit_stats: pd.DataFrame) -> None:
    # The busiest restaurants have the best-identified indices; showing them
    # is a fair test of smoothness (sparse restaurants are flagged in the
    # summary table instead).
    busiest = fit_stats.nlargest(N_INDEX_PANELS, "n_obs")["restaurant_id"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 6), sharex=True)
    # strict=False: fewer restaurants than panels is fine (extras stay blank).
    for ax, restaurant_id in zip(axes.ravel(), busiest, strict=False):
        series = indices[indices["restaurant_id"] == restaurant_id]
        ax.plot(pd.to_datetime(series["date"]), series["price_index"])
        ax.set_title(f"restaurant {restaurant_id}")
        ax.tick_params(axis="x", rotation=45)
    fig.suptitle("Price index (euros per entree-equivalent), 6 busiest restaurants")
    fig.tight_layout()
    fig.savefig(paths.FIGURES_DIR / "fig3_price_indices.png", dpi=150)
    plt.close(fig)


def build_summary(orders: pd.DataFrame, fit_stats: pd.DataFrame,
                  indices: pd.DataFrame) -> pd.DataFrame:
    prices_per_product = orders.groupby(
        ["restaurant_id", "product_key"])["product_unit_price"].nunique()
    share_repriced = prices_per_product.gt(1).groupby("restaurant_id").mean() \
                                       .rename("share_products_repriced")

    obs_per_product = orders.groupby(["restaurant_id", "product_key"]).size()
    share_single_obs = obs_per_product.eq(1).groupby("restaurant_id").mean() \
                                      .rename("share_single_obs_products")
    median_obs_per_day = orders.groupby(["restaurant_id", "date"]).size() \
                               .groupby("restaurant_id").median() \
                               .rename("median_obs_per_day")

    # CV within the main component only: across components the index level is
    # arbitrary, so pooled volatility would mix identification artifacts with
    # real price movement.
    main_component = indices[indices["component_id"] == 0]
    index_cv = main_component.groupby("restaurant_id")["price_index"] \
                             .agg(lambda s: s.std() / s.mean()).rename("index_cv")

    summary = fit_stats.merge(share_repriced, on="restaurant_id") \
                       .merge(share_single_obs, on="restaurant_id") \
                       .merge(median_obs_per_day, on="restaurant_id") \
                       .merge(index_cv, on="restaurant_id")
    # Flags, not exclusions: users of the index should know where it is flat
    # by construction, unidentified across components, unusually volatile,
    # thinly identified per day, or dominated by single-observation sizes.
    summary["flag_flat_index"] = summary["share_products_repriced"] == 0
    summary["flag_disconnected"] = summary["n_components"] > 1
    summary["flag_volatile_index"] = summary["index_cv"] > VOLATILE_INDEX_CV
    summary["flag_sparse_days"] = summary["median_obs_per_day"] < SPARSE_DAY_MEDIAN_OBS
    summary["flag_many_single_obs"] = (
        summary["share_single_obs_products"] > SINGLE_OBS_SHARE_LIMIT)
    return summary


if __name__ == "__main__":  # pragma: no cover
    main()
