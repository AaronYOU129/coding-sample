# Purpose:    Decompose product prices into a product-specific size and a
#             restaurant-date price index via a two-way fixed-effects
#             regression of log price on product and date dummies, run
#             separately for each restaurant. Log prices make the assumed
#             multiplicative structure (price = size x index) additive and
#             estimable by OLS. Unlike a naive daily average price, product
#             fixed effects absorb the stable cross-product price differences,
#             so the index is identified only from within-product price
#             changes and is immune to day-to-day composition shifts.
#             Estimation uses the normalized product_key (case/whitespace
#             variants are one product); the modal raw name is kept for
#             display. Because fixed-effect LEVELS are only identified within
#             a connected component of the product-date graph, normalization
#             is done PER COMPONENT (entree mean size = 1 where the component
#             contains entrees, component mean size = 1 otherwise) and every
#             output row carries component_id and anchor so users never
#             compare sizes or indices across components.
# Inputs:     work/orders_clean.csv  (cleaned order lines from prepare_data.py)
# Outputs:    output/product_sizes.csv   (one row per restaurant-product_key)
#             output/price_indices.csv   (one row per restaurant-date)
#             work/fit_stats.csv         (per-restaurant fit summary)
#             work/residuals.csv         (order-line fit for diagnostics)
# Key Steps:  For each restaurant: build dummy design matrix (first date is
#             the baseline) -> least squares on log price -> label connected
#             components of the product-date graph (union-find) -> normalize
#             within each component -> export.
# How to Run: python estimate.py   (from the task1/ directory; run
#             prepare_data.py first), or python run_all.py.

from typing import Any

import numpy as np
import pandas as pd

import paths
import prepare_data

ENTREE_MEAN_TOLERANCE = 1e-8


def main() -> None:
    paths.ensure_directories()
    orders = prepare_data.read_clean_orders()
    display_names = modal_value_by_product(orders, "product_name")
    categories = modal_value_by_product(orders, "category")

    size_rows, index_rows, stat_rows, residual_frames = [], [], [], []
    for restaurant_id, orders_r in orders.groupby("restaurant_id"):
        fit = estimate_restaurant(orders_r)
        # Use the same modal category that is exported below, so the reported
        # entree group and the normalization group cannot disagree when raw
        # rows contain conflicting category labels for one product.
        entree_products = {
            product for product in fit["products"]
            if categories[(restaurant_id, product)] == "entree"
        }
        product_table, date_table = normalize(fit, entree_products)
        validate_reconstruction(fit, product_table, date_table, orders_r)

        obs_per_product = orders_r["product_key"].value_counts()
        for product_key, row in product_table.iterrows():
            size_rows.append({
                "restaurant_id": restaurant_id,
                "product_name": display_names[(restaurant_id, product_key)],
                "product_key": product_key,
                "category": categories[(restaurant_id, product_key)],
                "size": row["size"], "component_id": int(row["component_id"]),
                "anchor": row["anchor"], "n_obs": obs_per_product[product_key],
            })

        obs_per_date = orders_r["date"].value_counts()
        for date, row in date_table.iterrows():
            index_rows.append({
                "restaurant_id": restaurant_id, "date": date,
                "price_index": row["price_index"],
                "component_id": int(row["component_id"]),
                "anchor": row["anchor"], "n_obs": obs_per_date[date],
            })

        stat_rows.append({
            "restaurant_id": restaurant_id, "n_obs": len(orders_r),
            "n_products": len(fit["products"]), "n_days": len(fit["dates"]),
            "r_squared": fit["r_squared"],
            "rmse_log_points": np.sqrt(np.mean(fit["residual"] ** 2)),
            "n_components": fit["n_components"],
        })

        residual_frames.append(pd.DataFrame({
            "restaurant_id": restaurant_id,
            "product_key": orders_r["product_key"].to_numpy(),
            "date": orders_r["date"].to_numpy(),
            "log_price": np.log(orders_r["product_unit_price"].to_numpy()),
            "fitted": fit["fitted"], "residual": fit["residual"],
        }))

    pd.DataFrame(size_rows).to_csv(paths.PRODUCT_SIZES, index=False)
    pd.DataFrame(index_rows).to_csv(paths.PRICE_INDICES, index=False)
    fit_stats = pd.DataFrame(stat_rows)
    fit_stats.to_csv(paths.FIT_STATS, index=False)
    pd.concat(residual_frames).to_csv(paths.RESIDUALS, index=False)

    print(f"Restaurants estimated: {len(stat_rows)}")
    print(f"Product sizes written: {len(size_rows)}")
    print(f"Price indices written: {len(index_rows)}")
    print(f"Median R^2: {fit_stats['r_squared'].median():.4f}")
    print(f"Restaurants with disconnected product-date graph: "
          f"{(fit_stats['n_components'] > 1).sum()}")


def build_design_matrix(products_idx: np.ndarray, dates_idx: np.ndarray,
                        n_products: int, n_dates: int) -> np.ndarray:
    # One dummy per product, one per date except the first: with both full
    # sets the matrix would be collinear (a constant shifts freely between
    # them). Dropping the first date makes it the baseline; the arbitrary
    # level this introduces is removed later by the per-component normalization.
    n_obs = len(products_idx)
    design = np.zeros((n_obs, n_products + n_dates - 1))
    design[np.arange(n_obs), products_idx] = 1.0
    later_dates = dates_idx > 0
    design[np.where(later_dates)[0], n_products + dates_idx[later_dates] - 1] = 1.0
    return design


def component_labels(product_indices: np.ndarray, date_indices: np.ndarray,
                     product_count: int,
                     date_count: int) -> tuple[np.ndarray, np.ndarray]:
    # Date effects are comparable across days only if products link the days
    # together (same logic as repeat-sales indices). Union-find over the
    # bipartite product-date graph; separate components mean levels are only
    # identified within each component, so downstream normalization must be
    # done per component. Returned labels are sequential ids with the
    # highest-observation component labeled 0.
    parent_nodes = list(range(product_count + date_count))

    def find_root(node: int) -> int:
        while parent_nodes[node] != node:
            parent_nodes[node] = parent_nodes[parent_nodes[node]]
            node = parent_nodes[node]
        return node

    for product_index, date_index in zip(product_indices, date_indices, strict=True):
        product_root, date_root = (
            find_root(product_index), find_root(product_count + date_index))
        if product_root != date_root:
            parent_nodes[product_root] = date_root

    product_roots = np.array([find_root(node) for node in range(product_count)])
    date_roots = np.array(
        [find_root(product_count + node) for node in range(date_count)])
    # Rank components by number of order lines so component 0 is the main one.
    roots_by_observation_count = pd.Series(
        product_roots[product_indices]).value_counts().index
    root_to_component_id = {
        root: component_id
        for component_id, root in enumerate(roots_by_observation_count)
    }
    return (np.array([root_to_component_id[root] for root in product_roots]),
            np.array([root_to_component_id[root] for root in date_roots]))


def estimate_restaurant(orders_r: pd.DataFrame) -> dict[str, Any]:
    products = np.sort(orders_r["product_key"].unique())
    dates = np.sort(orders_r["date"].unique())
    # .codes is int8 for small category sets; cast up so that column
    # arithmetic like n_products + date_idx cannot overflow.
    products_idx = pd.Categorical(orders_r["product_key"],
                                  categories=products).codes.astype(np.intp)
    dates_idx = pd.Categorical(orders_r["date"],
                               categories=dates).codes.astype(np.intp)
    log_price = np.log(orders_r["product_unit_price"].to_numpy())

    design = build_design_matrix(products_idx, dates_idx, len(products), len(dates))
    coef, _, _, _ = np.linalg.lstsq(design, log_price, rcond=None)
    alpha = coef[: len(products)]                      # log size + baseline level
    delta = np.concatenate([[0.0], coef[len(products):]])  # log index vs first date

    fitted = design @ coef
    residual = log_price - fitted
    total_ss = np.sum((log_price - log_price.mean()) ** 2)
    r_squared = 1.0 - np.sum(residual ** 2) / total_ss if total_ss > 0 else np.nan

    product_component, date_component = component_labels(
        products_idx, dates_idx, len(products), len(dates))
    return {
        "products": products, "dates": dates, "alpha": alpha, "delta": delta,
        "fitted": fitted, "residual": residual, "r_squared": r_squared,
        "product_component": product_component, "date_component": date_component,
        "n_components": int(product_component.max()) + 1,
    }


def normalize(fit: dict[str, Any],
              entree_products: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    # The regression pins down only relative effects, and only WITHIN a
    # connected component. Rescale each component separately: entree mean
    # size = 1 where the component has entrees (index = euros per
    # entree-equivalent), component mean size = 1 otherwise (anchor column
    # marks these as not comparable in entree units). The offsetting factor
    # moves into the index so size * index still reproduces fitted prices.
    raw_sizes = pd.Series(np.exp(fit["alpha"]), index=fit["products"])
    raw_index = pd.Series(np.exp(fit["delta"]), index=fit["dates"])
    product_component = pd.Series(fit["product_component"], index=fit["products"])
    date_component = pd.Series(fit["date_component"], index=fit["dates"])
    assert raw_sizes.index.isin(entree_products).any(), \
        "no entree products; normalization impossible"

    sizes = raw_sizes.copy()
    index = raw_index.copy()
    anchors: dict[int, str] = {}
    for component in sorted(product_component.unique()):
        in_component = (product_component == component).to_numpy()
        is_entree = in_component & raw_sizes.index.isin(entree_products)
        anchors[component] = "entree_mean" if is_entree.any() else "component_mean"
        scale = raw_sizes[is_entree if is_entree.any() else in_component].mean()
        sizes[in_component] = raw_sizes[in_component] / scale
        on_dates = (date_component == component).to_numpy()
        index[on_dates] = raw_index[on_dates] * scale
        if anchors[component] == "entree_mean":
            assert abs(sizes[is_entree].mean() - 1.0) < ENTREE_MEAN_TOLERANCE

    product_table = pd.DataFrame({
        "size": sizes, "component_id": product_component,
        "anchor": product_component.map(anchors),
    })
    date_table = pd.DataFrame({
        "price_index": index, "component_id": date_component,
        "anchor": date_component.map(anchors),
    })
    return product_table, date_table


def validate_reconstruction(fit: dict[str, Any], product_table: pd.DataFrame,
                            date_table: pd.DataFrame,
                            orders_r: pd.DataFrame) -> None:
    # Per-component scales cancel because each order line's product and date
    # sit in the same component by construction.
    reconstructed = (product_table["size"][orders_r["product_key"]].to_numpy()
                     * date_table["price_index"][orders_r["date"]].to_numpy())
    assert np.allclose(reconstructed, np.exp(fit["fitted"])), \
        "size * index does not reproduce fitted prices"


def modal_value_by_product(orders: pd.DataFrame, column: str) -> pd.Series:
    # Display name / category per normalized product: the modal raw value.
    return orders.groupby(["restaurant_id", "product_key"])[column] \
                 .agg(lambda values: values.mode().iat[0])


if __name__ == "__main__":  # pragma: no cover
    main()
