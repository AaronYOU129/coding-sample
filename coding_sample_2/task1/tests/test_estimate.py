# Purpose:    Unit tests for the estimation module. The synthetic restaurant
#             follows price = size * index exactly, so the tests assert exact
#             recovery of the known decomposition -- the strongest possible
#             check of the estimator, normalization, and reconstruction.
#             Component handling is tested explicitly: per-component
#             normalization and the no-entree anchor fallback.
# Inputs:     Fixtures from conftest.py.
# Outputs:    None (test results).
# Key Steps:  Design matrix -> component labeling -> exact recovery ->
#             per-component normalization -> end-to-end main().
# How to Run: pytest tests/test_estimate.py   (from the task1/ directory)

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import estimate
import paths
import prepare_data
from tests.conftest import TRUE_INDEX, TRUE_SIZES


def cleaned(raw: pd.DataFrame) -> pd.DataFrame:
    frame = raw.copy()
    frame["product_name"] = frame["product_name"].str.strip()
    return prepare_data.clean_orders(frame)


def toy_orders(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame["product_key"] = frame["product_name"].map(
        prepare_data.normalize_product_name)
    return frame


def test_build_design_matrix_places_dummies_correctly() -> None:
    # 2 products, 3 dates, one observation of (product 1, date 2).
    design = estimate.build_design_matrix(
        products_idx=np.array([1]), dates_idx=np.array([2]),
        n_products=2, n_dates=3,
    )
    # Columns: [product0, product1, date1, date2]; date0 is the baseline.
    assert design.shape == (1, 4)
    assert design.tolist() == [[0.0, 1.0, 0.0, 1.0]]


def test_baseline_date_gets_no_dummy() -> None:
    design = estimate.build_design_matrix(
        products_idx=np.array([0]), dates_idx=np.array([0]),
        n_products=2, n_dates=3,
    )
    assert design.tolist() == [[1.0, 0.0, 0.0, 0.0]]


def test_component_labels_connected() -> None:
    # Product 0 sold on both dates links them: one component, labeled 0.
    product_labels, date_labels = estimate.component_labels(
        product_indices=np.array([0, 0, 1]),
        date_indices=np.array([0, 1, 1]),
        product_count=2, date_count=2,
    )
    assert product_labels.tolist() == [0, 0]
    assert date_labels.tolist() == [0, 0]


def test_component_labels_disconnected_ranked_by_obs() -> None:
    # Product 0 has two order lines on date 0; product 1 has one on date 1.
    # The larger component must get id 0.
    product_labels, date_labels = estimate.component_labels(
        product_indices=np.array([0, 0, 1]),
        date_indices=np.array([0, 0, 1]),
        product_count=2, date_count=2,
    )
    assert product_labels.tolist() == [0, 1]
    assert date_labels.tolist() == [0, 1]


def test_exact_recovery_of_sizes_and_index(raw_orders: pd.DataFrame) -> None:
    orders_r = cleaned(raw_orders)
    fit = estimate.estimate_restaurant(orders_r)
    product_table, date_table = estimate.normalize(fit, entree_products={"paella"})

    # Prices were built as size * index exactly, so recovery must be exact
    # (up to float precision) and the fit perfect.
    assert fit["r_squared"] == pytest.approx(1.0)
    assert fit["n_components"] == 1
    for product, true_size in TRUE_SIZES.items():
        assert product_table.loc[product.casefold(), "size"] == pytest.approx(true_size)
    for date, true_index in TRUE_INDEX.items():
        assert date_table.loc[date, "price_index"] == pytest.approx(true_index)
    # Zero-quantity Agua (1.0 euro on a day with index 10) has size 0.1.
    assert product_table.loc["agua", "size"] == pytest.approx(0.1)
    assert (product_table["anchor"] == "entree_mean").all()


def test_normalize_fails_without_entrees(raw_orders: pd.DataFrame) -> None:
    orders_r = cleaned(raw_orders)
    fit = estimate.estimate_restaurant(orders_r)
    with pytest.raises(AssertionError, match="no entree"):
        estimate.normalize(fit, entree_products=set())


def test_disconnected_components_normalized_separately() -> None:
    # Two entrees that never share a date: sizes must be 1 within EACH
    # component and each date's index must equal its own entree's price.
    orders_r = toy_orders([
        {"restaurant_id": 1, "product_name": "A", "product_unit_price": 10.0,
         "date": "2022-10-01", "category": "entree"},
        {"restaurant_id": 1, "product_name": "B", "product_unit_price": 12.0,
         "date": "2022-10-02", "category": "entree"},
    ])
    fit = estimate.estimate_restaurant(orders_r)
    product_table, date_table = estimate.normalize(fit, entree_products={"a", "b"})

    assert fit["n_components"] == 2
    assert product_table["size"].tolist() == pytest.approx([1.0, 1.0])
    assert date_table.loc["2022-10-01", "price_index"] == pytest.approx(10.0)
    assert date_table.loc["2022-10-02", "price_index"] == pytest.approx(12.0)
    # Product and date of the same order line share a component id.
    assert (product_table.loc["a", "component_id"]
            == date_table.loc["2022-10-01", "component_id"])
    assert set(product_table["component_id"]) == {0, 1}


def test_component_without_entree_uses_component_mean_anchor() -> None:
    orders_r = toy_orders([
        {"restaurant_id": 1, "product_name": "A", "product_unit_price": 10.0,
         "date": "2022-10-01", "category": "entree"},
        {"restaurant_id": 1, "product_name": "C", "product_unit_price": 2.0,
         "date": "2022-10-02", "category": "drink"},
    ])
    fit = estimate.estimate_restaurant(orders_r)
    product_table, date_table = estimate.normalize(fit, entree_products={"a"})

    assert product_table.loc["a", "anchor"] == "entree_mean"
    # C's component has no entree: its size is normalized to the component
    # mean (= itself here) and marked so it is never read in entree units.
    assert product_table.loc["c", "anchor"] == "component_mean"
    assert product_table.loc["c", "size"] == pytest.approx(1.0)
    assert date_table.loc["2022-10-02", "price_index"] == pytest.approx(2.0)


def test_reconstruction_validates(raw_orders: pd.DataFrame) -> None:
    orders_r = cleaned(raw_orders)
    fit = estimate.estimate_restaurant(orders_r)
    product_table, date_table = estimate.normalize(fit, entree_products={"paella"})
    estimate.validate_reconstruction(fit, product_table, date_table, orders_r)


def test_main_writes_all_outputs(tmp_paths: Path) -> None:
    prepare_data.main()
    estimate.main()

    sizes = pd.read_csv(paths.PRODUCT_SIZES)
    indices = pd.read_csv(paths.PRICE_INDICES)
    fit_stats = pd.read_csv(paths.FIT_STATS)
    residuals = pd.read_csv(paths.RESIDUALS)

    assert len(sizes) == 3        # paella, cola, agua
    assert len(indices) == 3      # three dates
    assert len(fit_stats) == 1    # one restaurant
    assert len(residuals) == 601  # one row per clean order line
    assert fit_stats["n_components"].iloc[0] == 1
    assert {"component_id", "anchor", "product_key"} <= set(sizes.columns)
    assert {"component_id", "anchor"} <= set(indices.columns)
    paella = sizes.loc[sizes["product_key"] == "paella"]
    assert paella["size"].iloc[0] == pytest.approx(1.0)
    assert paella["product_name"].iloc[0] == "Paella"  # display keeps raw form
