# Purpose:    Shared fixtures for the Task 1 test suite. Builds a tiny
#             synthetic restaurant whose prices follow the multiplicative
#             model EXACTLY, so estimation tests can assert exact recovery
#             of known sizes and indices rather than loose approximations.
# Inputs:     None (synthetic data defined here).
# Outputs:    None (fixtures consumed by tests).
# Key Steps:  Small dirty raw frame for cleaning tests -> replicated large
#             frame so the <1% drop guard passes end-to-end -> tmp_paths
#             fixture that redirects every path constant into pytest's
#             tmp_path so tests never touch real data.
# How to Run: Not run directly; used by pytest.

from pathlib import Path

import pandas as pd
import pytest

import paths

# Ground truth for the synthetic restaurant: price = size * index exactly.
TRUE_INDEX = {"2022-10-01": 10.0, "2022-10-02": 10.0, "2022-10-03": 11.0}
TRUE_SIZES = {"Paella": 1.0, "Cola": 0.2}  # Paella is the only entree


def _clean_rows() -> list[dict]:
    rows = []
    for date, index_value in TRUE_INDEX.items():
        for product, size in TRUE_SIZES.items():
            rows.append({
                "restaurant_id": 1,
                "product_name": f" {product}",  # leading space: stripping is tested
                "product_unit_price": size * index_value,
                "bought_product_quantity": 1,
                "date": date,
                "category": "entree" if product == "Paella" else "drink",
            })
    return rows


def _dirty_rows() -> list[dict]:
    return [
        {"restaurant_id": 1, "product_name": None, "product_unit_price": 5.0,
         "bought_product_quantity": 1, "date": "2022-10-01", "category": "entree"},
        {"restaurant_id": 1, "product_name": "Flan", "product_unit_price": None,
         "bought_product_quantity": 1, "date": "2022-10-01", "category": "dessert"},
        {"restaurant_id": 1, "product_name": "Flan", "product_unit_price": 3.0,
         "bought_product_quantity": 1, "date": None, "category": "dessert"},
        {"restaurant_id": 1, "product_name": "Agua", "product_unit_price": 1.0,
         "bought_product_quantity": 0, "date": "2022-10-02", "category": "drink"},
    ]


@pytest.fixture
def raw_orders() -> pd.DataFrame:
    """Small frame (6 clean + 4 dirty rows) for unit tests of cleaning logic."""
    return pd.DataFrame(_clean_rows() + _dirty_rows())


@pytest.fixture
def raw_orders_large() -> pd.DataFrame:
    """Clean rows replicated so dirty rows are <1% and main() passes its guard."""
    replicated = _clean_rows() * 100
    return pd.DataFrame(replicated + _dirty_rows())


@pytest.fixture
def tmp_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
              raw_orders_large: pd.DataFrame) -> Path:
    """Redirect every path constant into tmp_path and write synthetic raw data."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    raw_file = data_dir / "orders_sample.csv"
    raw_orders_large.to_csv(raw_file, index=False)

    monkeypatch.setattr(paths, "RAW_ORDERS", raw_file)
    monkeypatch.setattr(paths, "WORK_DIR", tmp_path / "work")
    monkeypatch.setattr(paths, "OUTPUT_DIR", tmp_path / "output")
    monkeypatch.setattr(paths, "FIGURES_DIR", tmp_path / "figures")
    monkeypatch.setattr(paths, "CLEAN_ORDERS", tmp_path / "work" / "orders_clean.csv")
    monkeypatch.setattr(paths, "CLEANING_LOG", tmp_path / "work" / "cleaning_log.txt")
    monkeypatch.setattr(paths, "RESIDUALS", tmp_path / "work" / "residuals.csv")
    monkeypatch.setattr(paths, "FIT_STATS", tmp_path / "work" / "fit_stats.csv")
    monkeypatch.setattr(paths, "PRODUCT_SIZES", tmp_path / "output" / "product_sizes.csv")
    monkeypatch.setattr(paths, "PRICE_INDICES", tmp_path / "output" / "price_indices.csv")
    monkeypatch.setattr(paths, "DIAGNOSTICS_SUMMARY",
                        tmp_path / "output" / "diagnostics_summary.csv")
    return tmp_path
