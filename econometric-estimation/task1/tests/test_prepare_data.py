# Purpose:    Unit tests for the cleaning module: dropping rules, flags,
#             validation guards, and the end-to-end main() on synthetic data.
# Inputs:     Fixtures from conftest.py.
# Outputs:    None (test results).
# Key Steps:  Test each cleaning function in isolation, then main() with
#             paths redirected to tmp_path.
# How to Run: pytest tests/test_prepare_data.py   (from the task1/ directory)

from pathlib import Path

import pandas as pd
import pytest

import paths
import prepare_data


def test_clean_orders_drops_only_rows_missing_key_fields(raw_orders: pd.DataFrame) -> None:
    clean = prepare_data.clean_orders(raw_orders)
    # 6 exact-model rows + the zero-quantity row survive; 3 rows with missing
    # name/price/date are dropped.
    assert len(clean) == 7
    assert clean[prepare_data.KEY_FIELDS].notna().all().all()


def test_zero_quantity_rows_are_kept_and_flagged(raw_orders: pd.DataFrame) -> None:
    clean = prepare_data.clean_orders(raw_orders)
    agua = clean[clean["product_name"] == "Agua"]
    assert len(agua) == 1
    assert bool(agua["is_zero_quantity"].iloc[0])
    assert clean["is_zero_quantity"].sum() == 1


def test_describe_dropped_rows_reports_counts(raw_orders: pd.DataFrame) -> None:
    log_text = prepare_data.describe_dropped_rows(raw_orders)
    assert "3 of 10" in log_text
    assert "missing product_name: 1" in log_text
    assert "missing product_unit_price: 1" in log_text
    assert "missing date: 1" in log_text


def test_normalize_product_name_merges_case_and_whitespace() -> None:
    assert prepare_data.normalize_product_name("ARROZ AL  HORNO") == "arroz al horno"
    assert prepare_data.normalize_product_name(" Pizza New York ") == "pizza new york"


def test_clean_orders_builds_product_key(raw_orders: pd.DataFrame) -> None:
    clean = prepare_data.clean_orders(raw_orders)
    # " Paella" and "Paella" collapse to one key.
    assert set(clean["product_key"]) == {"paella", "cola", "agua"}


def test_describe_name_merges_lists_variants() -> None:
    frame = pd.DataFrame([
        {"restaurant_id": 1, "product_name": "Pizza New York",
         "product_unit_price": 8.0, "bought_product_quantity": 1,
         "date": "2022-10-01", "category": "entree"},
        {"restaurant_id": 1, "product_name": "Pizza new york",
         "product_unit_price": 8.0, "bought_product_quantity": 1,
         "date": "2022-10-02", "category": "entree"},
    ])
    clean = prepare_data.clean_orders(frame)
    log_text = prepare_data.describe_name_merges(clean)
    assert "merged by product_key: 1" in log_text
    assert "Pizza New York" in log_text and "Pizza new york" in log_text


def test_validate_rejects_excessive_drops(raw_orders: pd.DataFrame) -> None:
    clean = prepare_data.clean_orders(raw_orders)
    # 3 of 10 rows dropped (30%) violates the <1% guard on the small frame.
    with pytest.raises(AssertionError, match="more than 1%"):
        prepare_data.validate(clean, n_raw=len(raw_orders))


def test_main_writes_clean_file_and_log(tmp_paths: Path) -> None:
    prepare_data.main()
    clean = pd.read_csv(paths.CLEAN_ORDERS)
    # 600 replicated clean rows + 1 zero-quantity row; names are stripped.
    assert len(clean) == 601
    assert set(clean["product_name"]) == {"Paella", "Cola", "Agua"}
    assert paths.CLEANING_LOG.read_text().startswith("Rows dropped")
