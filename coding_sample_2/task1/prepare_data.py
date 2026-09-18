# Purpose:    Clean the raw order-line data for the price decomposition.
#             Rows missing product_name, price, or date are dropped because the
#             two-way fixed-effects model needs all three to assign an
#             observation to a (product, restaurant-date) cell; at <0.25% of
#             the sample this loss is immaterial, but we log where the dropped
#             rows come from to verify the missingness is not concentrated in
#             one restaurant (which could bias that restaurant's estimates).
#             Zero-quantity rows are KEPT: the decomposition uses only prices,
#             and these rows still carry a valid posted unit price.
#             Beyond stripping whitespace, product names are normalized into a
#             product_key (Unicode NFKC, collapsed whitespace, casefold):
#             labels like "ARROZ AL HORNO" vs "Arroz al Horno" are the same
#             product, and treating them as different splits its price history,
#             weakens cross-date links, and can artificially disconnect the
#             product-date graph. The raw name is kept for display.
# Inputs:     ../data/orders_sample.csv  (raw order lines)
# Outputs:    work/orders_clean.csv      (estimation-ready order lines)
#             work/cleaning_log.txt      (what was dropped/merged and why)
# Key Steps:  Load -> strip names -> drop rows with missing key fields
#             (logging their distribution) -> build normalized product_key
#             (logging merged name variants) -> flag zero-quantity rows ->
#             validate -> write.
# How to Run: python prepare_data.py   (from the task1/ directory), or
#             python run_all.py for the full pipeline.

import re
import unicodedata

import pandas as pd

import paths

KEY_FIELDS: list[str] = ["product_name", "product_unit_price", "date"]


def main() -> None:
    paths.ensure_directories()
    orders = load_raw_orders()
    log_text = describe_dropped_rows(orders)

    clean = clean_orders(orders)
    validate(clean, n_raw=len(orders))

    clean.to_csv(paths.CLEAN_ORDERS, index=False)
    log_text += (
        f"\n{describe_name_merges(clean)}"
        f"\nZero-quantity rows kept (flagged): {clean['is_zero_quantity'].sum()}"
        f"\nClean rows written: {len(clean)}"
    )
    paths.CLEANING_LOG.write_text(log_text)
    print(log_text)


def normalize_product_name(name: str) -> str:
    # NFKC unifies visually-identical Unicode variants; collapsing whitespace
    # and casefolding merges labels that differ only in typing style.
    name = unicodedata.normalize("NFKC", name)
    return re.sub(r"\s+", " ", name).strip().casefold()


def read_clean_orders() -> pd.DataFrame:
    # product_key is re-derived from product_name rather than read from the
    # CSV: a product literally named "Nan" casefolds to "nan", which pandas
    # would otherwise parse back as a missing value on load. Deriving it here
    # keeps every downstream reader consistent and round-trip-safe.
    orders = pd.read_csv(paths.CLEAN_ORDERS)
    orders["product_key"] = orders["product_name"].map(normalize_product_name)
    return orders


def load_raw_orders() -> pd.DataFrame:
    orders = pd.read_csv(paths.RAW_ORDERS)
    # Product names carry leading whitespace in the raw file; without stripping,
    # " Paella" and "Paella" would be treated as two different products.
    orders["product_name"] = orders["product_name"].str.strip()
    return orders


def describe_dropped_rows(orders: pd.DataFrame) -> str:
    dropped = orders[orders[KEY_FIELDS].isna().any(axis=1)]
    lines = [f"Rows dropped for missing key fields: {len(dropped)} of {len(orders)}"]
    for field in KEY_FIELDS:
        lines.append(f"  missing {field}: {orders[field].isna().sum()}")
    # If missingness clusters in one restaurant, dropping is not innocuous;
    # we report the distribution so the report can address it.
    by_restaurant = dropped["restaurant_id"].value_counts()
    lines.append("Dropped rows by restaurant (top 5):")
    for restaurant_id, count in by_restaurant.head(5).items():
        lines.append(f"  restaurant {restaurant_id}: {count}")
    n_restaurants = orders["restaurant_id"].nunique()
    lines.append(f"Restaurants affected: {by_restaurant.size} of {n_restaurants}")
    return "\n".join(lines)


def clean_orders(orders: pd.DataFrame) -> pd.DataFrame:
    clean = orders.dropna(subset=KEY_FIELDS).copy()
    clean["product_key"] = clean["product_name"].map(normalize_product_name)
    # Kept but flagged: likely refunds or freebies; used in estimation
    # (their price is valid) and revisited in diagnostics.
    clean["is_zero_quantity"] = clean["bought_product_quantity"] == 0
    return clean


def describe_name_merges(clean: pd.DataFrame) -> str:
    variants = clean.groupby(["restaurant_id", "product_key"])["product_name"].unique()
    merged = variants[variants.map(len) > 1]
    lines = [f"Name-variant groups merged by product_key: {len(merged)}"]
    for (restaurant_id, _), names in merged.items():
        lines.append(f"  restaurant {restaurant_id}: {list(names)}")
    return "\n".join(lines)


def validate(clean: pd.DataFrame, n_raw: int) -> None:
    assert clean[KEY_FIELDS].notna().all().all(), "missing values survived cleaning"
    assert (clean["product_unit_price"] > 0).all(), \
        "non-positive prices would break the log transform"
    assert len(clean) > 0.99 * n_raw, "dropped more than 1% of rows; cleaning rules need review"


if __name__ == "__main__":  # pragma: no cover
    main()
