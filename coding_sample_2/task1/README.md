# Task 1 — Restaurant Price Index

Task 1 decomposes each order-line price into a product-specific size and a
restaurant-date price index. The model is estimated separately by restaurant,
and product sizes are normalized within each connected product-date component.

## Input

- `../data/orders_sample.csv` — order lines with restaurant, product, unit
  price, quantity, date, and category information.

## Method

The estimator regresses log unit price on product and date dummies separately
for each restaurant. Product effects determine relative sizes and date effects
determine price indices. Within a connected component, mean entree size is
normalized to one when entrees are available; otherwise mean product size is
normalized to one. Component identifiers and anchors are retained because
levels from disconnected components are not directly comparable.

## Pipeline

1. `eda.py` explores raw prices, categories, daily coverage, and data quality.
2. `prepare_data.py` removes rows missing a required field, normalizes product
   names, keeps and flags zero-quantity rows, and writes a cleaning log.
3. `estimate.py` estimates product sizes and price indices and saves fit
   statistics and order-line residuals.
4. `diagnostics.py` evaluates fit, category-level sizes, index paths, and
   identification warnings.
5. `run_all.py` runs these four stages in order.

`paths.py` contains all Task 1 input and output paths.

## Run

From `task1/`:

```bash
python run_all.py
```

## Generated files

Intermediate files:

- `work/orders_clean.csv`
- `work/cleaning_log.txt`
- `work/fit_stats.csv`
- `work/residuals.csv`

Final outputs:

- `output/eda_summary.csv`
- `output/eda_summary.tex`
- `output/product_sizes.csv`
- `output/price_indices.csv`
- `output/diagnostics_summary.csv`

Figures:

- `figures/eda.png`
- `figures/fig1_fit_quality.png`
- `figures/fig2_sizes_by_category.png`
- `figures/fig3_price_indices.png`

## Checks

The 21 tests cover cleaning, product-name normalization, fixed-effect recovery
on simulated data, component-specific normalization, reconstruction of fitted
prices, diagnostics, and figure generation.

```bash
ruff check .
mypy paths.py eda.py prepare_data.py estimate.py diagnostics.py run_all.py
pytest -q
```

