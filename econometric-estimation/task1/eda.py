# Purpose:    Explore the raw order data before modeling, focusing on price
#             distributions, category differences, and daily coverage because
#             these features motivate the log-price transformation, product
#             fixed effects, and checks for sparse restaurant-days.
# Inputs:     ../data/orders_sample.csv (raw order-line data).
# Outputs:    figures/eda.png (four-panel exploratory figure);
#             output/eda_summary.csv (reusable summary statistics);
#             output/eda_summary.tex (report-ready LaTeX table).
# Key Steps:  Load raw data -> summarize data quality, coverage, prices, and
#             quantities -> export CSV and LaTeX tables -> plot raw and log
#             prices, daily order counts, and prices by category.
# How to Run: python eda.py   (from the task1/ directory).

import matplotlib

# A non-interactive backend makes the script reproducible on headless machines.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import paths  # noqa: E402

CATEGORY_ORDER = [
    "appetizer",
    "dessert",
    "drink",
    "entree",
    "side",
    "uncertain",
    "unclassified",
]
KEY_FIELDS = ["product_name", "product_unit_price", "date"]


def main() -> None:
    """Run the reproducible EDA workflow and report where outputs are saved."""
    paths.ensure_directories()
    orders = load_orders()
    sample_summary, variable_summary = build_summary_tables(orders)
    print_summary_tables(sample_summary, variable_summary)
    save_summary_tables(sample_summary, variable_summary)
    create_eda_figure(orders)
    print(f"EDA summary CSV written to {paths.EDA_SUMMARY}")
    print(f"EDA summary LaTeX written to {paths.EDA_SUMMARY_TEX}")
    print(f"EDA figure written to {paths.EDA_FIGURE}")


def load_orders() -> pd.DataFrame:
    """Load raw rows without dropping problems so the EDA reveals data quality."""
    orders = pd.read_csv(paths.RAW_ORDERS)
    # Coercion exposes invalid dates as missing rather than stopping exploration.
    orders["parsed_date"] = pd.to_datetime(orders["date"], errors="coerce")
    return orders


def build_summary_tables(orders: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build sample and variable summaries that directly inform modeling choices."""
    analysis_orders = orders.dropna(subset=KEY_FIELDS).copy()
    analysis_orders["product_key"] = (
        analysis_orders["product_name"]
        .str.normalize("NFKC")
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .str.casefold()
    )
    price_counts = analysis_orders.groupby(
        ["restaurant_id", "product_key"],
    )["product_unit_price"].nunique()
    restaurant_has_repricing = price_counts.gt(1).groupby(level="restaurant_id").any()

    date_min = orders["parsed_date"].min().date()
    date_max = orders["parsed_date"].max().date()
    sample_summary = pd.DataFrame([
        ("Raw order lines", f"{len(orders):,}"),
        ("Clean order lines", f"{len(analysis_orders):,}"),
        ("Restaurants", f"{orders['restaurant_id'].nunique()}"),
        ("Unique dates", f"{orders['parsed_date'].nunique()}"),
        ("Date range", f"{date_min} to {date_max}"),
        (
            "Restaurant-product pairs",
            f"{analysis_orders[['restaurant_id', 'product_key']].drop_duplicates().shape[0]:,}",
        ),
        (
            "Restaurant-date pairs",
            f"{analysis_orders[['restaurant_id', 'date']].drop_duplicates().shape[0]:,}",
        ),
        ("Rows missing product, price, or date", f"{len(orders) - len(analysis_orders):,}"),
        ("Zero-quantity rows", f"{(orders['bought_product_quantity'] == 0).sum():,}"),
        ("Restaurants with no repricing", f"{(~restaurant_has_repricing).sum():,}"),
    ], columns=["Statistic", "Value"])

    variables = {
        "Unit price": orders.loc[
            orders["product_unit_price"] > 0,
            "product_unit_price",
        ].dropna(),
        "Quantity": orders["bought_product_quantity"].dropna(),
    }
    variable_rows = []
    for variable_name, values in variables.items():
        variable_rows.append({
            "Variable": variable_name,
            "N": int(values.count()),
            "Mean": values.mean(),
            "SD": values.std(),
            "Min": values.min(),
            "P25": values.quantile(0.25),
            "Median": values.median(),
            "P75": values.quantile(0.75),
            "Max": values.max(),
        })
    variable_summary = pd.DataFrame(variable_rows)
    return sample_summary, variable_summary


def print_summary_tables(
    sample_summary: pd.DataFrame,
    variable_summary: pd.DataFrame,
) -> None:
    """Print the same summaries exported for the report for quick inspection."""
    print("Sample structure and data quality")
    print(sample_summary.to_string(index=False))
    print("\nVariable summary statistics")
    print(variable_summary.round(2).to_string(index=False))


def save_summary_tables(
    sample_summary: pd.DataFrame,
    variable_summary: pd.DataFrame,
) -> None:
    """Save a reusable CSV and a report-ready LaTeX table from one source."""
    csv_rows: list[dict[str, object]] = []
    for statistic, value in sample_summary.itertuples(index=False, name=None):
        csv_rows.append({"Panel": "Sample", "Statistic": statistic, "Value": value})
    for row in variable_summary.to_dict(orient="records"):
        csv_rows.append({
            "Panel": "Variables",
            "Statistic": row.pop("Variable"),
            **row,
        })
    pd.DataFrame(csv_rows).to_csv(paths.EDA_SUMMARY, index=False)

    sample_rows = [
        f"{statistic} & {value} \\\\"
        for statistic, value in sample_summary.itertuples(index=False, name=None)
    ]
    variable_rows = []
    for row in variable_summary.itertuples(index=False):
        variable_rows.append(
            f"{row.Variable} & {row.N:,} & {row.Mean:.2f} & {row.SD:.2f} & "
            f"{row.Min:.2f} & {row.P25:.2f} & {row.Median:.2f} & "
            f"{row.P75:.2f} & {row.Max:.2f} \\\\"
        )

    latex_table = "\n".join([
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Task 1: summary statistics for restaurant order lines.}",
        r"\label{tab:t1summary}",
        r"\begin{tabular}{lr}",
        r"\toprule",
        r"\multicolumn{2}{l}{\textit{Panel A: Sample structure and data quality}} \\",
        r"\midrule",
        *sample_rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{0.5em}",
        r"\small",
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r"\multicolumn{9}{l}{\textit{Panel B: Variable summary statistics}} \\",
        r"\midrule",
        r"Variable & $N$ & Mean & SD & Min & P25 & Median & P75 & Max \\",
        r"\midrule",
        *variable_rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    paths.EDA_SUMMARY_TEX.write_text(latex_table + "\n", encoding="utf-8")


def create_eda_figure(orders: pd.DataFrame) -> None:
    """Create four plots that motivate the transformation and fixed effects.

    Raw and log-price histograms assess skewness and the usefulness of logs.
    Daily counts reveal thin dates, while category boxplots show persistent
    price differences that should be absorbed by product fixed effects.
    """
    valid_prices = orders.loc[
        orders["product_unit_price"] > 0,
        ["product_unit_price", "category"],
    ].dropna()
    daily_counts = orders.dropna(subset=["parsed_date"]).groupby("parsed_date").size()
    present_categories = [
        category for category in CATEGORY_ORDER
        if (valid_prices["category"] == category).any()
    ]
    category_prices = [
        valid_prices.loc[
            valid_prices["category"] == category,
            "product_unit_price",
        ]
        for category in present_categories
    ]

    figure, axes = plt.subplots(2, 2, figsize=(12, 9))

    # Raw prices reveal skewness and extreme values that averages may hide.
    axes[0, 0].hist(valid_prices["product_unit_price"], bins=50)
    axes[0, 0].set(title="Unit price distribution", xlabel="price")

    # Log prices show whether the multiplicative model has a usable scale.
    axes[0, 1].hist(np.log(valid_prices["product_unit_price"]), bins=50)
    axes[0, 1].set(title="Log unit price distribution", xlabel="log(price)")

    # Daily coverage identifies dates where a daily fixed effect may be noisy.
    axes[1, 0].plot(daily_counts.index, daily_counts.to_numpy())
    axes[1, 0].set(title="Order lines per day", xlabel="date", ylabel="order lines")
    axes[1, 0].tick_params(axis="x", rotation=45)

    # Persistent category price gaps motivate controlling for product identity.
    axes[1, 1].boxplot(category_prices)
    axes[1, 1].set_xticklabels(present_categories, rotation=45, ha="right")
    axes[1, 1].set(title="Unit price by category", xlabel="category", ylabel="price")

    figure.tight_layout()
    figure.savefig(paths.EDA_FIGURE, dpi=150)
    plt.close(figure)


if __name__ == "__main__":  # pragma: no cover
    main()
