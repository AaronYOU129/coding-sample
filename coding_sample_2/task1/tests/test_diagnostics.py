# Purpose:    Tests for the diagnostics module: full coverage of the summary
#             table logic (flags are decisions users act on), smoke tests for
#             the figures (they only need to render and hit disk).
# Inputs:     Fixtures from conftest.py.
# Outputs:    None (test results).
# Key Steps:  Run the pipeline on synthetic data -> check summary flags ->
#             check all figures exist and are non-empty.
# How to Run: pytest tests/test_diagnostics.py   (from the task1/ directory)

from pathlib import Path

import pandas as pd
import pytest

import diagnostics
import estimate
import paths
import prepare_data


@pytest.fixture
def pipeline_outputs(tmp_paths: Path) -> Path:
    prepare_data.main()
    estimate.main()
    return tmp_paths


def test_summary_flags(pipeline_outputs: Path) -> None:
    orders = pd.read_csv(paths.CLEAN_ORDERS)
    fit_stats = pd.read_csv(paths.FIT_STATS)
    indices = pd.read_csv(paths.PRICE_INDICES)

    summary = diagnostics.build_summary(orders, fit_stats, indices)

    assert len(summary) == 1
    row = summary.iloc[0]
    # Paella and Cola change price on 2022-10-03; Agua never reprices, so 2 of
    # 3 products are repriced and the index is not flat...
    assert row["share_products_repriced"] == pytest.approx(2 / 3)
    assert not row["flag_flat_index"]
    # ...but a 10% single move keeps the coefficient of variation moderate.
    assert not row["flag_volatile_index"]
    assert not row["flag_disconnected"]
    # 600 replicated lines over 3 dates => plenty per day, not sparse.
    assert not row["flag_sparse_days"]
    # Agua appears once (the zero-quantity line), so 1 of 3 products is
    # single-observation (0.33 > 0.25): the flag correctly fires.
    assert row["flag_many_single_obs"]


def test_figures_render_to_disk(pipeline_outputs: Path) -> None:
    diagnostics.main()
    for name in ["fig1_fit_quality.png", "fig2_sizes_by_category.png",
                 "fig3_price_indices.png"]:
        figure = paths.FIGURES_DIR / name
        assert figure.exists() and figure.stat().st_size > 0


def test_summary_written_by_main(pipeline_outputs: Path) -> None:
    diagnostics.main()
    summary = pd.read_csv(paths.DIAGNOSTICS_SUMMARY)
    assert {"flag_flat_index", "flag_disconnected", "flag_volatile_index",
            "flag_sparse_days", "flag_many_single_obs"} <= set(summary.columns)
