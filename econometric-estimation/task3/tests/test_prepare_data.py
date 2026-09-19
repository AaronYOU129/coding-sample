# Purpose:    Verify the substantive data decisions that determine the estimation
#             sample and outcomes. These checks target mistakes that would change
#             the research answer, not every implementation branch.
# Inputs:     The three raw CSV files in data/ through prepare_data.py loaders.
# Outputs:    None; pytest assertions confirm row counts and variable definitions.
# Key Steps:  Prepare stories -> merge daily shocks -> check sample loss, outcome
#             construction, and the domain required by logarithmic transformations.
# How to Run: `pytest tests/test_prepare_data.py` from task3/.

import numpy as np

import prepare_data


# Check sample sizes and outcome construction on real data, catching the follow-up off-by-one.
def test_analysis_sample_and_constructed_variables() -> None:
    stories = prepare_data.load_and_prepare_stories()
    rainfall = prepare_data.load_rainfall()
    power_shortage = prepare_data.load_power_shortage()
    analysis_data, _ = prepare_data.merge_daily_shocks(
        stories,
        rainfall,
        power_shortage,
    )

    assert len(stories) == 37_327
    assert len(analysis_data) == 33_094
    assert len(stories) - len(analysis_data) == 4_233
    assert (stories["followup_count"] == stories["nnarticles"] - 1).all()
    assert (stories["duration_hours"] >= 0).all()
    np.testing.assert_allclose(stories["log_views"], np.log(stories["views"]))
