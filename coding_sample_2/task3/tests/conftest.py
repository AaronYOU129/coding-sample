# Purpose:    Provide one simulated dataset with known causal structure so tests
#             can verify the Task 3 specifications rather than third-party package
#             internals. Endogeneity makes OLS biased while strong external
#             instruments allow 2SLS to recover the known coefficient.
# Inputs:     A fixed NumPy random seed; no project data files.
# Outputs:    Pytest fixture containing a reproducible analysis-format DataFrame.
# Key Steps:  Draw date-level instruments -> create endogenous clicks -> create an
#             outcome with known causal effect -> add every control used by formulas.
# How to Run: Loaded automatically by pytest.

import numpy as np
import pandas as pd
import pytest

TRUE_CAUSAL_EFFECT = 2.0


# Simulate data with known effect and biased OLS, so 2SLS recovery can be verified.
@pytest.fixture
def simulated_analysis_data() -> pd.DataFrame:
    random_generator = np.random.default_rng(20260724)
    cluster_count = 80
    observations_per_cluster = 10
    observation_count = cluster_count * observations_per_cluster
    cluster_indices = np.repeat(np.arange(cluster_count), observations_per_cluster)

    date_instruments = random_generator.normal(size=(cluster_count, 3))
    instruments = date_instruments[cluster_indices]
    omitted_story_quality = random_generator.normal(size=observation_count)
    log_word_count = random_generator.normal(size=observation_count)
    structural_error = (
        0.8 * omitted_story_quality
        + random_generator.normal(scale=0.5, size=observation_count)
    )

    log_views = (
        instruments @ np.array([1.0, 0.8, 0.6])
        + 0.4 * log_word_count
        + omitted_story_quality
    )
    outcome = (
        TRUE_CAUSAL_EFFECT * log_views
        + 0.3 * log_word_count
        + structural_error
    )

    dates = pd.Timestamp("2012-01-01") + pd.to_timedelta(cluster_indices, unit="D")
    return pd.DataFrame(
        {
            "log1p_duration": outcome,
            "log1p_followups": 0.5 * outcome,
            "log_views": log_views,
            "totshort": instruments[:, 0],
            "rain_1": instruments[:, 1],
            "rain_2": instruments[:, 2],
            "PTI": random_generator.integers(0, 2, observation_count),
            "home_page": random_generator.integers(0, 2, observation_count),
            "front_page": random_generator.integers(0, 2, observation_count),
            "editor_pick": random_generator.integers(0, 2, observation_count),
            "video": random_generator.integers(0, 2, observation_count),
            "image": random_generator.integers(0, 2, observation_count),
            "log_word_count": log_word_count,
            "story_class": random_generator.choice(
                ["Business", "National News", "Sports"],
                observation_count,
            ),
            "month": dates.month,
            "day_of_week": dates.dayofweek,
            "date": dates.strftime("%Y-%m-%d"),
        }
    )
