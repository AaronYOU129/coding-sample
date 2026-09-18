# Purpose:    Create a transparent analysis dataset from three files whose dates
#             use incompatible formats. Stories are matched to daily shocks using
#             their start date because the identification strategy concerns reader
#             attention when a story begins. Missing power days are dropped rather
#             than imputed so the instrument is never artificially constructed.
# Inputs:     data/newspaper.csv — story records; data/power.csv — daily electricity
#             shortage; data/raindata.csv — daily rainfall in two cities.
# Outputs:    work/analysis_data.csv — merged analysis sample;
#             work/merge_log.txt — auditable row counts and data checks.
# Key Steps:  Parse the three date formats -> validate raw fields -> construct outcomes
#             and logs -> merge rainfall and power by start date -> record exclusions.
# How to Run: `python prepare_data.py` from task3/, or `python run_all.py`.

import numpy as np
import pandas as pd

import paths

NEWSPAPER_DATETIME_FORMAT = "%m/%d/%y %H:%M"
POWER_DATE_FORMAT = "%d%b%Y"
RAIN_DATE_FORMAT = "%d-%b-%y"


# Run the full prepare-and-merge pipeline and persist the analysis sample plus log.
def main() -> None:
    paths.ensure_directories()
    stories = load_and_prepare_stories()
    rainfall = load_rainfall()
    power_shortage = load_power_shortage()
    analysis_data, merge_messages = merge_daily_shocks(stories, rainfall, power_shortage)
    analysis_data.to_csv(paths.ANALYSIS_DATA, index=False)
    paths.MERGE_LOG.write_text("\n".join(merge_messages) + "\n")
    print("\n".join(merge_messages))


# Load stories, parse timestamps, and derive outcomes/logs the regressions need.
def load_and_prepare_stories() -> pd.DataFrame:
    stories = pd.read_csv(paths.NEWSPAPER_DATA)
    stories["start_time"] = pd.to_datetime(
        stories["st_starttime"],
        format=NEWSPAPER_DATETIME_FORMAT,
    )
    stories["end_time"] = pd.to_datetime(
        stories["st_endtime"],
        format=NEWSPAPER_DATETIME_FORMAT,
    )
    validate_story_data(stories)

    prepared_stories = stories.copy()
    prepared_stories["date"] = prepared_stories["start_time"].dt.normalize()
    prepared_stories["duration_hours"] = (
        prepared_stories["end_time"] - prepared_stories["start_time"]
    ).dt.total_seconds() / 3600
    prepared_stories["followup_count"] = prepared_stories["nnarticles"] - 1
    prepared_stories["log_views"] = np.log(prepared_stories["views"])
    prepared_stories["log1p_duration"] = np.log1p(
        prepared_stories["duration_hours"]
    )
    prepared_stories["log1p_followups"] = np.log1p(
        prepared_stories["followup_count"]
    )
    prepared_stories["log_word_count"] = np.log1p(prepared_stories["word_count"])
    prepared_stories["month"] = prepared_stories["start_time"].dt.month
    prepared_stories["day_of_week"] = prepared_stories["start_time"].dt.dayofweek
    prepared_stories["story_class"] = prepared_stories["class"]
    prepared_stories["image"] = (
        prepared_stories["image"].astype(str).str.upper().eq("TRUE").astype(int)
    )
    return prepared_stories


# Reject impossible rows early, so bad data fails loudly instead of silently biasing estimates.
def validate_story_data(stories: pd.DataFrame) -> None:
    if (stories["end_time"] < stories["start_time"]).any():
        raise ValueError("A story ends before it starts.")
    if (stories["views"] < 1).any():
        raise ValueError("views must be positive before taking log(views).")
    if (stories["nnarticles"] < 1).any():
        raise ValueError("nnarticles must include at least the initial article.")


# Load two-city rain instruments; reject missing/duplicate dates before they reach the merge.
def load_rainfall() -> pd.DataFrame:
    rainfall = pd.read_csv(paths.RAIN_DATA)
    rainfall["date"] = pd.to_datetime(
        rainfall["raindate"],
        format=RAIN_DATE_FORMAT,
    )
    validate_unique_daily_dates(rainfall, "raindata.csv")
    if rainfall[["rain_1", "rain_2"]].isna().any().any():
        raise ValueError("Rainfall instruments contain missing values.")
    return rainfall[["date", "rain_1", "rain_2"]]


# Load the daily shortage instrument; enforce unique dates so the merge key is clean.
def load_power_shortage() -> pd.DataFrame:
    power_shortage = pd.read_csv(paths.POWER_DATA)
    power_shortage["date"] = pd.to_datetime(
        power_shortage["date"],
        format=POWER_DATE_FORMAT,
    )
    validate_unique_daily_dates(power_shortage, "power.csv")
    return power_shortage[["date", "totshort"]]


# Guard the daily merge key; duplicate dates would fan out rows and distort the sample.
def validate_unique_daily_dates(daily_data: pd.DataFrame, file_name: str) -> None:
    if daily_data["date"].duplicated().any():
        raise ValueError(f"{file_name} contains duplicate dates.")


# Attach instruments by start date and log row counts, so sample exclusions stay auditable.
def merge_daily_shocks(
    stories: pd.DataFrame,
    rainfall: pd.DataFrame,
    power_shortage: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    stories_with_rain = stories.merge(
        rainfall,
        on="date",
        how="left",
        validate="many_to_one",
    )
    missing_rainfall_count = int(stories_with_rain["rain_1"].isna().sum())
    if missing_rainfall_count:
        raise ValueError("Rainfall does not cover every story start date.")

    analysis_data = stories_with_rain.merge(
        power_shortage,
        on="date",
        how="inner",
        validate="many_to_one",
    )
    dropped_story_count = len(stories) - len(analysis_data)
    messages = [
        f"Stories loaded: {len(stories)}",
        f"Stories matched to rainfall: {len(stories_with_rain)}",
        f"Stories in analysis sample: {len(analysis_data)}",
        f"Stories dropped because power data are missing: {dropped_story_count}",
    ]
    return analysis_data, messages


if __name__ == "__main__":
    main()
