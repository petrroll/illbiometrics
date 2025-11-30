"""Sleep data analytics module. No external API dependencies."""

from dataclasses import dataclass
from datetime import date
from typing import Optional
import pandas as pd
import numpy as np

from app.oura_client import SleepData, HeartRateData

# Sources considered "active" heart rate (not resting/sleep)
NON_SLEEP_HR_SOURCES = {"awake", "live", "workout"}

# Maximum gap (in seconds) between consecutive HR points to consider them connected
# Points further apart than this are not interpolated between
HR_MAX_GAP_SECONDS = 300

# Resampling interval for heart rate data
HR_RESAMPLE_INTERVAL = "1min"


@dataclass
class SleepAnalytics:
    """Sleep analytics results."""
    start_date: Optional[date]
    end_date: Optional[date]
    median_sleep_duration: float
    median_avg_hr: float
    median_avg_hrv: float
    hr_20th_percentile: float
    hr_80th_percentile: float
    hrv_20th_percentile: float
    hrv_80th_percentile: float


@dataclass
class HeartRateAnalytics:
    """Heart rate analytics results (aggregated across all days)."""
    start_date: Optional[date]
    end_date: Optional[date]
    average_hr: float
    hr_20th_percentile: float
    hr_50th_percentile: float
    hr_80th_percentile: float
    hr_95th_percentile: float
    hr_99th_percentile: float


@dataclass
class DailyHeartRateAnalytics:
    """Heart rate analytics for a single day."""
    day: date
    average_hr: float
    hr_20th_percentile: float
    hr_50th_percentile: float
    hr_80th_percentile: float
    hr_95th_percentile: float
    hr_99th_percentile: float


# Only include "long_sleep" type for nightly sleep analytics.
# Sleep type statistics from historical data:
# | Type       | Count | Avg Duration | Avg Start Time |
# |------------|-------|--------------|----------------|
# | late_nap   | 11    | 0h 30m       | 19:16          |
# | long_sleep | 722   | 9h 52m       | 21:40          |
# | rest       | 45    | 0h 34m       | 16:47          |
# | sleep      | 843   | 0h 15m       | 15:58          |
# Only "long_sleep" represents actual overnight sleep periods.
NIGH_SLEEP_SLEEP_TYPE = "long_sleep"

def oura_sleep_to_dataframe(sleep_data: list[SleepData]) -> pd.DataFrame:
    """Convert Oura sleep API response to DataFrame."""
    records = []
    for sleep in sleep_data:
        record = {
            "id": sleep.id,
            "day": sleep.day,
            "type": sleep.type,
            "total_sleep_duration": sleep.total_sleep_duration,
            "average_heart_rate": sleep.average_heart_rate,
            "average_hrv": sleep.average_hrv,
            "heart_rate_samples": sleep.heart_rate.items if sleep.heart_rate else [],
            "hrv_samples": sleep.hrv.items if sleep.hrv else [],
        }
        records.append(record)
    return pd.DataFrame(records)


def analyze_sleep(df: pd.DataFrame) -> SleepAnalytics:
    """Compute sleep analytics from DataFrame.
    
    Filters to only include 'long_sleep' type entries for analysis.
    See LONG_SLEEP_TYPE comment for sleep type statistics.
    """
    # Filter to only long_sleep type for meaningful nightly statistics
    if not df.empty and "type" in df.columns:
        df = df.loc[df["type"] == NIGH_SLEEP_SLEEP_TYPE]
    
    # Get actual date range from the data
    if not df.empty:
        dates = pd.to_datetime(df["day"]).dt.date
        actual_start = dates.min()
        actual_end = dates.max()
    else:
        actual_start = None
        actual_end = None

    # Median sleep duration (convert seconds to hours)
    median_duration = df["total_sleep_duration"].median()

    # Median of average HR and HRV
    median_avg_hr = df["average_heart_rate"].median()
    median_avg_hrv = df["average_hrv"].median()

    # Flatten all HR samples for global percentiles
    all_hr = []
    for samples in df["heart_rate_samples"]:
        if samples:
            all_hr.extend([s for s in samples if s is not None])
    
    # Flatten all HRV samples for global percentiles
    all_hrv = []
    for samples in df["hrv_samples"]:
        if samples:
            all_hrv.extend([s for s in samples if s is not None])

    hr_arr = np.array(all_hr) if all_hr else np.array([0])
    hrv_arr = np.array(all_hrv) if all_hrv else np.array([0])

    return SleepAnalytics(
        start_date=actual_start,
        end_date=actual_end,
        median_sleep_duration=float(round(median_duration, 2)),
        median_avg_hr=float(round(median_avg_hr, 1)),
        median_avg_hrv=float(round(median_avg_hrv, 1)),
        hr_20th_percentile=float(round(np.percentile(hr_arr, 20), 1)),
        hr_80th_percentile=float(round(np.percentile(hr_arr, 80), 1)),
        hrv_20th_percentile=float(round(np.percentile(hrv_arr, 20), 1)),
        hrv_80th_percentile=float(round(np.percentile(hrv_arr, 80), 1)),
    )


def oura_heartrate_to_dataframe(heartrate_data: HeartRateData) -> pd.DataFrame:
    """Convert Oura heart rate API response to DataFrame."""
    records = []
    for sample in heartrate_data.data:
        records.append({
            "bpm": sample.bpm,
            "source": sample.source,
            "timestamp": sample.timestamp,
        })
    df = pd.DataFrame(records)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["day"] = df["timestamp"].dt.date
    return df


def resample_heartrate(df: pd.DataFrame) -> pd.DataFrame:
    """Resample heart rate data to 1-minute intervals.
    
    Rules:
    - If two consecutive points are more than HR_MAX_GAP_SECONDS apart,
      they are not considered consecutive and we don't resample between them.
    - If multiple points fall within the same minute, take their max.
    - Resamples to 1-minute intervals, forward-filling values for gaps <= HR_MAX_GAP_SECONDS.
      E.g., a reading at 10:00 with next reading at 10:03 produces values at 10:00, 10:01, 10:02.
    
    Args:
        df: DataFrame with 'timestamp', 'bpm', 'source', 'day' columns.
            Should already be filtered to desired sources.
    
    Returns:
        Resampled DataFrame with 1-minute interval HR values.
    """
    if df.empty:
        return df
    
    # Work with a copy and ensure sorted by timestamp
    df = df.copy()
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # Calculate time gaps between consecutive points
    df["time_diff"] = df["timestamp"].diff().dt.total_seconds()
    
    # Mark segment boundaries where gap > HR_MAX_GAP_SECONDS
    df["segment"] = (df["time_diff"] > HR_MAX_GAP_SECONDS).cumsum()
    
    resampled_segments = []
    
    for segment_id, segment_df in df.groupby("segment"):
        if segment_df.empty:
            continue
        
        # Set timestamp as index for resampling
        segment_df = segment_df.set_index("timestamp")
        
        # Create a complete 1-minute range from first to last timestamp in segment
        start_time = segment_df.index.min().floor(HR_RESAMPLE_INTERVAL)
        end_time = segment_df.index.max().floor(HR_RESAMPLE_INTERVAL)
        
        # First, aggregate multiple readings within same minute by taking max
        aggregated = segment_df["bpm"].resample(HR_RESAMPLE_INTERVAL, closed="left", label="left").max()
        
        # Create complete minute range and reindex with forward fill
        full_range = pd.date_range(start=start_time, end=end_time, freq=HR_RESAMPLE_INTERVAL)
        resampled = aggregated.reindex(full_range).ffill()
        
        # Drop any remaining NaN (shouldn't happen, but safety)
        resampled = resampled.dropna()
        
        if not resampled.empty:
            segment_result = pd.DataFrame({
                "timestamp": resampled.index,
                "bpm": resampled.values,
            })
            resampled_segments.append(segment_result)
    
    if not resampled_segments:
        return pd.DataFrame(columns=["timestamp", "bpm", "day"])
    
    # Combine all segments
    result = pd.concat(resampled_segments, ignore_index=True)
    result["day"] = result["timestamp"].dt.date
    
    return result


def analyze_heart_rate(df: pd.DataFrame) -> HeartRateAnalytics:
    """Compute aggregate heart rate analytics from DataFrame (active heart rate only).
    
    Filters to active sources: awake, live, workout (excludes rest/sleep data).
    Resamples to 1-minute intervals before computing statistics.
    """
    # Filter to active sources only
    if not df.empty:
        active_df = df.loc[df["source"].isin(NON_SLEEP_HR_SOURCES)]
    else:
        active_df = df
    
    # Resample to 1-minute intervals
    resampled_df = resample_heartrate(active_df)
    
    # Get actual date range from the data
    if not resampled_df.empty:
        min_day = resampled_df["day"].min()
        max_day = resampled_df["day"].max()
        actual_start: date | None = date(min_day.year, min_day.month, min_day.day)
        actual_end: date | None = date(max_day.year, max_day.month, max_day.day)
        hr_values = np.array(resampled_df["bpm"].tolist())
    else:
        actual_start = None
        actual_end = None
        hr_values = np.array([0])

    average_hr = float(np.mean(hr_values)) if len(hr_values) > 0 else 0.0

    return HeartRateAnalytics(
        start_date=actual_start,
        end_date=actual_end,
        average_hr=float(round(average_hr, 1)),
        hr_20th_percentile=float(round(np.percentile(hr_values, 20), 1)),
        hr_50th_percentile=float(round(np.percentile(hr_values, 50), 1)),
        hr_80th_percentile=float(round(np.percentile(hr_values, 80), 1)),
        hr_95th_percentile=float(round(np.percentile(hr_values, 95), 1)),
        hr_99th_percentile=float(round(np.percentile(hr_values, 99), 1)),
    )


def analyze_heart_rate_daily(df: pd.DataFrame) -> list[DailyHeartRateAnalytics]:
    """Compute daily heart rate analytics from DataFrame (active heart rate only).
    
    Filters to active sources: awake, live, workout (excludes rest/sleep data).
    Resamples to 1-minute intervals before computing statistics.
    """
    if df.empty:
        return []
    
    # Filter to active sources only
    active_df = df.loc[df["source"].isin(NON_SLEEP_HR_SOURCES)]
    
    if active_df.empty:
        return []
    
    # Resample to 1-minute intervals
    resampled_df = resample_heartrate(active_df)
    
    if resampled_df.empty:
        return []
    
    results: list[DailyHeartRateAnalytics] = []
    for day, group in resampled_df.groupby("day"):
        hr_values = np.array(group["bpm"].tolist())
        if len(hr_values) == 0:
            continue
        
        results.append(DailyHeartRateAnalytics(
            day=date(day.year, day.month, day.day),  # type: ignore[union-attr]
            average_hr=float(round(np.mean(hr_values), 1)),
            hr_20th_percentile=float(round(np.percentile(hr_values, 20), 1)),
            hr_50th_percentile=float(round(np.percentile(hr_values, 50), 1)),
            hr_80th_percentile=float(round(np.percentile(hr_values, 80), 1)),
            hr_95th_percentile=float(round(np.percentile(hr_values, 95), 1)),
            hr_99th_percentile=float(round(np.percentile(hr_values, 99), 1)),
        ))
    
    # Sort by day
    results.sort(key=lambda x: x.day)
    return results

