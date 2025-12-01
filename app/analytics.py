"""Sleep data analytics module. No external API dependencies."""

from dataclasses import dataclass
from datetime import date
from typing import Optional
import pandas as pd
import numpy as np

from app.oura_client import SleepData, HeartRateData

# Sources considered "active" heart rate (not resting/sleep)
NON_SLEEP_HR_SOURCES = {"awake", "live", "workout"}

# Default maximum gap (in seconds) between consecutive HR points to consider them connected
# Points further apart than this are not interpolated between
DEFAULT_HR_MAX_GAP_SECONDS = 300

# Default resampling interval for heart rate data
DEFAULT_HR_RESAMPLE_INTERVAL = "1min"


@dataclass
class SleepAnalytics:
    """Sleep analytics results."""
    start_date: Optional[date]
    end_date: Optional[date]
    median_sleep_duration: float
    sleep_duration_std: float
    median_avg_hr: float
    avg_hr_std: float
    median_avg_hrv: float
    avg_hrv_std: float
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
    average_hr_std: float
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
    sleep_duration_std = df["total_sleep_duration"].std()

    # Median of average HR and HRV
    median_avg_hr = df["average_heart_rate"].median()
    avg_hr_std = df["average_heart_rate"].std()
    median_avg_hrv = df["average_hrv"].median()
    avg_hrv_std = df["average_hrv"].std()

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
        median_sleep_duration=float(round(median_duration, 2)) if not np.isnan(median_duration) else 0.0,
        sleep_duration_std=float(round(sleep_duration_std, 2)) if not np.isnan(sleep_duration_std) else 0.0,
        median_avg_hr=float(round(median_avg_hr, 1)) if not np.isnan(median_avg_hr) else 0.0,
        avg_hr_std=float(round(avg_hr_std, 1)) if not np.isnan(avg_hr_std) else 0.0,
        median_avg_hrv=float(round(median_avg_hrv, 1)) if not np.isnan(median_avg_hrv) else 0.0,
        avg_hrv_std=float(round(avg_hrv_std, 1)) if not np.isnan(avg_hrv_std) else 0.0,
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


def resample_heartrate(
    df: pd.DataFrame,
    max_gap_seconds: int = DEFAULT_HR_MAX_GAP_SECONDS,
    resample_interval: str = DEFAULT_HR_RESAMPLE_INTERVAL,
) -> pd.DataFrame:
    """Resample heart rate data to regular intervals.
    
    This function resamples heart rate data across all sources as a single merged stream.
    Consecutive points from different sources (e.g., rest->awake) are connected if within
    the max gap threshold, allowing for continuous resampling across source transitions.
    
    Rules:
    - If two consecutive points are more than max_gap_seconds apart,
      they are not considered consecutive and we don't resample between them.
    - If multiple points fall within the same interval, take their max.
    - Forward-fills values for gaps <= max_gap_seconds.
      E.g., with 1min interval, a reading at 10:00 with next reading at 10:03 
      produces values at 10:00, 10:01, 10:02.
    
    Args:
        df: DataFrame with 'timestamp', 'bpm', 'source', 'day' columns.
        max_gap_seconds: Maximum gap in seconds between consecutive points to consider 
            them connected. Points further apart create separate segments.
            Default is 300 seconds (5 minutes).
        resample_interval: Pandas frequency string for resampling interval.
            Default is "1min".
    
    Returns:
        Resampled DataFrame with interval HR values, containing columns:
        'timestamp', 'bpm', 'day'.
    """
    if df.empty:
        return df
    
    # Work with a copy and ensure sorted by timestamp
    df = df.copy()
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # Calculate time gaps between consecutive points
    df["time_diff"] = df["timestamp"].diff().dt.total_seconds()
    
    # Mark segment boundaries where gap > max_gap_seconds
    df["segment"] = (df["time_diff"] > max_gap_seconds).cumsum()
    
    resampled_segments = []
    
    for segment_id, segment_df in df.groupby("segment"):
        if segment_df.empty:
            continue
        
        # Set timestamp as index for resampling
        segment_df = segment_df.set_index("timestamp")
        
        # Create a complete range from first to last timestamp in segment
        start_time = segment_df.index.min().floor(resample_interval)
        end_time = segment_df.index.max().floor(resample_interval)
        
        # First, aggregate multiple readings within same interval by taking max
        aggregated = segment_df["bpm"].resample(resample_interval, closed="left", label="left").max()
        
        # Create complete range and reindex with forward fill
        full_range = pd.date_range(start=start_time, end=end_time, freq=resample_interval)
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


def analyze_heart_rate(
    df: pd.DataFrame,
    max_gap_seconds: int = DEFAULT_HR_MAX_GAP_SECONDS,
    resample_interval: str = DEFAULT_HR_RESAMPLE_INTERVAL,
) -> HeartRateAnalytics:
    """Compute aggregate heart rate analytics from DataFrame (active heart rate only).
    
    Resamples ALL sources as a merged stream to ensure continuous data across
    source transitions, then filters to only include timestamps that had active
    sources (awake, live, workout) in the original data.
    
    Args:
        df: DataFrame with 'timestamp', 'bpm', 'source', 'day' columns.
        max_gap_seconds: Maximum gap in seconds between consecutive points to consider 
            them connected during resampling. Default is 300 seconds (5 minutes).
        resample_interval: Pandas frequency string for resampling interval.
            Default is "1min".
    
    Returns:
        HeartRateAnalytics with aggregate statistics from active HR data.
    """
    if df.empty:
        return HeartRateAnalytics(
            start_date=None,
            end_date=None,
            average_hr=0.0,
            average_hr_std=0.0,
            hr_20th_percentile=0.0,
            hr_50th_percentile=0.0,
            hr_80th_percentile=0.0,
            hr_95th_percentile=0.0,
            hr_99th_percentile=0.0,
        )
    
    # Get timestamps that have active sources
    active_timestamps = set(
        df.loc[df["source"].isin(NON_SLEEP_HR_SOURCES), "timestamp"]
        .dt.floor(resample_interval)
    )
    
    if not active_timestamps:
        return HeartRateAnalytics(
            start_date=None,
            end_date=None,
            average_hr=0.0,
            average_hr_std=0.0,
            hr_20th_percentile=0.0,
            hr_50th_percentile=0.0,
            hr_80th_percentile=0.0,
            hr_95th_percentile=0.0,
            hr_99th_percentile=0.0,
        )
    
    # Resample ALL sources as a merged stream
    resampled_df = resample_heartrate(df, max_gap_seconds, resample_interval)
    
    if resampled_df.empty:
        return HeartRateAnalytics(
            start_date=None,
            end_date=None,
            average_hr=0.0,
            average_hr_std=0.0,
            hr_20th_percentile=0.0,
            hr_50th_percentile=0.0,
            hr_80th_percentile=0.0,
            hr_95th_percentile=0.0,
            hr_99th_percentile=0.0,
        )
    
    # Filter resampled data to only timestamps that had active sources
    resampled_df = resampled_df[resampled_df["timestamp"].isin(active_timestamps)]
    
    if resampled_df.empty:
        return HeartRateAnalytics(
            start_date=None,
            end_date=None,
            average_hr=0.0,
            average_hr_std=0.0,
            hr_20th_percentile=0.0,
            hr_50th_percentile=0.0,
            hr_80th_percentile=0.0,
            hr_95th_percentile=0.0,
            hr_99th_percentile=0.0,
        )
    
    # Get actual date range from the data
    min_day = resampled_df["day"].min()
    max_day = resampled_df["day"].max()
    actual_start: date | None = date(min_day.year, min_day.month, min_day.day)
    actual_end: date | None = date(max_day.year, max_day.month, max_day.day)
    hr_values = np.array(resampled_df["bpm"].tolist())

    average_hr = float(np.mean(hr_values)) if len(hr_values) > 0 else 0.0
    average_hr_std = float(np.std(hr_values)) if len(hr_values) > 0 else 0.0

    return HeartRateAnalytics(
        start_date=actual_start,
        end_date=actual_end,
        average_hr=float(round(average_hr, 1)),
        average_hr_std=float(round(average_hr_std, 1)),
        hr_20th_percentile=float(round(np.percentile(hr_values, 20), 1)),
        hr_50th_percentile=float(round(np.percentile(hr_values, 50), 1)),
        hr_80th_percentile=float(round(np.percentile(hr_values, 80), 1)),
        hr_95th_percentile=float(round(np.percentile(hr_values, 95), 1)),
        hr_99th_percentile=float(round(np.percentile(hr_values, 99), 1)),
    )


def analyze_heart_rate_daily(
    df: pd.DataFrame,
    max_gap_seconds: int = DEFAULT_HR_MAX_GAP_SECONDS,
    resample_interval: str = DEFAULT_HR_RESAMPLE_INTERVAL,
) -> list[DailyHeartRateAnalytics]:
    """Compute daily heart rate analytics from DataFrame (active heart rate only).
    
    Resamples ALL sources as a merged stream to ensure continuous data across
    source transitions, then filters to only include timestamps that had active
    sources (awake, live, workout) in the original data.
    
    Args:
        df: DataFrame with 'timestamp', 'bpm', 'source', 'day' columns.
        max_gap_seconds: Maximum gap in seconds between consecutive points to consider 
            them connected during resampling. Default is 300 seconds (5 minutes).
        resample_interval: Pandas frequency string for resampling interval.
            Default is "1min".
    
    Returns:
        List of DailyHeartRateAnalytics sorted by day.
    """
    if df.empty:
        return []
    
    # Get timestamps that have active sources
    active_timestamps = set(
        df.loc[df["source"].isin(NON_SLEEP_HR_SOURCES), "timestamp"]
        .dt.floor(resample_interval)
    )
    
    if not active_timestamps:
        return []
    
    # Resample ALL sources as a merged stream
    resampled_df = resample_heartrate(df, max_gap_seconds, resample_interval)
    
    if resampled_df.empty:
        return []
    
    # Filter resampled data to only timestamps that had active sources
    resampled_df = resampled_df[resampled_df["timestamp"].isin(active_timestamps)]
    
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

