"""Sleep data analytics module. No external API dependencies."""

from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional, Sequence
import pandas as pd
import numpy as np

from app.oura_client import SleepData, HeartRateData

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
    nights_count: int
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

    @classmethod
    def empty(cls) -> "SleepAnalytics":
        """Create an empty SleepAnalytics instance with default values."""
        return cls(
            start_date=None,
            end_date=None,
            nights_count=0,
            median_sleep_duration=0.0,
            sleep_duration_std=0.0,
            median_avg_hr=0.0,
            avg_hr_std=0.0,
            median_avg_hrv=0.0,
            avg_hrv_std=0.0,
            hr_20th_percentile=0.0,
            hr_80th_percentile=0.0,
            hrv_20th_percentile=0.0,
            hrv_80th_percentile=0.0,
        )


@dataclass
class HeartRateAnalytics:
    """Heart rate analytics results (aggregated across all days)."""
    start_date: Optional[date]
    end_date: Optional[date]
    hours_with_good_data: int
    sleep_hours_filtered: float
    average_hr: float
    average_hr_std: float
    hr_20th_percentile: float
    hr_50th_percentile: float
    hr_80th_percentile: float
    hr_95th_percentile: float
    hr_99th_percentile: float

    @classmethod
    def empty(cls) -> "HeartRateAnalytics":
        """Create an empty HeartRateAnalytics instance with default values."""
        return cls(
            start_date=None,
            end_date=None,
            hours_with_good_data=0,
            sleep_hours_filtered=0.0,
            average_hr=0.0,
            average_hr_std=0.0,
            hr_20th_percentile=0.0,
            hr_50th_percentile=0.0,
            hr_80th_percentile=0.0,
            hr_95th_percentile=0.0,
            hr_99th_percentile=0.0,
        )


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
            "bedtime_start": sleep.bedtime_start,
            "bedtime_end": sleep.bedtime_end,
            "total_sleep_duration": sleep.total_sleep_duration,
            "average_heart_rate": sleep.average_heart_rate,
            "average_hrv": sleep.average_hrv,
            "heart_rate_samples": sleep.heart_rate.items if sleep.heart_rate else [],
            "hrv_samples": sleep.hrv.items if sleep.hrv else [],
        }
        records.append(record)
    return pd.DataFrame(records)


def get_sleep_intervals(sleep_df: pd.DataFrame) -> list[tuple[datetime, datetime]]:
    """Extract sleep time intervals from sleep DataFrame.
    
    Only considers 'long_sleep' type entries (actual overnight sleep).
    
    Args:
        sleep_df: DataFrame with 'type', 'bedtime_start', 'bedtime_end' columns.
    
    Returns:
        List of (start, end) datetime tuples representing sleep periods.
    """
    if sleep_df.empty:
        return []
    
    # Filter to long_sleep type only
    if "type" in sleep_df.columns:
        sleep_df = sleep_df[sleep_df["type"] == NIGH_SLEEP_SLEEP_TYPE]
    
    intervals = []
    for _, row in sleep_df.iterrows():
        start = row.get("bedtime_start")
        end = row.get("bedtime_end")
        if start and end:
            try:
                start_dt = pd.to_datetime(start)
                end_dt = pd.to_datetime(end)
                intervals.append((start_dt, end_dt))
            except (ValueError, TypeError):
                continue
    
    return intervals


def filter_hr_outside_sleep(hr_df: pd.DataFrame, sleep_intervals: Sequence[tuple[datetime, datetime]]) -> tuple[pd.DataFrame, float]:
    """Filter heart rate DataFrame to exclude samples during sleep periods.
    
    Note: This implementation is O(n_samples * n_intervals) which scales poorly with
    many sleep intervals. However, this is acceptable for now because:
    - We have very few users
    - Oura history is typically not that long
    - We expect to analyze ~1 month of data at a time (so ~30 sleep intervals max)
    
    Args:
        hr_df: DataFrame with 'timestamp' column.
        sleep_intervals: List of (start, end) datetime tuples representing sleep periods.
    
    Returns:
        Tuple of (filtered DataFrame, total sleep hours filtered out).
    """
    # Calculate total sleep hours from intervals
    total_sleep_hours = sum(
        (end - start).total_seconds() / 3600
        for start, end in sleep_intervals
    )
    
    if hr_df.empty or not sleep_intervals:
        return hr_df, total_sleep_hours
    
    # Create a mask for samples that are NOT during any sleep period
    mask = pd.Series(True, index=hr_df.index)
    
    for start, end in sleep_intervals:
        # Mark samples during this sleep period as False
        sleep_mask = (hr_df["timestamp"] >= start) & (hr_df["timestamp"] <= end)
        mask = mask & ~sleep_mask
    
    return hr_df[mask].copy(), total_sleep_hours


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
    
    # Count the number of nights (rows in filtered dataframe)
    nights_count = len(df)

    return SleepAnalytics(
        start_date=actual_start,
        end_date=actual_end,
        nights_count=nights_count,
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
        df["day"] = pd.to_datetime(df["timestamp"]).dt.date
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
    df["time_diff"] = pd.to_timedelta(df["timestamp"].diff()).dt.total_seconds()
    
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
    hr_df: pd.DataFrame,
    sleep_intervals: Sequence[tuple[datetime, datetime]],
    max_gap_seconds: int = DEFAULT_HR_MAX_GAP_SECONDS,
    resample_interval: str = DEFAULT_HR_RESAMPLE_INTERVAL,
) -> HeartRateAnalytics:
    """Compute aggregate heart rate analytics from DataFrame (excluding sleep periods).
    
    Filters out HR samples that occur during sleep periods, then resamples.
    
    Args:
        hr_df: DataFrame with 'timestamp', 'bpm', 'source', 'day' columns.
        sleep_intervals: List of (start, end) datetime tuples representing sleep periods.
        max_gap_seconds: Maximum gap in seconds between consecutive points to consider 
            them connected during resampling. Default is 300 seconds (5 minutes).
        resample_interval: Pandas frequency string for resampling interval.
            Default is "1min".
    
    Returns:
        HeartRateAnalytics with aggregate statistics from non-sleep HR data.
    """
    if hr_df.empty:
        return HeartRateAnalytics.empty()
    
    # Filter out HR samples during sleep periods
    filtered_df, sleep_hours_filtered = filter_hr_outside_sleep(hr_df, sleep_intervals)
    
    if filtered_df.empty:
        return HeartRateAnalytics.empty()
    
    # Resample the filtered data
    resampled_df = resample_heartrate(filtered_df, max_gap_seconds, resample_interval)
    
    if resampled_df.empty:
        return HeartRateAnalytics.empty()
    
    # Get actual date range from the data
    min_day = resampled_df["day"].min()
    max_day = resampled_df["day"].max()
    actual_start: date | None = date(min_day.year, min_day.month, min_day.day)
    actual_end: date | None = date(max_day.year, max_day.month, max_day.day)
    hr_values = np.array(resampled_df["bpm"].tolist())

    average_hr = float(np.mean(hr_values)) if len(hr_values) > 0 else 0.0
    average_hr_std = float(np.std(hr_values)) if len(hr_values) > 0 else 0.0
    
    # Count hours with more than 10 samples (based on resampled data)
    resampled_df = resampled_df.copy()
    resampled_df["hour"] = pd.to_datetime(resampled_df["timestamp"]).dt.floor("h")
    hourly_counts = resampled_df.groupby("hour").size()
    hours_with_good_data = int((hourly_counts > 10).sum())

    return HeartRateAnalytics(
        start_date=actual_start,
        end_date=actual_end,
        hours_with_good_data=hours_with_good_data,
        sleep_hours_filtered=float(round(sleep_hours_filtered, 1)),
        average_hr=float(round(average_hr, 1)),
        average_hr_std=float(round(average_hr_std, 1)),
        hr_20th_percentile=float(round(np.percentile(hr_values, 20), 1)),
        hr_50th_percentile=float(round(np.percentile(hr_values, 50), 1)),
        hr_80th_percentile=float(round(np.percentile(hr_values, 80), 1)),
        hr_95th_percentile=float(round(np.percentile(hr_values, 95), 1)),
        hr_99th_percentile=float(round(np.percentile(hr_values, 99), 1)),
    )


def analyze_heart_rate_daily(
    hr_df: pd.DataFrame,
    sleep_intervals: Sequence[tuple[datetime, datetime]],
    max_gap_seconds: int = DEFAULT_HR_MAX_GAP_SECONDS,
    resample_interval: str = DEFAULT_HR_RESAMPLE_INTERVAL,
) -> list[DailyHeartRateAnalytics]:
    """Compute daily heart rate analytics from DataFrame (excluding sleep periods).
    
    Filters out HR samples that occur during sleep periods, then resamples.
    
    Args:
        hr_df: DataFrame with 'timestamp', 'bpm', 'source', 'day' columns.
        sleep_intervals: List of (start, end) datetime tuples representing sleep periods.
        max_gap_seconds: Maximum gap in seconds between consecutive points to consider 
            them connected during resampling. Default is 300 seconds (5 minutes).
        resample_interval: Pandas frequency string for resampling interval.
            Default is "1min".
    
    Returns:
        List of DailyHeartRateAnalytics sorted by day.
    """
    if hr_df.empty:
        return []
    
    # Filter out HR samples during sleep periods
    filtered_df, _ = filter_hr_outside_sleep(hr_df, sleep_intervals)
    
    if filtered_df.empty:
        return []
    
    # Resample the filtered data
    resampled_df = resample_heartrate(filtered_df, max_gap_seconds, resample_interval)
    
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


@dataclass
class CombinedAnalytics:
    """Combined sleep and heart rate analytics."""
    sleep: SleepAnalytics
    heart_rate: HeartRateAnalytics


@dataclass
class CombinedDailyAnalytics:
    """Combined daily analytics."""
    days: list[DailyHeartRateAnalytics]


@dataclass
class PeriodComparisonResult:
    """Result of comparing two time periods (e.g., pre/post treatment)."""
    pre_period: CombinedAnalytics
    post_period: CombinedAnalytics
    # Period date ranges
    pre_start: date
    pre_end: date
    post_start: date
    post_end: date
    # Deltas (post - pre)
    sleep_duration_diff: float  # seconds
    median_hr_diff: float  # bpm (during sleep)
    median_hrv_diff: float  # ms (during sleep)
    hr_50th_percentile_diff: float  # bpm (waking)
    # Significance assessment
    suggested_effect: str  # "improved", "worsened", "no_effect"
    significant_change: bool
    significance_details: str  # Explanation of what changed significantly


def analyze_combined(
    sleep_df: pd.DataFrame,
    hr_df: pd.DataFrame,
    max_gap_seconds: int = DEFAULT_HR_MAX_GAP_SECONDS,
    resample_interval: str = DEFAULT_HR_RESAMPLE_INTERVAL,
) -> CombinedAnalytics:
    """Compute combined sleep and heart rate analytics.
    
    Uses sleep periods from sleep_df to filter out HR samples during sleep.
    
    Args:
        sleep_df: DataFrame with sleep data (from oura_sleep_to_dataframe).
        hr_df: DataFrame with heart rate data (from oura_heartrate_to_dataframe).
        max_gap_seconds: Maximum gap in seconds between consecutive HR points.
        resample_interval: Pandas frequency string for HR resampling interval.
    
    Returns:
        CombinedAnalytics with both sleep and heart rate analytics.
    """
    sleep_analytics = analyze_sleep(sleep_df)
    
    sleep_intervals = get_sleep_intervals(sleep_df)
    hr_analytics = analyze_heart_rate(hr_df, sleep_intervals, max_gap_seconds, resample_interval)
    
    return CombinedAnalytics(sleep=sleep_analytics, heart_rate=hr_analytics)


def analyze_combined_daily(
    sleep_df: pd.DataFrame,
    hr_df: pd.DataFrame,
    max_gap_seconds: int = DEFAULT_HR_MAX_GAP_SECONDS,
    resample_interval: str = DEFAULT_HR_RESAMPLE_INTERVAL,
) -> CombinedDailyAnalytics:
    """Compute combined daily analytics.
    
    Uses sleep periods from sleep_df to filter out HR samples during sleep.
    
    Args:
        sleep_df: DataFrame with sleep data (from oura_sleep_to_dataframe).
        hr_df: DataFrame with heart rate data (from oura_heartrate_to_dataframe).
        max_gap_seconds: Maximum gap in seconds between consecutive HR points.
        resample_interval: Pandas frequency string for HR resampling interval.
    
    Returns:
        CombinedDailyAnalytics with daily heart rate analytics.
    """
    sleep_intervals = get_sleep_intervals(sleep_df)
    daily_hr = analyze_heart_rate_daily(hr_df, sleep_intervals, max_gap_seconds, resample_interval)
    
    return CombinedDailyAnalytics(days=daily_hr)


def compare_periods(
    pre_sleep_df: pd.DataFrame,
    post_sleep_df: pd.DataFrame,
    pre_hr_df: pd.DataFrame,
    post_hr_df: pd.DataFrame,
    pre_start: date,
    pre_end: date,
    post_start: date,
    post_end: date,
    max_gap_seconds: int = DEFAULT_HR_MAX_GAP_SECONDS,
    resample_interval: str = DEFAULT_HR_RESAMPLE_INTERVAL,
) -> PeriodComparisonResult:
    """Compare biometric data between two time periods.
    
    Useful for assessing treatment effects by comparing pre-treatment vs post-treatment periods.
    Uses 2-sigma rule: changes beyond 2x standard deviation of baseline are significant.
    
    Args:
        pre_sleep_df: DataFrame with pre-period sleep data (from oura_sleep_to_dataframe).
        post_sleep_df: DataFrame with post-period sleep data (from oura_sleep_to_dataframe).
        pre_hr_df: DataFrame with pre-period heart rate data (from oura_heartrate_to_dataframe).
        post_hr_df: DataFrame with post-period heart rate data (from oura_heartrate_to_dataframe).
        pre_start: Start date of pre-treatment period.
        pre_end: End date of pre-treatment period.
        post_start: Start date of post-treatment period.
        post_end: End date of post-treatment period.
        max_gap_seconds: Maximum gap in seconds between consecutive HR points.
        resample_interval: Pandas frequency string for HR resampling interval.
    
    Returns:
        PeriodComparisonResult with comparison metrics and significance assessment.
    """
    # Analyze each period
    pre_analytics = analyze_combined(pre_sleep_df, pre_hr_df, max_gap_seconds, resample_interval)
    post_analytics = analyze_combined(post_sleep_df, post_hr_df, max_gap_seconds, resample_interval)
    
    # Calculate deltas
    sleep_duration_diff = post_analytics.sleep.median_sleep_duration - pre_analytics.sleep.median_sleep_duration
    median_hr_diff = post_analytics.sleep.median_avg_hr - pre_analytics.sleep.median_avg_hr
    median_hrv_diff = post_analytics.sleep.median_avg_hrv - pre_analytics.sleep.median_avg_hrv
    hr_50th_diff = post_analytics.heart_rate.hr_50th_percentile - pre_analytics.heart_rate.hr_50th_percentile
    
    # Assess significance using 2-sigma rule
    # For each metric, check if the change exceeds 2x the baseline std
    significant_changes = []
    
    # Sleep duration: significant if change > 2 * std
    if pre_analytics.sleep.sleep_duration_std > 0:
        if abs(sleep_duration_diff) > 2 * pre_analytics.sleep.sleep_duration_std:
            direction = "increased" if sleep_duration_diff > 0 else "decreased"
            significant_changes.append(f"Sleep duration {direction}")
    
    # Sleep HR: significant if change > 2 * std
    if pre_analytics.sleep.avg_hr_std > 0:
        if abs(median_hr_diff) > 2 * pre_analytics.sleep.avg_hr_std:
            direction = "increased" if median_hr_diff > 0 else "decreased"
            significant_changes.append(f"Sleep heart rate {direction}")
    
    # Sleep HRV: significant if change > 2 * std
    if pre_analytics.sleep.avg_hrv_std > 0:
        if abs(median_hrv_diff) > 2 * pre_analytics.sleep.avg_hrv_std:
            direction = "increased" if median_hrv_diff > 0 else "decreased"
            significant_changes.append(f"Heart rate variability {direction}")
    
    # Waking HR: significant if change > 2 * std
    if pre_analytics.heart_rate.average_hr_std > 0:
        if abs(hr_50th_diff) > 2 * pre_analytics.heart_rate.average_hr_std:
            direction = "increased" if hr_50th_diff > 0 else "decreased"
            significant_changes.append(f"Waking heart rate {direction}")
    
    significant_change = len(significant_changes) > 0
    significance_details = "; ".join(significant_changes) if significant_changes else "No significant changes detected"
    
    # Determine suggested effect
    # Improved: lower HR (sleep or waking), higher HRV, or longer sleep
    # Worsened: higher HR, lower HRV, or shorter sleep
    improvement_score = 0
    if significant_change:
        # Lower HR is generally better
        if median_hr_diff < 0 and "Sleep heart rate" in significance_details:
            improvement_score += 1
        elif median_hr_diff > 0 and "Sleep heart rate" in significance_details:
            improvement_score -= 1
        
        if hr_50th_diff < 0 and "Waking heart rate" in significance_details:
            improvement_score += 1
        elif hr_50th_diff > 0 and "Waking heart rate" in significance_details:
            improvement_score -= 1
        
        # Higher HRV is generally better
        if median_hrv_diff > 0 and "Heart rate variability" in significance_details:
            improvement_score += 1
        elif median_hrv_diff < 0 and "Heart rate variability" in significance_details:
            improvement_score -= 1
        
        # Longer sleep is generally better (within reason)
        if sleep_duration_diff > 0 and "Sleep duration" in significance_details:
            improvement_score += 1
        elif sleep_duration_diff < 0 and "Sleep duration" in significance_details:
            improvement_score -= 1
    
    if improvement_score > 0:
        suggested_effect = "improved"
    elif improvement_score < 0:
        suggested_effect = "worsened"
    else:
        suggested_effect = "no_effect"
    
    return PeriodComparisonResult(
        pre_period=pre_analytics,
        post_period=post_analytics,
        pre_start=pre_start,
        pre_end=pre_end,
        post_start=post_start,
        post_end=post_end,
        sleep_duration_diff=float(round(sleep_duration_diff, 2)),
        median_hr_diff=float(round(median_hr_diff, 1)),
        median_hrv_diff=float(round(median_hrv_diff, 1)),
        hr_50th_percentile_diff=float(round(hr_50th_diff, 1)),
        suggested_effect=suggested_effect,
        significant_change=significant_change,
        significance_details=significance_details,
    )

