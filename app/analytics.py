"""Sleep data analytics module. No external API dependencies."""

from dataclasses import dataclass
from datetime import date
from typing import Optional
import pandas as pd
import numpy as np

from app.oura_client import SleepData, HeartRateData

# Sources considered "active" heart rate (not resting/sleep)
ACTIVE_HR_SOURCES = {"awake", "live", "workout"}


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


def oura_sleep_to_dataframe(sleep_data: list[SleepData]) -> pd.DataFrame:
    """Convert Oura sleep API response to DataFrame."""
    records = []
    for sleep in sleep_data:
        record = {
            "id": sleep.id,
            "day": sleep.day,
            "total_sleep_duration": sleep.total_sleep_duration,
            "average_heart_rate": sleep.average_heart_rate,
            "average_hrv": sleep.average_hrv,
            "heart_rate_samples": sleep.heart_rate.items if sleep.heart_rate else [],
            "hrv_samples": sleep.hrv.items if sleep.hrv else [],
        }
        records.append(record)
    return pd.DataFrame(records)


def analyze_sleep(df: pd.DataFrame) -> SleepAnalytics:
    """Compute sleep analytics from DataFrame."""
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


def analyze_heart_rate(df: pd.DataFrame) -> HeartRateAnalytics:
    """Compute aggregate heart rate analytics from DataFrame (active heart rate only).
    
    Filters to active sources: awake, live, workout (excludes rest/sleep data).
    """
    # Filter to active sources only
    if not df.empty:
        active_df = df[df["source"].isin(ACTIVE_HR_SOURCES)]
    else:
        active_df = df
    
    # Get actual date range from the data
    if not active_df.empty:
        min_day = active_df["day"].min()
        max_day = active_df["day"].max()
        actual_start: date | None = date(min_day.year, min_day.month, min_day.day)
        actual_end: date | None = date(max_day.year, max_day.month, max_day.day)
        hr_values = np.array(active_df["bpm"].tolist())
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
    """
    if df.empty:
        return []
    
    # Filter to active sources only
    active_df = df[df["source"].isin(ACTIVE_HR_SOURCES)]
    
    if active_df.empty:
        return []
    
    results: list[DailyHeartRateAnalytics] = []
    for day, group in active_df.groupby("day"):
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

