"""Garmin data analytics module. No external API dependencies."""

from dataclasses import dataclass
from datetime import date
from typing import Optional
import pandas as pd
import numpy as np

from app.garmin_client import SleepHRVData, StressData, DailySleepData


@dataclass
class GarminSleepAnalytics:
    """Sleep analytics results from Garmin data."""
    start_date: Optional[date]
    end_date: Optional[date]
    median_sleep_duration: float  # in seconds
    median_deep_sleep: float  # in seconds
    median_rem_sleep: float  # in seconds
    median_light_sleep: float  # in seconds
    median_avg_stress: float


@dataclass
class GarminHRVAnalytics:
    """HRV analytics results from Garmin data."""
    start_date: Optional[date]
    end_date: Optional[date]
    median_weekly_avg: float
    median_last_night_avg: float
    median_5_min_high: float
    hrv_20th_percentile: float
    hrv_50th_percentile: float
    hrv_80th_percentile: float


@dataclass
class GarminStressAnalytics:
    """Stress analytics results from Garmin data."""
    start_date: Optional[date]
    end_date: Optional[date]
    average_stress_level: float
    median_stress_level: float
    stress_20th_percentile: float
    stress_50th_percentile: float
    stress_80th_percentile: float
    stress_95th_percentile: float
    # Duration statistics (in seconds)
    avg_rest_duration: float
    avg_low_stress_duration: float
    avg_medium_stress_duration: float
    avg_high_stress_duration: float


def garmin_sleep_to_dataframe(sleep_data: list[DailySleepData]) -> pd.DataFrame:
    """Convert Garmin sleep data to DataFrame."""
    records = []
    for sleep in sleep_data:
        record = {
            "calendar_date": sleep.calendar_date,
            "sleep_time_seconds": sleep.sleep_time_seconds,
            "deep_sleep_seconds": sleep.deep_sleep_seconds,
            "light_sleep_seconds": sleep.light_sleep_seconds,
            "rem_sleep_seconds": sleep.rem_sleep_seconds,
            "awake_sleep_seconds": sleep.awake_sleep_seconds,
            "avg_sleep_stress": sleep.avg_sleep_stress,
            "average_sp_o2_value": sleep.average_sp_o2_value,
            "average_respiration_value": sleep.average_respiration_value,
        }
        records.append(record)
    return pd.DataFrame(records)


def garmin_hrv_to_dataframe(hrv_data: list[SleepHRVData]) -> pd.DataFrame:
    """Convert Garmin HRV data to DataFrame."""
    records = []
    for hrv in hrv_data:
        record = {
            "calendar_date": hrv.calendar_date,
            "weekly_avg": hrv.weekly_avg,
            "last_night_avg": hrv.last_night_avg,
            "last_night_5_min_high": hrv.last_night_5_min_high,
            "baseline_low_upper": hrv.baseline_low_upper,
            "baseline_balanced_low": hrv.baseline_balanced_low,
            "baseline_balanced_upper": hrv.baseline_balanced_upper,
            "status": hrv.status,
            "feedback_phrase": hrv.feedback_phrase,
        }
        records.append(record)
    return pd.DataFrame(records)


def garmin_stress_to_dataframe(stress_data: list[StressData]) -> pd.DataFrame:
    """Convert Garmin stress data to DataFrame."""
    records = []
    for stress in stress_data:
        record = {
            "calendar_date": stress.calendar_date,
            "overall_stress_level": stress.overall_stress_level,
            "rest_stress_duration": stress.rest_stress_duration,
            "low_stress_duration": stress.low_stress_duration,
            "medium_stress_duration": stress.medium_stress_duration,
            "high_stress_duration": stress.high_stress_duration,
        }
        records.append(record)
    return pd.DataFrame(records)


def analyze_garmin_sleep(df: pd.DataFrame) -> GarminSleepAnalytics:
    """Compute sleep analytics from Garmin sleep DataFrame."""
    if df.empty:
        return GarminSleepAnalytics(
            start_date=None,
            end_date=None,
            median_sleep_duration=0.0,
            median_deep_sleep=0.0,
            median_rem_sleep=0.0,
            median_light_sleep=0.0,
            median_avg_stress=0.0,
        )
    
    # Get actual date range from the data
    dates = pd.to_datetime(df["calendar_date"]).dt.date
    actual_start = dates.min()
    actual_end = dates.max()
    
    # Compute medians, handling missing data
    median_sleep = df["sleep_time_seconds"].median()
    median_deep = df["deep_sleep_seconds"].median()
    median_rem = df["rem_sleep_seconds"].median()
    median_light = df["light_sleep_seconds"].median()
    median_stress = df["avg_sleep_stress"].median()
    
    return GarminSleepAnalytics(
        start_date=actual_start,
        end_date=actual_end,
        median_sleep_duration=float(round(median_sleep, 2)) if pd.notna(median_sleep) else 0.0,
        median_deep_sleep=float(round(median_deep, 2)) if pd.notna(median_deep) else 0.0,
        median_rem_sleep=float(round(median_rem, 2)) if pd.notna(median_rem) else 0.0,
        median_light_sleep=float(round(median_light, 2)) if pd.notna(median_light) else 0.0,
        median_avg_stress=float(round(median_stress, 1)) if pd.notna(median_stress) else 0.0,
    )


def analyze_garmin_hrv(df: pd.DataFrame) -> GarminHRVAnalytics:
    """Compute HRV analytics from Garmin HRV DataFrame."""
    if df.empty:
        return GarminHRVAnalytics(
            start_date=None,
            end_date=None,
            median_weekly_avg=0.0,
            median_last_night_avg=0.0,
            median_5_min_high=0.0,
            hrv_20th_percentile=0.0,
            hrv_50th_percentile=0.0,
            hrv_80th_percentile=0.0,
        )
    
    # Get actual date range from the data
    dates = pd.to_datetime(df["calendar_date"]).dt.date
    actual_start = dates.min()
    actual_end = dates.max()
    
    # Compute medians
    median_weekly = df["weekly_avg"].median()
    median_last_night = df["last_night_avg"].median()
    median_5_min = df["last_night_5_min_high"].median()
    
    # Compute percentiles from last_night_avg values
    hrv_values = df["last_night_avg"].dropna().values
    if len(hrv_values) == 0:
        hrv_values = np.array([0])
    
    return GarminHRVAnalytics(
        start_date=actual_start,
        end_date=actual_end,
        median_weekly_avg=float(round(median_weekly, 1)) if pd.notna(median_weekly) else 0.0,
        median_last_night_avg=float(round(median_last_night, 1)) if pd.notna(median_last_night) else 0.0,
        median_5_min_high=float(round(median_5_min, 1)) if pd.notna(median_5_min) else 0.0,
        hrv_20th_percentile=float(round(np.percentile(hrv_values, 20), 1)),
        hrv_50th_percentile=float(round(np.percentile(hrv_values, 50), 1)),
        hrv_80th_percentile=float(round(np.percentile(hrv_values, 80), 1)),
    )


def analyze_garmin_stress(df: pd.DataFrame) -> GarminStressAnalytics:
    """Compute stress analytics from Garmin stress DataFrame.
    
    Computes percentiles for stress levels as specified in the requirements.
    """
    if df.empty:
        return GarminStressAnalytics(
            start_date=None,
            end_date=None,
            average_stress_level=0.0,
            median_stress_level=0.0,
            stress_20th_percentile=0.0,
            stress_50th_percentile=0.0,
            stress_80th_percentile=0.0,
            stress_95th_percentile=0.0,
            avg_rest_duration=0.0,
            avg_low_stress_duration=0.0,
            avg_medium_stress_duration=0.0,
            avg_high_stress_duration=0.0,
        )
    
    # Get actual date range from the data
    dates = pd.to_datetime(df["calendar_date"]).dt.date
    actual_start = dates.min()
    actual_end = dates.max()
    
    # Compute stress level statistics
    stress_values = df["overall_stress_level"].dropna().values
    if len(stress_values) == 0:
        stress_values = np.array([0])
    
    average_stress = float(np.mean(stress_values))
    median_stress = float(np.median(stress_values))
    
    # Duration statistics
    avg_rest = df["rest_stress_duration"].mean()
    avg_low = df["low_stress_duration"].mean()
    avg_medium = df["medium_stress_duration"].mean()
    avg_high = df["high_stress_duration"].mean()
    
    return GarminStressAnalytics(
        start_date=actual_start,
        end_date=actual_end,
        average_stress_level=float(round(average_stress, 1)),
        median_stress_level=float(round(median_stress, 1)),
        stress_20th_percentile=float(round(np.percentile(stress_values, 20), 1)),
        stress_50th_percentile=float(round(np.percentile(stress_values, 50), 1)),
        stress_80th_percentile=float(round(np.percentile(stress_values, 80), 1)),
        stress_95th_percentile=float(round(np.percentile(stress_values, 95), 1)),
        avg_rest_duration=float(round(avg_rest, 2)) if pd.notna(avg_rest) else 0.0,
        avg_low_stress_duration=float(round(avg_low, 2)) if pd.notna(avg_low) else 0.0,
        avg_medium_stress_duration=float(round(avg_medium, 2)) if pd.notna(avg_medium) else 0.0,
        avg_high_stress_duration=float(round(avg_high, 2)) if pd.notna(avg_high) else 0.0,
    )
