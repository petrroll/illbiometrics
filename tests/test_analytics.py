"""Tests for the analytics module."""

import pytest
import pandas as pd
import numpy as np
from datetime import date, datetime

from app.analytics import (
    oura_sleep_to_dataframe,
    analyze_sleep,
    SleepAnalytics,
    oura_heartrate_to_dataframe,
    analyze_heart_rate,
    analyze_heart_rate_daily,
    resample_heartrate,
    HeartRateAnalytics,
    DailyHeartRateAnalytics,
    NIGH_SLEEP_SLEEP_TYPE,
    DEFAULT_HR_MAX_GAP_SECONDS,
    get_sleep_intervals,
    filter_hr_outside_sleep,
    analyze_combined,
    analyze_combined_daily,
    CombinedAnalytics,
    CombinedDailyAnalytics,
    get_monthly_avg_sleep_times,
    generate_fallback_sleep_intervals,
)
from app.oura_client import SleepData, SleepHRData, SleepHRVData, _parse_sleep_data, HeartRateData, HeartRateSample


class TestOuraSleepToDataframe:
    """Tests for oura_sleep_to_dataframe function."""

    def test_converts_sleep_data_to_dataframe(self, sleep_data_list: list[dict]):
        """Test that sleep data is correctly converted to DataFrame."""
        # Parse fixture data into SleepData objects
        sleep_records = [_parse_sleep_data(record) for record in sleep_data_list]
        
        df = oura_sleep_to_dataframe(sleep_records)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(sleep_data_list)
        assert "id" in df.columns
        assert "day" in df.columns
        assert "total_sleep_duration" in df.columns
        assert "average_heart_rate" in df.columns
        assert "average_hrv" in df.columns
        assert "heart_rate_samples" in df.columns
        assert "hrv_samples" in df.columns

    def test_empty_sleep_data_returns_empty_dataframe(self):
        """Test that empty input returns empty DataFrame."""
        df = oura_sleep_to_dataframe([])
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_dataframe_preserves_sleep_values(self, sleep_data_list: list[dict]):
        """Test that DataFrame preserves the original values."""
        sleep_records = [_parse_sleep_data(record) for record in sleep_data_list]
        
        df = oura_sleep_to_dataframe(sleep_records)
        
        # Check first record values match
        first_record = sleep_data_list[0]
        assert df.iloc[0]["id"] == first_record["id"]
        assert df.iloc[0]["day"] == first_record["day"]
        assert df.iloc[0]["total_sleep_duration"] == first_record["total_sleep_duration"]
        assert df.iloc[0]["average_heart_rate"] == first_record["average_heart_rate"]
        assert df.iloc[0]["average_hrv"] == first_record["average_hrv"]


class TestAnalyzeSleep:
    """Tests for analyze_sleep function."""

    def test_returns_sleep_analytics_dataclass(self, sleep_data_list: list[dict]):
        """Test that analyze_sleep returns SleepAnalytics dataclass."""
        sleep_records = [_parse_sleep_data(record) for record in sleep_data_list]
        df = oura_sleep_to_dataframe(sleep_records)
        
        result = analyze_sleep(df)
        
        assert isinstance(result, SleepAnalytics)
        # Dates should be extracted from actual data
        assert result.start_date is not None
        assert result.end_date is not None
        assert result.start_date <= result.end_date

    def test_computes_median_sleep_duration(self, sleep_data_list: list[dict]):
        """Test that median sleep duration is computed correctly."""
        sleep_records = [_parse_sleep_data(record) for record in sleep_data_list]
        df = oura_sleep_to_dataframe(sleep_records)
        
        result = analyze_sleep(df)
        
        # Filter to long_sleep only (same as analyze_sleep does)
        filtered_df = df[df["type"] == NIGH_SLEEP_SLEEP_TYPE]
        expected_median = filtered_df["total_sleep_duration"].median()
        assert result.median_sleep_duration == round(expected_median, 2)

    def test_computes_median_avg_hr(self, sleep_data_list: list[dict]):
        """Test that median average HR is computed correctly."""
        sleep_records = [_parse_sleep_data(record) for record in sleep_data_list]
        df = oura_sleep_to_dataframe(sleep_records)
        
        result = analyze_sleep(df)
        
        # Filter to long_sleep only (same as analyze_sleep does)
        filtered_df = df[df["type"] == NIGH_SLEEP_SLEEP_TYPE]
        expected_median = filtered_df["average_heart_rate"].median()
        assert result.median_avg_hr == round(expected_median, 1)

    def test_computes_median_avg_hrv(self, sleep_data_list: list[dict]):
        """Test that median average HRV is computed correctly."""
        sleep_records = [_parse_sleep_data(record) for record in sleep_data_list]
        df = oura_sleep_to_dataframe(sleep_records)
        
        result = analyze_sleep(df)
        
        # Filter to long_sleep only (same as analyze_sleep does)
        filtered_df = df[df["type"] == NIGH_SLEEP_SLEEP_TYPE]
        expected_median = filtered_df["average_hrv"].median()
        assert result.median_avg_hrv == round(expected_median, 1)

    def test_computes_hr_percentiles(self, sleep_data_list: list[dict]):
        """Test that HR percentiles are computed from all samples."""
        sleep_records = [_parse_sleep_data(record) for record in sleep_data_list]
        df = oura_sleep_to_dataframe(sleep_records)
        
        result = analyze_sleep(df)
        
        # HR percentiles should be reasonable values (not 0 unless no data)
        assert result.hr_20th_percentile <= result.hr_80th_percentile

    def test_computes_hrv_percentiles(self, sleep_data_list: list[dict]):
        """Test that HRV percentiles are computed from all samples."""
        sleep_records = [_parse_sleep_data(record) for record in sleep_data_list]
        df = oura_sleep_to_dataframe(sleep_records)
        
        result = analyze_sleep(df)
        
        # HRV percentiles should be reasonable values
        assert result.hrv_20th_percentile <= result.hrv_80th_percentile

    def test_handles_empty_hr_samples(self):
        """Test that empty HR samples are handled gracefully."""
        sleep_data = SleepData(
            id="test-1",
            day="2025-01-01",
            type="long_sleep",
            bedtime_start="2025-01-01T22:00:00+00:00",
            bedtime_end="2025-01-02T06:00:00+00:00",
            total_sleep_duration=28800,
            average_heart_rate=60.0,
            average_hrv=50,
            heart_rate=None,
            hrv=None,
        )
        df = oura_sleep_to_dataframe([sleep_data])
        
        result = analyze_sleep(df)
        
        # Should return 0 for percentiles when no samples
        assert result.hr_20th_percentile == 0.0
        assert result.hr_80th_percentile == 0.0
        # Date should be extracted from the single record
        assert result.start_date == date(2025, 1, 1)
        assert result.end_date == date(2025, 1, 1)

    def test_handles_none_values_in_samples(self):
        """Test that None values in HR/HRV samples are filtered out."""
        sleep_data = SleepData(
            id="test-1",
            day="2025-01-01",
            type="long_sleep",
            bedtime_start="2025-01-01T22:00:00+00:00",
            bedtime_end="2025-01-02T06:00:00+00:00",
            total_sleep_duration=28800,
            average_heart_rate=60.0,
            average_hrv=50,
            heart_rate=SleepHRData(
                interval=60.0,
                items=[60, None, 65, None, 70],
                timestamp="2025-01-01T00:00:00+00:00",
            ),
            hrv=SleepHRVData(
                interval=60.0,
                items=[40, None, 50, None, 60],
                timestamp="2025-01-01T00:00:00+00:00",
            ),
        )
        df = oura_sleep_to_dataframe([sleep_data])
        
        result = analyze_sleep(df)
        
        # Should compute percentiles from non-None values only
        assert result.hr_20th_percentile > 0
        assert result.hrv_20th_percentile > 0

    def test_computes_sleep_duration_std(self, sleep_data_list: list[dict]):
        """Test that standard deviation of sleep duration is computed."""
        sleep_records = [_parse_sleep_data(record) for record in sleep_data_list]
        df = oura_sleep_to_dataframe(sleep_records)
        
        result = analyze_sleep(df)
        
        # Filter to long_sleep only (same as analyze_sleep does)
        filtered_df = df[df["type"] == NIGH_SLEEP_SLEEP_TYPE]
        expected_std = filtered_df["total_sleep_duration"].std()
        assert result.sleep_duration_std == round(expected_std, 2)
        # Std should be non-negative
        assert result.sleep_duration_std >= 0

    def test_computes_avg_hr_std(self, sleep_data_list: list[dict]):
        """Test that standard deviation of average HR is computed."""
        sleep_records = [_parse_sleep_data(record) for record in sleep_data_list]
        df = oura_sleep_to_dataframe(sleep_records)
        
        result = analyze_sleep(df)
        
        # Filter to long_sleep only (same as analyze_sleep does)
        filtered_df = df[df["type"] == NIGH_SLEEP_SLEEP_TYPE]
        expected_std = filtered_df["average_heart_rate"].std()
        assert result.avg_hr_std == round(expected_std, 1)
        # Std should be non-negative
        assert result.avg_hr_std >= 0

    def test_computes_avg_hrv_std(self, sleep_data_list: list[dict]):
        """Test that standard deviation of average HRV is computed."""
        sleep_records = [_parse_sleep_data(record) for record in sleep_data_list]
        df = oura_sleep_to_dataframe(sleep_records)
        
        result = analyze_sleep(df)
        
        # Filter to long_sleep only (same as analyze_sleep does)
        filtered_df = df[df["type"] == NIGH_SLEEP_SLEEP_TYPE]
        expected_std = filtered_df["average_hrv"].std()
        assert result.avg_hrv_std == round(expected_std, 1)
        # Std should be non-negative
        assert result.avg_hrv_std >= 0

    def test_handles_single_record_std(self):
        """Test that std returns 0 when there's only a single record (no variance possible)."""
        sleep_data = SleepData(
            id="test-1",
            day="2025-01-01",
            type="long_sleep",
            bedtime_start="2025-01-01T22:00:00+00:00",
            bedtime_end="2025-01-02T06:00:00+00:00",
            total_sleep_duration=28800,
            average_heart_rate=60.0,
            average_hrv=50,
            heart_rate=None,
            hrv=None,
        )
        df = oura_sleep_to_dataframe([sleep_data])
        
        result = analyze_sleep(df)
        
        # With only one record, std should be 0 (pandas returns NaN, but we convert to 0)
        assert result.sleep_duration_std == 0.0
        assert result.avg_hr_std == 0.0
        assert result.avg_hrv_std == 0.0


class TestOuraHeartrateToDataframe:
    """Tests for oura_heartrate_to_dataframe function."""

    def test_converts_heartrate_data_to_dataframe(self, heartrate_data_list: list[dict]):
        """Test that heart rate data is correctly converted to DataFrame."""
        # Parse fixture data into HeartRateData object
        samples = [
            HeartRateSample(
                bpm=item["bpm"],
                source=item["source"],
                timestamp=item["timestamp"],
            )
            for item in heartrate_data_list
        ]
        heartrate_data = HeartRateData(data=samples)
        
        df = oura_heartrate_to_dataframe(heartrate_data)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(heartrate_data_list)
        assert "bpm" in df.columns
        assert "source" in df.columns
        assert "timestamp" in df.columns
        assert "day" in df.columns

    def test_empty_heartrate_data_returns_empty_dataframe(self):
        """Test that empty input returns empty DataFrame."""
        heartrate_data = HeartRateData(data=[])
        
        df = oura_heartrate_to_dataframe(heartrate_data)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_dataframe_preserves_heartrate_values(self, heartrate_data_list: list[dict]):
        """Test that DataFrame preserves the original values."""
        samples = [
            HeartRateSample(
                bpm=item["bpm"],
                source=item["source"],
                timestamp=item["timestamp"],
            )
            for item in heartrate_data_list
        ]
        heartrate_data = HeartRateData(data=samples)
        
        df = oura_heartrate_to_dataframe(heartrate_data)
        
        # Check first record values match
        first_record = heartrate_data_list[0]
        assert df.iloc[0]["bpm"] == first_record["bpm"]
        assert df.iloc[0]["source"] == first_record["source"]


class TestGetSleepIntervals:
    """Tests for get_sleep_intervals function."""

    def test_extracts_sleep_intervals_from_long_sleep(self):
        """Test that sleep intervals are extracted from long_sleep entries."""
        sleep_data = [
            SleepData(
                id="test-1",
                day="2025-01-01",
                type="long_sleep",
                bedtime_start="2025-01-01T22:00:00+00:00",
                bedtime_end="2025-01-02T06:00:00+00:00",
                total_sleep_duration=28800,
                average_heart_rate=60.0,
                average_hrv=50,
                heart_rate=None,
                hrv=None,
            ),
        ]
        df = oura_sleep_to_dataframe(sleep_data)
        
        intervals = get_sleep_intervals(df)
        
        assert len(intervals) == 1
        assert intervals[0][0] == pd.Timestamp("2025-01-01T22:00:00+00:00")
        assert intervals[0][1] == pd.Timestamp("2025-01-02T06:00:00+00:00")

    def test_ignores_non_long_sleep_types(self):
        """Test that non-long_sleep types are ignored."""
        sleep_data = [
            SleepData(
                id="test-1",
                day="2025-01-01",
                type="rest",
                bedtime_start="2025-01-01T14:00:00+00:00",
                bedtime_end="2025-01-01T14:30:00+00:00",
                total_sleep_duration=1800,
                average_heart_rate=55.0,
                average_hrv=60,
                heart_rate=None,
                hrv=None,
            ),
            SleepData(
                id="test-2",
                day="2025-01-01",
                type="long_sleep",
                bedtime_start="2025-01-01T22:00:00+00:00",
                bedtime_end="2025-01-02T06:00:00+00:00",
                total_sleep_duration=28800,
                average_heart_rate=60.0,
                average_hrv=50,
                heart_rate=None,
                hrv=None,
            ),
        ]
        df = oura_sleep_to_dataframe(sleep_data)
        
        intervals = get_sleep_intervals(df)
        
        # Only long_sleep should be included
        assert len(intervals) == 1

    def test_handles_empty_dataframe(self):
        """Test that empty DataFrame returns empty list."""
        df = pd.DataFrame()
        
        intervals = get_sleep_intervals(df)
        
        assert intervals == []


class TestFilterHrOutsideSleep:
    """Tests for filter_hr_outside_sleep function."""

    def test_filters_out_samples_during_sleep(self):
        """Test that HR samples during sleep are filtered out."""
        samples = [
            HeartRateSample(bpm=70, source="awake", timestamp="2025-01-01T20:00:00+00:00"),  # Before sleep
            HeartRateSample(bpm=55, source="rest", timestamp="2025-01-01T23:00:00+00:00"),   # During sleep
            HeartRateSample(bpm=75, source="awake", timestamp="2025-01-02T08:00:00+00:00"),  # After sleep
        ]
        heartrate_data = HeartRateData(data=samples)
        hr_df = oura_heartrate_to_dataframe(heartrate_data)
        
        sleep_intervals = [
            (pd.Timestamp("2025-01-01T22:00:00+00:00"), pd.Timestamp("2025-01-02T06:00:00+00:00"))
        ]
        
        filtered_df, sleep_hours = filter_hr_outside_sleep(hr_df, sleep_intervals)
        
        # Only 2 samples should remain (before and after sleep)
        assert len(filtered_df) == 2
        assert list(filtered_df["bpm"]) == [70, 75]
        # 8 hours of sleep
        assert sleep_hours == 8.0

    def test_handles_multiple_sleep_intervals(self):
        """Test filtering with multiple sleep intervals."""
        samples = [
            HeartRateSample(bpm=70, source="awake", timestamp="2025-01-01T10:00:00+00:00"),
            HeartRateSample(bpm=55, source="rest", timestamp="2025-01-01T23:00:00+00:00"),   # Night 1
            HeartRateSample(bpm=75, source="awake", timestamp="2025-01-02T12:00:00+00:00"),
            HeartRateSample(bpm=52, source="rest", timestamp="2025-01-02T23:00:00+00:00"),   # Night 2
            HeartRateSample(bpm=80, source="awake", timestamp="2025-01-03T10:00:00+00:00"),
        ]
        heartrate_data = HeartRateData(data=samples)
        hr_df = oura_heartrate_to_dataframe(heartrate_data)
        
        sleep_intervals = [
            (pd.Timestamp("2025-01-01T22:00:00+00:00"), pd.Timestamp("2025-01-02T06:00:00+00:00")),
            (pd.Timestamp("2025-01-02T22:00:00+00:00"), pd.Timestamp("2025-01-03T06:00:00+00:00")),
        ]
        
        filtered_df, sleep_hours = filter_hr_outside_sleep(hr_df, sleep_intervals)
        
        # Only 3 samples should remain
        assert len(filtered_df) == 3
        # 16 hours of sleep total (2 nights * 8 hours)
        assert sleep_hours == 16.0

    def test_returns_all_samples_when_no_sleep_intervals(self):
        """Test that all samples are returned when there are no sleep intervals."""
        samples = [
            HeartRateSample(bpm=70, source="awake", timestamp="2025-01-01T10:00:00+00:00"),
            HeartRateSample(bpm=75, source="awake", timestamp="2025-01-01T12:00:00+00:00"),
        ]
        heartrate_data = HeartRateData(data=samples)
        hr_df = oura_heartrate_to_dataframe(heartrate_data)
        
        filtered_df, sleep_hours = filter_hr_outside_sleep(hr_df, [])
        
        assert len(filtered_df) == 2
        assert sleep_hours == 0.0


class TestAnalyzeHeartRate:
    """Tests for analyze_heart_rate function."""

    def test_returns_heart_rate_analytics_dataclass(self):
        """Test that analyze_heart_rate returns HeartRateAnalytics dataclass."""
        samples = [
            HeartRateSample(bpm=70, source="awake", timestamp="2025-01-01T10:00:00+00:00"),
            HeartRateSample(bpm=75, source="awake", timestamp="2025-01-01T10:01:00+00:00"),
            HeartRateSample(bpm=80, source="awake", timestamp="2025-01-01T10:02:00+00:00"),
        ]
        heartrate_data = HeartRateData(data=samples)
        df = oura_heartrate_to_dataframe(heartrate_data)
        
        result = analyze_heart_rate(df, sleep_intervals=[])
        
        assert isinstance(result, HeartRateAnalytics)
        assert result.start_date is not None
        assert result.end_date is not None

    def test_filters_out_sleep_periods(self):
        """Test that HR samples during sleep are excluded from analytics."""
        samples = [
            HeartRateSample(bpm=70, source="awake", timestamp="2025-01-01T20:00:00+00:00"),  # Before sleep
            HeartRateSample(bpm=55, source="rest", timestamp="2025-01-01T23:00:00+00:00"),   # During sleep
            HeartRateSample(bpm=75, source="awake", timestamp="2025-01-02T08:00:00+00:00"),  # After sleep
        ]
        heartrate_data = HeartRateData(data=samples)
        df = oura_heartrate_to_dataframe(heartrate_data)
        
        sleep_intervals = [
            (pd.Timestamp("2025-01-01T22:00:00+00:00"), pd.Timestamp("2025-01-02T06:00:00+00:00"))
        ]
        
        result = analyze_heart_rate(df, sleep_intervals)
        
        # Average should be from non-sleep samples only: (70 + 75) / 2 = 72.5
        assert result.average_hr == 72.5

    def test_computes_hr_percentiles(self):
        """Test that HR percentiles are computed correctly."""
        samples = [
            HeartRateSample(bpm=70, source="awake", timestamp="2025-01-01T10:00:00+00:00"),
            HeartRateSample(bpm=75, source="awake", timestamp="2025-01-01T10:01:00+00:00"),
            HeartRateSample(bpm=80, source="awake", timestamp="2025-01-01T10:02:00+00:00"),
            HeartRateSample(bpm=85, source="awake", timestamp="2025-01-01T10:03:00+00:00"),
            HeartRateSample(bpm=90, source="awake", timestamp="2025-01-01T10:04:00+00:00"),
        ]
        heartrate_data = HeartRateData(data=samples)
        df = oura_heartrate_to_dataframe(heartrate_data)
        
        result = analyze_heart_rate(df, sleep_intervals=[])
        
        # Percentiles should be in order
        assert result.hr_20th_percentile <= result.hr_50th_percentile
        assert result.hr_50th_percentile <= result.hr_80th_percentile
        assert result.hr_80th_percentile <= result.hr_95th_percentile
        assert result.hr_95th_percentile <= result.hr_99th_percentile

    def test_handles_empty_data(self):
        """Test that empty data is handled gracefully."""
        heartrate_data = HeartRateData(data=[])
        df = oura_heartrate_to_dataframe(heartrate_data)
        
        result = analyze_heart_rate(df, sleep_intervals=[])
        
        assert result.start_date is None
        assert result.end_date is None
        assert result.average_hr == 0.0

    def test_handles_all_data_during_sleep(self):
        """Test that data entirely during sleep returns empty result."""
        samples = [
            HeartRateSample(bpm=55, source="rest", timestamp="2025-01-01T23:00:00+00:00"),
            HeartRateSample(bpm=52, source="rest", timestamp="2025-01-02T03:00:00+00:00"),
        ]
        heartrate_data = HeartRateData(data=samples)
        df = oura_heartrate_to_dataframe(heartrate_data)
        
        sleep_intervals = [
            (pd.Timestamp("2025-01-01T22:00:00+00:00"), pd.Timestamp("2025-01-02T06:00:00+00:00"))
        ]
        
        result = analyze_heart_rate(df, sleep_intervals)
        
        assert result.start_date is None
        assert result.end_date is None
        assert result.average_hr == 0.0

    def test_computes_average_hr_std(self):
        """Test that standard deviation of heart rate is computed."""
        samples = [
            HeartRateSample(bpm=70, source="awake", timestamp="2025-01-01T10:00:00+00:00"),
            HeartRateSample(bpm=80, source="awake", timestamp="2025-01-01T10:01:00+00:00"),
            HeartRateSample(bpm=90, source="awake", timestamp="2025-01-01T10:02:00+00:00"),
        ]
        heartrate_data = HeartRateData(data=samples)
        df = oura_heartrate_to_dataframe(heartrate_data)
        
        result = analyze_heart_rate(df, sleep_intervals=[])
        
        # Std should be non-negative
        assert result.average_hr_std >= 0
        # For this data, std should be greater than 0 (there's variation)
        assert result.average_hr_std > 0

    def test_handles_empty_data_std(self):
        """Test that empty data returns 0 for standard deviation."""
        heartrate_data = HeartRateData(data=[])
        df = oura_heartrate_to_dataframe(heartrate_data)
        
        result = analyze_heart_rate(df, sleep_intervals=[])
        
        assert result.average_hr_std == 0.0


class TestAnalyzeHeartRateDaily:
    """Tests for analyze_heart_rate_daily function."""

    def test_returns_list_of_daily_analytics(self):
        """Test that analyze_heart_rate_daily returns list of DailyHeartRateAnalytics."""
        samples = [
            HeartRateSample(bpm=70, source="awake", timestamp="2025-01-01T10:00:00+00:00"),
            HeartRateSample(bpm=75, source="awake", timestamp="2025-01-01T10:01:00+00:00"),
            HeartRateSample(bpm=80, source="awake", timestamp="2025-01-02T10:00:00+00:00"),
        ]
        heartrate_data = HeartRateData(data=samples)
        df = oura_heartrate_to_dataframe(heartrate_data)
        
        result = analyze_heart_rate_daily(df, sleep_intervals=[])
        
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(day, DailyHeartRateAnalytics) for day in result)

    def test_returns_one_entry_per_day(self):
        """Test that each day has exactly one entry."""
        samples = [
            HeartRateSample(bpm=70, source="awake", timestamp="2025-01-01T10:00:00+00:00"),
            HeartRateSample(bpm=75, source="awake", timestamp="2025-01-01T14:00:00+00:00"),
            HeartRateSample(bpm=80, source="awake", timestamp="2025-01-02T10:00:00+00:00"),
        ]
        heartrate_data = HeartRateData(data=samples)
        df = oura_heartrate_to_dataframe(heartrate_data)
        
        result = analyze_heart_rate_daily(df, sleep_intervals=[])
        
        # Two unique days
        assert len(result) == 2

    def test_computes_daily_percentiles(self):
        """Test that daily percentiles are computed correctly."""
        samples = [
            HeartRateSample(bpm=70, source="awake", timestamp="2025-01-01T10:00:00+00:00"),
            HeartRateSample(bpm=75, source="awake", timestamp="2025-01-01T10:01:00+00:00"),
            HeartRateSample(bpm=80, source="awake", timestamp="2025-01-01T10:02:00+00:00"),
            HeartRateSample(bpm=85, source="awake", timestamp="2025-01-01T10:03:00+00:00"),
            HeartRateSample(bpm=90, source="awake", timestamp="2025-01-01T10:04:00+00:00"),
        ]
        heartrate_data = HeartRateData(data=samples)
        df = oura_heartrate_to_dataframe(heartrate_data)
        
        result = analyze_heart_rate_daily(df, sleep_intervals=[])
        
        for day in result:
            # Percentiles should be in order
            assert day.hr_20th_percentile <= day.hr_50th_percentile
            assert day.hr_50th_percentile <= day.hr_80th_percentile
            assert day.hr_80th_percentile <= day.hr_95th_percentile
            assert day.hr_95th_percentile <= day.hr_99th_percentile

    def test_results_sorted_by_day(self):
        """Test that results are sorted by day."""
        samples = [
            HeartRateSample(bpm=80, source="awake", timestamp="2025-01-03T10:00:00+00:00"),
            HeartRateSample(bpm=70, source="awake", timestamp="2025-01-01T10:00:00+00:00"),
            HeartRateSample(bpm=75, source="awake", timestamp="2025-01-02T10:00:00+00:00"),
        ]
        heartrate_data = HeartRateData(data=samples)
        df = oura_heartrate_to_dataframe(heartrate_data)
        
        result = analyze_heart_rate_daily(df, sleep_intervals=[])
        
        days = [day.day for day in result]
        assert days == sorted(days)

    def test_handles_empty_data(self):
        """Test that empty data returns empty list."""
        heartrate_data = HeartRateData(data=[])
        df = oura_heartrate_to_dataframe(heartrate_data)
        
        result = analyze_heart_rate_daily(df, sleep_intervals=[])
        
        assert result == []

    def test_filters_out_sleep_periods(self):
        """Test that HR samples during sleep are excluded from daily analytics."""
        samples = [
            HeartRateSample(bpm=70, source="awake", timestamp="2025-01-01T10:00:00+00:00"),
            HeartRateSample(bpm=55, source="rest", timestamp="2025-01-01T23:00:00+00:00"),   # During sleep
            HeartRateSample(bpm=75, source="awake", timestamp="2025-01-02T10:00:00+00:00"),
        ]
        heartrate_data = HeartRateData(data=samples)
        df = oura_heartrate_to_dataframe(heartrate_data)
        
        sleep_intervals = [
            (pd.Timestamp("2025-01-01T22:00:00+00:00"), pd.Timestamp("2025-01-02T06:00:00+00:00"))
        ]
        
        result = analyze_heart_rate_daily(df, sleep_intervals)
        
        # Two days of data outside sleep
        assert len(result) == 2
        assert result[0].average_hr == 70.0
        assert result[1].average_hr == 75.0


class TestResampleHeartrate:
    """Tests for resample_heartrate function."""

    def test_resamples_to_1_minute_intervals(self):
        """Test that data is resampled to 1-minute intervals."""
        samples = [
            HeartRateSample(bpm=70, source="awake", timestamp="2025-01-01T10:00:00+00:00"),
            HeartRateSample(bpm=75, source="awake", timestamp="2025-01-01T10:01:00+00:00"),
            HeartRateSample(bpm=80, source="awake", timestamp="2025-01-01T10:02:00+00:00"),
        ]
        heartrate_data = HeartRateData(data=samples)
        df = oura_heartrate_to_dataframe(heartrate_data)
        
        result = resample_heartrate(df)
        
        assert len(result) == 3
        # Check timestamps are 1 minute apart
        timestamps = result["timestamp"].tolist()
        for i in range(1, len(timestamps)):
            diff = (timestamps[i] - timestamps[i-1]).total_seconds()
            assert diff == 60

    def test_takes_max_when_multiple_points_in_same_minute(self):
        """Test that max BPM is taken when multiple readings fall within same minute."""
        samples = [
            HeartRateSample(bpm=70, source="awake", timestamp="2025-01-01T10:00:10+00:00"),
            HeartRateSample(bpm=85, source="awake", timestamp="2025-01-01T10:00:30+00:00"),
            HeartRateSample(bpm=75, source="awake", timestamp="2025-01-01T10:00:50+00:00"),
        ]
        heartrate_data = HeartRateData(data=samples)
        df = oura_heartrate_to_dataframe(heartrate_data)
        
        result = resample_heartrate(df)
        
        assert len(result) == 1
        assert result.iloc[0]["bpm"] == 85  # Max of 70, 85, 75

    def test_forward_fills_gaps_within_threshold(self):
        """Test that gaps within DEFAULT_HR_MAX_GAP_SECONDS are forward-filled."""
        # Two readings 3 minutes apart (within 300s threshold)
        samples = [
            HeartRateSample(bpm=70, source="awake", timestamp="2025-01-01T10:00:00+00:00"),
            HeartRateSample(bpm=80, source="awake", timestamp="2025-01-01T10:03:00+00:00"),
        ]
        heartrate_data = HeartRateData(data=samples)
        df = oura_heartrate_to_dataframe(heartrate_data)
        
        result = resample_heartrate(df)
        
        # Should have 4 entries: 10:00, 10:01, 10:02, 10:03
        assert len(result) == 4
        # First three should be forward-filled with 70
        assert result.iloc[0]["bpm"] == 70
        assert result.iloc[1]["bpm"] == 70  # forward-filled
        assert result.iloc[2]["bpm"] == 70  # forward-filled
        assert result.iloc[3]["bpm"] == 80

    def test_does_not_fill_gaps_exceeding_threshold(self):
        """Test that gaps exceeding DEFAULT_HR_MAX_GAP_SECONDS create separate segments."""
        # Two readings 6 minutes apart (exceeds 300s threshold)
        samples = [
            HeartRateSample(bpm=70, source="awake", timestamp="2025-01-01T10:00:00+00:00"),
            HeartRateSample(bpm=80, source="awake", timestamp="2025-01-01T10:06:00+00:00"),
        ]
        heartrate_data = HeartRateData(data=samples)
        df = oura_heartrate_to_dataframe(heartrate_data)
        
        result = resample_heartrate(df)
        
        # Should have 2 entries only (no filling between segments)
        assert len(result) == 2
        assert result.iloc[0]["bpm"] == 70
        assert result.iloc[1]["bpm"] == 80

    def test_handles_exactly_5_minute_gap(self):
        """Test that exactly 5-minute (300s) gap is still considered connected."""
        samples = [
            HeartRateSample(bpm=70, source="awake", timestamp="2025-01-01T10:00:00+00:00"),
            HeartRateSample(bpm=80, source="awake", timestamp="2025-01-01T10:05:00+00:00"),
        ]
        heartrate_data = HeartRateData(data=samples)
        df = oura_heartrate_to_dataframe(heartrate_data)
        
        result = resample_heartrate(df)
        
        # 300s gap is NOT greater than DEFAULT_HR_MAX_GAP_SECONDS, so should be connected
        # Should have 6 entries: 10:00, 10:01, 10:02, 10:03, 10:04, 10:05
        assert len(result) == 6

    def test_handles_just_over_5_minute_gap(self):
        """Test that gap just over 5 minutes creates separate segments."""
        samples = [
            HeartRateSample(bpm=70, source="awake", timestamp="2025-01-01T10:00:00+00:00"),
            HeartRateSample(bpm=80, source="awake", timestamp="2025-01-01T10:05:01+00:00"),
        ]
        heartrate_data = HeartRateData(data=samples)
        df = oura_heartrate_to_dataframe(heartrate_data)
        
        result = resample_heartrate(df)
        
        # 301s gap IS greater than DEFAULT_HR_MAX_GAP_SECONDS, so separate segments
        assert len(result) == 2

    def test_handles_empty_dataframe(self):
        """Test that empty DataFrame is handled gracefully."""
        heartrate_data = HeartRateData(data=[])
        df = oura_heartrate_to_dataframe(heartrate_data)
        
        result = resample_heartrate(df)
        
        assert len(result) == 0

    def test_handles_single_data_point(self):
        """Test that single data point returns single resampled point."""
        samples = [
            HeartRateSample(bpm=70, source="awake", timestamp="2025-01-01T10:00:30+00:00"),
        ]
        heartrate_data = HeartRateData(data=samples)
        df = oura_heartrate_to_dataframe(heartrate_data)
        
        result = resample_heartrate(df)
        
        assert len(result) == 1
        assert result.iloc[0]["bpm"] == 70

    def test_multiple_segments_with_gaps(self):
        """Test multiple segments separated by large gaps."""
        samples = [
            # Segment 1: 10:00-10:02
            HeartRateSample(bpm=70, source="awake", timestamp="2025-01-01T10:00:00+00:00"),
            HeartRateSample(bpm=72, source="awake", timestamp="2025-01-01T10:02:00+00:00"),
            # Gap > 300s
            # Segment 2: 10:10-10:11
            HeartRateSample(bpm=80, source="awake", timestamp="2025-01-01T10:10:00+00:00"),
            HeartRateSample(bpm=82, source="awake", timestamp="2025-01-01T10:11:00+00:00"),
        ]
        heartrate_data = HeartRateData(data=samples)
        df = oura_heartrate_to_dataframe(heartrate_data)
        
        result = resample_heartrate(df)
        
        # Segment 1: 10:00, 10:01, 10:02 (3 points, forward-filled)
        # Segment 2: 10:10, 10:11 (2 points)
        assert len(result) == 5
        
        # Check segment 1 values
        assert result.iloc[0]["bpm"] == 70
        assert result.iloc[1]["bpm"] == 70  # forward-filled
        assert result.iloc[2]["bpm"] == 72
        
        # Check segment 2 values
        assert result.iloc[3]["bpm"] == 80
        assert result.iloc[4]["bpm"] == 82

    def test_preserves_day_column(self):
        """Test that day column is correctly computed in resampled result."""
        samples = [
            HeartRateSample(bpm=70, source="awake", timestamp="2025-01-01T10:00:00+00:00"),
            HeartRateSample(bpm=75, source="awake", timestamp="2025-01-01T10:02:00+00:00"),
        ]
        heartrate_data = HeartRateData(data=samples)
        df = oura_heartrate_to_dataframe(heartrate_data)
        
        result = resample_heartrate(df)
        
        assert "day" in result.columns
        assert all(day == date(2025, 1, 1) for day in result["day"])

    def test_handles_unsorted_input(self):
        """Test that unsorted input is handled correctly."""
        samples = [
            HeartRateSample(bpm=80, source="awake", timestamp="2025-01-01T10:02:00+00:00"),
            HeartRateSample(bpm=70, source="awake", timestamp="2025-01-01T10:00:00+00:00"),
            HeartRateSample(bpm=75, source="awake", timestamp="2025-01-01T10:01:00+00:00"),
        ]
        heartrate_data = HeartRateData(data=samples)
        df = oura_heartrate_to_dataframe(heartrate_data)
        
        result = resample_heartrate(df)
        
        # Should be sorted and correct
        assert len(result) == 3
        assert result.iloc[0]["bpm"] == 70
        assert result.iloc[1]["bpm"] == 75
        assert result.iloc[2]["bpm"] == 80

    def test_custom_max_gap_seconds(self):
        """Test that custom max_gap_seconds parameter is respected."""
        # Two readings 3 minutes apart
        samples = [
            HeartRateSample(bpm=70, source="awake", timestamp="2025-01-01T10:00:00+00:00"),
            HeartRateSample(bpm=80, source="awake", timestamp="2025-01-01T10:03:00+00:00"),
        ]
        heartrate_data = HeartRateData(data=samples)
        df = oura_heartrate_to_dataframe(heartrate_data)
        
        # With default 300s threshold, should be connected (4 points)
        result_default = resample_heartrate(df)
        assert len(result_default) == 4
        
        # With 120s threshold (2 min), should be separate segments (2 points)
        result_custom = resample_heartrate(df, max_gap_seconds=120)
        assert len(result_custom) == 2

    def test_custom_resample_interval(self):
        """Test that custom resample_interval parameter is respected."""
        samples = [
            HeartRateSample(bpm=70, source="awake", timestamp="2025-01-01T10:00:00+00:00"),
            HeartRateSample(bpm=80, source="awake", timestamp="2025-01-01T10:04:00+00:00"),
        ]
        heartrate_data = HeartRateData(data=samples)
        df = oura_heartrate_to_dataframe(heartrate_data)
        
        # With 1-minute interval (default), should have 5 points
        result_1min = resample_heartrate(df)
        assert len(result_1min) == 5
        
        # With 2-minute interval, should have 3 points: 10:00, 10:02, 10:04
        result_2min = resample_heartrate(df, resample_interval="2min")
        assert len(result_2min) == 3

    def test_mixed_sources_treated_as_single_stream(self):
        """Test that different sources are resampled together as a single stream."""
        # Mix of sources that would create gaps if resampled separately
        samples = [
            HeartRateSample(bpm=55, source="rest", timestamp="2025-01-01T10:00:00+00:00"),
            HeartRateSample(bpm=75, source="awake", timestamp="2025-01-01T10:02:00+00:00"),
            HeartRateSample(bpm=52, source="rest", timestamp="2025-01-01T10:04:00+00:00"),
        ]
        heartrate_data = HeartRateData(data=samples)
        df = oura_heartrate_to_dataframe(heartrate_data)
        
        result = resample_heartrate(df)
        
        # All 5 minutes should be covered since gaps are <= 5 min
        assert len(result) == 5


class TestSleepBasedFiltering:
    """Tests for sleep-based HR filtering behavior."""

    def test_analyze_heart_rate_excludes_sleep_periods(self):
        """Test that analyze_heart_rate excludes data during sleep."""
        samples = [
            HeartRateSample(bpm=70, source="awake", timestamp="2025-01-01T10:00:00+00:00"),
            HeartRateSample(bpm=55, source="rest", timestamp="2025-01-01T23:00:00+00:00"),
            HeartRateSample(bpm=75, source="awake", timestamp="2025-01-02T08:00:00+00:00"),
        ]
        heartrate_data = HeartRateData(data=samples)
        df = oura_heartrate_to_dataframe(heartrate_data)
        
        sleep_intervals = [
            (pd.Timestamp("2025-01-01T22:00:00+00:00"), pd.Timestamp("2025-01-02T06:00:00+00:00"))
        ]
        
        result = analyze_heart_rate(df, sleep_intervals)
        
        # Should have non-None dates since there's data outside sleep
        assert result.start_date is not None
        assert result.end_date is not None
        # Average should be from non-sleep samples only: (70 + 75) / 2 = 72.5
        assert result.average_hr == 72.5

    def test_analyze_heart_rate_daily_excludes_sleep_periods(self):
        """Test that analyze_heart_rate_daily excludes data during sleep."""
        samples = [
            HeartRateSample(bpm=70, source="awake", timestamp="2025-01-01T10:00:00+00:00"),
            HeartRateSample(bpm=55, source="rest", timestamp="2025-01-01T23:00:00+00:00"),
            HeartRateSample(bpm=75, source="awake", timestamp="2025-01-02T08:00:00+00:00"),
        ]
        heartrate_data = HeartRateData(data=samples)
        df = oura_heartrate_to_dataframe(heartrate_data)
        
        sleep_intervals = [
            (pd.Timestamp("2025-01-01T22:00:00+00:00"), pd.Timestamp("2025-01-02T06:00:00+00:00"))
        ]
        
        result = analyze_heart_rate_daily(df, sleep_intervals)
        
        assert len(result) == 2
        assert result[0].day == date(2025, 1, 1)
        assert result[0].average_hr == 70.0
        assert result[1].day == date(2025, 1, 2)
        assert result[1].average_hr == 75.0

    def test_analyze_heart_rate_custom_parameters(self):
        """Test that analyze_heart_rate accepts custom resampling parameters."""
        samples = [
            HeartRateSample(bpm=70, source="awake", timestamp="2025-01-01T10:00:00+00:00"),
            HeartRateSample(bpm=80, source="awake", timestamp="2025-01-01T10:03:00+00:00"),
        ]
        heartrate_data = HeartRateData(data=samples)
        df = oura_heartrate_to_dataframe(heartrate_data)
        
        # With default parameters, should work
        result_default = analyze_heart_rate(df, sleep_intervals=[])
        assert result_default.start_date is not None
        
        # With custom max_gap_seconds=120, 3-minute gap creates separate segments
        result_custom = analyze_heart_rate(df, sleep_intervals=[], max_gap_seconds=120)
        assert result_custom.start_date is not None

    def test_analyze_heart_rate_daily_custom_parameters(self):
        """Test that analyze_heart_rate_daily accepts custom resampling parameters."""
        samples = [
            HeartRateSample(bpm=70, source="awake", timestamp="2025-01-01T10:00:00+00:00"),
            HeartRateSample(bpm=80, source="awake", timestamp="2025-01-01T10:03:00+00:00"),
        ]
        heartrate_data = HeartRateData(data=samples)
        df = oura_heartrate_to_dataframe(heartrate_data)
        
        # With default parameters
        result_default = analyze_heart_rate_daily(df, sleep_intervals=[])
        assert len(result_default) == 1
        
        # With custom resample_interval
        result_custom = analyze_heart_rate_daily(df, sleep_intervals=[], resample_interval="2min")
        assert len(result_custom) == 1

    def test_source_transitions_dont_break_resampling(self):
        """Test that source type changes don't create artificial gaps in resampling."""
        # Continuous data with source transitions every 2 minutes
        samples = [
            HeartRateSample(bpm=55, source="rest", timestamp="2025-01-01T10:00:00+00:00"),
            HeartRateSample(bpm=75, source="awake", timestamp="2025-01-01T10:02:00+00:00"),
            HeartRateSample(bpm=85, source="workout", timestamp="2025-01-01T10:04:00+00:00"),
            HeartRateSample(bpm=78, source="live", timestamp="2025-01-01T10:06:00+00:00"),
        ]
        heartrate_data = HeartRateData(data=samples)
        df = oura_heartrate_to_dataframe(heartrate_data)
        
        result = resample_heartrate(df)
        
        # All should be in one continuous segment (7 points: 10:00 through 10:06)
        assert len(result) == 7
        
        # Verify no gaps in timestamps
        timestamps = result["timestamp"].tolist()
        for i in range(1, len(timestamps)):
            diff = (timestamps[i] - timestamps[i-1]).total_seconds()
            assert diff == 60  # All consecutive

    def test_all_data_during_sleep_returns_empty(self):
        """Test that data entirely during sleep returns empty result."""
        samples = [
            HeartRateSample(bpm=55, source="rest", timestamp="2025-01-01T23:00:00+00:00"),
            HeartRateSample(bpm=52, source="rest", timestamp="2025-01-02T02:00:00+00:00"),
        ]
        heartrate_data = HeartRateData(data=samples)
        df = oura_heartrate_to_dataframe(heartrate_data)
        
        sleep_intervals = [
            (pd.Timestamp("2025-01-01T22:00:00+00:00"), pd.Timestamp("2025-01-02T06:00:00+00:00"))
        ]
        
        result = analyze_heart_rate(df, sleep_intervals)
        
        assert result.start_date is None
        assert result.end_date is None
        assert result.average_hr == 0.0


class TestCombinedAnalytics:
    """Tests for combined analytics functions."""

    def test_analyze_combined_returns_both_sleep_and_hr(self):
        """Test that analyze_combined returns both sleep and HR analytics."""
        sleep_data = [
            SleepData(
                id="test-1",
                day="2025-01-01",
                type="long_sleep",
                bedtime_start="2025-01-01T22:00:00+00:00",
                bedtime_end="2025-01-02T06:00:00+00:00",
                total_sleep_duration=28800,
                average_heart_rate=60.0,
                average_hrv=50,
                heart_rate=None,
                hrv=None,
            ),
        ]
        sleep_df = oura_sleep_to_dataframe(sleep_data)
        
        hr_samples = [
            HeartRateSample(bpm=70, source="awake", timestamp="2025-01-01T10:00:00+00:00"),
            HeartRateSample(bpm=75, source="awake", timestamp="2025-01-02T10:00:00+00:00"),
        ]
        heartrate_data = HeartRateData(data=hr_samples)
        hr_df = oura_heartrate_to_dataframe(heartrate_data)
        
        result = analyze_combined(sleep_df, hr_df)
        
        assert isinstance(result, CombinedAnalytics)
        assert isinstance(result.sleep, SleepAnalytics)
        assert isinstance(result.heart_rate, HeartRateAnalytics)
        assert result.sleep.nights_count == 1
        assert result.heart_rate.average_hr == 72.5

    def test_analyze_combined_filters_hr_during_sleep(self):
        """Test that combined analytics filters HR during sleep."""
        sleep_data = [
            SleepData(
                id="test-1",
                day="2025-01-01",
                type="long_sleep",
                bedtime_start="2025-01-01T22:00:00+00:00",
                bedtime_end="2025-01-02T06:00:00+00:00",
                total_sleep_duration=28800,
                average_heart_rate=60.0,
                average_hrv=50,
                heart_rate=None,
                hrv=None,
            ),
        ]
        sleep_df = oura_sleep_to_dataframe(sleep_data)
        
        hr_samples = [
            HeartRateSample(bpm=70, source="awake", timestamp="2025-01-01T10:00:00+00:00"),
            HeartRateSample(bpm=55, source="rest", timestamp="2025-01-01T23:00:00+00:00"),  # During sleep
            HeartRateSample(bpm=75, source="awake", timestamp="2025-01-02T10:00:00+00:00"),
        ]
        heartrate_data = HeartRateData(data=hr_samples)
        hr_df = oura_heartrate_to_dataframe(heartrate_data)
        
        result = analyze_combined(sleep_df, hr_df)
        
        # HR during sleep should be excluded
        assert result.heart_rate.average_hr == 72.5  # (70 + 75) / 2

    def test_analyze_combined_daily_returns_daily_hr(self):
        """Test that analyze_combined_daily returns daily HR analytics."""
        sleep_data = [
            SleepData(
                id="test-1",
                day="2025-01-01",
                type="long_sleep",
                bedtime_start="2025-01-01T22:00:00+00:00",
                bedtime_end="2025-01-02T06:00:00+00:00",
                total_sleep_duration=28800,
                average_heart_rate=60.0,
                average_hrv=50,
                heart_rate=None,
                hrv=None,
            ),
        ]
        sleep_df = oura_sleep_to_dataframe(sleep_data)
        
        hr_samples = [
            HeartRateSample(bpm=70, source="awake", timestamp="2025-01-01T10:00:00+00:00"),
            HeartRateSample(bpm=75, source="awake", timestamp="2025-01-02T10:00:00+00:00"),
        ]
        heartrate_data = HeartRateData(data=hr_samples)
        hr_df = oura_heartrate_to_dataframe(heartrate_data)
        
        result = analyze_combined_daily(sleep_df, hr_df)
        
        assert isinstance(result, CombinedDailyAnalytics)
        assert len(result.days) == 2
        assert result.days[0].average_hr == 70.0
        assert result.days[1].average_hr == 75.0


class TestGetMonthlyAvgSleepTimes:
    """Tests for get_monthly_avg_sleep_times function."""

    def test_calculates_monthly_averages(self):
        """Test that monthly average sleep times are calculated correctly."""
        sleep_data = [
            SleepData(
                id="test-1",
                day="2025-01-01",
                type="long_sleep",
                bedtime_start="2025-01-01T22:00:00+00:00",
                bedtime_end="2025-01-02T08:00:00+00:00",
                total_sleep_duration=36000,
                average_heart_rate=60.0,
                average_hrv=50,
                heart_rate=None,
                hrv=None,
            ),
            SleepData(
                id="test-2",
                day="2025-01-02",
                type="long_sleep",
                bedtime_start="2025-01-02T23:00:00+00:00",
                bedtime_end="2025-01-03T09:00:00+00:00",
                total_sleep_duration=36000,
                average_heart_rate=60.0,
                average_hrv=50,
                heart_rate=None,
                hrv=None,
            ),
        ]
        df = oura_sleep_to_dataframe(sleep_data)
        
        result = get_monthly_avg_sleep_times(df)
        
        assert "2025-01" in result
        avg_start, avg_end = result["2025-01"]
        # Average of 22:00 and 23:00 = 22:30 (22.5 hours)
        assert abs(avg_start - 22.5) < 0.1
        # Average of 08:00 and 09:00 = 08:30 (8.5 hours)
        assert abs(avg_end - 8.5) < 0.1

    def test_handles_after_midnight_start(self):
        """Test that bedtime after midnight is handled correctly."""
        sleep_data = [
            SleepData(
                id="test-1",
                day="2025-01-01",
                type="long_sleep",
                bedtime_start="2025-01-01T01:00:00+00:00",  # After midnight
                bedtime_end="2025-01-01T09:00:00+00:00",
                total_sleep_duration=28800,
                average_heart_rate=60.0,
                average_hrv=50,
                heart_rate=None,
                hrv=None,
            ),
        ]
        df = oura_sleep_to_dataframe(sleep_data)
        
        result = get_monthly_avg_sleep_times(df)
        
        assert "2025-01" in result
        avg_start, avg_end = result["2025-01"]
        # 01:00 after midnight should be treated as 25.0 (24 + 1)
        assert avg_start >= 24

    def test_ignores_non_long_sleep(self):
        """Test that non-long_sleep types are ignored."""
        sleep_data = [
            SleepData(
                id="test-1",
                day="2025-01-01",
                type="rest",
                bedtime_start="2025-01-01T14:00:00+00:00",
                bedtime_end="2025-01-01T14:30:00+00:00",
                total_sleep_duration=1800,
                average_heart_rate=55.0,
                average_hrv=60,
                heart_rate=None,
                hrv=None,
            ),
        ]
        df = oura_sleep_to_dataframe(sleep_data)
        
        result = get_monthly_avg_sleep_times(df)
        
        assert result == {}

    def test_handles_empty_dataframe(self):
        """Test that empty DataFrame returns empty dict."""
        df = pd.DataFrame()
        
        result = get_monthly_avg_sleep_times(df)
        
        assert result == {}


class TestGenerateFallbackSleepIntervals:
    """Tests for generate_fallback_sleep_intervals function."""

    def test_generates_intervals_for_missing_dates(self):
        """Test that fallback intervals are generated for dates without sleep records."""
        # Create sleep data for Jan 1 and Jan 3 (missing Jan 2)
        sleep_data = [
            SleepData(
                id="test-1",
                day="2025-01-01",
                type="long_sleep",
                bedtime_start="2024-12-31T22:00:00+00:00",
                bedtime_end="2025-01-01T06:00:00+00:00",
                total_sleep_duration=28800,
                average_heart_rate=60.0,
                average_hrv=50,
                heart_rate=None,
                hrv=None,
            ),
            SleepData(
                id="test-2",
                day="2025-01-03",
                type="long_sleep",
                bedtime_start="2025-01-02T22:00:00+00:00",
                bedtime_end="2025-01-03T06:00:00+00:00",
                total_sleep_duration=28800,
                average_heart_rate=60.0,
                average_hrv=50,
                heart_rate=None,
                hrv=None,
            ),
        ]
        sleep_df = oura_sleep_to_dataframe(sleep_data)
        existing_intervals = get_sleep_intervals(sleep_df)
        
        # Generate fallback for Jan 1-3 range
        fallback = generate_fallback_sleep_intervals(
            start_date=date(2025, 1, 1),
            end_date=date(2025, 1, 3),
            sleep_df=sleep_df,
            existing_intervals=existing_intervals,
        )
        
        # Should generate 1 fallback interval for Jan 2
        assert len(fallback) == 1
        start, end = fallback[0]
        assert end.date() == date(2025, 1, 2)

    def test_no_fallback_when_all_dates_covered(self):
        """Test that no fallback is generated when all dates have sleep records."""
        sleep_data = [
            SleepData(
                id="test-1",
                day="2025-01-01",
                type="long_sleep",
                bedtime_start="2024-12-31T22:00:00+00:00",
                bedtime_end="2025-01-01T06:00:00+00:00",
                total_sleep_duration=28800,
                average_heart_rate=60.0,
                average_hrv=50,
                heart_rate=None,
                hrv=None,
            ),
            SleepData(
                id="test-2",
                day="2025-01-02",
                type="long_sleep",
                bedtime_start="2025-01-01T22:00:00+00:00",
                bedtime_end="2025-01-02T06:00:00+00:00",
                total_sleep_duration=28800,
                average_heart_rate=60.0,
                average_hrv=50,
                heart_rate=None,
                hrv=None,
            ),
        ]
        sleep_df = oura_sleep_to_dataframe(sleep_data)
        existing_intervals = get_sleep_intervals(sleep_df)
        
        fallback = generate_fallback_sleep_intervals(
            start_date=date(2025, 1, 1),
            end_date=date(2025, 1, 2),
            sleep_df=sleep_df,
            existing_intervals=existing_intervals,
        )
        
        assert len(fallback) == 0

    def test_returns_empty_when_no_sleep_data_for_averages(self):
        """Test that empty list is returned when no sleep data exists to calculate averages."""
        sleep_df = pd.DataFrame()
        
        fallback = generate_fallback_sleep_intervals(
            start_date=date(2025, 1, 1),
            end_date=date(2025, 1, 3),
            sleep_df=sleep_df,
            existing_intervals=[],
        )
        
        assert fallback == []


class TestGetSleepIntervalsWithFallback:
    """Tests for get_sleep_intervals with fallback date range."""

    def test_includes_fallback_intervals_when_date_range_provided(self):
        """Test that fallback intervals are included when date range is provided."""
        # Sleep data for Jan 1 and Jan 3 only
        sleep_data = [
            SleepData(
                id="test-1",
                day="2025-01-01",
                type="long_sleep",
                bedtime_start="2024-12-31T22:00:00+00:00",
                bedtime_end="2025-01-01T06:00:00+00:00",
                total_sleep_duration=28800,
                average_heart_rate=60.0,
                average_hrv=50,
                heart_rate=None,
                hrv=None,
            ),
            SleepData(
                id="test-2",
                day="2025-01-03",
                type="long_sleep",
                bedtime_start="2025-01-02T22:00:00+00:00",
                bedtime_end="2025-01-03T06:00:00+00:00",
                total_sleep_duration=28800,
                average_heart_rate=60.0,
                average_hrv=50,
                heart_rate=None,
                hrv=None,
            ),
        ]
        df = oura_sleep_to_dataframe(sleep_data)
        
        # Without date range - should have 2 intervals
        intervals_no_fallback = get_sleep_intervals(df)
        assert len(intervals_no_fallback) == 2
        
        # With date range - should have 3 intervals (2 real + 1 fallback for Jan 2)
        intervals_with_fallback = get_sleep_intervals(
            df,
            start_date=date(2025, 1, 1),
            end_date=date(2025, 1, 3),
        )
        assert len(intervals_with_fallback) == 3

    def test_no_fallback_when_only_start_date_provided(self):
        """Test that fallback is not generated when only start_date is provided."""
        sleep_data = [
            SleepData(
                id="test-1",
                day="2025-01-01",
                type="long_sleep",
                bedtime_start="2024-12-31T22:00:00+00:00",
                bedtime_end="2025-01-01T06:00:00+00:00",
                total_sleep_duration=28800,
                average_heart_rate=60.0,
                average_hrv=50,
                heart_rate=None,
                hrv=None,
            ),
        ]
        df = oura_sleep_to_dataframe(sleep_data)
        
        intervals = get_sleep_intervals(df, start_date=date(2025, 1, 1))
        
        # Should only have the 1 real interval, no fallback
        assert len(intervals) == 1


class TestAnalyzeCombinedWithFallback:
    """Tests for analyze_combined with fallback filtering."""

    def test_filters_hr_with_fallback_when_date_range_provided(self):
        """Test that HR is filtered using fallback intervals when date range provided."""
        # Sleep data only for Jan 1 (missing Jan 2)
        sleep_data = [
            SleepData(
                id="test-1",
                day="2025-01-01",
                type="long_sleep",
                bedtime_start="2024-12-31T22:00:00+00:00",
                bedtime_end="2025-01-01T06:00:00+00:00",
                total_sleep_duration=28800,
                average_heart_rate=60.0,
                average_hrv=50,
                heart_rate=None,
                hrv=None,
            ),
        ]
        sleep_df = oura_sleep_to_dataframe(sleep_data)
        
        # HR samples including one during typical sleep on Jan 2 (missing sleep record)
        hr_samples = [
            HeartRateSample(bpm=70, source="awake", timestamp="2025-01-01T10:00:00+00:00"),
            HeartRateSample(bpm=50, source="rest", timestamp="2025-01-02T02:00:00+00:00"),  # Sleep hours on Jan 2
            HeartRateSample(bpm=75, source="awake", timestamp="2025-01-02T10:00:00+00:00"),
        ]
        heartrate_data = HeartRateData(data=hr_samples)
        hr_df = oura_heartrate_to_dataframe(heartrate_data)
        
        # Without date range - sleep sample on Jan 2 should NOT be filtered
        result_no_fallback = analyze_combined(sleep_df, hr_df)
        
        # With date range - sleep sample on Jan 2 SHOULD be filtered by fallback
        result_with_fallback = analyze_combined(
            sleep_df, hr_df,
            start_date=date(2025, 1, 1),
            end_date=date(2025, 1, 2),
        )
        
        # With fallback, the 02:00 sample should be filtered, so average should be different
        # Without fallback: (70 + 50 + 75) / 3 = 65 
        # With fallback: (70 + 75) / 2 = 72.5 (50 bpm sample filtered)
        assert result_no_fallback.heart_rate.average_hr != result_with_fallback.heart_rate.average_hr
        assert result_with_fallback.heart_rate.average_hr == 72.5
