"""Tests for the analytics module."""

import pytest
import pandas as pd
import numpy as np
from datetime import date

from app.analytics import (
    oura_sleep_to_dataframe,
    analyze_sleep,
    SleepAnalytics,
    oura_heartrate_to_dataframe,
    analyze_heart_rate,
    analyze_heart_rate_daily,
    HeartRateAnalytics,
    DailyHeartRateAnalytics,
    NIGH_SLEEP_SLEEP_TYPE,
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


class TestAnalyzeHeartRate:
    """Tests for analyze_heart_rate function."""

    def test_returns_heart_rate_analytics_dataclass(self, heartrate_data_list: list[dict]):
        """Test that analyze_heart_rate returns HeartRateAnalytics dataclass."""
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
        
        result = analyze_heart_rate(df)
        
        assert isinstance(result, HeartRateAnalytics)
        # Dates should be extracted from actual data
        assert result.start_date is not None
        assert result.end_date is not None
        assert result.start_date <= result.end_date

    def test_filters_to_active_sources_only(self, heartrate_data_list: list[dict]):
        """Test that only active source data (awake, live, workout) is used for analytics."""
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
        
        result = analyze_heart_rate(df)
        
        # Manually compute expected average from active sources only
        active_sources = {"awake", "live", "workout"}
        active_samples = [item for item in heartrate_data_list if item["source"] in active_sources]
        expected_avg = sum(s["bpm"] for s in active_samples) / len(active_samples)
        
        assert result.average_hr == round(expected_avg, 1)

    def test_computes_hr_percentiles(self, heartrate_data_list: list[dict]):
        """Test that HR percentiles are computed correctly."""
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
        
        result = analyze_heart_rate(df)
        
        # Percentiles should be in order
        assert result.hr_20th_percentile <= result.hr_50th_percentile
        assert result.hr_50th_percentile <= result.hr_80th_percentile
        assert result.hr_80th_percentile <= result.hr_95th_percentile
        assert result.hr_95th_percentile <= result.hr_99th_percentile

    def test_handles_empty_data(self):
        """Test that empty data is handled gracefully."""
        heartrate_data = HeartRateData(data=[])
        df = oura_heartrate_to_dataframe(heartrate_data)
        
        result = analyze_heart_rate(df)
        
        assert result.start_date is None
        assert result.end_date is None
        assert result.average_hr == 0.0

    def test_handles_only_rest_data(self):
        """Test that data with only rest source returns empty-like result."""
        samples = [
            HeartRateSample(bpm=55, source="rest", timestamp="2025-01-01T02:00:00+00:00"),
            HeartRateSample(bpm=52, source="rest", timestamp="2025-01-01T03:00:00+00:00"),
        ]
        heartrate_data = HeartRateData(data=samples)
        df = oura_heartrate_to_dataframe(heartrate_data)
        
        result = analyze_heart_rate(df)
        
        # No active data, should return None dates
        assert result.start_date is None
        assert result.end_date is None


class TestAnalyzeHeartRateDaily:
    """Tests for analyze_heart_rate_daily function."""

    def test_returns_list_of_daily_analytics(self, heartrate_data_list: list[dict]):
        """Test that analyze_heart_rate_daily returns list of DailyHeartRateAnalytics."""
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
        
        result = analyze_heart_rate_daily(df)
        
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(day, DailyHeartRateAnalytics) for day in result)

    def test_returns_one_entry_per_day(self, heartrate_data_list: list[dict]):
        """Test that each day has exactly one entry."""
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
        
        result = analyze_heart_rate_daily(df)
        
        # Get unique active days from fixture
        active_sources = {"awake", "live", "workout"}
        active_items = [item for item in heartrate_data_list if item["source"] in active_sources]
        unique_days = set(pd.to_datetime(item["timestamp"]).date() for item in active_items)
        
        assert len(result) == len(unique_days)

    def test_computes_daily_percentiles(self, heartrate_data_list: list[dict]):
        """Test that daily percentiles are computed correctly."""
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
        
        result = analyze_heart_rate_daily(df)
        
        for day in result:
            # Percentiles should be in order
            assert day.hr_20th_percentile <= day.hr_50th_percentile
            assert day.hr_50th_percentile <= day.hr_80th_percentile
            assert day.hr_80th_percentile <= day.hr_95th_percentile
            assert day.hr_95th_percentile <= day.hr_99th_percentile

    def test_results_sorted_by_day(self, heartrate_data_list: list[dict]):
        """Test that results are sorted by day."""
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
        
        result = analyze_heart_rate_daily(df)
        
        days = [day.day for day in result]
        assert days == sorted(days)

    def test_handles_empty_data(self):
        """Test that empty data returns empty list."""
        heartrate_data = HeartRateData(data=[])
        df = oura_heartrate_to_dataframe(heartrate_data)
        
        result = analyze_heart_rate_daily(df)
        
        assert result == []

    def test_filters_to_active_sources_only(self):
        """Test that only active source data (awake, live, workout) is used for daily analytics."""
        samples = [
            HeartRateSample(bpm=72, source="awake", timestamp="2025-01-01T08:00:00+00:00"),
            HeartRateSample(bpm=55, source="rest", timestamp="2025-01-01T02:00:00+00:00"),
            HeartRateSample(bpm=52, source="rest", timestamp="2025-01-02T02:00:00+00:00"),
        ]
        heartrate_data = HeartRateData(data=samples)
        df = oura_heartrate_to_dataframe(heartrate_data)
        
        result = analyze_heart_rate_daily(df)
        
        # Only one day with active data
        assert len(result) == 1
        assert result[0].day == date(2025, 1, 1)
        assert result[0].average_hr == 72.0
