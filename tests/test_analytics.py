"""Tests for the analytics module."""

import pytest
import pandas as pd
import numpy as np

from app.analytics import oura_sleep_to_dataframe, analyze_sleep, SleepAnalytics
from app.oura_client import SleepData, SleepHRData, SleepHRVData, _parse_sleep_data


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

    def test_computes_median_sleep_duration(self, sleep_data_list: list[dict]):
        """Test that median sleep duration is computed correctly."""
        sleep_records = [_parse_sleep_data(record) for record in sleep_data_list]
        df = oura_sleep_to_dataframe(sleep_records)
        
        result = analyze_sleep(df)
        
        expected_median = df["total_sleep_duration"].median()
        assert result.median_sleep_duration == round(expected_median, 2)

    def test_computes_median_avg_hr(self, sleep_data_list: list[dict]):
        """Test that median average HR is computed correctly."""
        sleep_records = [_parse_sleep_data(record) for record in sleep_data_list]
        df = oura_sleep_to_dataframe(sleep_records)
        
        result = analyze_sleep(df)
        
        expected_median = df["average_heart_rate"].median()
        assert result.median_avg_hr == round(expected_median, 1)

    def test_computes_median_avg_hrv(self, sleep_data_list: list[dict]):
        """Test that median average HRV is computed correctly."""
        sleep_records = [_parse_sleep_data(record) for record in sleep_data_list]
        df = oura_sleep_to_dataframe(sleep_records)
        
        result = analyze_sleep(df)
        
        expected_median = df["average_hrv"].median()
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

    def test_handles_none_values_in_samples(self):
        """Test that None values in HR/HRV samples are filtered out."""
        sleep_data = SleepData(
            id="test-1",
            day="2025-01-01",
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
