"""Tests for Garmin-specific analytics."""

import pytest
import pandas as pd
import numpy as np
from datetime import date

from app.analytics import (
    StressAnalytics,
    DailyStressAnalytics,
    garmin_stress_to_dataframe,
    analyze_stress,
    analyze_stress_daily,
    garmin_sleep_to_dataframe,
    garmin_heartrate_to_dataframe,
)
from app.garmin_client import (
    DailyStressData,
    SleepData,
    HeartRateData,
    HeartRateSample,
    SleepHRData,
    _parse_garmin_sleep_data,
)


class TestGarminStressToDataframe:
    """Tests for garmin_stress_to_dataframe function."""

    def test_converts_stress_data_to_dataframe(self, garmin_stress_data_list: list[dict]):
        """Test that stress data is correctly converted to DataFrame."""
        # Parse fixture data into DailyStressData objects
        stress_records = [
            DailyStressData(
                calendar_date=date.fromisoformat(record["calendar_date"]),
                overall_stress_level=record["overall_stress_level"],
                rest_stress_duration=record.get("rest_stress_duration"),
                low_stress_duration=record.get("low_stress_duration"),
                medium_stress_duration=record.get("medium_stress_duration"),
                high_stress_duration=record.get("high_stress_duration"),
            )
            for record in garmin_stress_data_list
        ]
        
        df = garmin_stress_to_dataframe(stress_records)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(garmin_stress_data_list)
        assert "calendar_date" in df.columns
        assert "overall_stress_level" in df.columns
        assert "rest_stress_duration" in df.columns
        assert "low_stress_duration" in df.columns
        assert "medium_stress_duration" in df.columns
        assert "high_stress_duration" in df.columns

    def test_empty_stress_data_returns_empty_dataframe(self):
        """Test that empty input returns empty DataFrame."""
        df = garmin_stress_to_dataframe([])
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_dataframe_preserves_stress_values(self, garmin_stress_data_list: list[dict]):
        """Test that DataFrame preserves the original values."""
        stress_records = [
            DailyStressData(
                calendar_date=date.fromisoformat(record["calendar_date"]),
                overall_stress_level=record["overall_stress_level"],
                rest_stress_duration=record.get("rest_stress_duration"),
                low_stress_duration=record.get("low_stress_duration"),
                medium_stress_duration=record.get("medium_stress_duration"),
                high_stress_duration=record.get("high_stress_duration"),
            )
            for record in garmin_stress_data_list
        ]
        
        df = garmin_stress_to_dataframe(stress_records)
        
        # Check first record values match
        first_record = garmin_stress_data_list[0]
        assert df.iloc[0]["overall_stress_level"] == first_record["overall_stress_level"]
        assert df.iloc[0]["rest_stress_duration"] == first_record["rest_stress_duration"]


class TestAnalyzeStress:
    """Tests for analyze_stress function."""

    def test_returns_stress_analytics_dataclass(self, garmin_stress_data_list: list[dict]):
        """Test that analyze_stress returns StressAnalytics dataclass."""
        stress_records = [
            DailyStressData(
                calendar_date=date.fromisoformat(record["calendar_date"]),
                overall_stress_level=record["overall_stress_level"],
                rest_stress_duration=record.get("rest_stress_duration"),
                low_stress_duration=record.get("low_stress_duration"),
                medium_stress_duration=record.get("medium_stress_duration"),
                high_stress_duration=record.get("high_stress_duration"),
            )
            for record in garmin_stress_data_list
        ]
        df = garmin_stress_to_dataframe(stress_records)
        
        result = analyze_stress(df)
        
        assert isinstance(result, StressAnalytics)
        assert result.start_date is not None
        assert result.end_date is not None
        assert result.start_date <= result.end_date

    def test_computes_average_stress_level(self, garmin_stress_data_list: list[dict]):
        """Test that average stress level is computed correctly."""
        stress_records = [
            DailyStressData(
                calendar_date=date.fromisoformat(record["calendar_date"]),
                overall_stress_level=record["overall_stress_level"],
                rest_stress_duration=record.get("rest_stress_duration"),
                low_stress_duration=record.get("low_stress_duration"),
                medium_stress_duration=record.get("medium_stress_duration"),
                high_stress_duration=record.get("high_stress_duration"),
            )
            for record in garmin_stress_data_list
        ]
        df = garmin_stress_to_dataframe(stress_records)
        
        result = analyze_stress(df)
        
        expected_avg = df["overall_stress_level"].mean()
        assert result.average_stress_level == round(expected_avg, 1)

    def test_computes_stress_percentiles(self, garmin_stress_data_list: list[dict]):
        """Test that stress percentiles are computed correctly."""
        stress_records = [
            DailyStressData(
                calendar_date=date.fromisoformat(record["calendar_date"]),
                overall_stress_level=record["overall_stress_level"],
                rest_stress_duration=record.get("rest_stress_duration"),
                low_stress_duration=record.get("low_stress_duration"),
                medium_stress_duration=record.get("medium_stress_duration"),
                high_stress_duration=record.get("high_stress_duration"),
            )
            for record in garmin_stress_data_list
        ]
        df = garmin_stress_to_dataframe(stress_records)
        
        result = analyze_stress(df)
        
        # Percentiles should be in order
        assert result.stress_20th_percentile <= result.stress_50th_percentile
        assert result.stress_50th_percentile <= result.stress_80th_percentile

    def test_handles_empty_data(self):
        """Test that empty data is handled gracefully."""
        df = pd.DataFrame()
        
        result = analyze_stress(df)
        
        assert result.start_date is None
        assert result.end_date is None
        assert result.average_stress_level == 0.0
        assert result.days_count == 0

    def test_computes_stress_duration_medians(self, garmin_stress_data_list: list[dict]):
        """Test that stress duration medians are computed."""
        stress_records = [
            DailyStressData(
                calendar_date=date.fromisoformat(record["calendar_date"]),
                overall_stress_level=record["overall_stress_level"],
                rest_stress_duration=record.get("rest_stress_duration"),
                low_stress_duration=record.get("low_stress_duration"),
                medium_stress_duration=record.get("medium_stress_duration"),
                high_stress_duration=record.get("high_stress_duration"),
            )
            for record in garmin_stress_data_list
        ]
        df = garmin_stress_to_dataframe(stress_records)
        
        result = analyze_stress(df)
        
        # Duration medians should be non-negative
        assert result.median_rest_stress_duration >= 0
        assert result.median_low_stress_duration >= 0
        assert result.median_medium_stress_duration >= 0
        assert result.median_high_stress_duration >= 0


class TestAnalyzeStressDaily:
    """Tests for analyze_stress_daily function."""

    def test_returns_list_of_daily_analytics(self, garmin_stress_data_list: list[dict]):
        """Test that analyze_stress_daily returns list of DailyStressAnalytics."""
        stress_records = [
            DailyStressData(
                calendar_date=date.fromisoformat(record["calendar_date"]),
                overall_stress_level=record["overall_stress_level"],
                rest_stress_duration=record.get("rest_stress_duration"),
                low_stress_duration=record.get("low_stress_duration"),
                medium_stress_duration=record.get("medium_stress_duration"),
                high_stress_duration=record.get("high_stress_duration"),
            )
            for record in garmin_stress_data_list
        ]
        df = garmin_stress_to_dataframe(stress_records)
        
        result = analyze_stress_daily(df)
        
        assert isinstance(result, list)
        assert len(result) == len(garmin_stress_data_list)
        assert all(isinstance(day, DailyStressAnalytics) for day in result)

    def test_results_sorted_by_day(self, garmin_stress_data_list: list[dict]):
        """Test that results are sorted by day."""
        # Create records in reverse order
        stress_records = [
            DailyStressData(
                calendar_date=date.fromisoformat(record["calendar_date"]),
                overall_stress_level=record["overall_stress_level"],
                rest_stress_duration=record.get("rest_stress_duration"),
                low_stress_duration=record.get("low_stress_duration"),
                medium_stress_duration=record.get("medium_stress_duration"),
                high_stress_duration=record.get("high_stress_duration"),
            )
            for record in reversed(garmin_stress_data_list)
        ]
        df = garmin_stress_to_dataframe(stress_records)
        
        result = analyze_stress_daily(df)
        
        days = [day.day for day in result]
        assert days == sorted(days)

    def test_handles_empty_data(self):
        """Test that empty data returns empty list."""
        df = pd.DataFrame()
        
        result = analyze_stress_daily(df)
        
        assert result == []


class TestGarminSleepToDataframe:
    """Tests for garmin_sleep_to_dataframe function."""

    def test_converts_sleep_data_to_dataframe(self, garmin_sleep_data_list: list[dict]):
        """Test that sleep data is correctly converted to DataFrame."""
        sleep_records = [_parse_garmin_sleep_data(record) for record in garmin_sleep_data_list]
        
        df = garmin_sleep_to_dataframe(sleep_records)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(garmin_sleep_data_list)
        assert "id" in df.columns
        assert "day" in df.columns
        assert "total_sleep_duration" in df.columns
        assert "deep_sleep_seconds" in df.columns
        assert "light_sleep_seconds" in df.columns

    def test_empty_sleep_data_returns_empty_dataframe(self):
        """Test that empty input returns empty DataFrame."""
        df = garmin_sleep_to_dataframe([])
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_dataframe_preserves_sleep_values(self, garmin_sleep_data_list: list[dict]):
        """Test that DataFrame preserves the original values."""
        sleep_records = [_parse_garmin_sleep_data(record) for record in garmin_sleep_data_list]
        
        df = garmin_sleep_to_dataframe(sleep_records)
        
        # Check first record values match
        first_dto = garmin_sleep_data_list[0]["daily_sleep_dto"]
        assert df.iloc[0]["total_sleep_duration"] == first_dto["sleep_time_seconds"]


class TestGarminHeartrateToDataframe:
    """Tests for garmin_heartrate_to_dataframe function."""

    def test_converts_heartrate_data_to_dataframe(self):
        """Test that heart rate data is correctly converted to DataFrame."""
        samples = [
            HeartRateSample(bpm=70, source="garmin", timestamp="2025-01-01T10:00:00+00:00"),
            HeartRateSample(bpm=75, source="garmin", timestamp="2025-01-01T10:01:00+00:00"),
            HeartRateSample(bpm=80, source="garmin", timestamp="2025-01-01T10:02:00+00:00"),
        ]
        heartrate_data = HeartRateData(data=samples)
        
        df = garmin_heartrate_to_dataframe(heartrate_data)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "bpm" in df.columns
        assert "source" in df.columns
        assert "timestamp" in df.columns
        assert "day" in df.columns

    def test_empty_heartrate_data_returns_empty_dataframe(self):
        """Test that empty input returns empty DataFrame."""
        heartrate_data = HeartRateData(data=[])
        
        df = garmin_heartrate_to_dataframe(heartrate_data)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0


class TestGarminClientDataParsing:
    """Tests for Garmin client data parsing functions."""

    def test_parse_garmin_sleep_data(self, garmin_sleep_data_list: list[dict]):
        """Test that Garmin sleep data is parsed correctly."""
        for raw_data in garmin_sleep_data_list:
            sleep_data = _parse_garmin_sleep_data(raw_data)
            
            dto = raw_data["daily_sleep_dto"]
            assert sleep_data.id == str(dto["id"])
            assert sleep_data.day == str(dto["calendar_date"])
            assert sleep_data.type == "long_sleep"
            assert sleep_data.total_sleep_duration == dto["sleep_time_seconds"]
            assert sleep_data.deep_sleep_seconds == dto.get("deep_sleep_seconds")

    def test_parse_garmin_sleep_data_handles_missing_fields(self):
        """Test that parsing handles missing optional fields."""
        minimal_data = {
            "daily_sleep_dto": {
                "id": "123",
                "calendar_date": "2025-01-01",
                "sleep_time_seconds": 28800,
                "sleep_start_timestamp_gmt": 1704067200000,
                "sleep_end_timestamp_gmt": 1704096000000,
            }
        }
        
        sleep_data = _parse_garmin_sleep_data(minimal_data)
        
        assert sleep_data.id == "123"
        assert sleep_data.day == "2025-01-01"
        assert sleep_data.total_sleep_duration == 28800
        assert sleep_data.deep_sleep_seconds is None
        assert sleep_data.sleep_score is None
