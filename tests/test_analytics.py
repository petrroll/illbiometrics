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
    resample_heartrate,
    HeartRateAnalytics,
    DailyHeartRateAnalytics,
    NIGH_SLEEP_SLEEP_TYPE,
    HR_MAX_GAP_SECONDS,
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

    def test_computes_average_hr_std(self, heartrate_data_list: list[dict]):
        """Test that standard deviation of heart rate is computed."""
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
        
        # Std should be non-negative
        assert result.average_hr_std >= 0
        # For realistic data, std should be greater than 0 (there's variation)
        assert result.average_hr_std > 0

    def test_handles_empty_data_std(self):
        """Test that empty data returns 0 for standard deviation."""
        heartrate_data = HeartRateData(data=[])
        df = oura_heartrate_to_dataframe(heartrate_data)
        
        result = analyze_heart_rate(df)
        
        assert result.average_hr_std == 0.0


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
        """Test that gaps within HR_MAX_GAP_SECONDS are forward-filled."""
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
        """Test that gaps exceeding HR_MAX_GAP_SECONDS create separate segments."""
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
        
        # 300s gap is NOT greater than HR_MAX_GAP_SECONDS, so should be connected
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
        
        # 301s gap IS greater than HR_MAX_GAP_SECONDS, so separate segments
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


class TestMergedStreamResampling:
    """Tests for merged stream resampling behavior across sources."""

    def test_analyze_heart_rate_uses_merged_stream_for_segments(self):
        """Test that analyze_heart_rate uses merged stream for segment detection."""
        # Pattern: rest - awake - rest where awake would be isolated if filtered first
        samples = [
            HeartRateSample(bpm=55, source="rest", timestamp="2025-01-01T10:00:00+00:00"),
            HeartRateSample(bpm=75, source="awake", timestamp="2025-01-01T10:02:00+00:00"),
            HeartRateSample(bpm=52, source="rest", timestamp="2025-01-01T10:04:00+00:00"),
        ]
        heartrate_data = HeartRateData(data=samples)
        df = oura_heartrate_to_dataframe(heartrate_data)
        
        result = analyze_heart_rate(df)
        
        # Should have non-None dates since there's active data
        assert result.start_date is not None
        assert result.end_date is not None
        assert result.average_hr == 75.0  # Only the awake sample

    def test_analyze_heart_rate_daily_uses_merged_stream_for_segments(self):
        """Test that analyze_heart_rate_daily uses merged stream for segment detection."""
        samples = [
            HeartRateSample(bpm=55, source="rest", timestamp="2025-01-01T10:00:00+00:00"),
            HeartRateSample(bpm=75, source="awake", timestamp="2025-01-01T10:02:00+00:00"),
            HeartRateSample(bpm=52, source="rest", timestamp="2025-01-01T10:04:00+00:00"),
        ]
        heartrate_data = HeartRateData(data=samples)
        df = oura_heartrate_to_dataframe(heartrate_data)
        
        result = analyze_heart_rate_daily(df)
        
        assert len(result) == 1
        assert result[0].day == date(2025, 1, 1)
        assert result[0].average_hr == 75.0  # Only the awake sample

    def test_analyze_heart_rate_custom_parameters(self):
        """Test that analyze_heart_rate accepts custom resampling parameters."""
        samples = [
            HeartRateSample(bpm=70, source="awake", timestamp="2025-01-01T10:00:00+00:00"),
            HeartRateSample(bpm=80, source="awake", timestamp="2025-01-01T10:03:00+00:00"),
        ]
        heartrate_data = HeartRateData(data=samples)
        df = oura_heartrate_to_dataframe(heartrate_data)
        
        # With default parameters, should work
        result_default = analyze_heart_rate(df)
        assert result_default.start_date is not None
        
        # With custom max_gap_seconds=120, 3-minute gap creates separate segments
        result_custom = analyze_heart_rate(df, max_gap_seconds=120)
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
        result_default = analyze_heart_rate_daily(df)
        assert len(result_default) == 1
        
        # With custom resample_interval
        result_custom = analyze_heart_rate_daily(df, resample_interval="2min")
        assert len(result_custom) == 1

    def test_source_transitions_dont_break_resampling(self):
        """Test that source type changes don't create artificial gaps in resampling."""
        # Continuous data with source transitions every 2 minutes
        # Without merged stream, this would create 3 segments
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

    def test_analyze_filters_to_active_sources_after_merged_resample(self):
        """Test that analysis only includes active source timestamps after resampling."""
        # The merged resample allows continuous data, but final stats should only
        # include timestamps where active sources were present
        samples = [
            HeartRateSample(bpm=55, source="rest", timestamp="2025-01-01T10:00:00+00:00"),
            HeartRateSample(bpm=75, source="awake", timestamp="2025-01-01T10:01:00+00:00"),
            HeartRateSample(bpm=80, source="awake", timestamp="2025-01-01T10:02:00+00:00"),
            HeartRateSample(bpm=52, source="rest", timestamp="2025-01-01T10:03:00+00:00"),
        ]
        heartrate_data = HeartRateData(data=samples)
        df = oura_heartrate_to_dataframe(heartrate_data)
        
        result = analyze_heart_rate(df)
        
        # Average should be from awake timestamps only: (75 + 80) / 2 = 77.5
        assert result.average_hr == 77.5

    def test_only_rest_data_returns_empty_result(self):
        """Test that data with only rest source returns empty result for active analysis."""
        samples = [
            HeartRateSample(bpm=55, source="rest", timestamp="2025-01-01T10:00:00+00:00"),
            HeartRateSample(bpm=52, source="rest", timestamp="2025-01-01T10:02:00+00:00"),
        ]
        heartrate_data = HeartRateData(data=samples)
        df = oura_heartrate_to_dataframe(heartrate_data)
        
        result = analyze_heart_rate(df)
        
        assert result.start_date is None
        assert result.end_date is None
        assert result.average_hr == 0.0
