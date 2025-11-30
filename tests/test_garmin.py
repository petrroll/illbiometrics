"""Tests for the Garmin client and analytics modules."""

import pytest
import asyncio
from datetime import date

from app.garmin_client import (
    GarminClient,
    GarminDataSource,
    NotAuthenticatedError,
    SleepHRVData,
    StressData,
    DailySleepData,
)
from app.garmin_analytics import (
    garmin_sleep_to_dataframe,
    garmin_hrv_to_dataframe,
    garmin_stress_to_dataframe,
    analyze_garmin_sleep,
    analyze_garmin_hrv,
    analyze_garmin_stress,
    GarminSleepAnalytics,
    GarminHRVAnalytics,
    GarminStressAnalytics,
)


def run_async(coro):
    """Helper to run async code in tests."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestGarminClientInit:
    """Tests for GarminClient initialization."""
    
    def test_user_mode_requires_token_getter(self):
        """Test that USER mode requires a token_getter."""
        with pytest.raises(ValueError, match="token_getter is required"):
            GarminClient(data_source=GarminDataSource.USER)
    
    def test_sandbox_mode_works_without_token_getter(self):
        """Test that SANDBOX mode works without token_getter."""
        client = GarminClient(data_source=GarminDataSource.SANDBOX)
        assert client.data_source == GarminDataSource.SANDBOX
    
    def test_user_mode_with_token_getter(self):
        """Test USER mode with token_getter provided."""
        def token_getter(client_key: str):
            return None
        
        client = GarminClient(
            data_source=GarminDataSource.USER,
            token_getter=token_getter,
        )
        assert client.data_source == GarminDataSource.USER


class TestGarminClientSandboxMode:
    """Tests for GarminClient in sandbox mode."""
    
    @pytest.fixture
    def sandbox_client(self):
        """Create a sandbox client with test data."""
        test_data = {
            "stress": [
                {
                    "calendar_date": "2025-01-01",
                    "overall_stress_level": 30,
                    "rest_stress_duration": 30000,
                    "low_stress_duration": 20000,
                    "medium_stress_duration": 5000,
                    "high_stress_duration": 1000,
                },
                {
                    "calendar_date": "2025-01-02",
                    "overall_stress_level": 35,
                    "rest_stress_duration": 28000,
                    "low_stress_duration": 22000,
                    "medium_stress_duration": 6000,
                    "high_stress_duration": 2000,
                },
            ],
            "hrv": [
                {
                    "calendar_date": "2025-01-01",
                    "weekly_avg": 45,
                    "last_night_avg": 42,
                    "last_night_5_min_high": 60,
                    "baseline": {"low_upper": 35, "balanced_low": 40, "balanced_upper": 55},
                    "status": "BALANCED",
                    "feedback_phrase": "HRV_BALANCED",
                },
                {
                    "calendar_date": "2025-01-02",
                    "weekly_avg": 46,
                    "last_night_avg": 44,
                    "last_night_5_min_high": 62,
                    "baseline": {"low_upper": 35, "balanced_low": 40, "balanced_upper": 55},
                    "status": "BALANCED",
                    "feedback_phrase": "HRV_BALANCED",
                },
            ],
            "sleep": [
                {
                    "calendar_date": "2025-01-01",
                    "sleep_time_seconds": 28800,
                    "deep_sleep_seconds": 5400,
                    "light_sleep_seconds": 18000,
                    "rem_sleep_seconds": 5400,
                    "awake_sleep_seconds": 0,
                    "avg_sleep_stress": 15.0,
                    "average_sp_o2_value": 96.0,
                    "average_respiration_value": 14.0,
                },
            ],
        }
        
        def sandbox_data_getter(client_key: str, data_type: str, start_date: date, end_date: date):
            return test_data.get(data_type)
        
        return GarminClient(
            data_source=GarminDataSource.SANDBOX,
            sandbox_data_getter=sandbox_data_getter,
        )
    
    def test_get_stress_data(self, sandbox_client):
        """Test getting stress data in sandbox mode."""
        stress_data, start_date, end_date = run_async(sandbox_client.get_stress_data())
        
        assert len(stress_data) == 2
        assert all(isinstance(s, StressData) for s in stress_data)
        assert stress_data[0].overall_stress_level == 30
        assert stress_data[1].overall_stress_level == 35
    
    def test_get_hrv_data(self, sandbox_client):
        """Test getting HRV data in sandbox mode."""
        hrv_data, start_date, end_date = run_async(sandbox_client.get_hrv_data())
        
        assert len(hrv_data) == 2
        assert all(isinstance(h, SleepHRVData) for h in hrv_data)
        assert hrv_data[0].last_night_avg == 42
        assert hrv_data[1].last_night_avg == 44
    
    def test_get_sleep_data(self, sandbox_client):
        """Test getting sleep data in sandbox mode."""
        sleep_data, start_date, end_date = run_async(sandbox_client.get_sleep_data())
        
        assert len(sleep_data) == 1
        assert isinstance(sleep_data[0], DailySleepData)
        assert sleep_data[0].sleep_time_seconds == 28800
    
    def test_get_stress_data_raw(self, sandbox_client):
        """Test getting raw stress data in sandbox mode."""
        raw_data, start_date, end_date = run_async(sandbox_client.get_stress_data_raw())
        
        assert len(raw_data) == 2
        assert isinstance(raw_data[0], dict)
        assert raw_data[0]["overall_stress_level"] == 30


class TestGarminClientUserMode:
    """Tests for GarminClient in user mode."""
    
    def test_raises_error_when_no_tokens(self):
        """Test that NotAuthenticatedError is raised when no tokens available."""
        def token_getter(client_key: str):
            return None
        
        client = GarminClient(
            data_source=GarminDataSource.USER,
            token_getter=token_getter,
        )
        
        with pytest.raises(NotAuthenticatedError):
            run_async(client.get_stress_data())


class TestGarminStressAnalytics:
    """Tests for Garmin stress analytics."""
    
    def test_analyze_stress_empty_data(self):
        """Test stress analytics with empty data."""
        df = garmin_stress_to_dataframe([])
        result = analyze_garmin_stress(df)
        
        assert isinstance(result, GarminStressAnalytics)
        assert result.start_date is None
        assert result.end_date is None
        assert result.average_stress_level == 0.0
        assert result.stress_20th_percentile == 0.0
    
    def test_analyze_stress_with_data(self):
        """Test stress analytics with data."""
        stress_data = [
            StressData(
                calendar_date=date(2025, 1, 1),
                overall_stress_level=30,
                rest_stress_duration=30000,
                low_stress_duration=20000,
                medium_stress_duration=5000,
                high_stress_duration=1000,
            ),
            StressData(
                calendar_date=date(2025, 1, 2),
                overall_stress_level=40,
                rest_stress_duration=28000,
                low_stress_duration=22000,
                medium_stress_duration=6000,
                high_stress_duration=2000,
            ),
            StressData(
                calendar_date=date(2025, 1, 3),
                overall_stress_level=50,
                rest_stress_duration=25000,
                low_stress_duration=25000,
                medium_stress_duration=8000,
                high_stress_duration=3000,
            ),
        ]
        
        df = garmin_stress_to_dataframe(stress_data)
        result = analyze_garmin_stress(df)
        
        assert result.start_date == date(2025, 1, 1)
        assert result.end_date == date(2025, 1, 3)
        assert result.average_stress_level == 40.0
        assert result.median_stress_level == 40.0
        # Percentiles should be ordered
        assert result.stress_20th_percentile <= result.stress_50th_percentile
        assert result.stress_50th_percentile <= result.stress_80th_percentile
        assert result.stress_80th_percentile <= result.stress_95th_percentile


class TestGarminHRVAnalytics:
    """Tests for Garmin HRV analytics."""
    
    def test_analyze_hrv_empty_data(self):
        """Test HRV analytics with empty data."""
        df = garmin_hrv_to_dataframe([])
        result = analyze_garmin_hrv(df)
        
        assert isinstance(result, GarminHRVAnalytics)
        assert result.start_date is None
        assert result.end_date is None
        assert result.median_weekly_avg == 0.0
    
    def test_analyze_hrv_with_data(self):
        """Test HRV analytics with data."""
        hrv_data = [
            SleepHRVData(
                calendar_date=date(2025, 1, 1),
                weekly_avg=45,
                last_night_avg=42,
                last_night_5_min_high=60,
                baseline_low_upper=35,
                baseline_balanced_low=40,
                baseline_balanced_upper=55,
                status="BALANCED",
                feedback_phrase="HRV_BALANCED",
            ),
            SleepHRVData(
                calendar_date=date(2025, 1, 2),
                weekly_avg=47,
                last_night_avg=50,
                last_night_5_min_high=65,
                baseline_low_upper=35,
                baseline_balanced_low=40,
                baseline_balanced_upper=55,
                status="BALANCED",
                feedback_phrase="HRV_BALANCED",
            ),
        ]
        
        df = garmin_hrv_to_dataframe(hrv_data)
        result = analyze_garmin_hrv(df)
        
        assert result.start_date == date(2025, 1, 1)
        assert result.end_date == date(2025, 1, 2)
        assert result.median_weekly_avg == 46.0
        assert result.median_last_night_avg == 46.0
        # Percentiles should be ordered
        assert result.hrv_20th_percentile <= result.hrv_50th_percentile
        assert result.hrv_50th_percentile <= result.hrv_80th_percentile


class TestGarminSleepAnalytics:
    """Tests for Garmin sleep analytics."""
    
    def test_analyze_sleep_empty_data(self):
        """Test sleep analytics with empty data."""
        df = garmin_sleep_to_dataframe([])
        result = analyze_garmin_sleep(df)
        
        assert isinstance(result, GarminSleepAnalytics)
        assert result.start_date is None
        assert result.end_date is None
        assert result.median_sleep_duration == 0.0
    
    def test_analyze_sleep_with_data(self):
        """Test sleep analytics with data."""
        sleep_data = [
            DailySleepData(
                calendar_date=date(2025, 1, 1),
                sleep_time_seconds=28800,  # 8 hours
                deep_sleep_seconds=5400,
                light_sleep_seconds=18000,
                rem_sleep_seconds=5400,
                awake_sleep_seconds=0,
                avg_sleep_stress=15.0,
                average_sp_o2_value=96.0,
                average_respiration_value=14.0,
            ),
            DailySleepData(
                calendar_date=date(2025, 1, 2),
                sleep_time_seconds=25200,  # 7 hours
                deep_sleep_seconds=4800,
                light_sleep_seconds=15600,
                rem_sleep_seconds=4800,
                awake_sleep_seconds=0,
                avg_sleep_stress=18.0,
                average_sp_o2_value=95.0,
                average_respiration_value=15.0,
            ),
        ]
        
        df = garmin_sleep_to_dataframe(sleep_data)
        result = analyze_garmin_sleep(df)
        
        assert result.start_date == date(2025, 1, 1)
        assert result.end_date == date(2025, 1, 2)
        assert result.median_sleep_duration == 27000.0  # Median of 28800 and 25200
        assert result.median_avg_stress == 16.5  # Median of 15 and 18
