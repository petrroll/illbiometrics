import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Callable, Optional
import json
import logging

import garth
from garth import SleepData as GarthSleepData
from garth import DailyStress as GarthDailyStress
from garth import DailyHRV as GarthDailyHRV

from app.config import GARMIN_TOKEN_DIR, SANDBOX_CACHE_DIR

logger = logging.getLogger(__name__)

# Thread pool for running blocking garth API calls
_garmin_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="garmin_api")

# Default query timespan is 28 days. This is chosen because:
# 1. It's exactly 4 weeks, making it unbiased to day-of-week cycles
# 2. Consistent with Oura client behavior
DEFAULT_QUERY_DAYS = 28


class DataSource(str, Enum):
    """Data source options for Garmin API."""
    USER = "user"  # User's actual data (requires auth)
    SANDBOX = "sandbox"  # Sandbox data (uses cache if available)


class NotAuthenticatedError(Exception):
    """Raised when user data is requested but no auth token is available."""
    pass


@dataclass
class HeartRateSample:
    """Heart rate sample with timestamp."""
    bpm: int
    source: str
    timestamp: str


@dataclass
class HeartRateData:
    """Collection of heart rate samples."""
    data: list[HeartRateSample]


@dataclass
class DailyHeartRateData:
    """Daily heart rate data from Garmin."""
    calendar_date: date
    resting_heart_rate: Optional[int]
    max_heart_rate: Optional[int]
    min_heart_rate: Optional[int]
    last_seven_days_avg_resting_heart_rate: Optional[int]
    # Heart rate values are [timestamp_ms, bpm] pairs
    heart_rate_values: list[list[int]]


@dataclass
class SleepHRData:
    """Sleep heart rate data."""
    interval: float
    items: list[Optional[int]]
    timestamp: str


@dataclass
class SleepHRVData:
    """Sleep HRV data."""
    interval: float
    items: list[Optional[int]]
    timestamp: str


@dataclass
class SleepData:
    """Sleep data from Garmin."""
    id: str
    day: str
    type: str
    bedtime_start: Optional[str]
    bedtime_end: Optional[str]
    total_sleep_duration: Optional[int]
    average_heart_rate: Optional[float]
    average_hrv: Optional[int]
    heart_rate: Optional[SleepHRData]
    hrv: Optional[SleepHRVData]
    # Garmin-specific fields
    deep_sleep_seconds: Optional[int] = None
    light_sleep_seconds: Optional[int] = None
    rem_sleep_seconds: Optional[int] = None
    awake_sleep_seconds: Optional[int] = None
    sleep_score: Optional[int] = None
    avg_sleep_stress: Optional[float] = None


@dataclass
class DailyStressData:
    """Daily stress data from Garmin."""
    calendar_date: date
    overall_stress_level: int
    rest_stress_duration: Optional[int] = None
    low_stress_duration: Optional[int] = None
    medium_stress_duration: Optional[int] = None
    high_stress_duration: Optional[int] = None


def _get_sandbox_cache_path(endpoint: str, start_date: date, end_date: date) -> Path:
    """Get the cache file path for a sandbox endpoint and date range."""
    safe_name = endpoint.replace("/", "_").strip("_")
    return SANDBOX_CACHE_DIR / f"garmin_{safe_name}_{start_date}_{end_date}.json"


def _load_from_sandbox_cache(endpoint: str, start_date: date, end_date: date) -> Optional[dict]:
    """Load data from sandbox cache file for the given date range, excluding metadata."""
    cache_path = _get_sandbox_cache_path(endpoint, start_date, end_date)
    if cache_path.exists():
        with open(cache_path, "r") as f:
            data = json.load(f)
            # Remove metadata before returning (it's for human readability only)
            data.pop("_metadata", None)
            return data
    return None


def _save_to_sandbox_cache(endpoint: str, data: dict, start_date: date, end_date: date) -> None:
    """Save data to sandbox cache file with metadata."""
    cache_path = _get_sandbox_cache_path(endpoint, start_date, end_date)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Add metadata to the cached data
    cache_data = {
        "_metadata": {
            "source": "garmin",
            "endpoint": endpoint.replace("_", "/"),
            "start_date": str(start_date),
            "end_date": str(end_date),
            "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        **data,
    }
    
    with open(cache_path, "w") as f:
        json.dump(cache_data, f, indent=2)


def _parse_sleep_hr_data(data: Optional[dict]) -> Optional[SleepHRData]:
    if not data:
        return None
    return SleepHRData(
        interval=data.get("interval", 0),
        items=data.get("items", []),
        timestamp=data.get("timestamp", ""),
    )


def _parse_sleep_hrv_data(data: Optional[dict]) -> Optional[SleepHRVData]:
    if not data:
        return None
    return SleepHRVData(
        interval=data.get("interval", 0),
        items=data.get("items", []),
        timestamp=data.get("timestamp", ""),
    )


def _parse_garmin_sleep_data(data: dict) -> SleepData:
    """Parse Garmin sleep data from the garth library format."""
    # Handle both raw dict format and garth DailySleepDTO format
    daily_dto = data.get("daily_sleep_dto", data)
    
    # Extract calendar_date
    calendar_date = daily_dto.get("calendar_date")
    if isinstance(calendar_date, date):
        day_str = calendar_date.isoformat()
    else:
        day_str = str(calendar_date) if calendar_date else ""
    
    # Convert timestamps to ISO format strings
    def to_iso_string(ts: int | datetime | str | None) -> Optional[str]:
        if ts is None:
            return None
        if isinstance(ts, datetime):
            return ts.isoformat()
        if isinstance(ts, int):
            # Garmin uses milliseconds since epoch
            return datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat()
        return str(ts)
    
    sleep_start = to_iso_string(daily_dto.get("sleep_start_timestamp_gmt"))
    sleep_end = to_iso_string(daily_dto.get("sleep_end_timestamp_gmt"))
    
    return SleepData(
        id=str(daily_dto.get("id", "")),
        day=day_str,
        type="long_sleep",  # Garmin primary sleep is always "long_sleep"
        bedtime_start=sleep_start,
        bedtime_end=sleep_end,
        total_sleep_duration=daily_dto.get("sleep_time_seconds"),
        average_heart_rate=daily_dto.get("average_sp_o2_hr_sleep"),
        average_hrv=None,  # HRV is fetched separately in Garmin
        heart_rate=None,  # Detailed HR is in a separate call
        hrv=None,  # Detailed HRV is in a separate call
        deep_sleep_seconds=daily_dto.get("deep_sleep_seconds"),
        light_sleep_seconds=daily_dto.get("light_sleep_seconds"),
        rem_sleep_seconds=daily_dto.get("rem_sleep_seconds"),
        awake_sleep_seconds=daily_dto.get("awake_sleep_seconds"),
        sleep_score=daily_dto.get("sleep_scores", {}).get("overall", {}).get("value") if daily_dto.get("sleep_scores") else None,
        avg_sleep_stress=daily_dto.get("avg_sleep_stress"),
    )


def _parse_stress_data(data: dict) -> DailyStressData:
    """Parse Garmin stress data."""
    calendar_date = data.get("calendar_date")
    if isinstance(calendar_date, str):
        calendar_date = date.fromisoformat(calendar_date)
    elif not isinstance(calendar_date, date):
        calendar_date = date.today()
    
    return DailyStressData(
        calendar_date=calendar_date,
        overall_stress_level=data.get("overall_stress_level", 0),
        rest_stress_duration=data.get("rest_stress_duration"),
        low_stress_duration=data.get("low_stress_duration"),
        medium_stress_duration=data.get("medium_stress_duration"),
        high_stress_duration=data.get("high_stress_duration"),
    )


class GarminClient:
    """
    Garmin API client using the garth library.
    
    Supports two modes:
    - USER: Fetches from user's actual data (requires garth auth)
    - SANDBOX: Uses cached sandbox data if available
    
    For USER mode, garth must be authenticated via 'uvx garth login' before use.
    """
    
    def __init__(
        self,
        data_source: DataSource,
        token_dir: str | None = None,
    ):
        """
        Initialize the client.
        
        Args:
            data_source: Where to fetch data from (user, sandbox)
            token_dir: Directory where garth tokens are stored (default: ~/.garth)
        """
        self._data_source = data_source
        self._token_dir = token_dir or GARMIN_TOKEN_DIR
        self._garth_client = None
        
        if data_source == DataSource.USER:
            self._init_garth_client()
    
    def _init_garth_client(self) -> None:
        """Initialize the garth client with stored credentials."""
        try:
            # Try to resume from saved session
            garth.resume(self._token_dir)
            
            # Test that we have valid credentials
            _ = garth.client.username
            self._garth_client = garth.client
            logger.info(f"Garmin client initialized for user: {self._garth_client.username}")
        except Exception as e:
            logger.warning(f"Failed to initialize Garmin client: {e}")
            self._garth_client = None
    
    @property
    def data_source(self) -> DataSource:
        """Get the configured data source."""
        return self._data_source
    
    def _ensure_authenticated(self) -> None:
        """Ensure the client is authenticated for user data access."""
        if self._data_source != DataSource.USER:
            return
        
        if self._garth_client is None:
            raise NotAuthenticatedError(
                "Not authenticated. Please run 'uvx garth login' to authenticate with Garmin."
            )
    
    async def get_sleep_data_raw(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        client_key: str = "default",
    ) -> tuple[dict, date, date]:
        """
        Fetch raw sleep data from Garmin API without parsing.
        
        Args:
            start_date: Start date for data range
            end_date: End date for data range
            client_key: Ignored for Garmin (for API compatibility)
        
        Returns:
            Tuple of (raw_json_data, start_date, end_date)
        """
        if end_date is None:
            end_date = date.today()
        if start_date is None:
            start_date = end_date - timedelta(days=DEFAULT_QUERY_DAYS)
        
        cache_endpoint = "sleep"
        
        # For sandbox mode, try cache first
        if self._data_source == DataSource.SANDBOX:
            cached = _load_from_sandbox_cache(cache_endpoint, start_date, end_date)
            if cached is not None:
                logger.info(f"Sandbox cache hit for {cache_endpoint} ({start_date} to {end_date})")
                return cached, start_date, end_date
            else:
                logger.info(f"Sandbox cache miss for {cache_endpoint} ({start_date} to {end_date})")
                # Return empty data for sandbox without cache
                return {"data": []}, start_date, end_date
        
        # User mode - fetch from Garmin via garth
        self._ensure_authenticated()
        
        # Run blocking garth API calls in thread pool
        loop = asyncio.get_event_loop()
        data_list = await loop.run_in_executor(
            _garmin_executor,
            self._fetch_sleep_data_sync,
            start_date,
            end_date
        )
        
        json_data = {"data": data_list}
        return json_data, start_date, end_date
    
    def _fetch_sleep_data_sync(self, start_date: date, end_date: date) -> list[dict]:
        """Fetch sleep data synchronously. Must be run in a thread pool."""
        data_list = []
        current_date = start_date
        while current_date <= end_date:
            try:
                # Use the authenticated client
                sleep_data = GarthSleepData.get(str(current_date), client=self._garth_client)
                if sleep_data:
                    # Convert to dict format
                    data_list.append({
                        "daily_sleep_dto": {
                            "id": sleep_data.daily_sleep_dto.id,
                            "calendar_date": str(sleep_data.daily_sleep_dto.calendar_date),
                            "sleep_time_seconds": sleep_data.daily_sleep_dto.sleep_time_seconds,
                            "sleep_start_timestamp_gmt": sleep_data.daily_sleep_dto.sleep_start_timestamp_gmt,
                            "sleep_end_timestamp_gmt": sleep_data.daily_sleep_dto.sleep_end_timestamp_gmt,
                            "deep_sleep_seconds": sleep_data.daily_sleep_dto.deep_sleep_seconds,
                            "light_sleep_seconds": sleep_data.daily_sleep_dto.light_sleep_seconds,
                            "rem_sleep_seconds": sleep_data.daily_sleep_dto.rem_sleep_seconds,
                            "awake_sleep_seconds": sleep_data.daily_sleep_dto.awake_sleep_seconds,
                            "average_sp_o2_hr_sleep": sleep_data.daily_sleep_dto.average_sp_o2_hr_sleep,
                            "avg_sleep_stress": sleep_data.daily_sleep_dto.avg_sleep_stress,
                            "sleep_scores": {
                                "overall": {"value": sleep_data.daily_sleep_dto.sleep_scores.overall.value}
                            } if sleep_data.daily_sleep_dto.sleep_scores else None,
                        }
                    })
            except Exception as e:
                logger.debug(f"No sleep data for {current_date}: {e}")
            current_date += timedelta(days=1)
        
        return data_list
    
    async def get_sleep_data(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        client_key: str = "default",
    ) -> tuple[list[SleepData], date, date]:
        """
        Fetch sleep data from Garmin API.
        
        Args:
            start_date: Start date for data range
            end_date: End date for data range
            client_key: Ignored for Garmin (for API compatibility)
        
        Returns:
            Tuple of (sleep_data_list, start_date, end_date)
        """
        json_data, start_date, end_date = await self.get_sleep_data_raw(
            start_date, end_date, client_key
        )
        
        return [_parse_garmin_sleep_data(item) for item in json_data.get("data", [])], start_date, end_date
    
    async def get_stress_data_raw(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        client_key: str = "default",
    ) -> tuple[dict, date, date]:
        """
        Fetch raw stress data from Garmin API without parsing.
        
        Args:
            start_date: Start date for data range
            end_date: End date for data range
            client_key: Ignored for Garmin (for API compatibility)
        
        Returns:
            Tuple of (raw_json_data, start_date, end_date)
        """
        if end_date is None:
            end_date = date.today()
        if start_date is None:
            start_date = end_date - timedelta(days=DEFAULT_QUERY_DAYS)
        
        cache_endpoint = "stress"
        
        # For sandbox mode, try cache first
        if self._data_source == DataSource.SANDBOX:
            cached = _load_from_sandbox_cache(cache_endpoint, start_date, end_date)
            if cached is not None:
                logger.info(f"Sandbox cache hit for {cache_endpoint} ({start_date} to {end_date})")
                return cached, start_date, end_date
            else:
                logger.info(f"Sandbox cache miss for {cache_endpoint} ({start_date} to {end_date})")
                # Return empty data for sandbox without cache
                return {"data": []}, start_date, end_date
        
        # User mode - fetch from Garmin via garth
        self._ensure_authenticated()
        
        # Calculate number of days
        days = (end_date - start_date).days + 1
        
        # Run blocking garth API calls in thread pool
        loop = asyncio.get_event_loop()
        data_list = await loop.run_in_executor(
            _garmin_executor,
            self._fetch_stress_data_sync,
            end_date,
            days
        )
        
        json_data = {"data": data_list}
        return json_data, start_date, end_date
    
    def _fetch_stress_data_sync(self, end_date: date, days: int) -> list[dict]:
        """Fetch stress data synchronously. Must be run in a thread pool."""
        data_list = []
        try:
            stress_list = GarthDailyStress.list(str(end_date), days, client=self._garth_client)
            for stress in stress_list:
                data_list.append({
                    "calendar_date": str(stress.calendar_date),
                    "overall_stress_level": stress.overall_stress_level,
                    "rest_stress_duration": stress.rest_stress_duration,
                    "low_stress_duration": stress.low_stress_duration,
                    "medium_stress_duration": stress.medium_stress_duration,
                    "high_stress_duration": stress.high_stress_duration,
                })
        except Exception as e:
            logger.warning(f"Failed to fetch stress data: {e}")
        return data_list
    
    async def get_stress_data(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        client_key: str = "default",
    ) -> tuple[list[DailyStressData], date, date]:
        """
        Fetch stress data from Garmin API.
        
        Args:
            start_date: Start date for data range
            end_date: End date for data range
            client_key: Ignored for Garmin (for API compatibility)
        
        Returns:
            Tuple of (stress_data_list, start_date, end_date)
        """
        json_data, start_date, end_date = await self.get_stress_data_raw(
            start_date, end_date, client_key
        )
        
        return [_parse_stress_data(item) for item in json_data.get("data", [])], start_date, end_date
    
    async def get_hrv_data_raw(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        client_key: str = "default",
    ) -> tuple[dict, date, date]:
        """
        Fetch raw HRV data from Garmin API without parsing.
        
        Args:
            start_date: Start date for data range
            end_date: End date for data range
            client_key: Ignored for Garmin (for API compatibility)
        
        Returns:
            Tuple of (raw_json_data, start_date, end_date)
        """
        if end_date is None:
            end_date = date.today()
        if start_date is None:
            start_date = end_date - timedelta(days=DEFAULT_QUERY_DAYS)
        
        cache_endpoint = "hrv"
        
        # For sandbox mode, try cache first
        if self._data_source == DataSource.SANDBOX:
            cached = _load_from_sandbox_cache(cache_endpoint, start_date, end_date)
            if cached is not None:
                logger.info(f"Sandbox cache hit for {cache_endpoint} ({start_date} to {end_date})")
                return cached, start_date, end_date
            else:
                logger.info(f"Sandbox cache miss for {cache_endpoint} ({start_date} to {end_date})")
                # Return empty data for sandbox without cache
                return {"data": []}, start_date, end_date
        
        # User mode - fetch from Garmin via garth
        self._ensure_authenticated()
        
        # Calculate number of days
        days = (end_date - start_date).days + 1
        
        # Run blocking garth API calls in thread pool
        loop = asyncio.get_event_loop()
        data_list = await loop.run_in_executor(
            _garmin_executor,
            self._fetch_hrv_data_sync,
            start_date,
            end_date,
            days
        )
        
        json_data = {"data": data_list}
        return json_data, start_date, end_date
    
    def _fetch_hrv_data_sync(self, start_date: date, end_date: date, days: int) -> list[dict]:
        """Fetch HRV data synchronously. Must be run in a thread pool."""
        data_list = []
        try:
            hrv_list = GarthDailyHRV.list(period=days, client=self._garth_client)
            for hrv in hrv_list:
                # Filter to date range
                if start_date <= hrv.calendar_date <= end_date:
                    data_list.append({
                        "calendar_date": str(hrv.calendar_date),
                        "weekly_avg": hrv.weekly_avg,
                        "last_night_avg": hrv.last_night_avg,
                        "last_night_5_min_high": hrv.last_night_5_min_high,
                        "status": hrv.status,
                    })
        except Exception as e:
            logger.warning(f"Failed to fetch HRV data: {e}")
        return data_list
    
    async def get_heartrate_data_raw(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        client_key: str = "default",
    ) -> tuple[dict, date, date]:
        """
        Fetch raw heart rate data from Garmin API without parsing.
        
        Uses the dailyHeartRate endpoint which provides time series heart rate data
        along with daily summary stats (resting, min, max HR).
        
        Args:
            start_date: Start date for data range
            end_date: End date for data range
            client_key: Ignored for Garmin (for API compatibility)
        
        Returns:
            Tuple of (raw_json_data, start_date, end_date)
        """
        if end_date is None:
            end_date = date.today()
        if start_date is None:
            start_date = end_date - timedelta(days=DEFAULT_QUERY_DAYS)
        
        cache_endpoint = "heartrate"
        
        # For sandbox mode, try cache first
        if self._data_source == DataSource.SANDBOX:
            cached = _load_from_sandbox_cache(cache_endpoint, start_date, end_date)
            if cached is not None:
                logger.info(f"Sandbox cache hit for {cache_endpoint} ({start_date} to {end_date})")
                return cached, start_date, end_date
            else:
                logger.info(f"Sandbox cache miss for {cache_endpoint} ({start_date} to {end_date})")
                # Return empty data for sandbox without cache
                return {"data": []}, start_date, end_date
        
        # User mode - fetch from Garmin via garth connectapi
        self._ensure_authenticated()
        
        # Run blocking garth API calls in thread pool
        loop = asyncio.get_event_loop()
        data_list = await loop.run_in_executor(
            _garmin_executor,
            self._fetch_heartrate_data_sync,
            start_date,
            end_date
        )
        
        json_data = {"data": data_list}
        return json_data, start_date, end_date
    
    def _fetch_heartrate_data_sync(self, start_date: date, end_date: date) -> list[dict]:
        """
        Fetch heart rate data synchronously. Must be run in a thread pool.
        
        Uses the /wellness-service/wellness/dailyHeartRate endpoint which returns
        daily heart rate data including time series values.
        """
        data_list = []
        current_date = start_date
        
        while current_date <= end_date:
            try:
                # Use the connectapi to fetch daily heart rate data
                # Endpoint based on https://github.com/matin/garth/issues/134
                path = f"/wellness-service/wellness/dailyHeartRate/?date={current_date}"
                hr_data = self._garth_client.connectapi(path)
                
                if hr_data and isinstance(hr_data, dict):
                    # Extract heart rate samples from heartRateValues
                    # Format is [[timestamp_ms, bpm], ...]
                    heart_rate_values = hr_data.get("heartRateValues", [])
                    
                    data_list.append({
                        "calendar_date": str(current_date),
                        "resting_heart_rate": hr_data.get("restingHeartRate"),
                        "max_heart_rate": hr_data.get("maxHeartRate"),
                        "min_heart_rate": hr_data.get("minHeartRate"),
                        "last_seven_days_avg_resting_heart_rate": hr_data.get("lastSevenDaysAvgRestingHeartRate"),
                        "heart_rate_values": heart_rate_values,
                    })
            except Exception as e:
                logger.debug(f"No heart rate data for {current_date}: {e}")
            
            current_date += timedelta(days=1)
        
        return data_list
    
    async def get_heartrate_data(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        client_key: str = "default",
    ) -> tuple[HeartRateData, date, date]:
        """
        Fetch heart rate data from Garmin API.
        
        Converts the daily heart rate data into a flat list of HeartRateSample objects.
        
        Args:
            start_date: Start date for data range
            end_date: End date for data range
            client_key: Ignored for Garmin (for API compatibility)
        
        Returns:
            Tuple of (heartrate_data, start_date, end_date)
        """
        json_data, start_date, end_date = await self.get_heartrate_data_raw(
            start_date, end_date, client_key
        )
        
        samples = []
        for day_data in json_data.get("data", []):
            # Convert heart_rate_values [[timestamp_ms, bpm], ...] to HeartRateSample objects
            for hr_value in day_data.get("heart_rate_values", []):
                if isinstance(hr_value, list) and len(hr_value) >= 2:
                    timestamp_ms, bpm = hr_value[0], hr_value[1]
                    # Convert timestamp from milliseconds to ISO format
                    if isinstance(timestamp_ms, int) and isinstance(bpm, int) and bpm > 0:
                        timestamp = datetime.fromtimestamp(
                            timestamp_ms / 1000, tz=timezone.utc
                        ).isoformat()
                        samples.append(HeartRateSample(
                            bpm=bpm,
                            source="garmin",
                            timestamp=timestamp,
                        ))
        
        return HeartRateData(data=samples), start_date, end_date
    
    async def get_daily_heartrate_data(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        client_key: str = "default",
    ) -> tuple[list[DailyHeartRateData], date, date]:
        """
        Fetch daily heart rate summary data from Garmin API.
        
        Returns structured daily heart rate data including resting HR, min/max,
        and the time series values.
        
        Args:
            start_date: Start date for data range
            end_date: End date for data range
            client_key: Ignored for Garmin (for API compatibility)
        
        Returns:
            Tuple of (daily_heartrate_data_list, start_date, end_date)
        """
        json_data, start_date, end_date = await self.get_heartrate_data_raw(
            start_date, end_date, client_key
        )
        
        daily_data = []
        for item in json_data.get("data", []):
            calendar_date = item.get("calendar_date")
            if isinstance(calendar_date, str):
                calendar_date = date.fromisoformat(calendar_date)
            elif not isinstance(calendar_date, date):
                continue
            
            daily_data.append(DailyHeartRateData(
                calendar_date=calendar_date,
                resting_heart_rate=item.get("resting_heart_rate"),
                max_heart_rate=item.get("max_heart_rate"),
                min_heart_rate=item.get("min_heart_rate"),
                last_seven_days_avg_resting_heart_rate=item.get("last_seven_days_avg_resting_heart_rate"),
                heart_rate_values=item.get("heart_rate_values", []),
            ))
        
        return daily_data, start_date, end_date
