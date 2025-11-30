from dataclasses import dataclass
from datetime import date, timedelta
from enum import Enum
from pathlib import Path
from typing import Callable, Optional
import json
import httpx

from app.config import OURA_API_BASE, OURA_SANDBOX_BASE, SANDBOX_CACHE_DIR


class DataSource(str, Enum):
    """Data source options for Oura API."""
    USER = "user"  # User's actual data (requires auth)
    SANDBOX = "sandbox"  # Sandbox data (uses cache if available, otherwise fetches and caches)


class NotAuthenticatedError(Exception):
    """Raised when user data is requested but no auth token is available."""
    pass


@dataclass
class HeartRateSample:
    bpm: int
    source: str
    timestamp: str


@dataclass
class HeartRateData:
    data: list[HeartRateSample]


@dataclass
class SleepHRData:
    interval: float
    items: list[Optional[int]]
    timestamp: str


@dataclass
class SleepHRVData:
    interval: float
    items: list[Optional[int]]
    timestamp: str


@dataclass
class SleepData:
    id: str
    day: str
    total_sleep_duration: Optional[int]
    average_heart_rate: Optional[float]
    average_hrv: Optional[int]
    heart_rate: Optional[SleepHRData]
    hrv: Optional[SleepHRVData]


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


def _parse_sleep_data(data: dict) -> SleepData:
    return SleepData(
        id=data.get("id", ""),
        day=data.get("day", ""),
        total_sleep_duration=data.get("total_sleep_duration"),
        average_heart_rate=data.get("average_heart_rate"),
        average_hrv=data.get("average_hrv"),
        heart_rate=_parse_sleep_hr_data(data.get("heart_rate")),
        hrv=_parse_sleep_hrv_data(data.get("hrv")),
    )


def _get_sandbox_cache_path(endpoint: str, start_date: date, end_date: date) -> Path:
    """Get the cache file path for a sandbox endpoint and date range."""
    safe_name = endpoint.replace("/", "_").strip("_")
    return SANDBOX_CACHE_DIR / f"{safe_name}_{start_date}_{end_date}.json"


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
    from datetime import datetime, timezone
    
    cache_path = _get_sandbox_cache_path(endpoint, start_date, end_date)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Add metadata to the cached data
    cache_data = {
        "_metadata": {
            "endpoint": endpoint.replace("_", "/"),
            "start_date": str(start_date),
            "end_date": str(end_date),
            "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        **data,
    }
    
    with open(cache_path, "w") as f:
        json.dump(cache_data, f, indent=2)


class OuraClient:
    """
    Oura API client that handles data source switching transparently.
    
    Supports two modes:
    - USER: Fetches from user's actual data (requires auth)
    - SANDBOX: Uses cached sandbox data if available, otherwise fetches from
               Oura sandbox API and caches the result
    
    For USER mode, supports multiple users via client_key which is passed to
    each public API method and forwarded to the token_getter callback.
    """
    
    def __init__(
        self,
        data_source: DataSource,
        token_getter: Callable[[str], Optional[str]] | None = None,
    ):
        """
        Initialize the client.
        
        Args:
            data_source: Where to fetch data from (user, sandbox, cached)
            token_getter: A callable that takes a client_key and returns the current
                         access token or None (required for USER data source)
        """
        self._data_source = data_source
        self._token_getter = token_getter
        
        if data_source == DataSource.USER and token_getter is None:
            raise ValueError("token_getter is required for USER data source")
    
    @property
    def data_source(self) -> DataSource:
        """Get the configured data source."""
        return self._data_source
    
    def _get_token_or_raise(self, client_key: str) -> str:
        """Get the access token or raise NotAuthenticatedError.
        
        For sandbox modes, returns a dummy token since the sandbox API
        requires an auth header but accepts any token value.
        
        Args:
            client_key: The client key to retrieve the token for.
        """
        if self._data_source != DataSource.USER:
            return "sandbox"
        
        if self._token_getter is None:
            raise NotAuthenticatedError("No token getter configured.")
        token = self._token_getter(client_key)
        if not token:
            raise NotAuthenticatedError(
                "Not authenticated. Please visit /auth/login first."
            )
        return token
    
    def _get_base_url(self) -> str:
        """Get the API base URL based on data source."""
        if self._data_source == DataSource.USER:
            return OURA_API_BASE
        return OURA_SANDBOX_BASE
    
    async def get_heartrate_data(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        client_key: str = "default",
    ) -> tuple[HeartRateData, date, date]:
        """
        Fetch heart rate data from Oura API.
        
        By default, fetches data from last night (yesterday to today).
        
        Args:
            start_date: Start date for data range
            end_date: End date for data range
            client_key: The client key to retrieve the token for (USER mode only)
        
        Returns:
            Tuple of (heartrate_data, start_date, end_date)
        
        Raises:
            NotAuthenticatedError: If USER data source is used without authentication
        """
        if end_date is None:
            end_date = date.today()
        if start_date is None:
            start_date = end_date - timedelta(days=1)
        
        cache_endpoint = "usercollection_heartrate"
        
        # For sandbox mode, try cache first
        if self._data_source == DataSource.SANDBOX:
            cached = _load_from_sandbox_cache(cache_endpoint, start_date, end_date)
            if cached is not None:
                samples = [
                    HeartRateSample(
                        bpm=item.get("bpm", 0),
                        source=item.get("source", ""),
                        timestamp=item.get("timestamp", ""),
                    )
                    for item in cached.get("data", [])
                ]
                return HeartRateData(data=samples), start_date, end_date
        
        url = f"{self._get_base_url()}/usercollection/heartrate"
        headers = {"Authorization": f"Bearer {self._get_token_or_raise(client_key)}"}
        
        params = {
            "start_datetime": f"{start_date}T00:00:00",
            "end_datetime": f"{end_date}T23:59:59",
        }

        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params, headers=headers)
            response.raise_for_status()
            json_data = response.json()
            
            # Cache sandbox data
            if self._data_source == DataSource.SANDBOX:
                _save_to_sandbox_cache(cache_endpoint, json_data, start_date, end_date)
            
            samples = [
                HeartRateSample(
                    bpm=item.get("bpm", 0),
                    source=item.get("source", ""),
                    timestamp=item.get("timestamp", ""),
                )
                for item in json_data.get("data", [])
            ]
            return HeartRateData(data=samples), start_date, end_date

    async def get_sleep_data(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        client_key: str = "default",
    ) -> tuple[list[SleepData], date, date]:
        """
        Fetch sleep data from Oura API.
        
        By default, fetches last 30 days from user's data.
        
        Args:
            start_date: Start date for data range
            end_date: End date for data range
            client_key: The client key to retrieve the token for (USER mode only)
        
        Returns:
            Tuple of (sleep_data_list, start_date, end_date)
        
        Raises:
            NotAuthenticatedError: If USER data source is used without authentication
        """
        if end_date is None:
            end_date = date.today()
        if start_date is None:
            start_date = end_date - timedelta(days=30)
        
        cache_endpoint = "usercollection_sleep"
        
        # For sandbox mode, try cache first
        if self._data_source == DataSource.SANDBOX:
            cached = _load_from_sandbox_cache(cache_endpoint, start_date, end_date)
            if cached is not None:
                return [_parse_sleep_data(item) for item in cached.get("data", [])], start_date, end_date
        
        url = f"{self._get_base_url()}/usercollection/sleep"
        headers = {"Authorization": f"Bearer {self._get_token_or_raise(client_key)}"}        
        params = {
            "start_date": str(start_date),
            "end_date": str(end_date),
        }

        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params, headers=headers)
            response.raise_for_status()
            json_data = response.json()
            
            # Cache sandbox data
            if self._data_source == DataSource.SANDBOX:
                _save_to_sandbox_cache(cache_endpoint, json_data, start_date, end_date)
            
            return [_parse_sleep_data(item) for item in json_data.get("data", [])], start_date, end_date
