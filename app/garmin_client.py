from dataclasses import dataclass
from datetime import date, timedelta
from enum import Enum
from typing import Callable, Optional
import logging

import garth
from garth.auth_tokens import OAuth1Token, OAuth2Token

logger = logging.getLogger(__name__)


class GarminDataSource(str, Enum):
    """Data source options for Garmin API."""
    USER = "user"  # User's actual data (requires garth auth)
    SANDBOX = "sandbox"  # Sandbox data (uses cached test fixtures via callback)


class NotAuthenticatedError(Exception):
    """Raised when user data is requested but auth is not available."""
    pass


@dataclass
class SleepHRVData:
    """HRV data from sleep."""
    calendar_date: date
    weekly_avg: Optional[int]
    last_night_avg: Optional[int]
    last_night_5_min_high: Optional[int]
    baseline_low_upper: Optional[int]
    baseline_balanced_low: Optional[int]
    baseline_balanced_upper: Optional[int]
    status: str
    feedback_phrase: str


@dataclass
class StressData:
    """Daily stress data."""
    calendar_date: date
    overall_stress_level: int
    rest_stress_duration: Optional[int]
    low_stress_duration: Optional[int]
    medium_stress_duration: Optional[int]
    high_stress_duration: Optional[int]


@dataclass
class DailySleepData:
    """Daily sleep summary."""
    calendar_date: date
    sleep_time_seconds: int
    deep_sleep_seconds: Optional[int]
    light_sleep_seconds: Optional[int]
    rem_sleep_seconds: Optional[int]
    awake_sleep_seconds: Optional[int]
    avg_sleep_stress: Optional[float]
    average_sp_o2_value: Optional[float]
    average_respiration_value: Optional[float]


# Type aliases for callbacks
# Token getter returns garth tokens dict or None
GarthTokenGetter = Callable[[str], Optional[dict]]
# Token setter stores garth tokens dict
GarthTokenSetter = Callable[[str, dict], None]
# Sandbox data getter returns cached data or None
SandboxDataGetter = Callable[[str, str, date, date], Optional[list[dict]]]


class GarminClient:
    """
    Garmin API client using the garth library.
    
    Supports two modes:
    - USER: Fetches from user's actual data (requires garth auth)
    - SANDBOX: Uses cached sandbox data via callback for testing
    
    Authentication is handled through callbacks for in-memory token storage.
    No data is stored to disk by this client.
    """
    
    def __init__(
        self,
        data_source: GarminDataSource,
        token_getter: Optional[GarthTokenGetter] = None,
        token_setter: Optional[GarthTokenSetter] = None,
        sandbox_data_getter: Optional[SandboxDataGetter] = None,
    ):
        """
        Initialize the client.
        
        Args:
            data_source: Where to fetch data from (user, sandbox)
            token_getter: Callback to get garth tokens for a client_key.
                         Returns dict with 'oauth1' and 'oauth2' keys, or None.
            token_setter: Callback to store garth tokens for a client_key.
            sandbox_data_getter: Callback to get sandbox data for testing.
                                 Receives (client_key, data_type, start_date, end_date).
        """
        self._data_source = data_source
        self._token_getter = token_getter
        self._token_setter = token_setter
        self._sandbox_data_getter = sandbox_data_getter
        
        if data_source == GarminDataSource.USER and token_getter is None:
            raise ValueError("token_getter is required for USER data source")
    
    @property
    def data_source(self) -> GarminDataSource:
        """Get the configured data source."""
        return self._data_source
    
    def _init_garth_for_client(self, client_key: str) -> bool:
        """
        Initialize garth with tokens for the given client.
        
        Returns True if authentication succeeded, False otherwise.
        """
        if self._token_getter is None:
            return False
        
        tokens = self._token_getter(client_key)
        if not tokens:
            return False
        
        try:
            # Load tokens into garth from the in-memory dict
            oauth1_token = tokens.get("oauth1")
            oauth2_token = tokens.get("oauth2")
            
            if oauth1_token and oauth2_token:
                garth.client.oauth1_token = OAuth1Token(**oauth1_token)
                garth.client.oauth2_token = OAuth2Token(**oauth2_token)
                return True
            return False
        except Exception as e:
            logger.warning(f"Failed to initialize Garmin authentication for {client_key}: {e}")
            return False
    
    def _ensure_authenticated(self, client_key: str) -> None:
        """Raise error if not authenticated for user data source."""
        if self._data_source == GarminDataSource.USER:
            if not self._init_garth_for_client(client_key):
                raise NotAuthenticatedError(
                    f"Not authenticated with Garmin for client '{client_key}'. "
                    "Please authenticate first."
                )
    
    def _get_sandbox_data(self, client_key: str, data_type: str, start_date: date, end_date: date) -> Optional[list[dict]]:
        """Get sandbox data via callback."""
        if self._sandbox_data_getter is None:
            return None
        return self._sandbox_data_getter(client_key, data_type, start_date, end_date)
    
    async def get_hrv_data(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        client_key: str = "default",
    ) -> tuple[list[SleepHRVData], date, date]:
        """
        Fetch HRV data from Garmin.
        
        Args:
            start_date: Start date for data range
            end_date: End date for data range
            client_key: The client key to retrieve tokens for (USER mode only)
        
        Returns:
            Tuple of (hrv_data_list, start_date, end_date)
        """
        if end_date is None:
            end_date = date.today()
        if start_date is None:
            start_date = end_date - timedelta(days=30)
        
        # For sandbox mode, get data via callback
        if self._data_source == GarminDataSource.SANDBOX:
            cached = self._get_sandbox_data(client_key, "hrv", start_date, end_date)
            if cached is not None:
                logger.info(f"Garmin sandbox data retrieved for hrv ({start_date} to {end_date})")
                return [self._parse_hrv_data(item) for item in cached], start_date, end_date
            else:
                logger.info(f"No Garmin sandbox data for hrv ({start_date} to {end_date})")
                return [], start_date, end_date
        
        self._ensure_authenticated(client_key)
        
        try:
            period = (end_date - start_date).days + 1
            daily_hrv_list = garth.DailyHRV.list(end_date, period)
            
            result = []
            for hrv in daily_hrv_list:
                result.append(SleepHRVData(
                    calendar_date=hrv.calendar_date,
                    weekly_avg=hrv.weekly_avg,
                    last_night_avg=hrv.last_night_avg,
                    last_night_5_min_high=hrv.last_night_5_min_high,
                    baseline_low_upper=hrv.baseline.low_upper if hrv.baseline else None,
                    baseline_balanced_low=hrv.baseline.balanced_low if hrv.baseline else None,
                    baseline_balanced_upper=hrv.baseline.balanced_upper if hrv.baseline else None,
                    status=hrv.status,
                    feedback_phrase=hrv.feedback_phrase,
                ))
            
            return result, start_date, end_date
        except Exception as e:
            logger.error(f"Failed to fetch HRV data from Garmin: {e}")
            raise
    
    def _parse_hrv_data(self, data: dict) -> SleepHRVData:
        """Parse raw HRV data dict into SleepHRVData."""
        baseline = data.get("baseline", {})
        cal_date = data["calendar_date"]
        return SleepHRVData(
            calendar_date=date.fromisoformat(cal_date) if isinstance(cal_date, str) else cal_date,
            weekly_avg=data.get("weekly_avg"),
            last_night_avg=data.get("last_night_avg"),
            last_night_5_min_high=data.get("last_night_5_min_high"),
            baseline_low_upper=baseline.get("low_upper") if baseline else None,
            baseline_balanced_low=baseline.get("balanced_low") if baseline else None,
            baseline_balanced_upper=baseline.get("balanced_upper") if baseline else None,
            status=data.get("status", ""),
            feedback_phrase=data.get("feedback_phrase", ""),
        )
    
    async def get_stress_data(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        client_key: str = "default",
    ) -> tuple[list[StressData], date, date]:
        """
        Fetch stress data from Garmin.
        
        Args:
            start_date: Start date for data range
            end_date: End date for data range
            client_key: The client key to retrieve tokens for (USER mode only)
        
        Returns:
            Tuple of (stress_data_list, start_date, end_date)
        """
        if end_date is None:
            end_date = date.today()
        if start_date is None:
            start_date = end_date - timedelta(days=30)
        
        # For sandbox mode, get data via callback
        if self._data_source == GarminDataSource.SANDBOX:
            cached = self._get_sandbox_data(client_key, "stress", start_date, end_date)
            if cached is not None:
                logger.info(f"Garmin sandbox data retrieved for stress ({start_date} to {end_date})")
                return [self._parse_stress_data(item) for item in cached], start_date, end_date
            else:
                logger.info(f"No Garmin sandbox data for stress ({start_date} to {end_date})")
                return [], start_date, end_date
        
        self._ensure_authenticated(client_key)
        
        try:
            period = (end_date - start_date).days + 1
            daily_stress_list = garth.DailyStress.list(end_date, period)
            
            result = []
            for stress in daily_stress_list:
                result.append(StressData(
                    calendar_date=stress.calendar_date,
                    overall_stress_level=stress.overall_stress_level,
                    rest_stress_duration=stress.rest_stress_duration,
                    low_stress_duration=stress.low_stress_duration,
                    medium_stress_duration=stress.medium_stress_duration,
                    high_stress_duration=stress.high_stress_duration,
                ))
            
            return result, start_date, end_date
        except Exception as e:
            logger.error(f"Failed to fetch stress data from Garmin: {e}")
            raise
    
    def _parse_stress_data(self, data: dict) -> StressData:
        """Parse raw stress data dict into StressData."""
        cal_date = data["calendar_date"]
        return StressData(
            calendar_date=date.fromisoformat(cal_date) if isinstance(cal_date, str) else cal_date,
            overall_stress_level=data.get("overall_stress_level", 0),
            rest_stress_duration=data.get("rest_stress_duration"),
            low_stress_duration=data.get("low_stress_duration"),
            medium_stress_duration=data.get("medium_stress_duration"),
            high_stress_duration=data.get("high_stress_duration"),
        )
    
    async def get_sleep_data(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        client_key: str = "default",
    ) -> tuple[list[DailySleepData], date, date]:
        """
        Fetch sleep data from Garmin.
        
        Args:
            start_date: Start date for data range
            end_date: End date for data range
            client_key: The client key to retrieve tokens for (USER mode only)
        
        Returns:
            Tuple of (sleep_data_list, start_date, end_date)
        """
        if end_date is None:
            end_date = date.today()
        if start_date is None:
            start_date = end_date - timedelta(days=30)
        
        # For sandbox mode, get data via callback
        if self._data_source == GarminDataSource.SANDBOX:
            cached = self._get_sandbox_data(client_key, "sleep", start_date, end_date)
            if cached is not None:
                logger.info(f"Garmin sandbox data retrieved for sleep ({start_date} to {end_date})")
                return [self._parse_sleep_data(item) for item in cached], start_date, end_date
            else:
                logger.info(f"No Garmin sandbox data for sleep ({start_date} to {end_date})")
                return [], start_date, end_date
        
        self._ensure_authenticated(client_key)
        
        try:
            period = (end_date - start_date).days + 1
            sleep_data_list = garth.SleepData.list(end_date, period)
            
            result = []
            for sleep in sleep_data_list:
                dto = sleep.daily_sleep_dto
                result.append(DailySleepData(
                    calendar_date=dto.calendar_date,
                    sleep_time_seconds=dto.sleep_time_seconds,
                    deep_sleep_seconds=dto.deep_sleep_seconds,
                    light_sleep_seconds=dto.light_sleep_seconds,
                    rem_sleep_seconds=dto.rem_sleep_seconds,
                    awake_sleep_seconds=dto.awake_sleep_seconds,
                    avg_sleep_stress=dto.avg_sleep_stress,
                    average_sp_o2_value=dto.average_sp_o2_value,
                    average_respiration_value=dto.average_respiration_value,
                ))
            
            return result, start_date, end_date
        except Exception as e:
            logger.error(f"Failed to fetch sleep data from Garmin: {e}")
            raise
    
    def _parse_sleep_data(self, data: dict) -> DailySleepData:
        """Parse raw sleep data dict into DailySleepData."""
        cal_date = data["calendar_date"]
        return DailySleepData(
            calendar_date=date.fromisoformat(cal_date) if isinstance(cal_date, str) else cal_date,
            sleep_time_seconds=data.get("sleep_time_seconds", 0),
            deep_sleep_seconds=data.get("deep_sleep_seconds"),
            light_sleep_seconds=data.get("light_sleep_seconds"),
            rem_sleep_seconds=data.get("rem_sleep_seconds"),
            awake_sleep_seconds=data.get("awake_sleep_seconds"),
            avg_sleep_stress=data.get("avg_sleep_stress"),
            average_sp_o2_value=data.get("average_sp_o2_value"),
            average_respiration_value=data.get("average_respiration_value"),
        )
    
    async def get_hrv_data_raw(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        client_key: str = "default",
    ) -> tuple[list[dict], date, date]:
        """
        Fetch raw HRV data from Garmin without parsing.
        
        Returns:
            Tuple of (raw_data_list, start_date, end_date)
        """
        hrv_data, start_date, end_date = await self.get_hrv_data(start_date, end_date, client_key)
        
        raw_data = []
        for hrv in hrv_data:
            raw_data.append({
                "calendar_date": str(hrv.calendar_date),
                "weekly_avg": hrv.weekly_avg,
                "last_night_avg": hrv.last_night_avg,
                "last_night_5_min_high": hrv.last_night_5_min_high,
                "baseline": {
                    "low_upper": hrv.baseline_low_upper,
                    "balanced_low": hrv.baseline_balanced_low,
                    "balanced_upper": hrv.baseline_balanced_upper,
                } if hrv.baseline_low_upper is not None else None,
                "status": hrv.status,
                "feedback_phrase": hrv.feedback_phrase,
            })
        
        return raw_data, start_date, end_date
    
    async def get_stress_data_raw(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        client_key: str = "default",
    ) -> tuple[list[dict], date, date]:
        """
        Fetch raw stress data from Garmin without parsing.
        
        Returns:
            Tuple of (raw_data_list, start_date, end_date)
        """
        stress_data, start_date, end_date = await self.get_stress_data(start_date, end_date, client_key)
        
        raw_data = []
        for stress in stress_data:
            raw_data.append({
                "calendar_date": str(stress.calendar_date),
                "overall_stress_level": stress.overall_stress_level,
                "rest_stress_duration": stress.rest_stress_duration,
                "low_stress_duration": stress.low_stress_duration,
                "medium_stress_duration": stress.medium_stress_duration,
                "high_stress_duration": stress.high_stress_duration,
            })
        
        return raw_data, start_date, end_date
    
    async def get_sleep_data_raw(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        client_key: str = "default",
    ) -> tuple[list[dict], date, date]:
        """
        Fetch raw sleep data from Garmin without parsing.
        
        Returns:
            Tuple of (raw_data_list, start_date, end_date)
        """
        sleep_data, start_date, end_date = await self.get_sleep_data(start_date, end_date, client_key)
        
        raw_data = []
        for sleep in sleep_data:
            raw_data.append({
                "calendar_date": str(sleep.calendar_date),
                "sleep_time_seconds": sleep.sleep_time_seconds,
                "deep_sleep_seconds": sleep.deep_sleep_seconds,
                "light_sleep_seconds": sleep.light_sleep_seconds,
                "rem_sleep_seconds": sleep.rem_sleep_seconds,
                "awake_sleep_seconds": sleep.awake_sleep_seconds,
                "avg_sleep_stress": sleep.avg_sleep_stress,
                "average_sp_o2_value": sleep.average_sp_o2_value,
                "average_respiration_value": sleep.average_respiration_value,
            })
        
        return raw_data, start_date, end_date
