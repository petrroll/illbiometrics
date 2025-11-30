from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional
import httpx

from app.config import OURA_API_BASE


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


async def get_heartrate_data(
    access_token: str,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> HeartRateData:
    """
    Fetch heart rate data from Oura API.
    
    By default, fetches data from last night (yesterday to today).
    """
    if end_date is None:
        end_date = date.today()
    if start_date is None:
        start_date = end_date - timedelta(days=1)
    
    url = f"{OURA_API_BASE}/usercollection/heartrate"
    params = {
        "start_datetime": f"{start_date}T00:00:00",
        "end_datetime": f"{end_date}T23:59:59",
    }
    headers = {"Authorization": f"Bearer {access_token}"}
    
    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params, headers=headers)
        response.raise_for_status()
        json_data = response.json()
        samples = [
            HeartRateSample(
                bpm=item.get("bpm", 0),
                source=item.get("source", ""),
                timestamp=item.get("timestamp", ""),
            )
            for item in json_data.get("data", [])
        ]
        return HeartRateData(data=samples)


async def get_sleep_data(
    access_token: str,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> list[SleepData]:
    """
    Fetch sleep data from Oura API.
    
    By default, fetches last 30 days.
    """
    if end_date is None:
        end_date = date.today()
    if start_date is None:
        start_date = end_date - timedelta(days=30)
    
    url = f"{OURA_API_BASE}/usercollection/sleep"
    params = {
        "start_date": str(start_date),
        "end_date": str(end_date),
    }
    headers = {"Authorization": f"Bearer {access_token}"}
    
    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params, headers=headers)
        response.raise_for_status()
        json_data = response.json()
        return [_parse_sleep_data(item) for item in json_data.get("data", [])]
