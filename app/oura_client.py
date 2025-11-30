from datetime import date, timedelta
from typing import Optional
import httpx

from app.config import OURA_API_BASE


async def get_heartrate_data(
    access_token: str,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> dict:
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
        return response.json()
