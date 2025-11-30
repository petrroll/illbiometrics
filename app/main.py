import argparse
import os
from contextlib import asynccontextmanager
from datetime import date
from typing import Optional

from fastapi import FastAPI, HTTPException, Query

from app.auth import router as auth_router, get_stored_token
from app.oura_client import OuraClient, DataSource, NotAuthenticatedError
from app.analytics import oura_sleep_to_dataframe, analyze_sleep

# Global client instance, initialized at startup
oura_client: OuraClient | None = None


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    # Environment variable takes precedence, then CLI arg, then default
    env_data_source = os.environ.get("DATA_SOURCE")
    default_source = env_data_source if env_data_source else DataSource.USER.value
    
    parser = argparse.ArgumentParser(description="Oura Biometrics API")
    parser.add_argument(
        "--data-source",
        type=str,
        choices=[ds.value for ds in DataSource],
        default=default_source,
        help="Data source: 'user' (your data with auth), 'sandbox' (cached sandbox data, fetches if empty)",
    )
    # Use parse_known_args to ignore uvicorn's arguments when running with uvicorn
    args, _ = parser.parse_known_args()
    return args


def create_oura_client(data_source: DataSource) -> OuraClient:
    """Create an OuraClient with the specified data source."""
    if data_source == DataSource.USER:
        return OuraClient(data_source=data_source, token_getter=get_stored_token)
    else:
        return OuraClient(data_source=data_source)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global oura_client
    args = parse_args()
    data_source = DataSource(args.data_source)
    oura_client = create_oura_client(data_source)
    print(f"Starting with data source: {data_source.value}")
    yield


app = FastAPI(
    title="Oura Biometrics API",
    description="API for accessing Oura ring biometric data",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(auth_router)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "message": "Oura Biometrics API",
        "data_source": oura_client.data_source.value if oura_client else "not initialized",
    }


@app.get("/heartrate")
async def heartrate(
    start_date: Optional[date] = Query(None, description="Start date (YYYY-MM-DD), defaults to yesterday"),
    end_date: Optional[date] = Query(None, description="End date (YYYY-MM-DD), defaults to today"),
):
    """
    Get heart rate data from Oura.
    
    By default, returns data from last night (yesterday to today).
    Data source is configured at application startup via --data-source flag.
    """
    if oura_client is None:
        raise HTTPException(status_code=500, detail="Client not initialized")
    
    try:
        data, actual_start, actual_end = await oura_client.get_heartrate_data(
            start_date, end_date
        )
        return {
            "data_source": oura_client.data_source.value,
            "start_date": str(actual_start),
            "end_date": str(actual_end),
            "data": [{"bpm": s.bpm, "source": s.source, "timestamp": s.timestamp} for s in data.data],
        }
    except NotAuthenticatedError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analytics/sleep")
async def sleep_analytics(
    start_date: Optional[date] = Query(None, description="Start date (YYYY-MM-DD), defaults to 30 days ago"),
    end_date: Optional[date] = Query(None, description="End date (YYYY-MM-DD), defaults to today"),
):
    """
    Get sleep analytics.
    
    Returns median sleep duration, HR, HRV, and percentiles.
    Data source is configured at application startup via --data-source flag.
    """
    if oura_client is None:
        raise HTTPException(status_code=500, detail="Client not initialized")
    
    try:
        sleep_data, actual_start, actual_end = await oura_client.get_sleep_data(
            start_date, end_date
        )
        df = oura_sleep_to_dataframe(sleep_data)
        analytics = analyze_sleep(df)
        return {
            "data_source": oura_client.data_source.value,
            "start_date": str(actual_start),
            "end_date": str(actual_end),
            "median_sleep_duration": analytics.median_sleep_duration,
            "median_avg_hr": analytics.median_avg_hr,
            "median_avg_hrv": analytics.median_avg_hrv,
            "hr_20th_percentile": analytics.hr_20th_percentile,
            "hr_80th_percentile": analytics.hr_80th_percentile,
            "hrv_20th_percentile": analytics.hrv_20th_percentile,
            "hrv_80th_percentile": analytics.hrv_80th_percentile,
        }
    except NotAuthenticatedError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
