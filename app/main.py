import argparse
import logging
import os
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse

# Configure logging to match uvicorn's format
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:     %(name)s - %(message)s",
)

from app.auth import router as auth_router, get_stored_token
from app.oura_client import OuraClient, DataSource, NotAuthenticatedError
from app.garmin_client import (
    GarminClient,
    GarminDataSource,
    NotAuthenticatedError as GarminNotAuthenticatedError,
)
from app.analytics import (
    oura_sleep_to_dataframe,
    analyze_sleep,
    oura_heartrate_to_dataframe,
    analyze_heart_rate,
    analyze_heart_rate_daily,
)
from app.garmin_analytics import (
    garmin_sleep_to_dataframe,
    garmin_hrv_to_dataframe,
    garmin_stress_to_dataframe,
    analyze_garmin_sleep,
    analyze_garmin_hrv,
    analyze_garmin_stress,
)

# Create routers for grouping endpoints
raw_router = APIRouter(prefix="/raw", tags=["raw"])
analytics_router = APIRouter(prefix="/analytics", tags=["analytics"])
garmin_router = APIRouter(prefix="/garmin", tags=["garmin"])

# Global client instances, initialized at startup
oura_client: OuraClient | None = None
garmin_client: GarminClient | None = None

# In-memory storage for Garmin tokens (keyed by client_key)
_garmin_token_store: dict[str, dict] = {}

# In-memory storage for Garmin sandbox data
_garmin_sandbox_store: dict[str, list[dict]] = {}

# Pre-load dashboard HTML template
_dashboard_html: str = (Path(__file__).parent / "templates" / "dashboard.html").read_text()


def get_garmin_token(client_key: str) -> Optional[dict]:
    """Get stored Garmin tokens for a client."""
    return _garmin_token_store.get(client_key)


def set_garmin_token(client_key: str, tokens: dict) -> None:
    """Store Garmin tokens for a client."""
    _garmin_token_store[client_key] = tokens


def get_garmin_sandbox_data(
    client_key: str, data_type: str, start_date: date, end_date: date
) -> Optional[list[dict]]:
    """Get Garmin sandbox data from in-memory store."""
    key = f"{data_type}_{start_date}_{end_date}"
    return _garmin_sandbox_store.get(key)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    # Environment variable takes precedence, then CLI arg, then default
    env_data_source = os.environ.get("DATA_SOURCE")
    default_source = env_data_source if env_data_source else DataSource.USER.value
    
    parser = argparse.ArgumentParser(description="Biometrics API")
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


def create_garmin_client(data_source: GarminDataSource) -> GarminClient:
    """Create a GarminClient with the specified data source."""
    if data_source == GarminDataSource.USER:
        return GarminClient(
            data_source=data_source,
            token_getter=get_garmin_token,
            token_setter=set_garmin_token,
        )
    else:
        return GarminClient(
            data_source=data_source,
            sandbox_data_getter=get_garmin_sandbox_data,
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global oura_client, garmin_client
    args = parse_args()
    data_source = DataSource(args.data_source)
    oura_client = create_oura_client(data_source)
    
    # Map Oura data source to Garmin data source
    garmin_data_source = (
        GarminDataSource.USER if data_source == DataSource.USER 
        else GarminDataSource.SANDBOX
    )
    garmin_client = create_garmin_client(garmin_data_source)
    
    print(f"Starting with data source: {data_source.value}")
    yield


app = FastAPI(
    title="Biometrics API",
    description="""
## Biometrics API

API for accessing and analyzing biometric data from wearable devices.

### Supported Data Sources
- **Oura Ring** (implemented)
- **Garmin** (implemented via garth library)

### Features
- **Sleep Analytics**: Get sleep duration, HR, HRV statistics and percentiles
- **Heart Rate Analytics**: Analyze heart rate data with daily and detailed breakdowns
- **Stress Analytics** (Garmin): Analyze stress levels with percentiles
- **OAuth Authentication**: Secure authentication with data source APIs

### Data Sources
Configure via `--data-source` flag or `DATA_SOURCE` environment variable:
- `user`: Your personal data (requires authentication)
- `sandbox`: Cached sandbox data for testing
    """,
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",  # Swagger UI
    redoc_url="/redoc",  # ReDoc
    openapi_url="/openapi.json",  # OpenAPI schema
)

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "message": "Biometrics API",
        "oura_data_source": oura_client.data_source.value if oura_client else "not initialized",
        "garmin_data_source": garmin_client.data_source.value if garmin_client else "not initialized",
    }


@analytics_router.get("/sleep")
async def sleep_analytics(
    start_date: Optional[date] = Query(None, description="Start date (YYYY-MM-DD), defaults to 30 days ago"),
    end_date: Optional[date] = Query(None, description="End date (YYYY-MM-DD), defaults to today"),
):
    """
    Get sleep analytics.
    
    Returns median sleep duration, HR, HRV, and percentiles.
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
            "start_date": str(analytics.start_date) if analytics.start_date else None,
            "end_date": str(analytics.end_date) if analytics.end_date else None,
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


@analytics_router.get("/heartrate")
async def heartrate_analytics(
    start_date: Optional[date] = Query(None, description="Start date (YYYY-MM-DD), defaults to 30 days ago"),
    end_date: Optional[date] = Query(None, description="End date (YYYY-MM-DD), defaults to today"),
):
    """
    Get aggregate non-sleep heart rate analytics.
    
    Returns average heart rate and p20, p50, p80, p95, p99 percentiles
    across all days in the date range.
    """
    if oura_client is None:
        raise HTTPException(status_code=500, detail="Client not initialized")
    
    try:
        heartrate_data, actual_start, actual_end = await oura_client.get_heartrate_data(
            start_date, end_date
        )
        df = oura_heartrate_to_dataframe(heartrate_data)
        analytics = analyze_heart_rate(df)
        return {
            "data_source": oura_client.data_source.value,
            "start_date": str(analytics.start_date) if analytics.start_date else None,
            "end_date": str(analytics.end_date) if analytics.end_date else None,
            "average_hr": analytics.average_hr,
            "hr_20th_percentile": analytics.hr_20th_percentile,
            "hr_50th_percentile": analytics.hr_50th_percentile,
            "hr_80th_percentile": analytics.hr_80th_percentile,
            "hr_95th_percentile": analytics.hr_95th_percentile,
            "hr_99th_percentile": analytics.hr_99th_percentile,
        }
    except NotAuthenticatedError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@analytics_router.get("/heart-rate-daily")
async def heartrate_daily_analytics(
    start_date: Optional[date] = Query(None, description="Start date (YYYY-MM-DD), defaults to 30 days ago"),
    end_date: Optional[date] = Query(None, description="End date (YYYY-MM-DD), defaults to today"),
):
    """
    Get daily non-sleep heart rate analytics.
    
    Returns average heart rate and p20, p50, p80, p95, p99 percentiles
    for each day in the date range.
    """
    if oura_client is None:
        raise HTTPException(status_code=500, detail="Client not initialized")
    
    try:
        heartrate_data, actual_start, actual_end = await oura_client.get_heartrate_data(
            start_date, end_date
        )
        df = oura_heartrate_to_dataframe(heartrate_data)
        daily_analytics = analyze_heart_rate_daily(df)
        return {
            "data_source": oura_client.data_source.value,
            "start_date": str(actual_start),
            "end_date": str(actual_end),
            "days": [
                {
                    "day": str(day.day),
                    "average_hr": day.average_hr,
                    "hr_20th_percentile": day.hr_20th_percentile,
                    "hr_50th_percentile": day.hr_50th_percentile,
                    "hr_80th_percentile": day.hr_80th_percentile,
                    "hr_95th_percentile": day.hr_95th_percentile,
                    "hr_99th_percentile": day.hr_99th_percentile,
                }
                for day in daily_analytics
            ],
        }
    except NotAuthenticatedError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@raw_router.get("/oura/heartrate")
async def raw_heartrate(
    start_date: Optional[date] = Query(None, description="Start date (YYYY-MM-DD), defaults to 30 days ago"),
    end_date: Optional[date] = Query(None, description="End date (YYYY-MM-DD), defaults to today"),
):
    """
    Get raw heart rate data from Oura.
    
    Returns all heart rate samples in the date range.
    Data source is configured at application startup via --data-source flag.
    """
    if oura_client is None:
        raise HTTPException(status_code=500, detail="Client not initialized")
    
    try:
        raw_response, actual_start, actual_end = await oura_client.get_heartrate_data_raw(
            start_date, end_date
        )
        return {
            "data_source": oura_client.data_source.value,
            "start_date": str(actual_start),
            "end_date": str(actual_end),
            "data": raw_response.get("data", []),
        }
    except NotAuthenticatedError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@raw_router.get("/oura/sleep")
async def raw_sleep(
    start_date: Optional[date] = Query(None, description="Start date (YYYY-MM-DD), defaults to 30 days ago"),
    end_date: Optional[date] = Query(None, description="End date (YYYY-MM-DD), defaults to today"),
):
    """
    Get raw sleep data from Oura.
    
    Returns all sleep records in the date range.
    Data source is configured at application startup via --data-source flag.
    """
    if oura_client is None:
        raise HTTPException(status_code=500, detail="Client not initialized")
    
    try:
        raw_response, actual_start, actual_end = await oura_client.get_sleep_data_raw(
            start_date, end_date
        )
        return {
            "data_source": oura_client.data_source.value,
            "start_date": str(actual_start),
            "end_date": str(actual_end),
            "data": raw_response.get("data", []),
        }
    except NotAuthenticatedError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """
    Display a dashboard with sleep and heart rate analytics.
    """
    return HTMLResponse(content=_dashboard_html)


# Garmin Analytics Endpoints

@garmin_router.get("/analytics/sleep")
async def garmin_sleep_analytics(
    start_date: Optional[date] = Query(None, description="Start date (YYYY-MM-DD), defaults to 30 days ago"),
    end_date: Optional[date] = Query(None, description="End date (YYYY-MM-DD), defaults to today"),
):
    """
    Get Garmin sleep analytics.
    
    Returns median sleep duration, deep/REM/light sleep, and stress during sleep.
    """
    if garmin_client is None:
        raise HTTPException(status_code=500, detail="Garmin client not initialized")
    
    try:
        sleep_data, actual_start, actual_end = await garmin_client.get_sleep_data(
            start_date, end_date
        )
        df = garmin_sleep_to_dataframe(sleep_data)
        analytics = analyze_garmin_sleep(df)
        return {
            "data_source": garmin_client.data_source.value,
            "start_date": str(analytics.start_date) if analytics.start_date else None,
            "end_date": str(analytics.end_date) if analytics.end_date else None,
            "median_sleep_duration_seconds": analytics.median_sleep_duration,
            "median_deep_sleep_seconds": analytics.median_deep_sleep,
            "median_rem_sleep_seconds": analytics.median_rem_sleep,
            "median_light_sleep_seconds": analytics.median_light_sleep,
            "median_avg_sleep_stress": analytics.median_avg_stress,
        }
    except GarminNotAuthenticatedError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@garmin_router.get("/analytics/hrv")
async def garmin_hrv_analytics(
    start_date: Optional[date] = Query(None, description="Start date (YYYY-MM-DD), defaults to 30 days ago"),
    end_date: Optional[date] = Query(None, description="End date (YYYY-MM-DD), defaults to today"),
):
    """
    Get Garmin HRV analytics (sleep HRV).
    
    Returns median HRV values and percentiles.
    """
    if garmin_client is None:
        raise HTTPException(status_code=500, detail="Garmin client not initialized")
    
    try:
        hrv_data, actual_start, actual_end = await garmin_client.get_hrv_data(
            start_date, end_date
        )
        df = garmin_hrv_to_dataframe(hrv_data)
        analytics = analyze_garmin_hrv(df)
        return {
            "data_source": garmin_client.data_source.value,
            "start_date": str(analytics.start_date) if analytics.start_date else None,
            "end_date": str(analytics.end_date) if analytics.end_date else None,
            "median_weekly_avg": analytics.median_weekly_avg,
            "median_last_night_avg": analytics.median_last_night_avg,
            "median_5_min_high": analytics.median_5_min_high,
            "hrv_20th_percentile": analytics.hrv_20th_percentile,
            "hrv_50th_percentile": analytics.hrv_50th_percentile,
            "hrv_80th_percentile": analytics.hrv_80th_percentile,
        }
    except GarminNotAuthenticatedError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@garmin_router.get("/analytics/stress")
async def garmin_stress_analytics(
    start_date: Optional[date] = Query(None, description="Start date (YYYY-MM-DD), defaults to 30 days ago"),
    end_date: Optional[date] = Query(None, description="End date (YYYY-MM-DD), defaults to today"),
):
    """
    Get Garmin stress analytics.
    
    Returns stress level statistics with percentiles (p20, p50, p80, p95)
    and average durations for rest, low, medium, and high stress.
    """
    if garmin_client is None:
        raise HTTPException(status_code=500, detail="Garmin client not initialized")
    
    try:
        stress_data, actual_start, actual_end = await garmin_client.get_stress_data(
            start_date, end_date
        )
        df = garmin_stress_to_dataframe(stress_data)
        analytics = analyze_garmin_stress(df)
        return {
            "data_source": garmin_client.data_source.value,
            "start_date": str(analytics.start_date) if analytics.start_date else None,
            "end_date": str(analytics.end_date) if analytics.end_date else None,
            "average_stress_level": analytics.average_stress_level,
            "median_stress_level": analytics.median_stress_level,
            "stress_20th_percentile": analytics.stress_20th_percentile,
            "stress_50th_percentile": analytics.stress_50th_percentile,
            "stress_80th_percentile": analytics.stress_80th_percentile,
            "stress_95th_percentile": analytics.stress_95th_percentile,
            "avg_rest_duration_seconds": analytics.avg_rest_duration,
            "avg_low_stress_duration_seconds": analytics.avg_low_stress_duration,
            "avg_medium_stress_duration_seconds": analytics.avg_medium_stress_duration,
            "avg_high_stress_duration_seconds": analytics.avg_high_stress_duration,
        }
    except GarminNotAuthenticatedError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Garmin Raw Endpoints

@garmin_router.get("/raw/sleep")
async def garmin_raw_sleep(
    start_date: Optional[date] = Query(None, description="Start date (YYYY-MM-DD), defaults to 30 days ago"),
    end_date: Optional[date] = Query(None, description="End date (YYYY-MM-DD), defaults to today"),
):
    """
    Get raw sleep data from Garmin.
    
    Returns all sleep records in the date range.
    """
    if garmin_client is None:
        raise HTTPException(status_code=500, detail="Garmin client not initialized")
    
    try:
        raw_response, actual_start, actual_end = await garmin_client.get_sleep_data_raw(
            start_date, end_date
        )
        return {
            "data_source": garmin_client.data_source.value,
            "start_date": str(actual_start),
            "end_date": str(actual_end),
            "data": raw_response,
        }
    except GarminNotAuthenticatedError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@garmin_router.get("/raw/hrv")
async def garmin_raw_hrv(
    start_date: Optional[date] = Query(None, description="Start date (YYYY-MM-DD), defaults to 30 days ago"),
    end_date: Optional[date] = Query(None, description="End date (YYYY-MM-DD), defaults to today"),
):
    """
    Get raw HRV data from Garmin.
    
    Returns all HRV records in the date range.
    """
    if garmin_client is None:
        raise HTTPException(status_code=500, detail="Garmin client not initialized")
    
    try:
        raw_response, actual_start, actual_end = await garmin_client.get_hrv_data_raw(
            start_date, end_date
        )
        return {
            "data_source": garmin_client.data_source.value,
            "start_date": str(actual_start),
            "end_date": str(actual_end),
            "data": raw_response,
        }
    except GarminNotAuthenticatedError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@garmin_router.get("/raw/stress")
async def garmin_raw_stress(
    start_date: Optional[date] = Query(None, description="Start date (YYYY-MM-DD), defaults to 30 days ago"),
    end_date: Optional[date] = Query(None, description="End date (YYYY-MM-DD), defaults to today"),
):
    """
    Get raw stress data from Garmin.
    
    Returns all stress records in the date range.
    """
    if garmin_client is None:
        raise HTTPException(status_code=500, detail="Garmin client not initialized")
    
    try:
        raw_response, actual_start, actual_end = await garmin_client.get_stress_data_raw(
            start_date, end_date
        )
        return {
            "data_source": garmin_client.data_source.value,
            "start_date": str(actual_start),
            "end_date": str(actual_end),
            "data": raw_response,
        }
    except GarminNotAuthenticatedError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


app.include_router(auth_router)
app.include_router(raw_router)
app.include_router(analytics_router)
app.include_router(garmin_router)
