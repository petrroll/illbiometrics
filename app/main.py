import argparse
import calendar
import logging
import os
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import APIRouter, FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse

# Configure logging to match uvicorn's format
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:     %(name)s - %(message)s",
)

from app.auth import router as auth_router, get_stored_token
from app.oura_client import OuraClient, DataSource, NotAuthenticatedError
from app.analytics import (
    oura_sleep_to_dataframe,
    oura_heartrate_to_dataframe,
    analyze_combined,
    analyze_combined_daily,
    compare_periods,
)

# Create routers for grouping endpoints
raw_router = APIRouter(prefix="/raw", tags=["raw"])
analytics_router = APIRouter(prefix="/analytics", tags=["analytics"])

# Global client instance, initialized at startup
oura_client: OuraClient | None = None

# Pre-load dashboard HTML template
_dashboard_html: str = (Path(__file__).parent / "templates" / "dashboard.html").read_text()


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
    title="Biometrics API",
    description="""
## Biometrics API

API for accessing and analyzing biometric data from wearable devices.

### Supported Data Sources
- **Oura Ring** (currently implemented)
- More sources coming soon (Garmin, Whoop, etc.)

### Features
- **Sleep Analytics**: Get sleep duration, HR, HRV statistics and percentiles
- **Heart Rate Analytics**: Analyze heart rate data with daily and detailed breakdowns
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
        "data_source": oura_client.data_source.value if oura_client else "not initialized",
    }


@analytics_router.get("")
async def combined_analytics(
    start_date: Optional[date] = Query(None, description="Start date (YYYY-MM-DD), defaults to 28 days ago"),
    end_date: Optional[date] = Query(None, description="End date (YYYY-MM-DD), defaults to today"),
):
    """
    Get combined sleep and heart rate analytics.
    
    Returns sleep analytics (median duration, HR, HRV, percentiles) and
    heart rate analytics (average, std, percentiles) for the date range.
    Heart rate samples during sleep periods are excluded.
    """
    if oura_client is None:
        raise HTTPException(status_code=500, detail="Client not initialized")
    
    try:
        # Fetch both sleep and heart rate data
        sleep_data, actual_start, actual_end = await oura_client.get_sleep_data(start_date, end_date)
        heartrate_data, _, _ = await oura_client.get_heartrate_data(start_date, end_date)
        
        sleep_df = oura_sleep_to_dataframe(sleep_data)
        hr_df = oura_heartrate_to_dataframe(heartrate_data)
        
        analytics = analyze_combined(sleep_df, hr_df, actual_start, actual_end)
        
        return {
            "data_source": oura_client.data_source.value,
            "sleep": {
                "start_date": str(analytics.sleep.start_date) if analytics.sleep.start_date else None,
                "end_date": str(analytics.sleep.end_date) if analytics.sleep.end_date else None,
                "nights_count": analytics.sleep.nights_count,
                "median_sleep_duration": analytics.sleep.median_sleep_duration,
                "sleep_duration_std": analytics.sleep.sleep_duration_std,
                "median_avg_hr": analytics.sleep.median_avg_hr,
                "avg_hr_std": analytics.sleep.avg_hr_std,
                "median_avg_hrv": analytics.sleep.median_avg_hrv,
                "avg_hrv_std": analytics.sleep.avg_hrv_std,
                "hr_20th_percentile": analytics.sleep.hr_20th_percentile,
                "hr_80th_percentile": analytics.sleep.hr_80th_percentile,
                "hrv_20th_percentile": analytics.sleep.hrv_20th_percentile,
                "hrv_80th_percentile": analytics.sleep.hrv_80th_percentile,
            },
            "heart_rate": {
                "start_date": str(analytics.heart_rate.start_date) if analytics.heart_rate.start_date else None,
                "end_date": str(analytics.heart_rate.end_date) if analytics.heart_rate.end_date else None,
                "hours_with_good_data": analytics.heart_rate.hours_with_good_data,
                "sleep_hours_filtered": analytics.heart_rate.sleep_hours_filtered,
                "average_hr": analytics.heart_rate.average_hr,
                "average_hr_std": analytics.heart_rate.average_hr_std,
                "hr_20th_percentile": analytics.heart_rate.hr_20th_percentile,
                "hr_50th_percentile": analytics.heart_rate.hr_50th_percentile,
                "hr_80th_percentile": analytics.heart_rate.hr_80th_percentile,
                "hr_95th_percentile": analytics.heart_rate.hr_95th_percentile,
                "hr_99th_percentile": analytics.heart_rate.hr_99th_percentile,
            },
        }
    except NotAuthenticatedError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@analytics_router.get("/treatment-comparison")
async def treatment_comparison(
    treatment_month: str = Query(..., description="Month treatment started (YYYY-MM format)"),
    pre_month: Optional[str] = Query(None, description="Pre-treatment month (YYYY-MM), defaults to month before treatment"),
    post_month: Optional[str] = Query(None, description="Post-treatment month (YYYY-MM), defaults to month after treatment"),
):
    """
    Compare biometric data before and after a treatment start date.
    
    Compares the full calendar month before treatment vs the full calendar month after.
    For example, if treatment started in March 2025, compares February 2025 vs April 2025.
    
    Returns comparison of sleep and heart rate metrics with significance assessment.
    """
    if oura_client is None:
        raise HTTPException(status_code=500, detail="Client not initialized")
    
    try:
        # Parse treatment month
        treatment_year, treatment_month_num = map(int, treatment_month.split("-"))
        
        # Calculate pre and post months (full calendar months)
        if pre_month:
            pre_year, pre_month_num = map(int, pre_month.split("-"))
        else:
            # Default: month before treatment
            if treatment_month_num == 1:
                pre_year, pre_month_num = treatment_year - 1, 12
            else:
                pre_year, pre_month_num = treatment_year, treatment_month_num - 1
        
        if post_month:
            post_year, post_month_num = map(int, post_month.split("-"))
        else:
            # Default: month after treatment
            if treatment_month_num == 12:
                post_year, post_month_num = treatment_year + 1, 1
            else:
                post_year, post_month_num = treatment_year, treatment_month_num + 1
        
        # Get full date ranges for each month
        pre_start = date(pre_year, pre_month_num, 1)
        pre_end = date(pre_year, pre_month_num, calendar.monthrange(pre_year, pre_month_num)[1])
        post_start = date(post_year, post_month_num, 1)
        post_end = date(post_year, post_month_num, calendar.monthrange(post_year, post_month_num)[1])
        
        # Fetch data for each period separately (Oura API limits to 30 days per request)
        pre_sleep_data, _, _ = await oura_client.get_sleep_data(pre_start, pre_end)
        pre_hr_data, _, _ = await oura_client.get_heartrate_data(pre_start, pre_end)
        post_sleep_data, _, _ = await oura_client.get_sleep_data(post_start, post_end)
        post_hr_data, _, _ = await oura_client.get_heartrate_data(post_start, post_end)
        
        # Prepare DataFrames for compare_periods
        pre_sleep_df = oura_sleep_to_dataframe(pre_sleep_data)
        post_sleep_df = oura_sleep_to_dataframe(post_sleep_data)
        pre_hr_df = oura_heartrate_to_dataframe(pre_hr_data)
        post_hr_df = oura_heartrate_to_dataframe(post_hr_data)
        
        # Compare periods
        comparison = compare_periods(
            pre_sleep_df, post_sleep_df,
            pre_hr_df, post_hr_df,
            pre_start, pre_end,
            post_start, post_end
        )
        
        # Helper to format period analytics
        def format_combined_analytics(analytics):
            return {
                "sleep": {
                    "start_date": str(analytics.sleep.start_date) if analytics.sleep.start_date else None,
                    "end_date": str(analytics.sleep.end_date) if analytics.sleep.end_date else None,
                    "nights_count": analytics.sleep.nights_count,
                    "median_sleep_duration": analytics.sleep.median_sleep_duration,
                    "sleep_duration_std": analytics.sleep.sleep_duration_std,
                    "median_avg_hr": analytics.sleep.median_avg_hr,
                    "avg_hr_std": analytics.sleep.avg_hr_std,
                    "median_avg_hrv": analytics.sleep.median_avg_hrv,
                    "avg_hrv_std": analytics.sleep.avg_hrv_std,
                },
                "heart_rate": {
                    "start_date": str(analytics.heart_rate.start_date) if analytics.heart_rate.start_date else None,
                    "end_date": str(analytics.heart_rate.end_date) if analytics.heart_rate.end_date else None,
                    "hours_with_good_data": analytics.heart_rate.hours_with_good_data,
                    "average_hr": analytics.heart_rate.average_hr,
                    "average_hr_std": analytics.heart_rate.average_hr_std,
                    "hr_50th_percentile": analytics.heart_rate.hr_50th_percentile,
                },
            }
        
        return {
            "data_source": oura_client.data_source.value,
            "treatment_month": treatment_month,
            "pre_period": {
                "start_date": str(comparison.pre_start),
                "end_date": str(comparison.pre_end),
                "analytics": format_combined_analytics(comparison.pre_period),
            },
            "post_period": {
                "start_date": str(comparison.post_start),
                "end_date": str(comparison.post_end),
                "analytics": format_combined_analytics(comparison.post_period),
            },
            "comparison": {
                "sleep_duration_diff": comparison.sleep_duration_diff,
                "median_hr_diff": comparison.median_hr_diff,
                "median_hrv_diff": comparison.median_hrv_diff,
                "hr_50th_percentile_diff": comparison.hr_50th_percentile_diff,
                "suggested_effect": comparison.suggested_effect,
                "significant_change": comparison.significant_change,
                "significance_details": comparison.significance_details,
            },
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {e}")
    except NotAuthenticatedError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@analytics_router.get("/daily")
async def combined_daily_analytics(
    start_date: Optional[date] = Query(None, description="Start date (YYYY-MM-DD), defaults to 28 days ago"),
    end_date: Optional[date] = Query(None, description="End date (YYYY-MM-DD), defaults to today"),
):
    """
    Get daily heart rate analytics.
    
    Returns average heart rate and percentiles for each day.
    Heart rate samples during sleep periods are excluded.
    """
    if oura_client is None:
        raise HTTPException(status_code=500, detail="Client not initialized")
    
    try:
        # Fetch both sleep and heart rate data
        sleep_data, actual_start, actual_end = await oura_client.get_sleep_data(start_date, end_date)
        heartrate_data, _, _ = await oura_client.get_heartrate_data(start_date, end_date)
        
        sleep_df = oura_sleep_to_dataframe(sleep_data)
        hr_df = oura_heartrate_to_dataframe(heartrate_data)
        
        analytics = analyze_combined_daily(sleep_df, hr_df, actual_start, actual_end)
        
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
                for day in analytics.days
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
    start_date: Optional[date] = Query(None, description="Start date (YYYY-MM-DD), defaults to 28 days ago"),
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
    start_date: Optional[date] = Query(None, description="Start date (YYYY-MM-DD), defaults to 28 days ago"),
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

app.include_router(auth_router)
app.include_router(raw_router)
app.include_router(analytics_router)
