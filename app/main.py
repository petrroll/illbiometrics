from datetime import date
from typing import Optional

from fastapi import FastAPI, HTTPException, Query

from app.auth import router as auth_router, get_stored_token
from app.oura_client import get_heartrate_data

app = FastAPI(
    title="Oura Biometrics API",
    description="API for accessing Oura ring biometric data",
    version="0.1.0",
)

app.include_router(auth_router)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "message": "Oura Biometrics API"}


@app.get("/heartrate")
async def heartrate(
    start_date: Optional[date] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[date] = Query(None, description="End date (YYYY-MM-DD)"),
):
    """
    Get heart rate data from Oura.
    
    By default, returns data from last night (yesterday to today).
    Requires authentication via /auth/login first.
    """
    access_token = get_stored_token()
    if not access_token:
        raise HTTPException(
            status_code=401,
            detail="Not authenticated. Please visit /auth/login first.",
        )
    
    try:
        data = await get_heartrate_data(access_token, start_date, end_date)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
