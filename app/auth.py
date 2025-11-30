from urllib.parse import urlencode
import httpx
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import RedirectResponse

from app.config import (
    OURA_AUTH_URL,
    OURA_CLIENT_ID,
    OURA_CLIENT_SECRET,
    OURA_REDIRECT_URI,
    OURA_TOKEN_URL,
)

router = APIRouter(prefix="/auth", tags=["auth"])

# In-memory token storage (replace with proper storage in production)
_token_store: dict = {}


def get_stored_token() -> str | None:
    """Get the stored access token."""
    return _token_store.get("access_token")


@router.get("/login")
async def login():
    """
    Redirect to Oura OAuth authorization page.
    """
    if not OURA_CLIENT_ID:
        raise HTTPException(status_code=500, detail="OURA_CLIENT_ID not configured")
    
    params = {
        "response_type": "code",
        "client_id": OURA_CLIENT_ID,
        "redirect_uri": OURA_REDIRECT_URI,
        "scope": "heartrate",
    }
    auth_url = f"{OURA_AUTH_URL}?{urlencode(params)}"
    return RedirectResponse(url=auth_url)


@router.get("/callback")
async def callback(code: str = Query(..., description="Authorization code from Oura")):
    """
    Handle OAuth callback from Oura.
    Exchange authorization code for access token.
    """
    if not OURA_CLIENT_ID or not OURA_CLIENT_SECRET:
        raise HTTPException(status_code=500, detail="OAuth credentials not configured")
    
    token_data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": OURA_REDIRECT_URI,
        "client_id": OURA_CLIENT_ID,
        "client_secret": OURA_CLIENT_SECRET,
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(OURA_TOKEN_URL, data=token_data)
        
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Failed to exchange token: {response.text}",
            )
        
        tokens = response.json()
        _token_store["access_token"] = tokens.get("access_token")
        _token_store["refresh_token"] = tokens.get("refresh_token")
        
        return {"message": "Authentication successful", "token_type": tokens.get("token_type")}


@router.get("/status")
async def auth_status():
    """Check if user is authenticated."""
    return {"authenticated": bool(_token_store.get("access_token"))}
