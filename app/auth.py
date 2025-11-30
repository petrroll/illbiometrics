from urllib.parse import urlencode, quote, unquote
import json
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
# Keyed by client_key -> {"access_token": ..., "refresh_token": ...}
_token_store: dict[str, dict[str, str]] = {}

DEFAULT_CLIENT_KEY = "default"


def get_stored_token(client_key: str = DEFAULT_CLIENT_KEY) -> str | None:
    """Get the stored access token for a client.
    
    Args:
        client_key: The client key to retrieve the token for.
    
    Returns:
        The access token if available, None otherwise.
    """
    client_tokens = _token_store.get(client_key, {})
    return client_tokens.get("access_token")


@router.get("/login")
async def login(
    client_key: str = Query(DEFAULT_CLIENT_KEY, description="Client key to associate with this auth session"),
    redirect_uri: str | None = Query(None, description="URI to redirect to after successful auth"),
):
    """
    Redirect to Oura OAuth authorization page.
    
    Args:
        client_key: The client key to associate tokens with after successful auth.
        redirect_uri: Optional URI to redirect to after successful authentication.
    """
    if not OURA_CLIENT_ID:
        raise HTTPException(status_code=500, detail="OURA_CLIENT_ID not configured")
    
    # Encode client_key and redirect_uri in state as JSON
    state_data = {"client_key": client_key}
    if redirect_uri:
        state_data["redirect_uri"] = redirect_uri
    state = quote(json.dumps(state_data))
    
    params = {
        "response_type": "code",
        "client_id": OURA_CLIENT_ID,
        "redirect_uri": OURA_REDIRECT_URI,
        "scope": "heartrate daily sleep",
        "state": state,
    }
    auth_url = f"{OURA_AUTH_URL}?{urlencode(params)}"
    return RedirectResponse(url=auth_url)


@router.get("/callback")
async def callback(
    code: str = Query(..., description="Authorization code from Oura"),
    state: str = Query("", description="State passed via OAuth containing client_key and optional redirect_uri"),
):
    """
    Handle OAuth callback from Oura.
    Exchange authorization code for access token.
    
    The client_key and optional redirect_uri are retrieved from the OAuth state parameter.
    """
    if not OURA_CLIENT_ID or not OURA_CLIENT_SECRET:
        raise HTTPException(status_code=500, detail="OAuth credentials not configured")
    
    # Parse state to extract client_key and redirect_uri
    client_key = DEFAULT_CLIENT_KEY
    redirect_uri = None
    if state:
        try:
            state_data = json.loads(unquote(state))
            client_key = state_data.get("client_key", DEFAULT_CLIENT_KEY)
            redirect_uri = state_data.get("redirect_uri")
        except (json.JSONDecodeError, TypeError):
            # Fallback: treat state as plain client_key for backwards compatibility
            client_key = state
    
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
        _token_store[client_key] = {
            "access_token": tokens.get("access_token"),
            "refresh_token": tokens.get("refresh_token"),
        }
        
        # Redirect to the original page if redirect_uri was provided
        if redirect_uri:
            return RedirectResponse(url=redirect_uri)
        
        return {"message": "Authentication successful", "client_key": client_key, "token_type": tokens.get("token_type")}


@router.get("/status")
async def auth_status(client_key: str = Query(DEFAULT_CLIENT_KEY, description="Client key to check auth status for")):
    """Check if user is authenticated."""
    return {"authenticated": bool(get_stored_token(client_key))}
