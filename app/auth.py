from urllib.parse import urlencode, quote, unquote
import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor
import httpx
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

import garth
import garth.sso
from garth.exc import GarthException, GarthHTTPError

# Thread pool for running blocking garth operations
_garmin_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="garmin_auth")

from app.config import (
    OURA_AUTH_URL,
    OURA_CLIENT_ID,
    OURA_CLIENT_SECRET,
    OURA_REDIRECT_URI,
    OURA_TOKEN_URL,
    GARMIN_TOKEN_DIR,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])

# In-memory token storage (replace with proper storage in production)
# Keyed by client_key -> {"access_token": ..., "refresh_token": ...}
_token_store: dict[str, dict[str, str]] = {}

# Garmin token storage (in-memory, stores serialized garth tokens)
# Keyed by client_key -> garth.client.dumps() base64 string
_garmin_token_store: dict[str, str] = {}

# Pending MFA sessions for Garmin (client_key -> client_state dict)
_garmin_mfa_sessions: dict[str, dict] = {}

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
    """Check if user is authenticated with Oura."""
    return {"authenticated": bool(get_stored_token(client_key))}


# =============================================================================
# Garmin Authentication Endpoints
# =============================================================================

def get_garmin_client(client_key: str = DEFAULT_CLIENT_KEY) -> garth.Client | None:
    """Get a garth client for the given client key.
    
    Returns:
        A garth.Client if tokens exist for the client_key, None otherwise.
    """
    token_string = _garmin_token_store.get(client_key)
    if not token_string:
        return None
    
    try:
        client = garth.Client()
        client.loads(token_string)
        return client
    except Exception as e:
        logger.warning(f"Failed to load Garmin tokens for {client_key}: {e}")
        return None


def is_garmin_authenticated(client_key: str = DEFAULT_CLIENT_KEY) -> bool:
    """Check if there are valid Garmin tokens for the client key."""
    client = get_garmin_client(client_key)
    if not client:
        return False
    
    try:
        # Check if we have valid tokens
        if client.oauth2_token and not client.oauth2_token.expired:
            return True
        # Try to refresh if expired
        if client.oauth1_token:
            client.refresh_oauth2()
            # Update stored tokens
            _garmin_token_store[client_key] = client.dumps()
            return True
    except Exception as e:
        logger.debug(f"Garmin auth check failed for {client_key}: {e}")
    
    return False


class GarminLoginRequest(BaseModel):
    """Request body for Garmin login."""
    email: str
    password: str
    client_key: str = DEFAULT_CLIENT_KEY


class GarminMFARequest(BaseModel):
    """Request body for Garmin MFA verification."""
    mfa_code: str
    client_key: str = DEFAULT_CLIENT_KEY


def _do_garmin_login(email: str, password: str) -> tuple:
    """
    Perform blocking Garmin login. Must be run in a thread pool.
    
    Returns:
        Tuple of (result_type, data) where:
        - ("success", client) for successful login
        - ("needs_mfa", client_state) if MFA is required
        - ("error", exception) on failure
    """
    try:
        client = garth.Client()
        result = client.login(email, password, return_on_mfa=True)
        
        if isinstance(result, tuple) and result[0] == "needs_mfa":
            return ("needs_mfa", result[1])
        
        return ("success", client)
    except Exception as e:
        return ("error", e)


@router.post("/garmin/login")
async def garmin_login(request: GarminLoginRequest):
    """
    Authenticate with Garmin Connect using email and password.
    
    If MFA is required, returns {"needs_mfa": true} and the client should
    call /auth/garmin/mfa with the MFA code.
    
    Returns:
        {"success": true} on successful login
        {"needs_mfa": true} if MFA is required
    """
    loop = asyncio.get_event_loop()
    
    # Run blocking garth login in thread pool to avoid blocking the event loop
    result_type, result_data = await loop.run_in_executor(
        _garmin_executor,
        _do_garmin_login,
        request.email,
        request.password
    )
    
    if result_type == "error":
        e = result_data
        if isinstance(e, GarthHTTPError):
            logger.warning(f"Garmin login failed: {e}")
            raise HTTPException(status_code=401, detail="Invalid email or password")
        elif isinstance(e, GarthException):
            logger.warning(f"Garmin login error: {e}")
            raise HTTPException(status_code=400, detail=str(e))
        else:
            logger.error(f"Unexpected Garmin login error: {e}")
            raise HTTPException(status_code=500, detail="Login failed unexpectedly")
    
    if result_type == "needs_mfa":
        # Store the client state for MFA completion
        _garmin_mfa_sessions[request.client_key] = result_data
        return {"needs_mfa": True, "message": "MFA code required"}
    
    # Login successful (no MFA needed)
    client = result_data
    _garmin_token_store[request.client_key] = client.dumps()
    
    # Get username for confirmation
    try:
        username = client.username
    except Exception:
        username = "unknown"
    
    logger.info(f"Garmin login successful for user: {username}")
    return {"success": True, "username": username}


def _do_garmin_mfa(client_state: dict, mfa_code: str) -> tuple:
    """
    Perform blocking Garmin MFA verification. Must be run in a thread pool.
    
    Returns:
        Tuple of (result_type, data) where:
        - ("success", client) for successful MFA
        - ("error", exception) on failure
    """
    try:
        oauth1, oauth2 = garth.sso.resume_login(client_state, mfa_code)
        client = garth.Client()
        client.configure(oauth1_token=oauth1, oauth2_token=oauth2)
        return ("success", client)
    except Exception as e:
        return ("error", e)


@router.post("/garmin/mfa")
async def garmin_mfa(request: GarminMFARequest):
    """
    Complete Garmin authentication with MFA code.
    
    Must be called after /auth/garmin/login returns {"needs_mfa": true}.
    
    Returns:
        {"success": true} on successful MFA verification
    """
    client_state = _garmin_mfa_sessions.get(request.client_key)
    if not client_state:
        raise HTTPException(
            status_code=400,
            detail="No pending MFA session. Please start login again."
        )
    
    loop = asyncio.get_event_loop()
    
    # Run blocking garth MFA in thread pool
    result_type, result_data = await loop.run_in_executor(
        _garmin_executor,
        _do_garmin_mfa,
        client_state,
        request.mfa_code
    )
    
    if result_type == "error":
        e = result_data
        if isinstance(e, GarthException):
            logger.warning(f"Garmin MFA failed: {e}")
            raise HTTPException(status_code=401, detail="Invalid MFA code")
        else:
            logger.error(f"Unexpected Garmin MFA error: {e}")
            raise HTTPException(status_code=500, detail="MFA verification failed")
    
    # Success
    client = result_data
    _garmin_token_store[request.client_key] = client.dumps()
    
    # Clean up MFA session
    del _garmin_mfa_sessions[request.client_key]
    
    # Get username for confirmation
    try:
        username = client.username
    except Exception:
        username = "unknown"
    
    logger.info(f"Garmin MFA successful for user: {username}")
    return {"success": True, "username": username}


@router.get("/garmin/status")
async def garmin_status(client_key: str = Query(DEFAULT_CLIENT_KEY, description="Client key to check Garmin auth status for")):
    """Check if user is authenticated with Garmin."""
    authenticated = is_garmin_authenticated(client_key)
    
    result = {"authenticated": authenticated}
    
    # If authenticated, try to get username
    if authenticated:
        client = get_garmin_client(client_key)
        if client:
            try:
                result["username"] = client.username
            except Exception:
                pass
    
    return result


@router.post("/garmin/logout")
async def garmin_logout(client_key: str = Query(DEFAULT_CLIENT_KEY, description="Client key to logout")):
    """Logout from Garmin (clear stored tokens)."""
    if client_key in _garmin_token_store:
        del _garmin_token_store[client_key]
    if client_key in _garmin_mfa_sessions:
        del _garmin_mfa_sessions[client_key]
    return {"success": True}
