import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Oura OAuth Configuration
OURA_CLIENT_ID = os.getenv("OURA_CLIENT_ID", "")
OURA_CLIENT_SECRET = os.getenv("OURA_CLIENT_SECRET", "")
OURA_REDIRECT_URI = os.getenv("OURA_REDIRECT_URI", "http://localhost:8000/auth/callback")

# Oura API endpoints
OURA_AUTH_URL = "https://cloud.ouraring.com/oauth/authorize"
OURA_TOKEN_URL = "https://api.ouraring.com/oauth/token"
OURA_API_BASE = "https://api.ouraring.com/v2"
OURA_SANDBOX_BASE = "https://api.ouraring.com/v2/sandbox"

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
SANDBOX_CACHE_DIR = PROJECT_ROOT / "tests" / "fixtures"
