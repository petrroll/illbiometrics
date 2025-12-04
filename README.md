# Biometrics API

A minimal FastAPI backend for accessing biometric data from wearable devices via OAuth.

Currently supported data sources:
- **Oura Ring** (implemented)
- **Garmin** (implemented via [garth](https://github.com/matin/garth))

## Setup

1. **Install dependencies** (requires [uv](https://github.com/astral-sh/uv)):
   ```bash
   uv sync
   ```

2. **Configure OAuth credentials**:
   
   **For Oura:**
   - Create an OAuth application at https://cloud.ouraring.com/oauth/applications
   - Copy `.env.example` to `.env` and fill in your credentials:
     ```bash
     cp .env.example .env
     ```

   **For Garmin:**
   - Authenticate using the garth CLI:
     ```bash
     uvx garth login
     ```
   - This saves credentials to `~/.garth` (configurable via `GARMIN_TOKEN_DIR` env var)

3. **Run the server**:
   ```bash
   uv run uvicorn app.main:app --reload
   ```

## Usage

1. Visit http://localhost:8000/docs to see the API documentation
2. Navigate to http://localhost:8000/auth/login to authenticate with Oura
3. For Garmin, ensure you've run `uvx garth login` before starting the server
4. After authentication, use the analytics endpoints to fetch biometric data

## API Endpoints

- `GET /` - Dashboard
- `GET /health` - Health check
- `GET /auth/login` - Redirect to OAuth login (Oura)
- `GET /auth/callback` - OAuth callback handler (Oura)
- `GET /auth/status` - Check authentication status (Oura)
- `GET /analytics` - Get combined sleep and heart rate analytics
- `GET /analytics/daily` - Get daily heart rate analytics
- `GET /raw/oura/heartrate` - Get raw heart rate data from Oura
- `GET /raw/oura/sleep` - Get raw sleep data from Oura

## Features

### Oura Ring
- Sleep analytics: duration, HR, HRV, and percentiles
- Heart rate analytics: daily and aggregate with percentiles
- OAuth authentication flow

### Garmin
- Sleep analytics: duration, deep/light/REM sleep stages, sleep score
- Stress analytics: daily stress levels with percentiles
- HRV data: weekly averages and nightly readings
- Username/password authentication via garth
