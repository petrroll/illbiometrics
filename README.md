# Biometrics API

A minimal FastAPI backend for accessing biometric data from wearable devices via OAuth.

Currently supported data sources:
- **Oura Ring** (implemented)
- **Garmin** (implemented via [garth](https://github.com/matin/garth) library)

Future planned integrations:
- Whoop
- and more...

## Setup

1. **Install dependencies** (requires [uv](https://github.com/astral-sh/uv)):
   ```bash
   uv sync
   ```

2. **Configure OAuth credentials** (for Oura):
   - Create an OAuth application at https://cloud.ouraring.com/oauth/applications
   - Copy `.env.example` to `.env` and fill in your credentials:
     ```bash
     cp .env.example .env
     ```

3. **Configure Garmin authentication**:
   - Garmin uses the garth library for authentication
   - Run `uvx garth login` to authenticate with Garmin Connect
   - Tokens are stored in `~/.garth` by default

4. **Run the server**:
   ```bash
   uv run uvicorn app.main:app --reload
   ```

## Usage

1. Visit http://localhost:8000/docs to see the API documentation
2. Navigate to http://localhost:8000/auth/login to authenticate with your data source
3. After authentication, use the analytics endpoints to fetch biometric data

## API Endpoints

### General
- `GET /` - Dashboard
- `GET /health` - Health check

### Oura Authentication
- `GET /auth/login` - Redirect to OAuth login
- `GET /auth/callback` - OAuth callback handler
- `GET /auth/status` - Check authentication status

### Oura Analytics
- `GET /analytics/sleep` - Get sleep analytics (duration, HR, HRV, percentiles)
- `GET /analytics/heartrate` - Get heart rate analytics (aggregate)
- `GET /analytics/heart-rate-daily` - Get daily heart rate analytics

### Oura Raw Data
- `GET /raw/oura/heartrate` - Get raw heart rate data from Oura
- `GET /raw/oura/sleep` - Get raw sleep data from Oura

### Garmin Analytics
- `GET /garmin/analytics/sleep` - Get Garmin sleep analytics (duration, deep/REM/light sleep, stress)
- `GET /garmin/analytics/hrv` - Get Garmin HRV analytics (sleep HRV with percentiles)
- `GET /garmin/analytics/stress` - Get Garmin stress analytics (with p20, p50, p80, p95 percentiles)

### Garmin Raw Data
- `GET /garmin/raw/sleep` - Get raw sleep data from Garmin
- `GET /garmin/raw/hrv` - Get raw HRV data from Garmin
- `GET /garmin/raw/stress` - Get raw stress data from Garmin
