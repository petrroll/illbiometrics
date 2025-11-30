# Biometrics API

A minimal FastAPI backend for accessing biometric data from wearable devices via OAuth.

Currently supported data sources:
- **Oura Ring** (implemented)

Future planned integrations:
- Garmin
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

3. **Run the server**:
   ```bash
   uv run uvicorn app.main:app --reload
   ```

## Usage

1. Visit http://localhost:8000/docs to see the API documentation
2. Navigate to http://localhost:8000/auth/login to authenticate with your data source
3. After authentication, use the analytics endpoints to fetch biometric data

## API Endpoints

- `GET /` - Dashboard
- `GET /health` - Health check
- `GET /auth/login` - Redirect to OAuth login
- `GET /auth/callback` - OAuth callback handler
- `GET /auth/status` - Check authentication status
- `GET /analytics/sleep` - Get sleep analytics
- `GET /analytics/heartrate` - Get heart rate analytics
- `GET /raw/oura/heartrate` - Get raw heart rate data from Oura
- `GET /raw/oura/sleep` - Get raw sleep data from Oura
