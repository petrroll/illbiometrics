# Oura Biometrics API

A minimal FastAPI backend for accessing Oura ring biometric data via OAuth.

## Setup

1. **Install dependencies** (requires [uv](https://github.com/astral-sh/uv)):
   ```bash
   uv sync
   ```

2. **Configure OAuth credentials**:
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
2. Navigate to http://localhost:8000/auth/login to authenticate with Oura
3. After authentication, use http://localhost:8000/heartrate to fetch heart rate data

## API Endpoints

- `GET /` - Health check
- `GET /auth/login` - Redirect to Oura OAuth login
- `GET /auth/callback` - OAuth callback handler
- `GET /auth/status` - Check authentication status
- `GET /heartrate` - Get heart rate data (supports `start_date` and `end_date` query params)
