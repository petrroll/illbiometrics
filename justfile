# Default recipe
default: build

# Prepare environment and run type checks
build:
    uv sync
    uv run pyright

# Run with sandbox data (no auth needed)
run:
    DATA_SOURCE=sandbox uv run uvicorn app.main:app --reload

# Run with real user data (requires auth)
run-user:
    echo "Login at https://${CODESPACE_NAME}-8000.app.github.dev/auth/login"
    DATA_SOURCE=user uv run uvicorn app.main:app --reload --port 8000

# Run tests
test:
    DATA_SOURCE=sandbox uv run pytest tests/ -v

# Clean all downloaded sandbox data (preserves .gitkeep)
clean:
    find tests/fixtures -name '*.json' -type f -delete

# Download last 5 years of user data in 30-day chunks (must run just run-user first & auth)
get-playground-user-data:
    echo "Must run just run-user first & auth."
    uv run python playground/get_playground_user_data.py