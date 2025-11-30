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
    DATA_SOURCE=user uv run uvicorn app.main:app --reload

# Run tests
test:
    DATA_SOURCE=sandbox uv run pytest tests/ -v
