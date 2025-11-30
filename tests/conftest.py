"""Pytest configuration and shared fixtures for tests."""

import json
import os
from pathlib import Path

import pytest

# Ensure sandbox mode for all tests
os.environ["DATA_SOURCE"] = "sandbox"

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def sleep_fixture_data() -> dict:
    """Load the sleep fixture data."""
    with open(FIXTURES_DIR / "usercollection_sleep_2025-10-31_2025-11-29.json") as f:
        return json.load(f)


@pytest.fixture
def sleep_data_list(sleep_fixture_data: dict) -> list[dict]:
    """Get the list of sleep records from fixture."""
    return sleep_fixture_data["data"]


@pytest.fixture
def heartrate_fixture_data() -> dict:
    """Load the heart rate fixture data."""
    with open(FIXTURES_DIR / "usercollection_heartrate_2025-10-31_2025-11-29.json") as f:
        return json.load(f)


@pytest.fixture
def heartrate_data_list(heartrate_fixture_data: dict) -> list[dict]:
    """Get the list of heart rate records from fixture."""
    return heartrate_fixture_data["data"]
