#!/usr/bin/env python3
"""
Script to download heartrate and sleep data from Oura API for the last 5 years.
Downloads data in 30-day intervals and combines into two JSON files.

Requires the server to be running with user auth (just run-user) and authenticated.
"""

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import httpx


def fetch_data_in_chunks(
    base_url: str,
    endpoint: str,
    start_date: date,
    end_date: date,
    chunk_days: int = 30,
) -> list[dict]:
    """
    Fetch data from the API in chunks of chunk_days.
    
    Args:
        base_url: Base URL of the running server
        endpoint: API endpoint (heartrate or sleep)
        start_date: Start date for fetching
        end_date: End date for fetching
        chunk_days: Number of days per chunk (max 30 for Oura API)
    
    Returns:
        Combined list of all data items
    """
    all_data = []
    current_start = start_date
    
    while current_start < end_date:
        current_end = min(current_start + timedelta(days=chunk_days - 1), end_date)
        
        url = f"{base_url}/raw/oura/{endpoint}"
        params = {
            "start_date": str(current_start),
            "end_date": str(current_end),
        }
        
        print(f"Fetching {endpoint}: {current_start} to {current_end}...", end=" ", flush=True)
        
        try:
            response = httpx.get(url, params=params, timeout=60.0)
            response.raise_for_status()
            data = response.json()
            
            items = data.get("data", [])
            all_data.extend(items)
            print(f"got {len(items)} items")
            
        except httpx.HTTPStatusError as e:
            print(f"HTTP error: {e.response.status_code}")
            if e.response.status_code == 401:
                print("Not authenticated. Please run 'just run-user' and authenticate first.")
                sys.exit(1)
            raise
        except httpx.ConnectError:
            print("Connection error. Is the server running? Try 'just run-user' first.")
            sys.exit(1)
        
        current_start = current_end + timedelta(days=1)
    
    return all_data


def main():
    base_url = "http://localhost:8000"
    output_dir = Path(__file__).parent / "user_data"
    output_dir.mkdir(exist_ok=True)
    
    # Calculate date range: last 5 years
    end_date = date.today()
    start_date = end_date - timedelta(days=5 * 365)
    
    print(f"Downloading data from {start_date} to {end_date} (5 years)")
    print(f"Output directory: {output_dir}")
    print()
    
    # Fetch heartrate data
    print("=== Fetching Heart Rate Data ===")
    heartrate_data = fetch_data_in_chunks(
        base_url, "heartrate", start_date, end_date, chunk_days=30
    )
    
    heartrate_output = output_dir / "heartrate.json"
    with open(heartrate_output, "w") as f:
        json.dump({"data": heartrate_data}, f, indent=2)
    print(f"Saved {len(heartrate_data)} heartrate samples to {heartrate_output}")
    print()
    
    # Fetch sleep data
    print("=== Fetching Sleep Data ===")
    sleep_data = fetch_data_in_chunks(
        base_url, "sleep", start_date, end_date, chunk_days=30
    )
    
    sleep_output = output_dir / "sleep.json"
    with open(sleep_output, "w") as f:
        json.dump({"data": sleep_data}, f, indent=2)
    print(f"Saved {len(sleep_data)} sleep records to {sleep_output}")
    print()
    
    print("Done!")


if __name__ == "__main__":
    main()
