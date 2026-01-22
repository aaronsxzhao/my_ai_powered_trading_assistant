#!/usr/bin/env python3
"""
Download futures data from Databento and store locally in DBN format.

Features:
- Checks existing local data before downloading
- Only downloads missing dates (no duplicates)
- Supports daily-split file format from Databento portal

Usage:
    # Download missing data for last 30 days
    python scripts/download_databento.py --symbols MES --days 30
    
    # Download specific date range
    python scripts/download_databento.py --symbols MES ES NQ --start 2025-01-01 --end 2025-12-31
    
    # Check what data is available locally
    python scripts/download_databento.py --check-local
    
    # Show what would be downloaded (dry run)
    python scripts/download_databento.py --symbols MES --days 30 --dry-run
"""

import argparse
import os
import sys
from datetime import datetime, timedelta, date
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import databento as db

# Default symbols to download (continuous front-month contracts)
DEFAULT_SYMBOLS = ["MES"]

# Schemas to download for each symbol
SCHEMAS = ["ohlcv-1d", "ohlcv-1h", "ohlcv-1m"]

# Dataset for CME Globex futures
DATASET = "GLBX.MDP3"

# Output directory
DATA_DIR = Path(__file__).parent.parent / "data" / "databento"


def get_continuous_symbol(base: str) -> str:
    """Convert base symbol to continuous front-month notation."""
    return f"{base}.c.0"


def get_local_dates(schema: str, data_dir: Path) -> set[date]:
    """
    Scan local files to find which dates we already have data for.
    
    Supports two formats:
    1. Custom: {SYMBOL}_{schema}_{start}_{end}.dbn.zst
    2. Databento daily: GLBX-*/glbx-mdp3-{YYYYMMDD}.{schema}.dbn.zst
    
    Returns:
        Set of date objects for which we have local data
    """
    available_dates = set()
    
    # Check custom format files
    for file_path in data_dir.glob(f"*_{schema}_*.dbn.zst"):
        try:
            name = file_path.stem.replace(".dbn", "")
            parts = name.split("_")
            if len(parts) >= 4:
                file_start = datetime.strptime(parts[-2], "%Y-%m-%d").date()
                file_end = datetime.strptime(parts[-1], "%Y-%m-%d").date()
                current = file_start
                while current <= file_end:
                    available_dates.add(current)
                    current += timedelta(days=1)
        except (ValueError, IndexError):
            continue
    
    # Check Databento daily format (in subfolders)
    for file_path in data_dir.glob(f"*/glbx-mdp3-*.{schema}.dbn.zst"):
        try:
            name = file_path.name
            date_part = name.split(".")[0].split("-")[-1]
            file_date = datetime.strptime(date_part, "%Y%m%d").date()
            available_dates.add(file_date)
        except (ValueError, IndexError):
            continue
    
    # Also check root level daily format files
    for file_path in data_dir.glob(f"glbx-mdp3-*.{schema}.dbn.zst"):
        try:
            name = file_path.name
            date_part = name.split(".")[0].split("-")[-1]
            file_date = datetime.strptime(date_part, "%Y%m%d").date()
            available_dates.add(file_date)
        except (ValueError, IndexError):
            continue
    
    return available_dates


def get_missing_dates(
    start: date, 
    end: date, 
    schema: str, 
    data_dir: Path
) -> list[date]:
    """
    Find dates that need to be downloaded.
    
    Returns:
        List of dates that don't have local data
    """
    local_dates = get_local_dates(schema, data_dir)
    
    missing = []
    current = start
    while current <= end:
        if current not in local_dates:
            missing.append(current)
        current += timedelta(days=1)
    
    return missing


def download_date_range(
    client: db.Historical,
    symbol: str,
    schema: str,
    start_date: date,
    end_date: date,
    output_dir: Path,
) -> Path | None:
    """
    Download data for a date range.
    
    Returns:
        Path to the downloaded file, or None if failed
    """
    continuous_symbol = get_continuous_symbol(symbol)
    
    # Create filename
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    filename = f"{symbol}_{schema}_{start_str}_{end_str}.dbn.zst"
    output_path = output_dir / filename
    
    try:
        print(f"    📥 Downloading {start_str} to {end_str}...")
        
        data = client.timeseries.get_range(
            dataset=DATASET,
            symbols=[continuous_symbol],
            schema=schema,
            start=start_str,
            end=(end_date + timedelta(days=1)).strftime("%Y-%m-%d"),  # End is exclusive
        )
        
        # Save to file
        data.to_file(str(output_path))
        
        size_kb = output_path.stat().st_size / 1024
        print(f"    ✅ Saved: {filename} ({size_kb:.1f} KB)")
        
        return output_path
        
    except Exception as e:
        print(f"    ❌ Error: {e}")
        return None


def check_local_data():
    """Display summary of locally available data."""
    print(f"\n📁 Local Data Directory: {DATA_DIR}")
    print("=" * 60)
    
    for schema in SCHEMAS:
        local_dates = get_local_dates(schema, DATA_DIR)
        
        if local_dates:
            sorted_dates = sorted(local_dates)
            print(f"\n📊 {schema}:")
            print(f"   Dates available: {len(sorted_dates)}")
            print(f"   Range: {sorted_dates[0]} to {sorted_dates[-1]}")
            
            # Find gaps
            gaps = []
            for i in range(1, len(sorted_dates)):
                expected = sorted_dates[i-1] + timedelta(days=1)
                # Skip weekends
                while expected.weekday() >= 5:
                    expected += timedelta(days=1)
                if sorted_dates[i] > expected:
                    gaps.append((sorted_dates[i-1], sorted_dates[i]))
            
            if gaps and len(gaps) <= 5:
                print(f"   Gaps: {len(gaps)}")
                for gap_start, gap_end in gaps:
                    print(f"      - {gap_start} to {gap_end}")
        else:
            print(f"\n📊 {schema}: No local data")


def download_missing(
    symbols: list[str],
    start: date,
    end: date,
    schemas: list[str],
    dry_run: bool = False,
):
    """Download only missing data."""
    
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📦 Databento Data Downloader")
    print(f"   Dataset: {DATASET}")
    print(f"   Symbols: {', '.join(symbols)}")
    print(f"   Schemas: {', '.join(schemas)}")
    print(f"   Requested range: {start} to {end}")
    print(f"   Output: {DATA_DIR}")
    
    if dry_run:
        print("\n🔍 DRY RUN - No data will be downloaded\n")
    
    # Check what's missing for each schema
    download_plan = {}
    
    for schema in schemas:
        local_dates = get_local_dates(schema, DATA_DIR)
        missing = get_missing_dates(start, end, schema, DATA_DIR)
        
        download_plan[schema] = {
            "local_count": len(local_dates),
            "missing": missing,
            "missing_count": len(missing),
        }
        
        print(f"\n📊 {schema}:")
        print(f"   Local dates: {len(local_dates)}")
        print(f"   Missing dates: {len(missing)}")
        
        if missing:
            # Group consecutive dates for display
            if len(missing) <= 10:
                print(f"   Missing: {', '.join(str(d) for d in missing)}")
            else:
                print(f"   Missing: {missing[0]} ... {missing[-1]}")
    
    # Calculate totals
    total_missing = sum(p["missing_count"] for p in download_plan.values())
    
    if total_missing == 0:
        print("\n✅ All requested data is already available locally!")
        return
    
    print(f"\n📥 Total dates to download: {total_missing}")
    
    if dry_run:
        print("\n🔍 Dry run complete. Run without --dry-run to download.")
        return
    
    # Initialize client
    print("\n🔌 Connecting to Databento...")
    client = db.Historical()
    
    downloaded = 0
    failed = 0
    
    for symbol in symbols:
        print(f"\n📊 {symbol}:")
        
        for schema in schemas:
            missing = download_plan[schema]["missing"]
            
            if not missing:
                print(f"   {schema}: ✅ Complete")
                continue
            
            print(f"   {schema}: Downloading {len(missing)} dates...")
            
            # Group consecutive dates into ranges for efficient downloading
            ranges = []
            range_start = missing[0]
            range_end = missing[0]
            
            for d in missing[1:]:
                # If consecutive (allowing for weekends)
                if (d - range_end).days <= 3:  # Allow 3 day gaps for weekends
                    range_end = d
                else:
                    ranges.append((range_start, range_end))
                    range_start = d
                    range_end = d
            ranges.append((range_start, range_end))
            
            for range_start, range_end in ranges:
                result = download_date_range(
                    client, symbol, schema, range_start, range_end, DATA_DIR
                )
                if result:
                    downloaded += 1
                else:
                    failed += 1
    
    print(f"\n{'='*60}")
    print(f"✅ Download complete: {downloaded} files")
    if failed:
        print(f"❌ Failed: {failed} files")


def main():
    parser = argparse.ArgumentParser(
        description="Download Databento futures data (only missing dates)"
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=DEFAULT_SYMBOLS,
        help=f"Symbols to download (default: {' '.join(DEFAULT_SYMBOLS)})",
    )
    parser.add_argument(
        "--schemas",
        nargs="+",
        default=SCHEMAS,
        help=f"Schemas to download (default: {' '.join(SCHEMAS)})",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="Number of days of history to download (from today)",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--check-local",
        action="store_true",
        help="Only check what data is available locally",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded without downloading",
    )
    
    args = parser.parse_args()
    
    # Just check local data
    if args.check_local:
        check_local_data()
        return
    
    # Determine date range
    if args.days:
        end = date.today()
        start = end - timedelta(days=args.days)
    elif args.start and args.end:
        start = datetime.strptime(args.start, "%Y-%m-%d").date()
        end = datetime.strptime(args.end, "%Y-%m-%d").date()
    else:
        # Default: last 30 days
        end = date.today()
        start = end - timedelta(days=30)
    
    download_missing(args.symbols, start, end, args.schemas, args.dry_run)


if __name__ == "__main__":
    main()
