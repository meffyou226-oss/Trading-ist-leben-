"""
data_loader.py
==============
Load, validate, and merge XAUUSD M5 CSV files from Dukascopy.
Provides data quality report and gap detection.
"""

import os
import glob
import logging
from typing import Tuple, Optional, List, Dict
import numpy as np
import pandas as pd

from pipeline.config import DATA_DIR, DATA_PATTERN, INSTRUMENT, TIMEFRAME

logger = logging.getLogger(__name__)


def load_data(
    data_dir: str = DATA_DIR,
    pattern: str = DATA_PATTERN,
) -> pd.DataFrame:
    """
    Load and merge all XAUUSD M5 CSV files from the data directory.

    Parameters
    ----------
    data_dir : str
        Directory containing the CSV files.
    pattern : str
        Glob pattern for CSV files.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame sorted by timestamp, with columns:
        timestamp, date, time, open, high, low, close, volume
    """
    files = sorted(glob.glob(os.path.join(data_dir, pattern)))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern} in {data_dir}")

    logger.info(f"Found {len(files)} CSV files to load")

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        dfs.append(df)
        logger.debug(f"Loaded {os.path.basename(f)}: {len(df)} rows")

    data = pd.concat(dfs, ignore_index=True)

    # Ensure correct types
    data["timestamp"] = data["timestamp"].astype(np.int64)
    data["open"] = data["open"].astype(np.float64)
    data["high"] = data["high"].astype(np.float64)
    data["low"] = data["low"].astype(np.float64)
    data["close"] = data["close"].astype(np.float64)
    data["volume"] = data["volume"].astype(np.float64)

    # Sort by timestamp and remove duplicates
    data = data.sort_values("timestamp").reset_index(drop=True)
    data = data.drop_duplicates(subset=["timestamp"], keep="first")

    logger.info(f"Total records after merge: {len(data)}")
    return data


def validate_data(data: pd.DataFrame) -> Dict:
    """
    Validate data quality and return a report dictionary.

    Checks:
    - OHLC consistency (high >= low, high >= open/close, low <= open/close)
    - No NaN values in critical columns
    - Monotonic timestamps
    - Price range sanity
    - Volume non-negativity
    """
    report = {
        "instrument": INSTRUMENT,
        "timeframe": TIMEFRAME,
        "total_records": len(data),
        "issues": [],
    }

    # Time range
    ts_min = pd.Timestamp(data["timestamp"].min(), unit="ms")
    ts_max = pd.Timestamp(data["timestamp"].max(), unit="ms")
    report["start_date"] = ts_min.isoformat()
    report["end_date"] = ts_max.isoformat()
    report["duration_days"] = (ts_max - ts_min).days

    # Price range
    report["close_min"] = float(data["close"].min())
    report["close_max"] = float(data["close"].max())
    report["close_mean"] = float(data["close"].mean())

    # OHLC consistency
    ohlc_issues = (
        (data["high"] < data["low"]) |
        (data["high"] < data["open"]) |
        (data["high"] < data["close"]) |
        (data["low"] > data["open"]) |
        (data["low"] > data["close"])
    )
    n_ohlc = ohlc_issues.sum()
    if n_ohlc > 0:
        report["issues"].append(f"OHLC inconsistencies: {n_ohlc} rows")

    # NaN check
    nan_counts = data[["open", "high", "low", "close", "volume"]].isna().sum()
    if nan_counts.any():
        report["issues"].append(f"NaN values: {nan_counts.to_dict()}")

    # Monotonic timestamps
    ts_diff = np.diff(data["timestamp"].values)
    if (ts_diff <= 0).any():
        report["issues"].append("Non-monotonic timestamps detected")

    # Volume check
    neg_vol = (data["volume"] < 0).sum()
    if neg_vol > 0:
        report["issues"].append(f"Negative volume: {neg_vol} rows")

    # Zero volume
    zero_vol = (data["volume"] == 0).sum()
    report["zero_volume_bars"] = int(zero_vol)

    return report


def detect_gaps(data: pd.DataFrame, gap_threshold_ms: int = 3600000) -> pd.DataFrame:
    """
    Detect data gaps (weekends, holidays, outages).

    Parameters
    ----------
    data : pd.DataFrame
        The OHLC data.
    gap_threshold_ms : int
        Minimum gap size in milliseconds to report (default 1 hour).

    Returns
    -------
    pd.DataFrame
        DataFrame with gap information: start, end, duration_hours
    """
    timestamps = data["timestamp"].values
    diffs = np.diff(timestamps)

    gap_indices = np.where(diffs > gap_threshold_ms)[0]

    gaps = []
    for idx in gap_indices:
        gap_start = pd.Timestamp(timestamps[idx], unit="ms")
        gap_end = pd.Timestamp(timestamps[idx + 1], unit="ms")
        duration_hours = diffs[idx] / 3600000
        gaps.append({
            "start": gap_start.isoformat(),
            "end": gap_end.isoformat(),
            "duration_hours": round(duration_hours, 1),
        })

    gaps_df = pd.DataFrame(gaps)
    logger.info(f"Detected {len(gaps_df)} data gaps > {gap_threshold_ms/3600000:.0f}h")
    return gaps_df


def add_time_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add basic time-derived columns needed for feature engineering
    and session detection.

    Adds:
    - datetime: pandas Timestamp
    - hour: hour of day (UTC)
    - dayofweek: 0=Monday, 6=Sunday
    """
    data = data.copy()
    data["datetime"] = pd.to_datetime(data["timestamp"], unit="ms", utc=True)
    data["hour"] = data["datetime"].dt.hour.astype(np.int8)
    data["dayofweek"] = data["datetime"].dt.dayofweek.astype(np.int8)
    return data


def get_data_summary(data: pd.DataFrame) -> str:
    """Generate a human-readable data summary string."""
    report = validate_data(data)
    gaps = detect_gaps(data)

    lines = [
        "=" * 60,
        f"  DATA SUMMARY: {report['instrument']} {report['timeframe']}",
        "=" * 60,
        f"  Period:        {report['start_date'][:10]} to {report['end_date'][:10]}",
        f"  Duration:      {report['duration_days']} days",
        f"  Total bars:    {report['total_records']:,}",
        f"  Close min:     {report['close_min']:.3f}",
        f"  Close max:     {report['close_max']:.3f}",
        f"  Close mean:    {report['close_mean']:.3f}",
        f"  Data gaps:     {len(gaps)} detected",
        f"  Zero-vol bars: {report.get('zero_volume_bars', 0):,}",
    ]

    if report["issues"]:
        lines.append(f"  Issues:")
        for issue in report["issues"]:
            lines.append(f"    - {issue}")
    else:
        lines.append("  Issues:        None")

    lines.append("=" * 60)
    return "\n".join(lines)
