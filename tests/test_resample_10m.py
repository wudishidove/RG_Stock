"""Tests for 10-minute resampling."""

import pandas as pd
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.resample_10m import resample_to_10min


def make_session_minutes(date: str = "2025-01-02", tz: str = "America/New_York") -> pd.DataFrame:
    """Generate synthetic 1-minute bars for a full regular session."""
    idx = pd.date_range(f"{date} 09:31", f"{date} 16:00", freq="1min", tz=tz)
    n = len(idx)
    rng = np.random.default_rng(0)
    close = 100.0 + np.cumsum(rng.standard_normal(n) * 0.01)
    return pd.DataFrame({
        "datetime_et": idx,
        "open": close * 0.999,
        "high": close * 1.001,
        "low": close * 0.998,
        "close": close,
        "volume": rng.integers(100, 1000, n).astype(float),
    })


def test_bar_count():
    df = make_session_minutes()
    bars = resample_to_10min(df)
    # Session: 09:40 through 16:00 = 39 bars
    assert len(bars) == 39, f"Expected 39 bars, got {len(bars)}"


def test_first_last_bar():
    df = make_session_minutes()
    bars = resample_to_10min(df)
    times = pd.to_datetime(bars["datetime_et"])
    assert times.iloc[0].strftime("%H:%M") == "09:40"
    assert times.iloc[-1].strftime("%H:%M") == "16:00"


def test_close_is_last():
    """Last 1-min close in each 10-min window should equal the 10-min close."""
    df = make_session_minutes()
    bars = resample_to_10min(df)
    # The 10-min bar ending at 09:40 should use the close from the 09:40 minute bar
    df_indexed = df.set_index("datetime_et")
    expected_close_0940 = df_indexed.loc["2025-01-02 09:40:00-05:00", "close"]
    actual_close_0940 = bars.set_index("datetime_et").loc["2025-01-02 09:40:00-05:00", "close"]
    assert abs(actual_close_0940 - expected_close_0940) < 1e-8


def test_volume_is_sum():
    df = make_session_minutes()
    bars = resample_to_10min(df)
    # First 10-min bar covers minutes 09:31–09:40 (10 minute bars)
    df_indexed = df.set_index("datetime_et")
    t_start = pd.Timestamp("2025-01-02 09:31", tz="America/New_York")
    t_end   = pd.Timestamp("2025-01-02 09:40", tz="America/New_York")
    expected_vol = df_indexed.loc[t_start:t_end, "volume"].sum()
    actual_vol = bars.set_index("datetime_et").loc[pd.Timestamp("2025-01-02 09:40", tz="America/New_York"), "volume"]
    assert abs(actual_vol - expected_vol) < 1e-6
