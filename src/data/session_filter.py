"""
Filter minute bars to the regular trading session: 09:30–16:00 ET.

The Massive API returns UTC timestamps. We convert to ET (America/New_York)
and keep bars whose close-time is in [09:31, 16:00] inclusive
(bar labeled 09:31 covers 09:30–09:31; bar labeled 16:00 covers 15:59–16:00).
"""

import pandas as pd

_ET_TZ = "America/New_York"
# Minute bar labels: 09:31 is the first bar (covers 09:30:00–09:31:00)
# 16:00 is the last bar (covers 15:59:00–16:00:00)
_SESSION_START = pd.Timedelta(hours=9, minutes=31)
_SESSION_END = pd.Timedelta(hours=16, minutes=0)


def filter_regular_session(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only minute bars within the regular trading session.

    Parameters
    ----------
    df : DataFrame with 'datetime_utc' column (UTC tz-aware datetime).

    Returns
    -------
    DataFrame with added 'datetime_et' column, filtered to [09:31, 16:00] ET,
    weekdays only.
    """
    df = df.copy()
    df["datetime_et"] = pd.to_datetime(df["datetime_utc"]).dt.tz_convert(_ET_TZ)

    # Weekdays only
    df = df[df["datetime_et"].dt.dayofweek < 5]

    # Time-of-day filter
    tod = df["datetime_et"].dt.hour * 3600 + df["datetime_et"].dt.minute * 60
    session_start_sec = 9 * 3600 + 31 * 60
    session_end_sec = 16 * 3600 + 0 * 60
    df = df[(tod >= session_start_sec) & (tod <= session_end_sec)]

    return df.reset_index(drop=True)


def add_session_date(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'session_date' column (ET date) for grouping by trading day."""
    df = df.copy()
    if "datetime_et" not in df.columns:
        df["datetime_et"] = pd.to_datetime(df["datetime_utc"]).dt.tz_convert(_ET_TZ)
    df["session_date"] = df["datetime_et"].dt.date
    return df
