"""
Resample 1-minute OHLCV bars to 10-minute bars.

Resampling rules:
  open  : first
  high  : max
  low   : min
  close : last
  volume: sum

Bar labeling: 10-minute bars are labeled by their CLOSE time (end-time).
- First 10-min bar: label 09:40 (covers 09:30–09:40, i.e. minute bars 09:31–09:40)
- Last  10-min bar: label 16:00 (covers 15:50–16:00, i.e. minute bars 15:51–16:00)
"""

import pandas as pd


def resample_to_10min(df: pd.DataFrame, datetime_col: str = "datetime_et") -> pd.DataFrame:
    """
    Resample a regular-session 1-minute bar DataFrame to 10-minute bars.

    Parameters
    ----------
    df : DataFrame with datetime_col (tz-aware ET), open/high/low/close/volume.
    datetime_col : name of the datetime column.

    Returns
    -------
    DataFrame with 10-minute bars, indexed by bar close-time (ET tz-aware).
    """
    df = df.set_index(datetime_col).sort_index()

    agg_dict: dict[str, str] = {}
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            agg_dict[col] = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}[col]

    # closed='right', label='right' → bar labeled by close time
    bars_10m = df[list(agg_dict.keys())].resample("10min", closed="right", label="right").agg(agg_dict)

    # Drop rows outside session (resampler may create edge bins)
    bars_10m = bars_10m.dropna(subset=["close"])

    # Keep only session bars: 09:40 through 16:00
    session_start = pd.Timedelta(hours=9, minutes=40)
    session_end = pd.Timedelta(hours=16, minutes=0)
    tod = bars_10m.index.hour * 3600 + bars_10m.index.minute * 60
    session_start_sec = 9 * 3600 + 40 * 60
    session_end_sec = 16 * 3600 + 0 * 60
    bars_10m = bars_10m[(tod >= session_start_sec) & (tod <= session_end_sec)]

    return bars_10m.reset_index().rename(columns={datetime_col: "datetime_et"})


def resample_ticker_10min(
    df: pd.DataFrame,
    ticker: str,
    datetime_col: str = "datetime_et",
) -> pd.DataFrame:
    """Resample a single ticker's session-filtered 1-min bars to 10-min bars."""
    result = resample_to_10min(df, datetime_col=datetime_col)
    result["ticker"] = ticker
    return result
