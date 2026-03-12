"""
Assemble (T × N) close-price and log-return panels from per-ticker 10-min bars.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_close_panel(
    bars_long: pd.DataFrame,
    tickers: list[str],
    datetime_col: str = "datetime_et",
) -> pd.DataFrame:
    """
    Pivot long-format 10-min bars to a (T × N) close price panel.

    Parameters
    ----------
    bars_long : DataFrame with columns [datetime_et, ticker, close].
    tickers   : ordered list of tickers (universe).
    datetime_col : timestamp column name.

    Returns
    -------
    DataFrame indexed by datetime_et, columns = tickers.
    Missing ticker/time combinations are NaN.
    """
    pivot = bars_long.pivot_table(
        index=datetime_col,
        columns="ticker",
        values="close",
        aggfunc="last",
    )
    # Align to universe order; add missing tickers as NaN
    pivot = pivot.reindex(columns=tickers)
    pivot.index = pd.to_datetime(pivot.index)
    pivot.sort_index(inplace=True)
    return pivot


def build_return_panel(close_panel: pd.DataFrame) -> pd.DataFrame:
    """Compute log returns from close panel: r_t = log(C_t / C_{t-1})."""
    ret = np.log(close_panel).diff()
    # First bar of each session should not carry overnight return;
    # handled downstream by session boundary masking.
    return ret


def build_future_return_panel(
    close_panel: pd.DataFrame,
    horizon_steps: int,
    session_dates: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Compute h-step-ahead log return: log(C_{t+h} / C_t).

    For EOD horizon (horizon_steps == -1), compute log(C_{eod} / C_t)
    where C_eod is the last bar of the same session.

    Parameters
    ----------
    close_panel   : (T × N) close price panel.
    horizon_steps : integer number of 10-min bars (use -1 for EOD).
    session_dates : pd.Series with same index as close_panel, mapping each
                    bar to its trading session date. Required for EOD.

    Returns
    -------
    DataFrame of same shape: future_ret[t] = log(C_{t+h} / C_t).
    Rows where target is unavailable (end of day/sample) are NaN.
    """
    log_close = np.log(close_panel)

    if horizon_steps >= 1:
        future_log = log_close.shift(-horizon_steps)
        future_ret = future_log - log_close
        # Zero out cross-session returns: if bar t and t+h are in different sessions,
        # the return spans overnight — mask as NaN.
        if session_dates is not None:
            future_session = session_dates.shift(-horizon_steps)
            cross_session = session_dates != future_session
            future_ret[cross_session] = np.nan
        return future_ret

    elif horizon_steps == -1:
        # EOD: log(C_{16:00} / C_t) for each bar t in the same session
        if session_dates is None:
            raise ValueError("session_dates required for EOD horizon")
        result = pd.DataFrame(np.nan, index=close_panel.index, columns=close_panel.columns)
        for date, grp_idx in close_panel.index.to_series().groupby(session_dates.values).groups.items():
            grp_idx = sorted(grp_idx)
            eod_close = log_close.loc[grp_idx[-1]]   # last bar of session
            for t in grp_idx:
                result.loc[t] = eod_close - log_close.loc[t]
        return result
    else:
        raise ValueError(f"Invalid horizon_steps: {horizon_steps}")
