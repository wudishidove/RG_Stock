"""
Preprocessing: within-session forward-fill, missing masks, return validation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def within_session_ffill(
    close_panel: pd.DataFrame,
    session_dates: pd.Series,
    max_ffill_bars: int = 3,
) -> pd.DataFrame:
    """
    Forward-fill NaN close prices within each trading session.

    Parameters
    ----------
    close_panel   : (T × N) close price panel (NaN = missing bar).
    session_dates : pd.Series same index as close_panel, session date per bar.
    max_ffill_bars: maximum consecutive bars to fill (L3 parameter).

    Returns
    -------
    Forward-filled close panel; gaps > max_ffill_bars remain NaN.
    """
    filled = close_panel.copy()

    for _, grp_idx in close_panel.index.to_series().groupby(session_dates.values).groups.items():
        grp_idx = sorted(grp_idx)
        sub = filled.loc[grp_idx]
        # Forward fill with limit
        sub_filled = sub.ffill(axis=0, limit=max_ffill_bars)
        filled.loc[grp_idx] = sub_filled

    return filled


def build_missing_mask(
    close_panel: pd.DataFrame,
    session_dates: pd.Series,
    max_missing_fraction: float = 0.20,
) -> pd.DataFrame:
    """
    Build boolean validity mask (True = valid) for each stock per estimation window.

    Stocks with fraction of missing bars > max_missing_fraction are excluded
    entirely from that window's computation (L3 parameter).

    Parameters
    ----------
    close_panel           : (T × N) panel (NaN where bar is missing).
    session_dates         : session date per bar.
    max_missing_fraction  : exclusion threshold.

    Returns
    -------
    Boolean DataFrame same shape as close_panel; True = usable bar.
    """
    is_missing = close_panel.isna()
    # For each stock, compute overall missing fraction
    miss_frac = is_missing.mean(axis=0)
    valid_tickers = miss_frac[miss_frac <= max_missing_fraction].index
    mask = ~is_missing
    # Exclude tickers that exceed threshold
    excluded = miss_frac[miss_frac > max_missing_fraction].index
    mask[excluded] = False
    return mask


def session_boundary_mask(
    return_panel: pd.DataFrame,
    session_dates: pd.Series,
) -> pd.DataFrame:
    """
    Create mask that is False for the first bar of each session
    (where log return spans overnight gap).
    """
    mask = pd.DataFrame(True, index=return_panel.index, columns=return_panel.columns)
    # First bar of each session
    first_bars = []
    prev_date = None
    for t in return_panel.index:
        d = session_dates.loc[t]
        if d != prev_date:
            first_bars.append(t)
            prev_date = d
    mask.loc[first_bars] = False
    return mask
