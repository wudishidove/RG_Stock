"""
Fixed stock universe construction.

Approximation (L2): select top-N stocks by average dollar volume
during the pre-sample (tuning) period, then hold fixed for the test period.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def compute_dollar_volume(
    bars_10m: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> pd.Series:
    """
    Compute average daily dollar volume per ticker over [start_date, end_date].

    Parameters
    ----------
    bars_10m : long-format DataFrame with columns ticker, datetime_et, close, volume.
    start_date, end_date : YYYY-MM-DD strings.

    Returns
    -------
    pd.Series indexed by ticker, values = avg daily dollar volume.
    """
    df = bars_10m.copy()
    df["datetime_et"] = pd.to_datetime(df["datetime_et"])
    mask = (
        (df["datetime_et"].dt.date >= pd.Timestamp(start_date).date())
        & (df["datetime_et"].dt.date <= pd.Timestamp(end_date).date())
    )
    df = df[mask].copy()
    df["dollar_volume"] = df["close"] * df["volume"]
    df["session_date"] = df["datetime_et"].dt.date

    daily = df.groupby(["ticker", "session_date"])["dollar_volume"].sum()
    avg_dv = daily.groupby("ticker").mean()
    return avg_dv.sort_values(ascending=False)


def select_universe(
    dollar_volume: pd.Series,
    n_stocks: int = 500,
    exclude: Sequence[str] = (),
) -> list[str]:
    """Return top-n tickers by dollar volume, excluding any in `exclude`."""
    filtered = dollar_volume.drop(labels=list(exclude), errors="ignore")
    return filtered.head(n_stocks).index.tolist()


def load_or_build_universe(
    universe_path: Path,
    bars_10m: pd.DataFrame,
    tuning_start: str,
    tuning_end: str,
    n_stocks: int = 500,
) -> list[str]:
    """Load universe from file or build from scratch and save."""
    if universe_path.exists():
        logger.info("Loading universe from %s", universe_path)
        with open(universe_path) as f:
            return [line.strip() for line in f if line.strip()]

    logger.info("Building universe via dollar-volume ranking (%d stocks)", n_stocks)
    dv = compute_dollar_volume(bars_10m, tuning_start, tuning_end)
    universe = select_universe(dv, n_stocks=n_stocks)
    universe_path.parent.mkdir(parents=True, exist_ok=True)
    with open(universe_path, "w") as f:
        f.write("\n".join(universe) + "\n")
    logger.info("Universe saved: %s (%d tickers)", universe_path, len(universe))
    return universe
