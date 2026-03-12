"""
Massive (Polygon.io-compatible) minute aggregate fetcher.

Per-ticker range requests with pagination → per-ticker parquet cache.
Rate-limited to 5 calls/minute (Basic tier).
"""

import os
import time
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://api.polygon.io"
_DEFAULT_RATE_DELAY = 12.0   # seconds between API calls (5/min limit)


class MassiveFetcher:
    """Fetch and cache per-ticker minute OHLCV bars from Massive/Polygon API."""

    def __init__(
        self,
        api_key: str,
        raw_dir: str | Path,
        base_url: str = _DEFAULT_BASE_URL,
        rate_delay: float = _DEFAULT_RATE_DELAY,
    ) -> None:
        self.api_key = api_key
        self.raw_dir = Path(raw_dir)
        self.base_url = base_url.rstrip("/")
        self.rate_delay = rate_delay
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ensure_ticker(self, ticker: str, from_date: str, to_date: str) -> Path:
        """Ensure local cache covers [from_date, to_date] for ticker.

        Only fetches missing date ranges. Appends to existing parquet file.
        Returns path to the parquet cache file.
        """
        cache_path = self.raw_dir / f"{ticker}.parquet"
        missing_ranges = self._missing_ranges(cache_path, from_date, to_date)
        if not missing_ranges:
            logger.info("%s: cache complete", ticker)
            return cache_path

        all_new: list[pd.DataFrame] = []
        for rng_start, rng_end in missing_ranges:
            logger.info("%s: fetching %s → %s", ticker, rng_start, rng_end)
            bars = self._fetch_all_pages(ticker, rng_start, rng_end)
            if bars is not None and not bars.empty:
                all_new.append(bars)
            time.sleep(self.rate_delay)

        if all_new:
            new_df = pd.concat(all_new, ignore_index=True)
            self._append_cache(cache_path, new_df)

        return cache_path

    def load_ticker(self, ticker: str) -> Optional[pd.DataFrame]:
        """Load cached parquet for a ticker; returns None if missing."""
        cache_path = self.raw_dir / f"{ticker}.parquet"
        if not cache_path.exists():
            return None
        return pd.read_parquet(cache_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_all_pages(self, ticker: str, from_date: str, to_date: str) -> Optional[pd.DataFrame]:
        """Fetch all pages of 1-minute bars for ticker in [from_date, to_date]."""
        url = (
            f"{self.base_url}/v2/aggs/ticker/{ticker}/range/1/minute"
            f"/{from_date}/{to_date}"
        )
        params = {
            "adjusted": "true",
            "sort": "asc",
            "limit": 50000,
            "apiKey": self.api_key,
        }
        frames: list[pd.DataFrame] = []

        while url:
            try:
                resp = requests.get(url, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:
                logger.error("%s: request failed — %s", ticker, exc)
                return None

            results = data.get("results", [])
            if results:
                frames.append(pd.DataFrame(results))

            next_url = data.get("next_url")
            if next_url:
                url = next_url
                params = {"apiKey": self.api_key}  # next_url already has other params
                time.sleep(self.rate_delay)
            else:
                url = None

        if not frames:
            logger.warning("%s: no data returned for %s → %s", ticker, from_date, to_date)
            return None

        df = pd.concat(frames, ignore_index=True)
        df = self._normalise_columns(df)
        return df

    @staticmethod
    def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names from Polygon response."""
        rename = {
            "t": "timestamp_ms",
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
            "vw": "vwap",
            "n": "n_trades",
        }
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
        # Convert millisecond timestamp to UTC datetime
        if "timestamp_ms" in df.columns:
            df["datetime_utc"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
        df = df.drop_duplicates(subset=["timestamp_ms"]).sort_values("timestamp_ms")
        return df

    def _missing_ranges(
        self, cache_path: Path, from_date: str, to_date: str
    ) -> list[tuple[str, str]]:
        """Determine date ranges not yet cached."""
        if not cache_path.exists():
            return [(from_date, to_date)]

        try:
            cached = pd.read_parquet(cache_path, columns=["datetime_utc"])
        except Exception:
            return [(from_date, to_date)]

        if cached.empty:
            return [(from_date, to_date)]

        cached_dates = pd.to_datetime(cached["datetime_utc"]).dt.date
        cached_min = str(cached_dates.min())
        cached_max = str(cached_dates.max())

        gaps = []
        if from_date < cached_min:
            day_before = str(
                (pd.Timestamp(cached_min) - pd.Timedelta(days=1)).date()
            )
            gaps.append((from_date, min(day_before, to_date)))
        if to_date > cached_max:
            day_after = str(
                (pd.Timestamp(cached_max) + pd.Timedelta(days=1)).date()
            )
            gaps.append((max(day_after, from_date), to_date))
        return gaps

    @staticmethod
    def _append_cache(cache_path: Path, new_df: pd.DataFrame) -> None:
        """Merge new_df into existing parquet cache (deduplicate by timestamp_ms)."""
        if cache_path.exists():
            try:
                existing = pd.read_parquet(cache_path)
                combined = pd.concat([existing, new_df], ignore_index=True)
                combined = combined.drop_duplicates(subset=["timestamp_ms"]).sort_values("timestamp_ms")
            except Exception:
                combined = new_df
        else:
            combined = new_df
        combined.to_parquet(cache_path, index=False)
        logger.info("Cache updated: %s (%d rows)", cache_path, len(combined))
