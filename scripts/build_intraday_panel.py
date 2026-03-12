"""
Script: Session filter + 10-min resample + panel assembly.

Reads per-ticker parquet caches, applies session filter,
resamples to 10-min bars, assembles (T, N) close panel.
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.session_filter import filter_regular_session, add_session_date
from src.data.resample_10m import resample_ticker_10min
from src.data.panel_builder import build_close_panel, build_return_panel, build_future_return_panel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build intraday 10-min panel")
    parser.add_argument("--raw-dir", default="data/raw/tickers")
    parser.add_argument("--output-dir", default="data/interim")
    parser.add_argument("--universe-file", default="data/raw/universe.txt")
    parser.add_argument("--from-date", default="2024-09-01")
    parser.add_argument("--to-date", default="2025-12-31")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    universe_file = Path(args.universe_file)
    if not universe_file.exists():
        logger.error("Universe file not found: %s", universe_file)
        sys.exit(1)

    tickers = [line.strip() for line in universe_file.read_text().splitlines() if line.strip()]
    logger.info("Building panel for %d tickers", len(tickers))

    frames = []
    for ticker in tickers:
        cache_path = raw_dir / f"{ticker}.parquet"
        if not cache_path.exists():
            logger.warning("Missing cache: %s", ticker)
            continue

        df = pd.read_parquet(cache_path)
        df_session = filter_regular_session(df)
        df_session = add_session_date(df_session)

        bars_10m = resample_ticker_10min(df_session, ticker=ticker)
        frames.append(bars_10m)

    if not frames:
        logger.error("No data loaded")
        sys.exit(1)

    bars_long = pd.concat(frames, ignore_index=True)
    bars_10m_path = output_dir / "bars_10m.parquet"
    bars_long.to_parquet(bars_10m_path, index=False)
    logger.info("Saved bars_10m: %s (%d rows)", bars_10m_path, len(bars_long))

    close_panel = build_close_panel(bars_long, tickers)
    close_panel.to_parquet(output_dir / "close_10m.parquet")
    logger.info("Saved close_10m: %s", close_panel.shape)

    returns_panel = build_return_panel(close_panel)
    returns_panel.to_parquet(output_dir / "returns_10m.parquet")
    logger.info("Saved returns_10m: %s", returns_panel.shape)


if __name__ == "__main__":
    main()
