"""
Script: Download and cache Massive (Polygon) minute bars per ticker.

Usage:
    python scripts/fetch_massive_data.py \
        --api-key YOUR_KEY \
        --tickers-file data/raw/candidate_tickers.txt \
        --from-date 2024-09-01 \
        --to-date 2025-12-31
"""

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.massive_fetcher import MassiveFetcher

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Massive minute bar data")
    parser.add_argument("--api-key", required=True, help="Massive/Polygon API key")
    parser.add_argument("--tickers-file", required=True, help="File with one ticker per line")
    parser.add_argument("--from-date", default="2024-09-01")
    parser.add_argument("--to-date", default="2025-12-31")
    parser.add_argument("--raw-dir", default="data/raw/tickers")
    parser.add_argument("--rate-delay", type=float, default=12.0)
    args = parser.parse_args()

    tickers_file = Path(args.tickers_file)
    if not tickers_file.exists():
        logger.error("Tickers file not found: %s", tickers_file)
        sys.exit(1)

    tickers = [line.strip() for line in tickers_file.read_text().splitlines() if line.strip()]
    logger.info("Fetching %d tickers from %s to %s", len(tickers), args.from_date, args.to_date)

    fetcher = MassiveFetcher(
        api_key=args.api_key,
        raw_dir=args.raw_dir,
        rate_delay=args.rate_delay,
    )

    for i, ticker in enumerate(tickers):
        logger.info("[%d/%d] %s", i + 1, len(tickers), ticker)
        try:
            fetcher.ensure_ticker(ticker, args.from_date, args.to_date)
        except Exception as e:
            logger.error("Failed %s: %s", ticker, e)


if __name__ == "__main__":
    main()
