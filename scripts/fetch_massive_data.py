"""
Script: Download and cache Massive (Polygon) minute bars per ticker.

Usage:
    # Read API key automatically from config/massive_key.txt (default)
    python scripts/fetch_massive_data.py --tickers-file data/raw/candidate_tickers.txt

    # Or pass key explicitly
    python scripts/fetch_massive_data.py --api-key YOUR_KEY --tickers-file data/raw/candidate_tickers.txt
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.massive_fetcher import MassiveFetcher

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def resolve_api_key(args) -> str:
    """Resolve API key: CLI argument takes priority, then key file."""
    if args.api_key:
        return args.api_key
    key_file = Path(args.api_key_file)
    if key_file.exists():
        key = key_file.read_text().strip().splitlines()[0].strip()
        if key:
            logger.info("API key loaded from %s", key_file)
            return key
    logger.error("No API key provided. Use --api-key or place key in %s", args.api_key_file)
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Massive minute bar data")
    parser.add_argument("--api-key", default=None, help="Massive/Polygon API key (overrides --api-key-file)")
    parser.add_argument("--api-key-file", default="config/massive_key.txt", help="Path to file containing API key")
    parser.add_argument("--tickers-file", required=True, help="File with one ticker per line")
    parser.add_argument("--from-date", default="2024-09-01")
    parser.add_argument("--to-date", default="2025-12-31")
    parser.add_argument("--raw-dir", default="data/raw/tickers")
    parser.add_argument("--rate-delay", type=float, default=12.0)
    args = parser.parse_args()

    api_key = resolve_api_key(args)

    tickers_file = Path(args.tickers_file)
    if not tickers_file.exists():
        logger.error("Tickers file not found: %s", tickers_file)
        sys.exit(1)

    tickers = [line.strip() for line in tickers_file.read_text().splitlines() if line.strip()]
    logger.info("Fetching %d tickers from %s to %s", len(tickers), args.from_date, args.to_date)

    fetcher = MassiveFetcher(
        api_key=api_key,
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
