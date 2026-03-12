"""
Full pipeline entry point.

Runs all stages sequentially:
  1. fetch_massive_data
  2. build_intraday_panel
  3. build_signals
  4. train_esn
  5. evaluate

Usage:
    # Read API key automatically from config/massive_key.txt (default)
    python scripts/run_all.py

    # Or pass key explicitly
    python scripts/run_all.py --api-key YOUR_KEY
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SCRIPTS_DIR = Path(__file__).parent


def run(cmd: list[str]) -> None:
    logger.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


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
    parser = argparse.ArgumentParser(description="Run full ESN pipeline")
    parser.add_argument("--api-key", default=None, help="Massive/Polygon API key (overrides --api-key-file)")
    parser.add_argument("--api-key-file", default="config/massive_key.txt", help="Path to file containing API key")
    parser.add_argument("--tickers-file", default="data/raw/candidate_tickers.txt")
    parser.add_argument("--from-date", default="2024-09-01")
    parser.add_argument("--to-date", default="2025-12-31")
    parser.add_argument("--skip-fetch", action="store_true")
    parser.add_argument("--skip-panel", action="store_true")
    parser.add_argument("--skip-signals", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    py = sys.executable

    if not args.skip_fetch:
        api_key = resolve_api_key(args)
        run([py, str(SCRIPTS_DIR / "fetch_massive_data.py"),
             "--api-key", api_key,
             "--tickers-file", args.tickers_file,
             "--from-date", args.from_date,
             "--to-date", args.to_date])

    if not args.skip_panel:
        run([py, str(SCRIPTS_DIR / "build_intraday_panel.py"),
             "--from-date", args.from_date,
             "--to-date", args.to_date])

    if not args.skip_signals:
        run([py, str(SCRIPTS_DIR / "build_signals.py")])

    if not args.skip_train:
        run([py, str(SCRIPTS_DIR / "train_esn.py")])

    run([py, str(SCRIPTS_DIR / "evaluate.py")])

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
