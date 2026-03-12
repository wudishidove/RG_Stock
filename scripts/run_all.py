"""
Full pipeline entry point.

Runs all stages sequentially:
  1. fetch_massive_data  (requires API key + tickers file)
  2. build_intraday_panel
  3. build_signals
  4. train_esn
  5. evaluate

Pass --help for options, or set environment variables.
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
    result = subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full ESN pipeline")
    parser.add_argument("--api-key", required=True)
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
        run([py, str(SCRIPTS_DIR / "fetch_massive_data.py"),
             "--api-key", args.api_key,
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
