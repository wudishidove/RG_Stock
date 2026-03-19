"""
Script: Build causal signal panel (T, N, D=6) — parallelised version.

Identical output to build_signals.py but uses multiprocessing for speed.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.signals.pipeline_fast import build_signals_causal

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build intraday ESN signals (fast)")
    parser.add_argument("--interim-dir", default="data/interim")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--pca-lookback", type=int, default=390,
                        help="Number of 10-min bars for PCA lookback (~1 week)")
    parser.add_argument("--n-pca", type=int, default=15)
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: cpu_count - 1)")
    args = parser.parse_args()

    interim_dir = Path(args.interim_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    returns = pd.read_parquet(interim_dir / "returns_10m.parquet")
    T, N = returns.shape
    logger.info("Returns panel: %s", returns.shape)

    # Session boundary: True at first bar of each session
    session_dates = returns.index.to_series().dt.date
    session_boundary = np.zeros(T, dtype=bool)
    prev_date = None
    for i, d in enumerate(session_dates):
        if d != prev_date:
            session_boundary[i] = True
            prev_date = d

    ret_arr = returns.values.astype(float)

    logger.info("Building signals (T=%d, N=%d)...", T, N)
    t0 = time.perf_counter()

    signals, validity = build_signals_causal(
        ret_arr,
        session_boundary=session_boundary,
        pca_lookback=args.pca_lookback,
        n_pca_components=args.n_pca,
        n_workers=args.workers,
    )

    elapsed = time.perf_counter() - t0
    logger.info("Done in %.1f seconds", elapsed)

    np.savez_compressed(
        output_dir / "signals_10m.npz",
        signals=signals,
        validity=validity,
    )
    logger.info("Saved signals: %s validity: %s", signals.shape, validity.shape)


if __name__ == "__main__":
    main()
