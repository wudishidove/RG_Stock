"""
Script: Train ESN for all horizons — vectorized reservoir version.

Identical output to train_esn.py but uses GEMM-based reservoir forward pass.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.esn_fast import ESNFast
from src.training.hyperparams import load_horizon_configs
from src.training.rolling_window import rolling_predictions, RollingWindowConfig
from src.data.panel_builder import build_future_return_panel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ESN for all horizons (fast)")
    parser.add_argument("--interim-dir", default="data/interim")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--horizons-config", default="config/horizons.yaml")
    parser.add_argument("--horizons", nargs="+",
                        default=["10min", "30min", "60min", "2hr", "EOD"])
    args = parser.parse_args()

    interim_dir = Path(args.interim_dir)
    processed_dir = Path(args.processed_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading data...")
    close_panel = pd.read_parquet(interim_dir / "close_10m.parquet")
    loaded = np.load(processed_dir / "signals_10m.npz")
    signals = loaded["signals"]     # (T, N, D)
    validity = loaded["validity"]   # (T, N)

    T, N, D = signals.shape
    logger.info("Signals: T=%d, N=%d, D=%d", T, N, D)

    horizon_configs = load_horizon_configs(args.horizons_config)

    t_total = time.perf_counter()

    for horizon_name in args.horizons:
        if horizon_name not in horizon_configs:
            logger.warning("Unknown horizon: %s", horizon_name)
            continue

        cfg = horizon_configs[horizon_name]
        logger.info("=== Horizon: %s (h=%d) ===", horizon_name, cfg.horizon_steps)

        # Build target returns
        session_dates = close_panel.index.to_series().dt.date
        targets_df = build_future_return_panel(
            close_panel, cfg.horizon_steps, session_dates=session_dates
        )
        targets = targets_df.values.astype(float)

        # Build ESN (fast version)
        esn = ESNFast(
            K=cfg.K, D=D,
            alpha=cfg.alpha, rho=cfg.rho, gamma=cfg.gamma,
            A_sparsity=cfg.A_sparsity, C_sparsity=cfg.C_sparsity,
            lambda_ridge=cfg.lambda_ridge,
        )

        t0 = time.perf_counter()
        logger.info("Computing reservoir states (vectorized)...")
        states = esn.compute_states(signals, validity)   # (T, N, K)
        t_res = time.perf_counter() - t0
        logger.info("Reservoir done in %.1f seconds", t_res)

        rw_cfg = RollingWindowConfig(
            train_window_bars=cfg.train_window_bars,
            buffer_bars=cfg.buffer_bars,
            horizon_steps=max(cfg.horizon_steps, 1),
            cv_lookback_days=cfg.cv_lookback_days,
            cv_split=cfg.cv_split,
        )

        # Compute session boundaries (first bar of each trading day)
        dates = close_panel.index.to_series().dt.date
        session_boundaries = (dates != dates.shift(1)).values

        t0 = time.perf_counter()
        logger.info("Rolling predictions (with daily CV for lambda)...")
        valid_targets = ~np.isnan(targets)
        train_valid = validity & valid_targets

        predictions, pred_valid = rolling_predictions(
            states, targets, train_valid, rw_cfg,
            lambda_=cfg.lambda_ridge,
            session_boundaries=session_boundaries,
        )
        t_roll = time.perf_counter() - t0
        logger.info("Rolling done in %.1f seconds", t_roll)

        # Save predictions
        pred_df = pd.DataFrame(predictions, index=close_panel.index, columns=close_panel.columns)
        out_path = results_dir / f"{horizon_name}_predictions.parquet"
        pred_df.to_parquet(out_path)
        logger.info("Saved predictions: %s", out_path)

    elapsed_total = time.perf_counter() - t_total
    logger.info("Total elapsed: %.1f seconds (%.1f minutes)", elapsed_total, elapsed_total / 60)


if __name__ == "__main__":
    main()
