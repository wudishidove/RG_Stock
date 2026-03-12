"""
Script: Train ESN for all horizons, produce per-horizon predictions parquet.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.esn import ESN
from src.training.hyperparams import load_horizon_configs
from src.training.rolling_window import rolling_predictions, RollingWindowConfig
from src.data.panel_builder import build_future_return_panel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ESN for all horizons")
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

        # Build ESN
        esn = ESN(
            K=cfg.K, D=D,
            alpha=cfg.alpha, rho=cfg.rho, gamma=cfg.gamma,
            A_sparsity=cfg.A_sparsity, C_sparsity=cfg.C_sparsity,
            lambda_ridge=cfg.lambda_ridge,
        )

        logger.info("Computing reservoir states...")
        states = esn.compute_states(signals, validity)   # (T, N, K)

        rw_cfg = RollingWindowConfig(
            train_window_bars=cfg.train_window_bars,
            buffer_bars=cfg.buffer_bars,
            horizon_steps=max(cfg.horizon_steps, 1),
        )

        logger.info("Rolling predictions...")
        # Target for training: shifted targets
        h = max(cfg.horizon_steps, 1)
        train_targets = np.full_like(targets, np.nan)
        train_targets[:-h] = targets[:-h]   # targets[t] = return at t+h

        valid_targets = ~np.isnan(targets)
        train_valid = validity & valid_targets

        predictions, pred_valid = rolling_predictions(
            states, targets, train_valid, rw_cfg, lambda_=cfg.lambda_ridge
        )

        # Save predictions
        pred_df = pd.DataFrame(predictions, index=close_panel.index, columns=close_panel.columns)
        out_path = results_dir / f"{horizon_name}_predictions.parquet"
        pred_df.to_parquet(out_path)
        logger.info("Saved predictions: %s", out_path)


if __name__ == "__main__":
    main()
