"""
Script: Compute metrics (MSFE, DM, MCS) and generate plots.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.metrics import msfe, cumulative_msfe_ratio, oos_r2
from src.evaluation.diebold_mariano import diebold_mariano_test
from src.training.hyperparams import load_horizon_configs

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ESN predictions")
    parser.add_argument("--interim-dir", default="data/interim")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--horizons-config", default="config/horizons.yaml")
    parser.add_argument("--horizons", nargs="+",
                        default=["10min", "30min", "60min", "2hr", "EOD"])
    args = parser.parse_args()

    interim_dir = Path(args.interim_dir)
    results_dir = Path(args.results_dir)
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    close_panel = pd.read_parquet(interim_dir / "close_10m.parquet")
    horizon_configs = load_horizon_configs(args.horizons_config)

    metrics_rows = []
    dm_rows = []

    for horizon_name in args.horizons:
        pred_path = results_dir / f"{horizon_name}_predictions.parquet"
        if not pred_path.exists():
            logger.warning("No predictions for %s", horizon_name)
            continue

        cfg = horizon_configs[horizon_name]
        h = max(cfg.horizon_steps, 1)

        predictions = pd.read_parquet(pred_path)
        actual_df = np.log(close_panel).diff(h).shift(-h)  # approximate targets
        actual = actual_df.reindex(columns=predictions.columns).values
        pred = predictions.values

        valid = ~np.isnan(actual) & ~np.isnan(pred)
        m = msfe(actual, pred, valid)
        r2 = oos_r2(actual, pred, valid)

        logger.info("%s: MSFE=%.6f  OOS-R²=%.4f", horizon_name, m, r2)
        metrics_rows.append({
            "horizon": horizon_name,
            "MSFE": m,
            "OOS_R2": r2,
        })

        # Cumulative MSFE plot (vs naive zero-forecast baseline)
        zero_pred = np.zeros_like(pred)
        cum_ratio = cumulative_msfe_ratio(actual, pred, zero_pred, valid)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(cum_ratio, lw=1.0)
        ax.axhline(1.0, color="r", ls="--", lw=0.8)
        ax.set_title(f"Cumulative MSFE ratio — {horizon_name} (ESN / zero-forecast)")
        ax.set_ylabel("Ratio")
        ax.set_xlabel("Time step")
        fig.tight_layout()
        fig.savefig(plots_dir / f"cum_msfe_{horizon_name}.png", dpi=120)
        plt.close(fig)

        # DM test vs zero-forecast
        dm = diebold_mariano_test(actual, pred, zero_pred, valid, h=h)
        dm_rows.append({
            "horizon": horizon_name,
            "dm_stat": dm["dm_stat"],
            "p_value": dm["p_value"],
            "n_obs": dm["n_obs"],
        })

    if metrics_rows:
        pd.DataFrame(metrics_rows).to_csv(results_dir / "metrics.csv", index=False)
        logger.info("Saved metrics.csv")

    if dm_rows:
        pd.DataFrame(dm_rows).to_csv(results_dir / "dm.csv", index=False)
        logger.info("Saved dm.csv")


if __name__ == "__main__":
    main()
