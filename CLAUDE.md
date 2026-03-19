# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Replication of arXiv paper 2504.19623 (Ballarin, Capra, Dellaportas) — **Multi-Horizon Echo State Network for Intraday Stock Return Prediction** — using Polygon.io minute-bar data instead of AlgoSeek.

- Paper summary (Chinese): [`docs/paper_2504.19623.md`](docs/paper_2504.19623.md)

## Commands

```bash
# Install dependencies (Python 3.10+)
pip install -e .

# Run full pipeline (fetch → panel → signals → train → evaluate)
python scripts/run_all.py
python scripts/run_all.py --api-key YOUR_KEY    # explicit API key
python scripts/run_all.py --skip-fetch --skip-panel  # skip stages

# Run individual pipeline stages
python scripts/fetch_massive_data.py --tickers-file data/raw/candidate_tickers.txt --from-date 2024-09-01 --to-date 2025-12-31
python scripts/build_intraday_panel.py --universe-file data/raw/universe.txt
python scripts/build_signals.py
python scripts/train_esn.py
python scripts/evaluate.py

# Run tests
python -m pytest tests/ -v
```

## Architecture

The pipeline has 5 sequential stages, each reading from the previous stage's output:

1. **Data Fetching** (`src/data/`) — Downloads per-ticker minute bars from Polygon.io API (rate-limited 5 calls/min), filters to 09:30–16:00 ET sessions, resamples to 10-min bars, selects top-500 universe by dollar volume, builds (T, N) panels of log returns.

2. **Signal Construction** (`src/signals/`) — Strictly causal pipeline: rolling PCA (J=15 components) → OLS factor regression → residuals → AR(1) OU parameter estimation at 6 lookback levels → modified z-score. Outputs a (T, N, D=6) signal array.

3. **ESN Model** (`src/model/`) — Echo State Network with fixed random reservoir (K=100 nodes). Reservoir state evolves via sparse recurrent matrix; readout is ridge regression. Also includes linear/ridge baselines.

4. **Training** (`src/training/`) — Rolling window with daily cross-validation (1-week lookback). Trains independently for 5 horizons: 10min, 30min, 60min, 2hr, EOD.

5. **Evaluation** (`src/evaluation/`) — MSFE, OOS R², cumulative MSFE ratios, Diebold-Mariano test (Newey-West HAC), Model Confidence Set.

## Configuration

- `config/default.yaml` — Global settings: random seed, PCA params, missing-data thresholds, directory paths
- `config/horizons.yaml` — Per-horizon ESN hyperparameters (alpha, rho, sparsity) from paper Table 1
- `config/sample.yaml` — Date ranges (tuning/test periods), session times, universe size, API rate limits
- `config/massive_key.txt` — Polygon.io API key (first line, gitignored)

## Data Flow

```
data/raw/tickers/{TICKER}.parquet   → per-ticker minute bars (cached)
data/interim/bars_10m.parquet       → 10-min OHLCV
data/interim/close_10m.parquet      → close price panel (T, N)
data/interim/returns_10m.parquet    → log returns (T, N)
data/processed/signals_10m.npz     → signal array (T, N, D=6)
results/{horizon}_predictions.parquet → predicted returns per horizon
results/metrics.csv, dm.csv        → evaluation outputs
results/plots/*.png                → cumulative MSFE ratio charts
```

## Known Limitations

- **L1**: Uses 16:00 minute-bar close as proxy for closing auction price (affects EOD targets)
- **L2**: Universe is top-500 by Q4-2024 dollar volume, not market cap as in original paper
- **L3**: Forward-fill cap (3 bars / ~30 min) and 20% missing-bar threshold are engineering defaults not from the paper
