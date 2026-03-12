# Multi-Horizon Echo State Network Stock Return Prediction

## Paper

- **arxiv**: 2504.19623 — *Multi-Horizon Echo State Network Prediction of Intraday Stock Returns*
- **Authors**: Ballarin, Capra, Dellaportas
- **Original data**: AlgoSeek 1-minute OHLC bars (SIP-consolidated), downsampled to 10-minute. Main test sample: full year 2013. Hyperparameter tuning pre-sample: Q4 2012.

---

## Positioning

> **Paper-style intraday ESN replication with Massive Basic as data-source substitute for AlgoSeek.**

This is not a daily-frequency adaptation. The goal is to preserve the paper's intraday structure as faithfully as possible while substituting the data vendor.

### What is faithfully preserved

- 1-minute → 10-minute downsampling
- All five prediction horizons: 10min / 30min / 60min / 2hr / EOD (target definition caveat — see Formal Research Limitations)
- Pooled forecasting across ~500 stocks
- 6-dimensional signal (PCA → residuals → OU → modified z-score)
- Fixed random ESN reservoir; only the linear readout is trained
- Short rolling estimation windows from paper Table 7
- Daily cross-validation (1-week lookback, 0.7/0.3 temporal split)
- MSFE / cumulative MSFE / Diebold-Mariano / MCS evaluation

### Necessary compromises

- Data vendor: Massive Basic instead of AlgoSeek
- Sample period: latest 2 years (Massive Basic limit) → Q4 2024 tuning + full year 2025 testing
- Universe selection: pre-sample dollar-volume approximation (see Formal Research Limitations)
- EOD target construction: microstructure substitution assumption (see Formal Research Limitations)

---

## Paper vs. This Implementation

| Item | Paper | This Plan (Massive Basic) |
|------|-------|--------------------------|
| Raw data | AlgoSeek 1-min OHLC (SIP consolidated) | Massive minute aggregates |
| Market coverage | All U.S. exchanges + FINRA | 100% market coverage (Massive claim) |
| Frequency | 1-min, downsampled to 10-min | 1-min, downsampled to 10-min |
| Main test sample | Full year 2013 | Full year 2025 |
| Tuning pre-sample | Q4 2012 | Q4 2024 |
| History limit | Deep history available | 2 years (Basic tier) |
| Prediction horizons | 10m / 30m / 60m / 2h / EOD | Identical |
| Rebalancing frequency | Every 10 minutes | Identical |

---

## Massive Basic — Practical Strategy

Massive Basic tier: **$0/month, 5 API calls/minute, 2 years history, minute aggregates, 100% market coverage**.

Data acquisition approach:

> **Per-ticker range request with pagination → local parquet cache per ticker → then session filter, resample, and assemble panel.**

Use the Massive aggregates endpoint per ticker:
```
GET /v2/aggs/ticker/{ticker}/range/1/minute/{from}/{to}?adjusted=true&limit=50000
```
Paginate via the `next_url` field until all bars for the requested date range are retrieved. With 5 calls/minute on the free tier, fetch tickers sequentially with a rate-limit delay (~12 s between calls). Skip any ticker whose cache already covers the requested range. Never re-fetch data already stored locally.

This approach is preferable to date-batched all-market downloads because the Massive free tier does not guarantee a bulk "all tickers for one date" endpoint, and per-ticker range requests are the documented Basic access pattern.

---

## Project Structure

```
D:\gitlab\boring\stock\
├── pyproject.toml
├── plan.md
├── config/
│   ├── default.yaml           # global settings: dates, PCA params, eval config
│   ├── horizons.yaml          # per-horizon ESN hyperparameters (from paper Table 1)
│   └── sample.yaml            # sample period, session definition, universe settings
├── src/
│   ├── data/
│   │   ├── massive_fetcher.py     # Massive minute aggregates batch download + cache
│   │   ├── session_filter.py      # keep regular session only (09:30–16:00 ET)
│   │   ├── resample_10m.py        # 1-minute → 10-minute OHLCV resampling
│   │   ├── universe.py            # fixed 500-stock universe construction
│   │   ├── panel_builder.py       # assemble (T, N) close and return panels
│   │   └── preprocess.py          # 10-min log returns, future returns, missing masks
│   ├── signals/
│   │   ├── pca_factors.py         # rolling causal PCA, J=15 market factors
│   │   ├── residuals.py           # rolling OLS factor regression → idiosyncratic residuals
│   │   ├── ou_estimation.py       # OU parameter estimation via AR(1) per window P
│   │   ├── zscore.py              # modified z-score construction
│   │   └── pipeline.py            # end-to-end causal signal orchestrator
│   ├── model/
│   │   ├── reservoir.py           # ESN reservoir: random matrices, state evolution, state decay
│   │   ├── readout.py             # diagonal anisotropic ridge regression readout
│   │   ├── esn.py                 # full ESN = reservoir + readout
│   │   └── baselines.py           # linear baseline + regularized benchmark ridge
│   ├── training/
│   │   ├── rolling_window.py      # rolling window train loop with buffer enforcement
│   │   ├── cross_validation.py    # daily CV: 1-week lookback, 0.7 temporal split
│   │   └── hyperparams.py         # HorizonConfig dataclass + YAML loader
│   ├── evaluation/
│   │   ├── metrics.py             # MSFE, cumulative MSFE, OOS R²
│   │   ├── diebold_mariano.py     # DM test with Newey-West HAC
│   │   └── model_confidence_set.py # MCS test
│   └── utils/
│       └── linalg.py              # spectral radius, sparse matrix generation
├── scripts/
│   ├── fetch_massive_data.py      # download + cache Massive minute data
│   ├── build_intraday_panel.py    # session filter + 10-min resample + panel assembly
│   ├── build_signals.py           # causal signal construction
│   ├── train_esn.py               # train ESN for all horizons
│   ├── evaluate.py                # compute metrics and generate reports
│   └── run_all.py                 # full pipeline entry point
└── tests/
    ├── test_resample_10m.py
    ├── test_reservoir.py
    ├── test_signals.py
    └── test_readout.py
```

---

## Sample Design

### Time periods

| Period | Dates |
|--------|-------|
| Hyperparameter tuning (pre-sample) | 2024-09-01 to 2024-12-31 |
| Main test sample | 2025-01-01 to 2025-12-31 |

This mirrors the paper's structure: Q4 for tuning, the following full year for evaluation — applied to the maximum available history under Massive Basic.

### Trading session

- Regular session only: **09:30–16:00 ET**
- Prediction timestamps: every 10 minutes from 09:30 to 15:50
- No overnight positions; 15:50 is the last actionable time point

### Prediction horizons (identical to paper)

| Horizon | Steps (10-min bars) | Daily prediction count |
|---------|--------------------|-----------------------|
| 10min | 1 | 39 |
| 30min | 3 | 37 |
| 60min | 6 | 34 |
| 2hr | 12 | 28 |
| EOD | variable | 39 |

---

## Stock Universe

**Problem**: using today's S&P 500 constituents to look back historically introduces survivorship bias. The paper does not disclose its exact universe construction method — it only states "500 largest US stocks" without specifying the ranking criterion or rebalancing rule.

**Approximation** (not a strict replication): select the top 500 stocks by average dollar volume during the pre-sample period (Q4 2024), then **hold the universe fixed** throughout the full year 2025 test period.

- Reduces (but does not eliminate) survivorship bias relative to using today's index membership
- Consistent with the paper's fixed-universe pooled forecasting spirit
- Does not require acquiring full historical index constituent data
- **Caveat**: dollar-volume ranking in Q4 2024 is not equivalent to the paper's "500 largest US stocks" criterion. The resulting universe may differ in composition, and this difference is unquantifiable without the original AlgoSeek universe list.

---

## Data Flow

```
Massive minute aggregates (per-ticker range + pagination)
    ↓ fetch_massive_data.py
data/raw/tickers/{TICKER}.parquet      (per-ticker cache, appended on re-run)
    ↓ session_filter + resample_10m
data/interim/bars_10m.parquet
    ↓ panel_builder
data/interim/close_10m.parquet         (T, N)
data/interim/returns_10m.parquet       (T, N)
data/interim/future_returns_{h}.parquet
    ↓ build_signals.py
data/processed/signals_10m.npz        (T, N, D=6)
    ↓ train_esn.py
results/{horizon}_predictions.parquet
    ↓ evaluate.py
results/metrics.csv + dm.csv + mcs.csv + plots/
```

---

## Signal Construction Pipeline (strictly causal)

> **Any signal, factor, OU parameter, or regularization choice must use only information available strictly before the current time point.**

No full-sample PCA, no full-sample residual fitting, no full-sample OU estimation. All estimation must respect the rolling time point — otherwise look-ahead bias is introduced.

### Step 1 — 10-minute log returns

- Build regular-session minute close price series
- Downsample to 10-minute bars (use period close)
- Compute log returns: `r_t = log(C_t / C_{t-1})`

### Step 1b — Forward-fill before PCA

Before computing cross-sectional statistics, apply within-session forward-fill to the 10-minute close price panel:

- Fill missing bars **within the same trading session only** (do not carry prices across the overnight gap)
- Cap forward-fill at a maximum of **3 consecutive bars** (~30 minutes); mark longer gaps as still-missing
- Stocks with more than **20% missing bars** in the estimation window are excluded from that window's PCA entirely

Forward-fill is applied to prices (before log-return computation), so the filled return for a multi-bar gap is zero, which is the standard no-trade assumption.

**These two thresholds (3-bar cap, 20% exclusion) are not specified by the paper — see L3 in Formal Research Limitations.**

### Step 2 — PCA market factors

- Apply PCA **only on data before the current estimation window**
- Standardize each stock's return series before PCA (per paper Appendix B)
- Extract J=15 principal components as market factors F[t, j]
- These 15 factors should explain >90% of cross-sectional variance

### Step 3 — Residual extraction

For each stock i, estimate via rolling OLS:

```
r_{t,i} = a_i + sum_{j=1}^{J} b_{j,i} * F_{t,j} + v_{t,i}
```

- Estimation uses only the rolling lookback window ending before t
- Outputs: drift `a_i`, factor loadings `b_{j,i}`, idiosyncratic residuals `v_{t,i}`

### Step 4 — OU parameter estimation

Six aggregation levels (units: 10-minute bars):

```
P ∈ {10, 20, 30, 60, 100, 150}
```

For each P, compute aggregated residuals:

```
U_hat[P, t, i] = sum_{s=t-P+1}^{t} v_hat[s, i]
```

Fit AR(1):

```
U_hat[P, t+1, i] = c_0 + c_u * U_hat[P, t, i] + eta
```

Extract OU parameters:

```
kappa_P = -log(c_u)
m_P = c_0 / (1 - c_u)
sigma_P = sqrt(Var(eta) / (2 * kappa_P))
```

Stability safeguard: clip `c_u` to `(eps, 1 - eps)` before taking log.

### Step 5 — Modified z-score

```
z_tilde[P, t, i] = (U_hat[P, t, i] - m_P[i]) / sigma_P[i]
                   - a[i] / (kappa_P[i] * sigma_P[i])
```

Apply floor: `sigma_P = max(sigma_P, eps)` to avoid division by zero.

Output per stock per time point: **6-dimensional signal vector** Z_t^(i) ∈ ℝ⁶.

---

## Missing Data Handling (state decay)

Do not simply drop time points with missing signals. Following the paper's Appendix C:

- When a stock's signal is missing at time t: **replace with zero vector**
- The reservoir state continues to evolve (state decay through the zero input)
- States and observations produced from missing inputs are **excluded from readout training**
- A boolean validity mask is maintained and propagated through training and evaluation

This keeps the time series continuous for the reservoir while cleanly excluding contaminated observations from the regression.

---

## ESN Reservoir

### State update equation

```
X_t = alpha * X_{t-1} + (1 - alpha) * phi(rho * A_bar * X_{t-1} + gamma * C_bar * Z_t)
```

| Symbol | Description | Value |
|--------|-------------|-------|
| X_t ∈ ℝ^K | reservoir state vector | K = 100 |
| alpha ∈ [0,1] | leak rate | horizon-specific |
| phi | nonlinear activation | tanh (default) |
| rho ∈ [0,1] | spectral radius | horizon-specific |
| gamma > 0 | input scaling | horizon-specific |
| A_bar (K×K) | normalized recurrent matrix | sparse Gaussian |
| C_bar (K×D) | normalized input matrix | sparse Uniform |

### Random matrix generation

- `A*`: sparse Gaussian — nonzero density = A_sparsity, nonzero values ~ N(0,1)
- `C*`: sparse Uniform — nonzero density = C_sparsity, nonzero values ~ U(-1,1)
- Normalize: `A_bar = A* / spectral_radius(A*)` then scale by `rho`
- Normalize: `C_bar = C* / ||C*||_2` then scale by `gamma`

**Key property**: reservoir weights are random and fixed. All reservoir states can be computed in a single forward pass before the rolling window loop. Only the readout refits at each step.

### Per-horizon hyperparameters (paper Table 1)

| Parameter | 10min | 30min | 60min | 2hr | EOD |
|-----------|-------|-------|-------|-----|-----|
| K | 100 | 100 | 100 | 100 | 100 |
| alpha (leak rate) | 0.9 | 0.2 | 0.0 | 0.0 | 0.0 |
| A sparsity | 0.15 | 0.15 | 0.15 | 0.65 | 0.35 |
| rho (spectral radius) | 0.4 | 0.6 | 0.6 | 0.6 | 0.0 |
| C sparsity | 0.95 | 0.55 | 0.75 | 0.85 | 0.25 |
| gamma (input scaling) | 0.005 | 0.005 | 0.005 | 0.005 | 0.015 |

Note: EOD horizon has rho=0, collapsing the recurrent connection to a purely feedforward nonlinear transform.

---

## Readout

### Mapping

```
y_hat_{t+h}^(i) = mu + theta^T * X_t^(i)
```

### Training objective — diagonal anisotropic ridge

Per paper Appendix C, the cross-validation selects a **diagonal anisotropic ridge penalty matrix**, not a scalar lambda:

```
beta_hat = argmin_beta ||y - X * beta||_2^2 + beta^T * Lambda * beta
```

where `X = [1, states]` and `Lambda` is diagonal (intercept unpunished, each readout weight can have a different penalty).

**Implementation phases**:
- Phase A: implement scalar ridge first (`(X'X + lambda*I)^{-1} X'y`) to get the pipeline running
- Phase B: upgrade to diagonal anisotropic ridge for paper-accurate results

### Pooled regression

All stocks share a single readout per horizon. Training pool = all valid stocks × all time steps in the rolling train window × non-missing mask. This matches the paper's pooled forecasting design.

---

## Rolling Window Training (paper's short-window settings)

The daily-frequency 252-day windows are removed. Use paper Appendix C / Table 7 settings:

| Parameter | 10min | 30min | 60min | 2hr | EOD |
|-----------|-------|-------|-------|-----|-----|
| Train window M | 30 min | 30 min | 1 hr | 1 hr | 1 day |
| Buffer tau_h | 10 min | 30 min | 1 hr | 2 hr | 1 day |
| CV frequency | daily | daily | daily | daily | daily |
| CV lookback | 1 week | 1 week | 1 week | 1 week | 1 week |
| CV split | 0.7/0.3 | 0.7/0.3 | 0.7/0.3 | 0.7/0.3 | 0.7/0.3 |

### Training timeline

```
|<-- train window M -->|<-- buffer tau_h -->|  t  |<-- h -->| t+h
|    training data     |   gap (excluded)   | now | predict | target
```

The short train windows are intentional — the authors found that using too much historical data does not improve intraday predictions. The buffer is set to the minimum legal value to avoid overlap with the target return interval.

---

## Cross-Validation

- Run CV **once per trading day** (not at every 10-min bar)
- Look back **1 week** of intraday bars
- Use a **0.7/0.3 temporal split** (no shuffling, strict time ordering)
- Purpose: select readout regularization parameter(s)
- Must use only data observable before the current trading day

---

## Formal Research Limitations

These are not engineering shortcuts — they are substantive differences from the paper's experimental design that must be disclosed in any results discussion.

### L1 — Closing auction substitution (market microstructure assumption)

The paper explicitly defines its trading protocol as:
- Predictions made every 10 minutes during regular session
- **Last actionable time point: 15:50 ET**
- Positions liquidated via **closing auction** at end of day
- EOD return defined as `log(closing_auction_price / C_t)`

This implementation uses **16:00 minute-bar close** as a proxy for the closing auction print. This substitution affects two components:

1. **All EOD horizon targets** — `r_{t, EOD}` is computed against the 16:00 bar close, not the true auction print. The closing auction typically clears at a price different from the last pre-close minute bar, particularly on index rebalancing days and option expiry dates.

2. **The 15:50 prediction's 10-minute horizon target** — the paper's 10-minute return at t=15:50 is `log(auction_price / C_{15:50})`. This implementation computes `log(C_{16:00} / C_{15:50})` instead. The 16:00 bar close and the auction print can diverge materially.

This is a market microstructure substitution assumption, not a minor approximation. It should be clearly stated when reporting EOD and 15:50 horizon results.

### L2 — Universe construction approximation

The paper describes "500 largest US stocks" without specifying the exact ranking metric, data source, or rebalancing frequency. This implementation uses pre-sample (Q4 2024) average dollar volume as a proxy. The resulting universe may differ in composition from the paper's, and this difference cannot be quantified without access to the original universe list.

### L3 — Forward-fill parameters are unspecified by the paper

The paper acknowledges missing data via "state decay" (replace missing signal with zero vector, continue evolving reservoir) but does not specify any price-level forward-fill step before signal construction. The 3-bar cap and 20% exclusion threshold in Step 1b are engineering defaults chosen for numerical stability. They are not grounded in the paper's methodology and are tunable. Sensitivity to these thresholds should be checked if results appear anomalous.

---

## Target Variable Definitions

### Bar labeling convention

Massive API minute bars use **close-time (end-time) labels**:
- Bar labeled `09:31` covers the period `09:30:00–09:31:00`
- The first regular-session bar is `09:31`

After 10-minute resampling (aggregate every 10 bars by taking period close):
- First 10-min bar labeled `09:40` covers `09:30:00–09:40:00`
- Second 10-min bar labeled `09:50` covers `09:40:00–09:50:00`
- Last intraday bar labeled `16:00` covers `15:50:00–16:00:00`

Signal time `t` refers to the **close of a 10-min bar**. Prediction at time `t` is made using information up to and including bar `t`, and targets the return beginning at `t`.

**Concrete example** (10-min horizon):
```
Signal time t = 09:40  (bar close)
Target return  = log(C_{09:50} / C_{09:40})
Next bar       : t+1 = 09:50
```

**Concrete example** (EOD horizon at t = 09:40):
```
r_{09:40, EOD} = log(C_{16:00} / C_{09:40})    [proxy — see L1]
```

### Standard horizons (10-min bar units)

```
10min : log(C_{t+1} / C_t)     e.g. t=09:40 → log(C_{09:50} / C_{09:40})
30min : log(C_{t+3} / C_t)     e.g. t=09:40 → log(C_{10:10} / C_{09:40})
60min : log(C_{t+6} / C_t)     e.g. t=09:40 → log(C_{10:40} / C_{09:40})
2hr   : log(C_{t+12} / C_t)    e.g. t=09:40 → log(C_{11:40} / C_{09:40})
```

For t = `15:50` with the 10-minute horizon, `C_{t+1}` = `C_{16:00}` bar close (see L1).

### EOD horizon

```
r_{t, EOD} = log(C_{16:00} / C_t)     [proxy — see L1]
```

The paper uses closing auction price as the denominator. This implementation uses the 16:00 minute-bar close. The substitution is documented in L1 and must be noted in all EOD result tables.

---

## Baselines

| Model | Description |
|-------|-------------|
| Linear baseline | OLS regression directly on Z_t, minimal window, no regularization |
| Benchmark ridge | Ridge regression on Z_t, same windowing and CV scheme as ESN |

Optional: feedforward-only ESN variant (rho=0 for all horizons) to isolate the value of recurrent connections. These match the paper's baseline comparisons.

---

## Evaluation

### 1. MSFE

```
MSFE_t = (1/N_t) * sum_i (r_{t+h}^(i) - r_hat_{t+h}^(i))^2
MSFE   = (1/T)   * sum_t MSFE_t
```

### 2. Out-of-sample R²

```
R^2 = 1 - sum(e^2) / sum((y - y_bar)^2)
```

### 3. Cumulative MSFE ratio

Running ratio of model MSFE vs. baseline MSFE — shows where in the test sample the ESN gains or loses predictive advantage.

### 4. Diebold-Mariano test

- Test equal predictive ability between two models
- Use Newey-West HAC standard errors
- Set lag parameter to align with the horizon length (overlapping returns require HAC correction)

### 5. Model Confidence Set (MCS)

- Paper uses MCS in addition to DM
- Identify the set of models not dominated at a given confidence level
- Implement `model_confidence_set.py`

---

## Memory and Compute Considerations

### Data layer

Do not load all raw minute data into a single dense in-memory tensor.

- Store raw 1-minute bars as **per-ticker parquet files** (`data/raw/tickers/{TICKER}.parquet`)
- Each file covers the full available history for that ticker; append new dates on re-run
- Apply session filter and 10-min resample when assembling the panel
- Assemble the (T, N) or (T, N, D) panel only for the active rolling window

### State layer

- Cache reservoir states per horizon in compressed numpy format
- During rolling window training, load only the slice needed for the current train window
- Propagate the validity mask alongside all state arrays

### Massive Basic rate limit

With only 5 calls/minute:
- Fetch per-ticker over a date range (not all-market per date)
- Implement a permanent local cache; check ticker file coverage before any API call
- Never re-query a date range already present in the local cache

---

## Known Challenges

| # | Challenge | Mitigation |
|---|-----------|------------|
| 1 | Massive Basic 5 calls/min rate limit | Per-ticker range requests with pagination + permanent parquet cache; ~12s delay between calls |
| 2 | EOD ≠ closing auction print (L1) | Use 16:00 bar close as proxy; formally disclosed in L1; flag in all EOD result tables |
| 3 | 15:50 10-min target also affected by L1 | Same proxy; note separately in results |
| 4 | Universe composition differs from paper (L2) | Pre-sample dollar-volume approximation; disclosed in L2 |
| 5 | NaN columns in PCA covariance matrix | Within-session forward-fill (max 3 bars) before PCA; exclude stocks >20% missing; both thresholds are L3 assumptions |
| 6 | Universe fixed but data has within-sample gaps | State decay (zero-input) + validity mask; do not drop rows |
| 7 | PCA / residual / OU must be causal | All estimation inside rolling window; no full-sample pre-computation |
| 8 | OU instability when c_u ≈ 0 or c_u ≥ 1 | Clip c_u to (eps, 1-eps) before log |
| 9 | Z-score denominator near zero | Floor: sigma_P = max(sigma_P, eps) |
| 10 | Anisotropic ridge more complex than scalar | Implement scalar first (Phase A), then upgrade (Phase B) |

---

## Dependencies

```
numpy>=1.24
pandas>=2.0
scipy>=1.10
scikit-learn>=1.3
pyyaml>=6.0
matplotlib>=3.7
tqdm>=4.65
pyarrow>=14.0
requests>=2.31
```

Add Massive official Python client if one exists; otherwise implement via REST + parquet cache.

---

## Implementation Order

| Phase | Content | Key files |
|-------|---------|-----------|
| 1 | Massive data fetch + cache + session filter | `massive_fetcher.py`, `session_filter.py` |
| 2 | 1-min → 10-min resample + future return definitions | `resample_10m.py`, `preprocess.py` |
| 3 | Fixed universe + panel assembly | `universe.py`, `panel_builder.py` |
| 4 | Causal signals: PCA → residuals → OU → z-score | `signals/*` |
| 5 | ESN reservoir + state evolution + state decay | `reservoir.py`, `esn.py` |
| 6 | Baselines + Phase-A scalar ridge readout | `readout.py`, `baselines.py` |
| 7 | Rolling window + daily CV | `rolling_window.py`, `cross_validation.py` |
| 8 | MSFE / DM / MCS + cumulative plots | `evaluation/*`, `evaluate.py` |
| 9 | Upgrade readout to diagonal anisotropic ridge (Phase B) | `readout.py` |

---

## One-line Summary

> **Paper-style intraday ESN replication (arXiv 2504.19623) using Massive Basic minute aggregates as a free-tier substitute for AlgoSeek, with two formally disclosed microstructure substitution assumptions: (L1) 16:00 bar close proxying for closing auction, and (L2) dollar-volume-based universe approximation.**