# Multi-Horizon Echo State Network — Intraday Stock Return Prediction

複現論文 [arXiv 2504.19623](https://arxiv.org/abs/2504.19623)（Ballarin, Capra, Dellaportas）的日內 ESN 預測系統，以 Polygon.io（Massive Basic）免費 API 取代原始 AlgoSeek 資料。

---

## 專案簡介

本專案對約 500 檔美國大型股，每 10 分鐘產生一次報酬預測，涵蓋五個預測時間長度：

| 預測視野 | 說明 |
|----------|------|
| 10min | 下一個 10 分鐘 bar |
| 30min | 未來 30 分鐘 |
| 60min | 未來 1 小時 |
| 2hr | 未來 2 小時 |
| EOD | 當日收盤（以 16:00 bar 收盤價代理，詳見 L1 限制） |

### 核心架構

```
1 分鐘 bar（Polygon API）
    ↓ 過濾正規交易時段 09:30–16:00 ET
    ↓ 重採樣為 10 分鐘 bar
    ↓ 因果信號建構（PCA → 殘差 → OU → 修正 z-score，D=6 維）
    ↓ ESN 儲層（固定隨機矩陣，K=100，僅訓練線性讀出層）
    ↓ 滾動窗口預測 + 每日 CV 選 lambda
    ↓ 評估（MSFE / OOS R² / Diebold-Mariano / MCS）
```

### 與論文的對應

| 項目 | 論文 | 本實作 |
|------|------|--------|
| 資料來源 | AlgoSeek 1-min SIP | Polygon.io Basic（免費） |
| 測試期 | 2013 全年 | 2025 全年 |
| 調參期 | Q4 2012 | Q4 2024 |
| 預測視野 | 5 種（相同） | 5 種（相同） |
| ESN 儲層 | 固定隨機 | 固定隨機（相同） |
| 宇宙股票數 | 500 | 500（依美元成交量選取，L2） |

### 已知限制（正式揭露）

- **L1**：16:00 bar 收盤價≠真實收盤競價價格，影響所有 EOD 目標及 15:50 的 10min 預測
- **L2**：以 Q4 2024 平均美元成交量選股，非論文原始「500 大市值」標準
- **L3**：前向填充上限（3 bar）與缺失排除門檻（20%）為工程預設值，論文未明確規定

---

## 專案結構

```
RG/
├── config/
│   ├── default.yaml        # 全局設定（PCA 參數、路徑、評估設定）
│   ├── horizons.yaml       # 各預測視野的 ESN 超參數（論文 Table 1）
│   └── sample.yaml         # 樣本期間、交易時段、股票池設定
├── src/
│   ├── data/               # 資料抓取、時段過濾、重採樣、面板建構
│   ├── signals/            # PCA → 殘差 → OU → z-score 因果信號
│   ├── model/              # ESN 儲層、讀出層、基準模型
│   ├── training/           # 滾動窗口、每日 CV、超參數載入
│   ├── evaluation/         # MSFE、DM 測試、MCS
│   └── utils/              # 稀疏矩陣、譜半徑工具
├── scripts/
│   ├── fetch_massive_data.py     # 抓取並快取 Polygon 分鐘資料
│   ├── build_intraday_panel.py   # 建立 10 分鐘面板
│   ├── build_signals.py          # 建構因果信號
│   ├── train_esn.py              # 訓練 ESN（全視野）
│   ├── evaluate.py               # 計算指標與繪圖
│   └── run_all.py                # 一鍵執行全管線
├── tests/                  # 單元測試（19 項）
├── data/                   # 執行後自動建立
│   ├── raw/tickers/        # 每檔股票的 parquet 快取
│   ├── interim/            # 10 分鐘面板
│   └── processed/          # 信號陣列
└── results/                # 預測結果、指標、圖表
```

---

## 安裝

```bash
pip install -r requirements.txt
# 或
pip install numpy pandas scipy scikit-learn pyyaml matplotlib tqdm pyarrow requests
```

> 需要 Python 3.10+

---

## 快速啟動

### 步驟一：準備候選股票清單

建立 `data/raw/candidate_tickers.txt`，每行一個股票代碼（建議從 Russell 1000 或 S&P 500 成分股開始）：

```
AAPL
MSFT
NVDA
...
```

### 步驟二：一鍵執行完整管線

```bash
python scripts/run_all.py --api-key YOUR_POLYGON_API_KEY
```

可加 `--skip-*` 跳過已完成的步驟：

```bash
# 資料已抓取，從建立面板開始
python scripts/run_all.py --api-key KEY --skip-fetch

# 面板已建立，從信號開始
python scripts/run_all.py --api-key KEY --skip-fetch --skip-panel
```

### 步驟三：逐步執行（可選）

```bash
# 1. 抓取 Polygon 分鐘資料（依速率限制自動控速，12 秒/請求）
python scripts/fetch_massive_data.py \
    --api-key YOUR_KEY \
    --tickers-file data/raw/candidate_tickers.txt \
    --from-date 2024-09-01 \
    --to-date 2025-12-31

# 2. 建立 10 分鐘面板
python scripts/build_intraday_panel.py \
    --universe-file data/raw/universe.txt

# 3. 建構因果信號（PCA + OU）
python scripts/build_signals.py

# 4. 訓練 ESN（全部五個視野）
python scripts/train_esn.py

# 5. 評估並產生報告
python scripts/evaluate.py
```

---

## 輸出結果

執行完成後，`results/` 目錄下會產生：

| 檔案 | 說明 |
|------|------|
| `{horizon}_predictions.parquet` | 各視野的預測報酬率 (T×N) |
| `metrics.csv` | MSFE 與 OOS R²（各視野） |
| `dm.csv` | Diebold-Mariano 測試結果 |
| `plots/cum_msfe_{horizon}.png` | 累積 MSFE 比率走勢圖 |

---

## 執行測試

```bash
python -m pytest tests/ -v
```

19 項單元測試，涵蓋重採樣正確性、儲層譜半徑、OU 參數估計、ridge 迴歸。

---

## Polygon API 金鑰

1. 至 [polygon.io](https://polygon.io) 免費註冊
2. Basic 方案：每分鐘 5 次請求、2 年歷史資料、1 分鐘聚合
3. 本系統已內建速率控制（12 秒間隔）與本地快取，不會重複請求已快取資料

---

## 參考論文

> Ballarin, G., Capra, L., & Dellaportas, P. (2025).
> *Multi-Horizon Echo State Network Prediction of Intraday Stock Returns*.
> arXiv:2504.19623
