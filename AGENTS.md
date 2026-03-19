# AGENTS.md — RG_Stock 專案指引

## 專案概要
複現論文 arXiv:2504.19623 的日內 ESN 預測系統。Polygon.io 資料 → 10min bar → PCA/OU 信號 → ESN 訓練 → 評估。

## 執行管線（依序）
1. `scripts/fetch_massive_data.py` — 抓取 Polygon 1-min bar
2. `scripts/build_intraday_panel.py` — 建立 10-min 面板
3. **`scripts/build_signals_fast.py`** — 建構因果信號（多進程平行化，優先使用此版本）
   - 舊版 `scripts/build_signals.py` 為單執行緒，僅作備用
4. `scripts/train_esn.py` — 訓練 ESN
5. `scripts/evaluate.py` — 評估指標
6. `scripts/run_all.py --skip-fetch` — 一鍵執行（跳過抓取）

## 重要慣例
- 信號建構優先用 `build_signals_fast.py`（multiprocessing，比單執行緒快 ~7x）
- fast 版本輸出與原版 bit-identical（已驗證 MD5 一致）
- API 金鑰放 `config/massive_key.txt`
- 測試：`python -m pytest tests/ -v`
