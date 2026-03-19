# 論文 vs 程式碼差異分析

## Context

專案目標是複製 arXiv 2504.19623 (Ballarin, Capra, Dellaportas) 的 Multi-Horizon ESN 日內股票報酬預測。目前模型效果不佳（所有 horizon 的 OOS R² 均為大幅負值）。以下是論文與程式碼之間所有顯著差異的完整清單，按影響程度排序。已知的資料集/股票數量差異不在分析範圍內。

---

## A. BUG（程式邏輯錯誤）

### A1. evaluate.py EOD 目標完全錯誤 [已在 plan.md]

| | 論文 | 程式碼 |
|---|---|---|
| EOD 目標 | log(C_eod / C_t) — 從 t 到收盤 | log(C_{t+1} / C_t) — 僅 1 根 bar |

- **檔案**: `scripts/evaluate.py:51-54`
- **原因**: `h = max(cfg.horizon_steps, 1)` → EOD 的 horizon_steps=-1 → h=1
- **影響**: EOD 評估指標完全無意義，R² ≈ -1.43 的根因

### A2. evaluate.py 未遮罩隔夜報酬 [已在 plan.md]

| | 論文 | 程式碼 |
|---|---|---|
| 跨 session 報酬 | 訓練時遮罩為 NaN | 評估時包含在 actual 中 |

- **檔案**: `scripts/evaluate.py:54` — `diff(h).shift(-h)` 不區分日內/隔夜
- **影響**: h=12 (2hr) 約 31% 的 bar 受汙染；h=6 (60min) 約 15%

### A3. oos_r2 NaN mask 錯位 [已在 plan.md]

- **檔案**: `src/evaluation/metrics.py:72-73`
- **問題**: 第 72 行過濾 `y` 後，第 73 行的 `~np.isnan(y)` 是對已縮短的 y 操作，mask 長度與 yhat 不一致
- **影響**: 所有 R² 計算結果都被破壞

### A4. OLS NaN 過濾過嚴（0% vs 20%）[已在 plan.md]

| | 論文 | 程式碼 |
|---|---|---|
| 缺失容忍 | per-stock 使用非 NaN row 做 OLS | 整檔股票只要有 1 個 NaN 就跳過 |

- **檔案**: `src/signals/residuals.py:37` — `np.any(np.isnan(returns), axis=0)`
- **影響**: 有效 universe 從 ~500 降到 ~197 檔

### A5. PCA NaN 過濾過嚴（同 A4 但在 PCA 層）[plan.md 未提及]

| | 論文 | 程式碼 |
|---|---|---|
| PCA 輸入 | 應使用通過 20% 門檻的所有股票 | `~np.any(np.isnan(X), axis=0)` 再排除有任何 NaN 的股票 |

- **檔案**: `src/signals/pca_factors.py:36-37`
- **影響**: PCA 因子從更少股票估計，降低因子品質。但因 PCA 只需 J+1=16 檔股票，如果仍有足夠清潔股票則影響相對較小
- **修正**: 用 `np.nanmean` 填補 NaN 或改用 drop NaN rows 而非 drop NaN columns

### A6. session_boundary 傳入但未使用 [已在 plan.md]

| | 論文 | 程式碼 |
|---|---|---|
| 隔夜 bar | 不應納入信號估計 | session_boundary 是函式參數但函式內從未使用 |

- **檔案**: `src/signals/pipeline.py:28,62-67` — 每日首根 bar 含隔夜跳空（~5.7x 日內波動），污染 PCA/OLS/OU
- **影響**: 高。隔夜 jump 嚴重偏置 PCA 第一主成分、OLS 載荷、OU 參數

---

## B. 顯著實作差異

### B1. 訓練視窗公式多扣了 h [已在 plan.md]

| | 論文 | 程式碼 |
|---|---|---|
| train_end | `t - τ_h`（Python exclusive index） | `t - τ_h - h + 1` |
| 30min 最後訓練 index | t-4 | t-6（早了 2 根） |
| 2hr 最後訓練 index | t-13 | t-24（早了 11 根!） |

- **檔案**: `src/training/rolling_window.py:48-49`
- **原因**: `train_end = t - cfg.buffer_bars - h + 1` 多減了 `h-1`
- **影響**: 中長 horizon 使用過時的訓練資料。以 2hr (M=6, h=12) 為例，程式用 t-29 到 t-24 的資料（~5 小時前），論文用 t-18 到 t-13（~2-3 小時前），差距巨大
- **修正**: `train_end = t - cfg.buffer_bars`（因為 τ_h ≥ h 已保證無洩漏）

### B2. Scalar Ridge vs Diagonal Anisotropic Ridge [已在 plan.md]

| | 論文 | 程式碼 |
|---|---|---|
| 正則化矩陣 | Λ_{t,h} = diag(λ_1, ..., λ_K) 每天 CV 選取 | λ·I 單一 scalar |

- **檔案**: `src/model/readout.py:6-7`（註解已標記 "Phase B (future)"）
- **影響**: 不同 reservoir 維度有不同 scale/信噪比，scalar 會過度正則化有用維度、不足正則化雜訊維度
- **修正**: 標準化 reservoir features 後再做 scalar ridge，等效於 Λ_k ∝ 1/σ²_k

### B3. 殘差聚合 Off-by-One [plan.md 未提及]

| | 論文 | 程式碼 |
|---|---|---|
| 公式 | Û_{P,t} = Σ_{s=t-P}^{t} v̂_s（P+1 項） | sum_{s=t-P+1}^{t}（P 項） |

- **檔案**: `src/signals/ou_estimation.py:16-37`
- **影響**: 低。每個聚合窗少一項，系統性但微小的偏差

---

## C. 中等/不確定差異

### C1. PCA lookback 長度

| | 論文 | 程式碼 |
|---|---|---|
| PCA 窗口 | 未明確指定（Avellaneda & Lee 2010 用 60 交易日 = ~2340 bars） | 390 bars（~10 交易日） |

- **檔案**: `scripts/build_signals.py` 預設參數
- **影響**: 較短窗口可能導致 PCA 因子不穩定，但也更靈活響應市場變化

### C2. Ridge 正規化慣例差異

| | 論文 | 程式碼 |
|---|---|---|
| 損失函數 | (1/MN) Σ(...)² + θ'Λθ（data term 除以 MN） | X'Xβ + λIβ（無正規化） |

- **影響**: 有效 λ 差 M×N 倍（~1500 for 10min）。CV 可補償，但預設 λ=1e-4 可能偏離最佳值

### C3. CV Lambda 搜索範圍

- **程式碼**: `[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]`
- **論文**: 未明確指定
- **影響**: 考慮 C2 的正規化差異，搜索範圍可能需要調整

---

## D. 環境差異（非程式碼問題）

| 差異 | 論文 | 程式碼 |
|---|---|---|
| 資料來源 | AlgoSeek（含收盤競價） | Polygon.io（16:00 bar 作為收盤代理） |
| 時間段 | 2013 年（後 QE 牛市） | 2025 年（不同市場環境） |
| Universe | 市值排名 top-500 | 美元交易量排名 top-500 |
| 超參調優 | 2012 Q4 Optuna 搜索 | 直接使用論文 Table 1 數值 |

---

## 影響排序與修正優先級

| 排名 | 問題 | 影響程度 | 修正成本 | 已在 plan.md |
|------|------|---------|---------|-------------|
| 1 | A1+A2: evaluate.py 目標錯誤 | 致命 | 低（不需重訓） | ✓ |
| 2 | A3: oos_r2 NaN mask 錯位 | 高 | 低 | ✓ |
| 3 | A6: session_boundary 未使用 | 高 | 中（需重建 signals） | ✓ |
| 4 | A4: OLS NaN 過嚴 | 高 | 中（需重建 signals） | ✓ |
| 5 | B1: 訓練視窗多扣 h | 高 | 中（需重訓） | ✓ |
| 6 | B2: Scalar vs Anisotropic Ridge | 中 | 中（需重訓） | ✓ |
| 7 | A5: PCA NaN 過嚴 | 中低 | 低 | ✗ 新發現 |
| 8 | B3: 殘差聚合 off-by-one | 低 | 低 | ✗ 新發現 |
| 9 | C1: PCA lookback 長度 | 不確定 | 低 | ✗ |
| 10 | C2/C3: Ridge 正規化慣例 | 低 | 低 | ✗ |

---

## 新發現（plan.md 未涵蓋）

### 需修正：

**A5. PCA NaN 過嚴** — `src/signals/pca_factors.py:36-37`
```python
# 現行：整欄刪除
valid_cols = ~np.any(np.isnan(X), axis=0)

# 建議改為：per-row drop NaN 或用 column mean 填補
nan_mask = np.isnan(X)
col_means = np.nanmean(X, axis=0)
X_imputed = np.where(nan_mask, col_means, X)
```

**B3. 殘差聚合 off-by-one** — `src/signals/ou_estimation.py:33-36`
```python
# 現行：P 項（t-P+1 到 t）
U[P - 1:] = cs[P - 1:]
U[P:] -= cs[:T - P]

# 論文：P+1 項（t-P 到 t），改為：
U[P:] = cs[P:]
U[P:] -= cs[:T - P]
# 注意：此修改也使得前 P 行（而非 P-1 行）為 NaN
```

### 可調參數（非必要修正）：

- C1: PCA lookback 可作為實驗參數，嘗試 390 / 780 / 1170 / 2340 bars
- C2/C3: CV grid 在修正 B2 (feature 標準化) 後可能需要重新調整範圍

---

## 結論

現有 `plan.md` 已涵蓋 6 個最關鍵的問題（排名 1-6）。本分析額外發現 2 個實作差異（A5、B3）和 2 個可調參數差異（C1、C2/C3）。

**建議執行順序**：按現有 plan.md Phase A → B → C 執行，同時在 Phase B 加入 A5（PCA NaN 修正）和 B3（聚合 off-by-one 修正），因為它們都在 signal 建構階段且成本低。
