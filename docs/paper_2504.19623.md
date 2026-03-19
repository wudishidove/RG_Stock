# Multi-Horizon Echo State Network Prediction of Intraday Stock Returns

**Authors:** Giovanni Ballarin (Universität Sankt Gallen), Jacopo Capra (UCL), Petros Dellaportas (UCL / Athens UEB)

**arXiv:** [2504.19623](https://arxiv.org/abs/2504.19623) (Computational Finance, Statistical Finance)

**arXiv:** 28 Apr 2025 | **Paper date:** April 29, 2025 | 27 pages, 3 figures, 7 tables

**DOI:** https://doi.org/10.48550/arXiv.2504.19623

**Keywords:** High-frequency data, reservoir computing, signal construction

**JEL:** G17, C45, C53

---

## Abstract

> Stock return prediction is a problem that has received much attention in the finance literature. In recent years, sophisticated machine learning methods have been shown to perform significantly better than "classical" prediction techniques. One downside of these approaches is that they are often very expensive to implement, for both training and inference, because of their high complexity. We propose a return prediction framework for intraday returns at multiple horizons based on Echo State Network (ESN) models, wherein a large portion of parameters are drawn at random and never trained. We show that this approach enjoys the benefits of recurrent neural network expressivity, inherently efficient implementation, and strong forecasting performance.

本文提出使用 Echo State Network (ESN) 模型來預測日內多時間尺度的股票報酬。ESN 是一類特殊的循環神經網路（RNN），其大部分參數在隨機初始化後固定不變，僅需訓練線性輸出層。作者展示此方法兼具 RNN 的表達能力、天然的高效實作特性、以及強勁的預測表現。

---

## 1. Introduction

時間序列預測在經濟和金融領域極為重要。在金融投資組合管理中，對股票特徵（價格、報酬、波動率、交易量）的預測是許多交易策略的核心。傳統上多依賴線性模型，近年來機器學習的發展大幅擴展了可用的預測方法。

在系統化交易產業中，多期投資組合建構方法越來越普遍，用以求解金融資產的最佳買賣決策（Meucci and Nicolosi, 2016），尤其在中頻交易情境中，投資經理會結合多種信號，各自對不同時間尺度的報酬有不同的預測能力（alpha decay profile）。

本文結合兩條研究脈絡，提出一種高效的非線性方法來預測日內多時間尺度報酬。具體而言，我們使用 **Echo State Network (ESN)**（Jaeger, 2001; Lukoševičius and Jaeger, 2009），一種特殊的循環神經網路，其內部狀態參數在隨機初始化後固定不變。使用線性輸出層時，ESN 可透過正則化線性回歸來擬合，遠比深度神經網路的訓練程序簡單。

ESN 在此框架中的三個關鍵優勢：
1. **表達能力**：ESN 是逼近非線性時間序列模型的通用工具
2. **多時間尺度適應性**：非線性狀態方程天然支援多步預測
3. **計算優勢**：可在短時間內解決高維數據問題

我們考慮兩個線性基準模型：(1) 基於信號向量的線性回歸；(2) 使用 Ridge 正則化的線性模型。ESN 方法先構建循環狀態方程，非線性地結合歷史信號，再用線性函數預測目標報酬。

在實證中，我們針對 500 檔美股在 5 個日內時間尺度（10min, 30min, 60min, 2hr, EOD）進行預測。結果顯示 ESN 在所有時間尺度上優於基準，10 分鐘尺度最高達 0.87% MSFE 降低，且對 ESN 隨機參數的抽樣具有高度穩健性。完整預測管線在一年數據上僅需數分鐘即可完成。

---

## 2. Related Literature

經典金融理論認為原始股票報酬在給定歷史資訊下不可預測。然而近期實證研究表明，在日內頻率上並非完全如此。Aït-Sahalia et al. (2022) 提供了毫秒級可預測性的廣泛證據。本文處理更溫和的設定（10 分鐘取樣），仍發現非微不足道的可預測性。

### 機器學習在金融中的應用

- **傳統 NN**：Olson and Mossman (2003)、Kwon and Moon (2007) 使用神經網路預測股票
- **DNN**：Yu and Yan (2020) 使用深度神經網路
- **LSTM**：Borovkova and Tsiamas (2019) 使用 LSTM 進行高頻股票分類
- **RNN**：Tölö (2020) 用 RNN 預測金融系統性危機（financial crises）
- **Bayesian NN**：Chandra and He (2021) 在 COVID-19 期間的股價預測
- **GAN**：Wang and Chen (2024) 提出因子型 GAN；Kim et al. (2024) 用 GAN 進行異常檢測以增強投資組合最佳化；Vuletić et al. (2024) 的 Fin-GAN 在 Sharpe ratio 上優於 LSTM 和 ARIMA
- **其他 ML 應用**：Abedin et al. (2021) 使用深度學習進行匯率預測；Kim et al. (2023)、Masi et al. (2023)、Nagy et al. (2023)、Acciaio et al. (2024)、Kwon and Lee (2024) 探索 GAN/VAE 合成金融數據生成；Cetingoz and Lehalle (2025) 對此方向提出理論批評

### 隨機權重神經網路與 Reservoir Computing

隨機參數在複雜非線性模型中的應用可追溯到 Lowe and Broomhead (1988) 和 Schmidt et al. (1992)。Huang et al. (2006) 提出 "extreme learning machine"；Rahimi and Recht (2008a,b) 發展了 "random kitchen sinks" 方法。近期 "lottery ticket" 假說（Frankle and Carbin, 2018; Malach et al., 2020; Ma et al., 2021; Sreenivasan et al., 2022）表明大型 NN 中小子網路在初始化時即可達到近似全網路的性能。Zhao et al. (2022) 和 Bolager et al. (2023) 進一步設計特殊取樣策略。RC 模型的計算成本可比完整 NN 訓練低數個數量級（Lohn and Musser, 2022）。

### Echo State Networks

ESN 是近年來發展的機器學習模型，旨在保持神經網路的廣泛有效性同時降低實作複雜度。ESN 已成功應用於（Sun et al., 2024）：
- 水位預測（Coulibaly, 2010）
- 電力負載預測（Deihimi and Showkati, 2012）
- 可再生能源發電預測（Hu et al., 2020）
- 深度 ESN 架構（Kim and King, 2020）
- 宏觀經濟預測（Ballarin et al., 2024）——混合頻率 ESN 達到 GDP 預測的 SOTA

金融應用方面文獻較少：Liu et al. (2018) 在金融數據上進行超參數優化研究；Trierweiler Ribeiro et al. (2021) 用 ESN 預測股票報酬波動率。其他基於隨機化的方法亦有探索，如 Akyildirim et al. (2023) 的隨機簽名方法和 Gonon (2023) 的理論保證。

---

## 3. Data and Setup

### 3.1 數據來源

日內股票數據來自 **AlgoSeek**，提供 2007 年 1 月至 2020 年 10 月的 1 分鐘 OHLC 資料，覆蓋所有美國交易所和 FINRA。

交易時段：
- **盤前**：4:00 AM – 9:29:59 AM (EST)
- **盤中**：9:30 AM – 4:00 PM (EST)
- **盤後**：4:00:01 PM – 8:00 PM (EST)

數據降採樣至 **10 分鐘解析度**。

### 3.2 Trading Setting

模擬日內交易簿，僅日內調倉，不持倉過夜。每 10 分鐘（9:30 AM 至 3:50 PM）進行投資組合再平衡和報酬預測。所有預測均為**重疊報酬**（overlapping returns）。

**5 個預測時間尺度：**

| 時間尺度 | 每日預測次數 | 說明 |
|---------|------------|------|
| 10 min  | 39         | 9:30 AM – 3:50 PM |
| 30 min  | 37         | |
| 60 min  | 34         | |
| 2 hr    | 28         | |
| EOD     | 39         | 從當前時點到收盤價 |

3:50 PM 提交收盤集合競價訂單，4:00 PM 以收盤競價價格完全出清持倉。

### 3.3 Modeling Setup

設交易宇宙中的股票索引為 $i = 1, \ldots, N$。對每檔股票，我們擁有：
- 未來報酬 $r_{t+h}^{(i)}$
- $D$ 維信號向量 $Z_t^{(i)} = (Z_{1,t}^{(i)}, \ldots, Z_{D,t}^{(i)})' \in \mathbb{R}^D$

預測目標為條件期望報酬：

$$\mathbb{E}\left[r_{t+h}^{(i)} \mid \mathcal{F}_t^{(i)}\right], \quad \text{where} \quad \mathcal{F}_t^{(i)} := \sigma(Z_t^{(i)}, Z_{t-1}^{(i)}, \ldots)$$

假設所有可觀測的預測資訊已完全嵌入信號中。對於缺失數據，使用 NaN 表示，實際截面大小為 $N_t \leq N$。

### 3.4 Signal Construction

信號構建的首要目標是將**股票特有特徵**與**一般市場結構**分離。遵循 Avellaneda and Lee (2010) 的方法，構建連續實值信號。

#### Step 1：因子分解

假設價格由連續時間隨機過程決定，報酬可分解為漂移、市場因子（系統性）和特異性成分：

$$\frac{dP_t^{(i)}}{P_t^{(i)}} = a^{(i)} dt + \sum_{j=1}^{J} b_j^{(i)} F_{j,t} + dU_t^{(i)}$$

其中：
- $a^{(i)}$：股票價格漂移
- $\{F_{j,t}\}_{j=1}^J$：$J$ 個市場風險因子的報酬
- $\{b_j^{(i)}\}_{j=1}^J$：因子載荷
- $dU_t^{(i)}$：股票特有的殘差成分

#### Step 2：OU 過程建模

殘差項建模為 **Ornstein-Uhlenbeck 過程**：

$$dU_t^{(i)} = \kappa^{(i)} (m^{(i)} - U_t^{(i)}) dt + \sigma^{(i)} dW_t$$

其中 $W_t$ 為標準 Wiener 過程，$\kappa^{(i)}$、$m^{(i)}$、$\sigma^{(i)}$ 為股票特有的緩慢變化參數。

#### Step 3：離散時間估計

**PCA 提取因子**：使用主成分分析從觀測報酬中提取市場因子（$J = 15$），然後回歸：

$$r_t^{(i)} = a^{(i)} + \sum_{j=1}^{J} b_j^{(i)} F_{j,t} + \upsilon_t^{(i)}$$

得到殘差 $\{\hat{\upsilon}_t^{(i)}\}$。

**離散化殘差**：對窗口大小 $P > 0$，定義：

$$\hat{U}_{P,t}^{(i)} := \sum_{s=t-P}^{t} \hat{\upsilon}_s^{(i)}$$

**AR(1) 回歸估計 OU 參數**：

$$\hat{U}_{P,t+1}^{(i)} = c_0^{(i)} + c_u^{(i)} \hat{U}_{P,t}^{(i)} + \eta_{P,t}^{(i)}$$

從而得到：
- $\kappa_P^{(i)} := -\log(c_u^{(i)})$
- $m_P^{(i)} := c_0^{(i)} / (1 - c_u^{(i)})$

#### Step 4：Z-score 信號

定義窗口大小 $P$ 的 z-score 信號：

$$z_{P,t}^{(i)} := \frac{\hat{U}_{P,t}^{(i)} - m_P^{(i)}}{\sigma_P^{(i)}}, \quad \text{where} \quad \sigma_P^{(i)} := \sqrt{\frac{\text{Var}(\eta_{P,t}^{(i)})}{2\kappa_P^{(i)}}}$$

考慮漂移的**修正 z-score**：

$$\tilde{z}_{P,t}^{(i)} := z_{P,t}^{(i)} - \frac{a^{(i)}}{\kappa_P^{(i)} \sigma_P^{(i)}}$$

#### 信號配置

- 15 個市場因子（解釋 >90% 的報酬變異）
- **6 個信號**，對應不同離散化窗口：$P \in \{10, 20, 30, 60, 100, 150\}$
- 核心信號使用 $P = 10$

### 3.5 Linear Return Prediction

線性基準模型：

$$r_{t+h}^{(i)} = \mu_{t,h} + \beta_{t,h}' Z_t^{(i)} + \epsilon_{t+h}^{(i)}$$

其中 $\mu_{t,h}$ 為市場平均報酬，$\beta_{t,h} \in \mathbb{R}^D$ 為特徵係數（假設跨截面不變）。線性預測為 $\hat{r}_{t+h}^{(i)} = \mu_{t,h} + \beta_{t,h}' Z_t^{(i)}$。

---

## 4. Echo State Networks

### 4.1 ESN Model Formulation

ESN 是一類循環神經網路，其內部參數隨機初始化後**固定不變**。核心狀態方程：

$$X_t := \alpha X_{t-1} + (1 - \alpha) \varphi(A X_t + C Z_t + b)$$

其中：
- $X_t \in \mathbb{R}^K$：循環狀態（reservoir state）
- $Z_t \in \mathbb{R}^D$：輸入信號
- $A \in \mathbb{R}^{K \times K}$：隨機遞迴矩陣
- $C \in \mathbb{R}^{K \times D}$：隨機輸入矩陣
- $b \in \mathbb{R}^K$：隨機偏置向量
- $\varphi$：逐元素非線性激活函數（e.g. hyperbolic tangent or ReLU）
- $\alpha \in [0, 1]$：洩漏率（leak rate）

**輸出層**（線性 readout）：

$$X_t \mapsto \mu_t + \theta_t' X_t$$

其中 $\theta_t \in \mathbb{R}^K$ 為輸出係數，$\mu_t \in \mathbb{R}$ 為截距。

#### 矩陣正規化

隨機抽取的矩陣 $A^*, C^*, b^*$ 經正規化處理，使用超參數：
- **譜半徑** $\rho \in [0, 1]$：控制遞迴矩陣的動態
- **輸入縮放** $\gamma > 0$：控制輸入影響
- **偏置縮放** $\zeta \geq 0$

正規化後的狀態方程：

$$X_t := \alpha X_{t-1} + (1 - \alpha) \varphi(\rho \bar{A} X_t + \gamma \bar{C} Z_t + \zeta \bar{b})$$

**Remark 4.2（隨機矩陣取樣）：** $A^*$ 的元素從**稀疏高斯分佈**抽取，$C^*$ 的元素從**稀疏均勻分佈**抽取（Ballarin et al., 2024）。所有 ESN 模型中，**$b$ 設為零向量**（不進行隨機抽取），因此不討論 $\zeta$ 的調參。實際使用的狀態方程為：

$$X_t := \alpha X_{t-1} + (1 - \alpha) \varphi(\rho \bar{A} X_t + \gamma \bar{C} Z_t)$$

$\|\cdot\|_A$ 使用最大絕對特徵值範數，使得 $\bar{A}$ 的譜半徑為 1。

### 4.2 Training and Estimation

ESN 訓練目標為最小化滾動窗口上的正則化經驗風險：

$$\mathcal{R}_{t,h}(\mu, \theta) := \frac{1}{M_t N} \sum_{s=t-\tau_h-M_t}^{t-\tau_h-1} \sum_{i=1}^{N} \left(r_{s+h}^{(i)} - \mu - \theta' X_s^{(i)}\right)^2 + \theta' \Lambda_{t,h} \theta$$

其中：
- $M_t$：訓練窗口大小
- $\tau_h$：緩衝區，防止多期報酬預測的資訊洩漏
- $\Lambda_{t,h}$：時變正則化矩陣，透過交叉驗證選取

由於目標為二次函數，最佳化有**封閉形式解**（ridge regression），無需迭代梯度下降。

#### 基準與對照線性模型

**Baseline**（無正則化，最小窗口 $M=1$）：

$$\mathcal{R}_{t,h}^{lin}(\mu, \beta) := \frac{1}{N} \sum_{i=1}^{N} \left(r_{t-\tau_h-1+h}^{(i)} - \mu - \beta' Z_{t-\tau_h-1}^{(i)}\right)^2$$

**Benchmark**（與 ESN 相同的窗口和正則化）：

$$\mathcal{R}_{t,h}^{reg\text{-}lin}(\mu, \beta) := \frac{1}{M_t N} \sum_{s=t-\tau_h-M_t}^{t-\tau_h-1} \sum_{i=1}^{N} \left(r_{s+h}^{(i)} - \mu - \beta' Z_s^{(i)}\right)^2 + \beta' \Lambda_{t,h} \beta$$

---

## 5. Forecasting Multi-horizon Returns

### ESN 模型配置

狀態維度固定為 $K = 100$。超參數在 2012 年 9 月至 12 月數據上使用 **Optuna** (Akiba et al., 2019) 優化以最小化 MSFE。

**Table 1: ESN Model Specifications**

| 超參數 | 10 min | 30 min | 1 hour | 2 hours | EOD |
|--------|--------|--------|--------|---------|-----|
| $K$（狀態維度）| 100 | 100 | 100 | 100 | 100 |
| $\alpha$（洩漏率）| 0.9 | 0.2 | 0 | 0 | 0 |
| $A$ sparsity | 0.15 | 0.15 | 0.15 | 0.65 | 0.35 |
| $\rho$（譜半徑）| 0.4 | 0.6 | 0.6 | 0.6 | 0 |
| $C$ sparsity | 0.95 | 0.55 | 0.75 | 0.85 | 0.25 |
| $\gamma$（輸入縮放）| 0.005 | 0.005 | 0.005 | 0.005 | 0.015 |

實驗使用 2013 年全年 12 個月數據（約 16,000 個數據點）。

### 5.1 Results

#### MSFE 定義

時間 $t$、預測尺度 $h$ 在 $N_t$ 檔股票上的均方預測誤差：

$$\text{MSFE}_{t,h} = \frac{1}{N_t} \sum_i \left(r_{t+h}^{(i)} - \hat{r}_{t+h}^{(i)}\right)^2$$

累積 MSFE：

$$\text{cuMSFE}_{t,h} = \frac{1}{t} \sum_s \text{MSFE}_{s,h}$$

#### Table 2: 2013 年 Q1-Q3 累積 MSFE

| 模型 | 10 min | 30 min | 1 hour | 2 hours | EOD |
|------|--------|--------|--------|---------|-----|
| Baseline | 0.0557 | 0.1402 | 0.2331 | 0.3704 | 0.7088 |
| Benchmark | 0.0557 [-0.0010%] | 0.1402 [-0.0007%] | 0.2331 [-0.0004%] | 0.3704 [-0.0003%] | 0.7088 [-0.0001%] |
| **ESN** | **0.0552 [-0.8775%]** | **0.1393 [-0.6059%]** | **0.2322 [-0.3890%]** | **0.3693 [-0.3023%]** | **0.7087 [-0.0148%]** |

- ESN 在 10 分鐘尺度達到最大 **0.87% MSFE 降低**
- 改善隨時間尺度增長而遞減
- 即使在 EOD，ESN 的改善仍比單純加正則化大**兩個數量級**

#### Table 3: 2013 年 Q1-Q3 預測 R²

| 模型 | 10 min | 30 min | 1 hour | 2 hours | EOD |
|------|--------|--------|--------|---------|-----|
| Baseline | -0.0766 | -0.1213 | -0.1128 | -0.1656 | -0.1760 |
| Benchmark | -0.0766 | -0.1213 | -0.1127 | -0.1656 | -0.1760 |
| **ESN** | **-0.0675** | **-0.1146** | **-0.1084** | **-0.1621** | **-0.1758** |

所有 R² 均為負值，但作者引用 Kelly et al. (2024) 指出：「負 R² 並不代表預測或策略效力差」。

### 5.2 Statistical Significance Testing

#### Table 4: Diebold-Mariano 檢定

| 比較 | 10 min | 30 min | 1 hour | 2 hours | EOD |
|------|--------|--------|--------|---------|-----|
| ESN vs. Baseline（統計量）| 56.2202 | 53.2776 | 44.5731 | 30.6386 | 2.566 |
| p-value | [0] | [0] | [0] | [0] | [0.0103] |
| ESN vs. Benchmark（統計量）| 56.1913 | 53.2462 | 44.5448 | 30.6180 | 2.558 |
| p-value | [0] | [0] | [0] | [0] | [0.0105] |

使用 **Newey-West HAC** 估計量進行 DM 檢定。等預測能力的虛無假設在所有時間尺度上被極度強烈地拒絕，僅 EOD 的 p 值略高於 1%。

#### Table 5: Model Confidence Set (MCS) 檢定

使用 $10^4$ 次 bootstrap 抽樣。

| 模型 | 10 min | 30 min | 1 hour | 2 hours | EOD |
|------|--------|--------|--------|---------|-----|
| Baseline | ○ (p=0) | ○ (p=0) | ○ (p=0) | ○ (p=0) | ○ (p=0.0011) |
| Benchmark | ○ (p=0) | ○ (p=0) | ○ (p=0) | ○ (p=0) | ● (p=0.5766) |
| **ESN** | **● (p=1)** | **● (p=1)** | **● (p=1)** | **● (p=1)** | **● (p=1)** |

（● = 包含在最佳模型集中；○ = 在 5% 水準下被排除）

ESN 在所有日內時間尺度上**穩健地主導**最佳模型集。EOD 時，ESN 和 Benchmark 均被納入。

### 5.3 Robustness to Parameter Sampling

以 100 組不同隨機種子重新抽樣 ESN 的 $A^*$ 和 $C^*$ 矩陣（保持 Table 1 的超參數不變），在 10 分鐘和 60 分鐘尺度上評估。

結果顯示 ESN 的預測增益**對隨機參數抽樣具有顯著的穩健性**，90% 和 50% 頻率帶非常緊密，表明在不同隨機初始化下性能高度一致。

---

## Appendix A: AlgoSeek OHLC Bar Schema

**Table 6: AlgoSeek OHLC bar table example**

| # | Field | Type | Description |
|---|-------|------|-------------|
| 1 | Date | YYYYMMDD | Trade Date |
| 2 | Ticker | String | Ticker Symbol |
| 3 | TimeBarStart | HHMM / HHMMSS / HHMMSSMMM | EST time-stamp |
| 4 | FirstTradePrice | Number | Price of first trade |
| 5 | HighTradePrice | Number | Price of highest trade |
| 6 | LowTradePrice | Number | Price of lowest trade |
| 7 | LastTradePrice | Number | Price of last trade |
| 8 | VolumeWeightPrice | Number | Trade volume weighted average price |
| 9 | Volume | Number | Total number of shares traded |
| 10 | TotalTrades | Number | Total number of trades |

## Appendix B: Signal Construction（PCA 細節）

從相關矩陣進行 PCA，得到特徵值和對應特徵向量 $v_{ij}$。每個索引 $j$ 的 eigenportfolio 和 eigenportfolio 報酬為：

$$Q_{ji} = \frac{v_{ij}}{\sigma^{(i)}}, \quad F_{jt} = \sum_{i=1}^{N} \frac{v_j^i}{\sigma^{(i)}} r_t^{(i)}, \quad j = 1, 2, \ldots, J$$

其中 $\sigma^{(i)}$ 為股票 $i$ 報酬的標準差。相關矩陣由標準化報酬（z-score）的成對協方差構成。

## Appendix C: Model and Estimation Details

### Missing Data and State Decay

當輸入信號缺失時，ESN 使用 **reservoir decay**：將 NaN 替換為零向量，繼續迭代狀態方程。例如對於信號序列 $\ldots, Z_{t-1}^{(i)}, \text{NaN}, Z_{t+1}^{(i)}, \ldots$，狀態計算為：

- $X_{t-1}^{(i)} := \alpha X_{t-2}^{(i)} + (1-\alpha)\varphi(\rho\bar{A}X_{t-2}^{(i)} + \gamma\bar{C}Z_{t-1}^{(i)})$（正常）
- $X_t^{(i)} := \alpha X_{t-1}^{(i)} + (1-\alpha)\varphi(\rho\bar{A}X_{t-1}^{(i)})$（缺失，輸入為零）
- $X_{t+1}^{(i)} := \alpha X_t^{(i)} + (1-\alpha)\varphi(\rho\bar{A}X_t^{(i)} + \gamma\bar{C}Z_{t+1}^{(i)})$（恢復正常）

只要 $\alpha \in [0,1)$ 且 $\bar{A}$ 的譜半徑小於 1，狀態會向零向量收縮（decay）。缺失期間的狀態**不納入**輸出係數的估計。

### Training Windows and Buffers

**Table 7: ESN training and cross-validation parameters**

| Training Parameter | 10 min | 30 min | 1 hour | 2 hours | EOD |
|-------------------|--------|--------|--------|---------|-----|
| $M_t$（Window size）| 30 min | 30 min | 1 hour | 1 hour | 1 day |
| $\tau_h$（Window buffer）| 10 min | 30 min | 1 hour | 2 hours | 1 day |
| CV frequency | 1 day | 1 day | 1 day | 1 day | 1 day |
| CV window size | 1 week | 1 week | 1 week | 1 week | 1 week |
| CV split ratio | 0.7 | 0.7 | 0.7 | 0.7 | 0.7 |

- **Window size**：在窗口長度和過多歷史數據對預測力的負面影響之間折衷
- **Window buffer**：$\tau_h \geq h$，選擇在給定預測時間尺度下的最小可能長度，防止資訊洩漏
- **Cross-validation**：每個交易日執行一次，使用過去一週的數據選取對角各向異性 ridge 正則化矩陣 $\Lambda_{t,h}$
- **Split ratio**：70% 訓練 / 30% 驗證，所有時間尺度一致

### ESN Hyperparameter Optimization

使用 Python 模型優化庫 **Optuna** (Akiba et al., 2019) 搜索 Table 1 中各時間尺度的 ESN 超參數。調優樣本僅包含 **2012 年 9 月至 12 月**的數據。使用與 Table 7 相同的訓練和交叉驗證參數，目標為最小化累積 MSFE。

---

## 6. Conclusion

本文提出了一種新方法，利用 ESN 模型在不同時間尺度預測日內股票報酬。

**主要發現：**
- ESN 方法在本質上是非參數的，能在不同時間尺度上顯著改善預測性能
- 唯一例外是 EOD 報酬的改善較為溫和
- DM 檢定在所有時間尺度上極強地拒絕等預測能力假設（僅 EOD 的 p 值略高於 1%）；MCS 檢定中 ESN 在日內時間尺度穩健主導最佳模型集，僅 EOD 時 ESN 與 Benchmark 同時被納入
- 儘管 ESN 模型本質上依賴隨機抽取的參數，預測增益在短期和中期日內時間尺度上對隨機性具有穩健性
- 計算效率極高：一年數據的完整預測管線僅需數分鐘

**未來研究方向：**
- 本文未討論 model aggregation 或 ensembles，未來可設計更精細的線上模型選擇機制，動態加權多個 ESN 以改善預測和獲利能力
- 本文將 ESN 輸出層限制為線性，未來可評估更靈活的輸出層（例如 shallow network）能否進一步提升性能

---

## References

1. Abedin, M. Z., Moon, M. H., Hassan, M. K., & Hajek, P. (2021). Deep learning-based exchange rate prediction during the COVID-19 pandemic. *Annals of Operations Research*, 1–52.
2. Acciaio, B., Eckstein, S., & Hou, S. (2024). Time-Causal VAE: Robust Financial Time Series Generator. Working Paper.
3. Aït-Sahalia, Y., Fan, J., Xue, L., & Zhou, Y. (2022). How and When are High-Frequency Stock Returns Predictable? *NBER Working Paper Series*.
4. Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A next-generation hyperparameter optimization framework. *KDD*.
5. Akyildirim, E., Gambara, M., Teichmann, J., & Zhou, S. (2023). Randomized Signature Methods in Optimal Portfolio Selection. Working Paper.
6. Aleti, S., Bollerslev, T., & Siggaard, M. (2025). Intraday Market Return Predictability Culled from the Factor Zoo. *Management Science*.
7. Anthony, M., & Bartlett, P. L. (2009). *Neural Network Learning: Theoretical Foundations*. Cambridge University Press.
8. Avellaneda, M., & Lee, J.-H. (2010). Statistical arbitrage in the US equities market. *Quantitative Finance*, 10(7), 761–782.
9. Ballarin, G., Dellaportas, P., Grigoryeva, L., Hirt, M., van Huellen, S., & Ortega, J.-P. (2024). Reservoir computing for macroeconomic forecasting with mixed-frequency data. *International Journal of Forecasting*, 40(3), 1206–1237.
10. Blake, C., Petrich, D., & Ulitsky, A. (2003). The Right Tool for the Job: Using Multiperiod Optimization in Transitions. *Trading*, 17(1), 33–37.
11. Bolager, E. L., Burak, I., Datar, C., Sun, Q., & Dietrich, F. (2023). Sampling weights of deep neural networks. *NeurIPS*, 36, 63075–63116.
12. Borovkova, S., & Tsiamas, I. (2019). An ensemble of LSTM neural networks for high-frequency stock market classification. *Journal of Forecasting*, 38(6), 600–619.
13. Cetingoz, A. R., & Lehalle, C.-A. (2025). Synthetic Data for Portfolios: A Throw of the Dice Will Never Abolish Chance. Working Paper.
14. Chandra, R., & He, Y. (2021). Bayesian neural networks for stock price forecasting before and during COVID-19 pandemic. *PLoS One*, 16(7), e0253217.
15. Coulibaly, P. (2010). Reservoir Computing approach to Great Lakes water level forecasting. *Journal of Hydrology*, 381(1), 76–88.
16. Deihimi, A., & Showkati, H. (2012). Application of echo state networks in short-term electric load forecasting. *Energy*, 39(1), 327–340.
17. Dhaene, G., & Wu, J. (2020). Incorporating overnight and intraday returns into multivariate GARCH volatility models. *Journal of Econometrics*, 217(2), 471–495.
18. Didisheim, A., Ke, S. B., Kelly, B. T., & Malamud, S. (2023). Complexity in Factor Pricing Models. *NBER Working Paper Series*.
19. Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*, 13(3), 253–263.
20. Frankle, J., & Carbin, M. (2018). The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks. *ICLR*.
21. Gârleanu, N., & Pedersen, L. H. (2013). Dynamic Trading with Predictable Returns and Transaction Costs. *Journal of Finance*, 68(6), 2309–2340.
22. Ghosn, J., & Bengio, Y. (1996). Multi-Task Learning for Stock Selection. *NeurIPS*, 9.
23. Gonon, L. (2023). Random Feature Neural Networks Learn Black-Scholes Type PDEs Without Curse of Dimensionality. *JMLR*, 24(189), 1–51.
24. Gonon, L., Grigoryeva, L., & Ortega, J.-P. (2020). Risk Bounds for Reservoir Computing. *JMLR*, 21(240), 1–61.
25. Gonon, L., Grigoryeva, L., & Ortega, J.-P. (2023). Approximation bounds for random neural networks and reservoir systems. *Annals of Applied Probability*, 33(1), 28–69.
26. Gonon, L., & Ortega, J. (2020). Reservoir Computing Universality With Stochastic Inputs. *IEEE TNNLS*, 31(1), 100–112.
27. Grigoryeva, L., & Ortega, J.-P. (2018). Echo state networks are universal. *Neural Networks*, 108, 495–508.
28. Grinold, R. (2007). Dynamic Portfolio Analysis. *Journal of Portfolio Management*, 34(1), 12–26.
29. Grinold, R. (2010). Signal Weighting. *Journal of Portfolio Management*, 36(4), 24–34.
30. Hansen, P. R., Lunde, A., & Nason, J. M. (2011). The Model Confidence Set. *Econometrica*, 79(2), 453–497.
31. Harvey, D., Leybourne, S., & Newbold, P. (1997). Testing the equality of prediction mean squared errors. *International Journal of Forecasting*, 13(2), 281–291.
32. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. Springer.
33. Hu, H., Wang, L., & Lv, S.-X. (2020). Forecasting energy consumption and wind power generation using deep echo state network. *Renewable Energy*, 154, 598–613.
34. Huang, G.-B., Chen, L., & Siew, C.-K. (2006). Universal approximation using incremental constructive feedforward networks with random hidden nodes. *IEEE TNN*, 17(4), 879–892.
35. Jaeger, H. (2001). The "echo state" approach to analysing and training recurrent neural networks. *Technical Report 34*, German National Research Center for Information Technology.
36. Kelly, B., Malamud, S., & Zhou, K. (2024). The Virtue of Complexity in Return Prediction. *Journal of Finance*, 79(1), 459–503.
37. Kim, J. H., Kim, S., Lee, Y., Kim, W. C., & Fabozzi, F. J. (2024). Enhancing mean-variance portfolio optimization through GANs-based anomaly detection. *Annals of Operations Research*, 1–28.
38. Kim, S., Hong, J., & Lee, Y. (2023). A GANs-Based Approach for Stock Price Anomaly Detection and Investment Risk Management. *ACM*, 1–9.
39. Kim, T., & King, B. R. (2020). Time series prediction using deep echo state networks. *Neural Computing and Applications*, 32(23), 17769–17787.
40. Kumbure, M. M., Lohrmann, C., Luukka, P., & Porras, J. (2022). Machine learning techniques and data for stock market forecasting: A literature review. *Expert Systems with Applications*, 197, 116659.
41. Kwon, S., & Lee, Y. (2024). Can GANs Learn the Stylized Facts of Financial Time Series? *ACM*, 126–133.
42. Kwon, Y.-K., & Moon, B.-R. (2007). A Hybrid Neurogenetic Approach for Stock Forecasting. *IEEE TNN*, 18(3), 851–864.
43. Liu, J., Sun, T., Luo, Y., Fu, Q., Cao, Y., Zhai, J., & Ding, X. (2018). Financial Data Forecasting Using Optimized Echo State Network. In *Neural Information Processing (ICONIP)*, Lecture Notes in Computer Science, 138–149. Springer.
44. Lohn, A. J., & Musser, M. (2022). AI and Compute: How Much Longer Can Computing Power Drive Artificial Intelligence Progress? Technical report, Center for Security and Emerging Technology.
45. Lowe, D., & Broomhead, D. (1988). Multivariable functional interpolation and adaptive networks. *Complex Systems*, 2(3), 321–355.
46. Lukoševičius, M. (2012). A Practical Guide to Applying Echo State Networks. In *Neural Networks: Tricks of the Trade*, Lecture Notes in Computer Science, 659–686. Springer.
47. Lukoševičius, M., & Jaeger, H. (2009). Reservoir computing approaches to recurrent neural network training. *Computer Science Review*, 3(3), 127–149.
48. Ma, X., Yuan, G., Shen, X., Chen, T., Chen, X., Chen, X., Liu, N., Qin, M., Liu, S., Wang, Z., & Wang, Y. (2021). Sanity Checks for Lottery Tickets: Does Your Winning Ticket Really Win the Jackpot? *NeurIPS*, 34, 12749–12760.
49. Malach, E., Yehudai, G., Shalev-Schwartz, S., & Shamir, O. (2020). Proving the Lottery Ticket Hypothesis: Pruning is All You Need. *ICML*, 6682–6691.
50. Markowitz, H. (1952). Portfolio Selection. *Journal of Finance*, 7(1), 77–91.
51. Masi, G., Prata, M., Conti, M., Bartolini, N., & Vyetrenko, S. (2023). On Correlated Stock Market Time Series Generation. *ACM*, 524–532.
52. Meucci, A., & Nicolosi, M. (2016). Dynamic portfolio management with views at multiple horizons. *Applied Mathematics and Computation*, 274, 495–518.
53. Nagy, P., Frey, S., Sapora, S., Li, K., Calinescu, A., Zohren, S., & Foerster, J. (2023). Generative AI for End-to-End Limit Order Book Modelling. *ACM*, 91–99.
54. Nicholas Refenes, A., Zapranis, A., & Francis, G. (1994). Stock performance modeling using neural networks: A comparative study with regression models. *Neural Networks*, 7(2), 375–388.
55. Olson, D., & Mossman, C. (2003). Neural network forecasts of Canadian stock returns using accounting ratios. *International Journal of Forecasting*, 19(3), 453–465.
56. Qian, E., Sorensen, E. H., & Hua, R. (2007). Information Horizon, Portfolio Turnover, and Optimal Alpha Models. *Journal of Portfolio Management*, 34(1), 27–40.
57. Rahimi, A., & Recht, B. (2008a). Uniform approximation of functions with random bases. In *46th Annual Allerton Conference on Communication, Control, and Computing*, 555–561. IEEE.
58. Rahimi, A., & Recht, B. (2008b). Weighted sums of random kitchen sinks: Replacing minimization with randomization in learning. *NeurIPS*, 21.
59. Saad, E. W., Prokhorov, D. V., & Wunsch, D. C. (1998). Comparative study of stock trend prediction using time delay, recurrent and probabilistic neural networks. *IEEE TNN*, 9(6), 1456–1470.
60. Schmidt, W. F., Kraaijveld, M. A., & Duin, R. P. W. (1992). Feedforward neural networks with random weights. In *11th IAPR International Conference on Pattern Recognition*. IEEE.
61. Sirignano, J., & Cont, R. (2019). Universal features of price formation in financial markets: Perspectives from deep learning. *Quantitative Finance*, 19(9), 1449–1459.
62. Sneddon, L. (2008). The Tortoise and the Hare: Portfolio Dynamics for Active Managers. *Journal of Investing*, 2008(4), 106–111.
63. Sreenivasan, K., Sohn, J.-y., Yang, L., Grinde, M., Nagle, A., Wang, H., Xing, E., Lee, K., & Papailiopoulos, D. (2022). Rare Gems: Finding Lottery Tickets at Initialization. *NeurIPS*, 35, 14529–14540.
64. Sun, C., Song, M., Cai, D., Zhang, B., Hong, S., & Li, H. (2024). A Systematic Review of Echo State Networks From Design to Application. *IEEE Transactions on Artificial Intelligence*, 5(1), 23–37.
65. Tanaka, G., et al. (2019). Recent advances in physical reservoir computing: A review. *Neural Networks*, 115, 100–123.
66. Tölö, E. (2020). Predicting systemic financial crises with recurrent neural networks. *Journal of Financial Stability*, 49, 100746.
67. Trierweiler Ribeiro, G., Alves Portela Santos, A., Cocco Mariani, V., & dos Santos Coelho, L. (2021). Novel hybrid model based on echo state neural network applied to the prediction of stock price return volatility. *Expert Systems with Applications*, 184, 115490.
68. Vuletić, M., Prenzel, F., & Cucuringu, M. (2024). Fin-GAN: Forecasting and classifying financial time series via generative adversarial networks. *Quantitative Finance*, 24(2), 175–199.
69. Wang, J., & Chen, Z. (2024). Factor-GAN: Enhancing stock price prediction and factor investment with Generative Adversarial Networks. *PLoS One*, 19(6), e0306094.
70. Yu, P., & Yan, X. (2020). Stock price prediction based on deep neural networks. *Neural Computing and Applications*, 32(6), 1609–1628.
71. Zhang, G., Eddy Patuwo, B., & Y. Hu, M. (1998). Forecasting with artificial neural networks: The state of the art. *International Journal of Forecasting*, 14(1), 35–62.
72. Zhang, T. (2023). *Mathematical Analysis of Machine Learning Algorithms*. Cambridge University Press.
73. Zhao, J., Schaefer, F. T., & Anandkumar, A. (2022). ZerO Initialization: Initializing Neural Networks with only Zeros and Ones. *Transactions on Machine Learning Research*.
