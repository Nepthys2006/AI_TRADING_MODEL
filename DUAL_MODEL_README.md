# 🤖 Dual-Timeframe AI Trading System

**Author:** Daryl James Padogdog  
**Version:** 2.0 (Phase 2 Optimized)  
**Asset:** Gold (XAU/USD)

---

## 📋 System Overview

This is a **hybrid dual-timeframe trading system** that uses two LSTM neural networks working together:

| Model | Timeframe | Purpose | Lookback |
|-------|-----------|---------|----------|
| **Big Brother** | 1-Hour (1H) | Determines overall market TREND (Bullish/Bearish) | 168 candles (1 week) |
| **Little Brother** | 5-Minute (5M) | Finds precise ENTRY points | 48 candles (4 hours) |

### Core Logic
```
TRADE ONLY when BOTH models AGREE:
- Big Brother says BULLISH + Little Brother says BUY → ✅ BUY NOW
- Big Brother says BEARISH + Little Brother says SELL → ✅ SELL NOW
- Otherwise → ⏸️ WAIT
```

---

## 📁 File Structure

```
train/
├── train_big_brother.py    # Trains the 1H trend model
├── train_little_brother.py # Trains the 5M entry model
demo_dual_model.py          # Runs backtest & generates signals
```

---

## 🔧 Step-by-Step: How Each File Works

### 1️⃣ `train_big_brother.py` (Run First)

**Purpose:** Train the 1-Hour trend detection model.

**Step-by-step process:**
1. **Load Data** - Reads `TRAIN_1h.csv` (semicolon-separated)
2. **Add Technical Indicators:**
   - EMA 20/50 and their difference
   - RSI (14-period)
   - MACD, Signal Line, Histogram
   - Bollinger Bands (Width & Position)
   - ATR (14-period)
   - Volatility (20-period rolling std)
3. **Normalize** - MinMaxScaler (0 to 1)
4. **Create Sequences** - Lookback of 168 candles (1 week of hourly data)
5. **Build Model:**
   ```
   Input (168, 17 features)
       ↓
   Bidirectional LSTM (128 units) + Dropout 0.3
       ↓
   Bidirectional LSTM (64 units) + Dropout 0.3
       ↓
   GlobalAveragePooling1D
       ↓
   Dense (64) → Dense (32) → Dense (1) [Output: Predicted Close]
   ```
6. **Train** - 100 epochs with EarlyStopping (patience=10)
7. **Save** - `big_brother_v2.keras`

**Output:** `big_brother_v2.keras`

---

### 2️⃣ `train_little_brother.py` (Run Second)

**Purpose:** Train the 5-Minute entry signal model.

**Step-by-step process:**
1. **Load Data** - Reads `TRAIN_5m.csv`
2. **Add Technical Indicators** - Same 17 features as Big Brother
3. **Normalize** - MinMaxScaler (0 to 1)
4. **Save Scalers:**
   - `scaler_small.pkl` - For feature scaling during prediction
   - `close_scaler.pkl` - For inverse-transforming predictions back to price
5. **Create Sequences** - Lookback of 48 candles (4 hours of 5M data)
6. **Build Model** - Same architecture as Big Brother
7. **Train** - 100 epochs with EarlyStopping
8. **Save** - `little_brother_v2.keras`

**Outputs:** 
- `little_brother_v2.keras`
- `scaler_small.pkl`
- `close_scaler.pkl`

---

### 3️⃣ `demo_dual_model.py` (Run for Backtest & Signals)

**Purpose:** Load both models, run backtest on 2025 data, display current signal.

**Step-by-step process:**

1. **Load Models & Scalers**
2. **Load Test Data** - `TEST_5m.csv` and `TEST_1h.csv`
3. **Generate Predictions** - Both models predict next Close price
4. **Run Hybrid Backtest with Phase 2 Optimizations:**

   **Optimizations Applied:**
   - **Dynamic Bias Correction:** Tracks prediction error over last 50 candles and corrects
   - **EMA Alignment Filter:** Trend must be confirmed by EMA 20 > EMA 50 (bullish) or vice versa
   - **RSI Filter:** No BUY if RSI > 70, No SELL if RSI < 30
   - **Minimum Threshold:** Predicted move must be > 0.15% to trigger signal

5. **Backtest Logic (per candle):**
   ```python
   # 1. Get Big Brother trend
   if pred_bullish AND ema_20 > ema_50 AND price > ema_50:
       trend = "BULLISH"
   elif pred_bearish AND ema_20 < ema_50 AND price < ema_50:
       trend = "BEARISH"
   else:
       trend = "NEUTRAL"
   
   # 2. Get Little Brother entry
   pred_change = (predicted_price - current_price) / current_price
   if pred_change > 0.0015:
       entry = "BUY"
   elif pred_change < -0.0015:
       entry = "SELL"
   
   # 3. Apply RSI filter
   if entry == "BUY" and RSI > 70: entry = "WAIT"
   if entry == "SELL" and RSI < 30: entry = "WAIT"
   
   # 4. Final decision
   if trend == "BULLISH" and entry == "BUY":
       → EXECUTE BUY
   elif trend == "BEARISH" and entry == "SELL":
       → EXECUTE SELL
   ```

6. **Paper Trading Simulation** - Starts with $200, compounds returns
7. **Display Results** - Win rate, total return, final balance
8. **Generate Current Signal** - Based on latest data

---

## 📊 Features Used (17 Total)

| # | Feature | Description |
|---|---------|-------------|
| 1 | Open | Opening price |
| 2 | High | Highest price |
| 3 | Low | Lowest price |
| 4 | Close | Closing price |
| 5 | Volume | Trading volume |
| 6 | Returns | Percentage change |
| 7 | EMA_20 | 20-period EMA |
| 8 | EMA_50 | 50-period EMA |
| 9 | EMA_Diff | EMA_20 - EMA_50 |
| 10 | RSI | Relative Strength Index |
| 11 | MACD | MACD line |
| 12 | MACD_Signal | Signal line |
| 13 | MACD_Hist | Histogram |
| 14 | BB_Width | Bollinger Band width |
| 15 | BB_Position | Position within bands (0-1) |
| 16 | ATR | Average True Range |
| 17 | Volatility | 20-period rolling std of returns |

---

## ⚙️ Current Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| LOOKBACK_BIG | 168 | 1 week of 1H candles |
| LOOKBACK_SMALL | 48 | 4 hours of 5M candles |
| LSTM Units | 128 → 64 | Bidirectional |
| Dropout | 0.3 / 0.2 | After LSTM / Dense |
| Learning Rate | 0.001 | Adam optimizer |
| Batch Size | 64 | Training |
| MIN_THRESHOLD | 0.15% | Minimum predicted move |
| RSI_BUY_MAX | 70 | Filter overbought |
| RSI_SELL_MIN | 30 | Filter oversold |
| BIAS_WINDOW | 50 | Candles for bias correction |

---

## ❓ Questions for the AI Council

Please analyze and suggest improvements for:

1. **Model Architecture:**
   - Is Bidirectional LSTM the best choice? Would Transformer/Attention help?
   - Is GlobalAveragePooling ideal or should we use the last hidden state?
   - Should we add more layers or use residual connections?

2. **Feature Engineering:**
   - Are there missing indicators that could improve predictions?
   - Should we add market session info (Asian/London/NY)?
   - Would order flow or sentiment data help?

3. **Signal Logic:**
   - Is the 0.15% threshold optimal?
   - Should RSI thresholds be dynamic based on volatility?
   - Should we add ADX filter for trend strength?

4. **Risk Management:**
   - Currently no stop-loss or take-profit in the backtest. How to add?
   - Should position sizing vary based on signal confidence?
   - How to handle consecutive losses?

5. **Training Process:**
   - Is 85/15 train/val split appropriate?
   - Should we use walk-forward validation instead?
   - Would transfer learning from pretrained models help?

6. **Data Issues:**
   - The models predict raw Close price - should they predict returns instead?
   - Is MinMaxScaler the best choice or should we use StandardScaler?
   - How to handle overnight gaps and weekends?

---

## 📈 Current Performance (Backtest Results)

*Run the demo to see latest metrics*

Target: **>60% Win Rate** with sustainable profitability.

---

## 🚀 How to Run

```bash
# 1. Train Big Brother (Google Colab recommended)
python train/train_big_brother.py

# 2. Train Little Brother
python train/train_little_brother.py

# 3. Run Demo/Backtest
python demo_dual_model.py
```

---

## 📞 Contact

Created by **Daryl James Padogdog**
