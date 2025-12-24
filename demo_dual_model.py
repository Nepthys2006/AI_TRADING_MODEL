# ================
# (INFO) HYBRID DUAL-TIMEFRAME DEMO
# ================
# Uses the same logic as Trading_AI_model.ipynb:
# - Big Brother: Determines overall market trend (BULLISH/BEARISH)
# - Little Brother: Confirms precise trade entries
# - Only trades when BOTH timeframes agree
# ===========================================

import numpy as np
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ================
# (INFO) CONFIGURATION
# ================
BIG_BROTHER_MODEL = 'big_brother_v2.keras'
LITTLE_BROTHER_MODEL = 'little_brother_v2.keras'
SCALER_SMALL = 'scaler_small.pkl'
CLOSE_SCALER = 'close_scaler.pkl'

TEST_5M = 'TEST_5m.csv'
TEST_1H = 'TEST_1h.csv'

LOOKBACK_BIG = 168   # 1 week of hourly data
LOOKBACK_SMALL = 48  # 4 hours of 5-min data

FEATURE_COLS = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'Returns', 'EMA_20', 'EMA_50', 'EMA_Diff',
    'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
    'BB_Width', 'BB_Position', 'ATR', 'Volatility'
]

# ================
# (INFO) TECHNICAL INDICATORS
# ================
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices, fast=12, slow=26, signal=9):
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculate_bollinger_bands(prices, period=20, std_dev=2):
    middle = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    return upper, middle, lower

def calculate_atr(high, low, close, period=14):
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def add_technical_indicators(df):
    df['Returns'] = df['Close'].pct_change()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA_Diff'] = df['EMA_20'] - df['EMA_50']
    df['RSI'] = calculate_rsi(df['Close'], 14)
    df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = calculate_macd(df['Close'])
    df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = calculate_bollinger_bands(df['Close'])
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    df['ATR'] = calculate_atr(df['High'], df['Low'], df['Close'])
    df['Volatility'] = df['Returns'].rolling(window=20).std()
    return df

def load_and_preprocess(filepath):
    print(f"Loading {filepath}...")
    df = pd.read_csv(filepath, sep=';')
    df['Date'] = pd.to_datetime(df['Date'], format='%Y.%m.%d %H:%M')
    df = df.sort_values('Date').reset_index(drop=True)
    df = add_technical_indicators(df)
    df = df.dropna().reset_index(drop=True)
    float_cols = df.select_dtypes(include=['float64']).columns
    df[float_cols] = df[float_cols].astype('float32')
    print(f"Loaded {len(df)} rows")
    return df

# ================
# (INFO) HYBRID BACKTEST LOGIC
# ================
def hybrid_backtest(df_5m, little_predictions, df_1h, big_predictions, lookback_5m, lookback_1h, initial_capital=200.0):
    """
    Backtest with PHASE 2 Optimizations & Paper Trading:
    1. Dynamic Bias Correction.
    2. Stricter Threshold & EMA Filter.
    3. Paper Trading ($200 Start).
    """
    results = {
        'total_signals': 0,
        'confirmed_trades': 0,
        'wins': 0,
        'losses': 0,
        'buy_signals': 0,
        'sell_signals': 0,
        'wait_signals': 0,
        'filtered_signals': 0,
        'initial_balance': initial_capital,
        'final_balance': initial_capital,
        'net_profit': 0.0
    }
    
    trades = []
    equity_curve = [initial_capital]
    signals_log = []
    
    # --- Bias Correction Setup ---
    bias_window = [] 
    BIAS_WINDOW_SIZE = 50 
    
    # --- Filters (PHASE 2) ---
    RSI_BUY_MAX = 70
    RSI_SELL_MIN = 30
    MIN_THRESHOLD = 0.0015
    
    # Align Data
    df_5m_subset = df_5m.iloc[lookback_5m-1:lookback_5m-1+len(little_predictions)].copy()
    df_5m_subset['Predicted_Raw'] = little_predictions
    
    df_1h_subset = df_1h.iloc[lookback_1h-1:lookback_1h-1+len(big_predictions)].copy()
    df_1h_subset['Predicted_Raw'] = big_predictions
    
    # 1H Mapping (Added EMAs for Trend Filter)
    df_1h_subset['Hour'] = df_1h_subset['Date'].dt.floor('H')
    hourly_trend = df_1h_subset.set_index('Hour')[
        ['Close', 'Predicted_Raw', 'EMA_20', 'EMA_50']
    ].to_dict('index')
    
    current_balance = initial_capital
    
    # --- MAIN LOOP ---
    for i in range(len(df_5m_subset) - 1):
        row = df_5m_subset.iloc[i]
        current_price = row['Close']
        actual_next = df_5m_subset.iloc[i + 1]['Close']
        
        # 1. Update Bias
        raw_pred_5m = row['Predicted_Raw']
        current_bias = current_price - raw_pred_5m
        bias_window.append(current_bias)
        if len(bias_window) > BIAS_WINDOW_SIZE:
            bias_window.pop(0)
        avg_bias = sum(bias_window) / len(bias_window)
        
        pred_5m_corrected = raw_pred_5m + avg_bias
        
        # 2. Big Brother Trend (STRICTER EMA FILTER)
        hour_key = row['Date'].floor('H')
        trend = "NEUTRAL"
        
        if hour_key in hourly_trend:
            h_data = hourly_trend[hour_key]
            pred_bullish = (h_data['Predicted_Raw'] - h_data['Close']) > -avg_bias
            ema_bullish = (h_data['Close'] > h_data['EMA_50']) and (h_data['EMA_20'] > h_data['EMA_50'])
            ema_bearish = (h_data['Close'] < h_data['EMA_50']) and (h_data['EMA_20'] < h_data['EMA_50'])
            
            if pred_bullish and ema_bullish:
                trend = "BULLISH"
            elif not pred_bullish and ema_bearish:
                trend = "BEARISH"
                
        # 3. Little Brother Entry
        pred_change = (pred_5m_corrected - current_price) / current_price
        
        entry_signal = "WAIT"
        if pred_change > MIN_THRESHOLD:
            entry_signal = "BUY"
        elif pred_change < -MIN_THRESHOLD:
            entry_signal = "SELL"
            
        # 4. RSI Filter
        if entry_signal == "BUY" and row['RSI'] > RSI_BUY_MAX:
            entry_signal = "WAIT"
            results['filtered_signals'] += 1
        if entry_signal == "SELL" and row['RSI'] < RSI_SELL_MIN:
            entry_signal = "WAIT"
            results['filtered_signals'] += 1
            
        # === FINAL DECISION ===
        results['total_signals'] += 1
        final_signal = "WAIT"
        trade_return = 0.0
        
        if trend == "BULLISH" and entry_signal == "BUY":
            final_signal = "BUY NOW"
            results['buy_signals'] += 1
            results['confirmed_trades'] += 1
            
            trade_return = (actual_next - current_price) / current_price
            trades.append(trade_return)
            
            if actual_next > current_price:
                results['wins'] += 1
            else:
                results['losses'] += 1
                
        elif trend == "BEARISH" and entry_signal == "SELL":
            final_signal = "SELL NOW"
            results['sell_signals'] += 1
            results['confirmed_trades'] += 1
            
            trade_return = (current_price - actual_next) / current_price
            trades.append(trade_return)
            
            if actual_next < current_price:
                results['wins'] += 1
            else:
                results['losses'] += 1
        
        elif entry_signal != "WAIT":
            results['wait_signals'] += 1
            
        # Update Paper Trading Balance
        if trade_return != 0:
            current_balance = current_balance * (1 + trade_return)
        
        equity_curve.append(current_balance)
            
        signals_log.append({
            'date': row['Date'],
            'price': current_price,
            'pred': pred_5m_corrected,
            'trend': trend,
            'final': final_signal
        })

    # Metrics
    results['final_balance'] = current_balance
    results['net_profit'] = current_balance - initial_capital
    
    if results['confirmed_trades'] > 0:
        results['win_rate'] = (results['wins'] / results['confirmed_trades']) * 100
        returns_series = pd.Series(trades)
        cumulative_returns = (1 + returns_series).cumprod()
        results['total_return'] = (cumulative_returns.iloc[-1] - 1) * 100
        results['sharpe_ratio'] = np.sqrt(252 * 288) * returns_series.mean() / returns_series.std() if returns_series.std() > 0 else 0
    else:
        results['win_rate'] = 0
        results['total_return'] = 0
        results['sharpe_ratio'] = 0
    
    return results, trades, signals_log, avg_bias, equity_curve

# ================
# (INFO) MAIN
# ================
if __name__ == "__main__":
    print("=" * 60)
    print("  HYBRID DUAL-TIMEFRAME DEMO (Phase 2 Optimized)")
    print("  Stricter Filters | EMA Alignment | Bias Correction")
    print("=" * 60)
    
    # ... (Model Loading Logic - Condensed for brevity, functionality maintained)
    # ===== LOAD MODELS =====
    print("\n[1/5] Loading models...")
    try:
        big_model = load_model(BIG_BROTHER_MODEL)
        print(f"  ✓ Big Brother (1H) loaded")
    except Exception as e:
        print(f"  ✗ Big Brother not found: {e}")
        big_model = None
    
    try:
        little_model = load_model(LITTLE_BROTHER_MODEL)
        print(f"  ✓ Little Brother (5M) loaded")
    except Exception as e:
        print(f"  ✗ Little Brother not found: {e}")
        little_model = None
    
    if not big_model or not little_model:
        exit()
    
    with open(SCALER_SMALL, 'rb') as f:
        scaler = pickle.load(f)
    with open(CLOSE_SCALER, 'rb') as f:
        close_scaler = pickle.load(f)
    print("  ✓ Scalers loaded")
    
    # ===== LOAD DATA =====
    print("\n[2/5] Loading 2025 test data...")
    df_5m = load_and_preprocess(TEST_5M)
    try:
        df_1h = load_and_preprocess(TEST_1H)
    except:
        print(" Using resampled 1H data...")
        pass 

    # ===== RUN PREDICTIONS =====
    print("\n[3/5] Running predictions...")
    
    # Little Brother (5M)
    print("  Running Little Brother (5M)...")
    df_5m_scaled = df_5m.copy()
    df_5m_scaled[FEATURE_COLS] = scaler.transform(df_5m[FEATURE_COLS])
    
    predict_5m = tf.keras.utils.timeseries_dataset_from_array(
        data=df_5m_scaled[FEATURE_COLS].values, targets=None,
        sequence_length=LOOKBACK_SMALL, batch_size=256, shuffle=False
    )
    pred_5m_scaled = little_model.predict(predict_5m, verbose=0)
    little_predictions = close_scaler.inverse_transform(pred_5m_scaled).flatten()
    print(f"  ✓ {len(little_predictions)} predictions")
    
    # Big Brother (1H)
    print("  Running Big Brother (1H)...")
    df_1h_scaled = df_1h.copy()
    df_1h_scaled[FEATURE_COLS] = scaler.transform(df_1h[FEATURE_COLS])
    
    predict_1h = tf.keras.utils.timeseries_dataset_from_array(
        data=df_1h_scaled[FEATURE_COLS].values, targets=None,
        sequence_length=LOOKBACK_BIG, batch_size=64, shuffle=False
    )
    pred_1h_scaled = big_model.predict(predict_1h, verbose=0)
    big_predictions = close_scaler.inverse_transform(pred_1h_scaled).flatten()
    print(f"  ✓ {len(big_predictions)} predictions")
    
    # ===== HYBRID BACKTEST =====
    print("\n[4/5] Running Phase 2 Optimization & Simulation...")
    
    metrics, trades, signals, final_bias, equity_curve = hybrid_backtest(
        df_5m, little_predictions,
        df_1h, big_predictions,
        LOOKBACK_SMALL, LOOKBACK_BIG,
        initial_capital=200.0
    )
    
    # ===== DISPLAY RESULTS =====
    print("\n" + "=" * 60)
    print("  HYBRID BACKTEST RESULTS (OPTIMIZED)")
    print("=" * 60)
    
    print(f"\n  📊 Signal Analysis:")
    print(f"     Total Candles:      {metrics['total_signals']}")
    print(f"     Confirmed Trades:   {metrics['confirmed_trades']} (Reduced Noise)")
    print(f"     Filtered Signals:   {metrics['filtered_signals']}")
    
    print(f"\n  💰 Trading Performance:")
    print(f"     BUY Trades:         {metrics['buy_signals']}")
    print(f"     SELL Trades:        {metrics['sell_signals']}")
    print(f"     Wins:               {metrics['wins']}")
    print(f"     Losses:             {metrics['losses']}")
    print(f"     Win Rate:           {metrics['win_rate']:.2f}%")
    print(f"     Total Return:       {metrics['total_return']:.2f}%")
    
    print(f"\n  💵 Paper Trade Simulation ($200 Start):")
    print(f"     Initial Balance:    ${metrics['initial_balance']:.2f}")
    print(f"     Final Balance:      ${metrics['final_balance']:.2f}")
    profit_sign = "+" if metrics['net_profit'] >= 0 else "-"
    print(f"     Net Profit:         {profit_sign}${abs(metrics['net_profit']):.2f}")
    print("=" * 60)
    
    print("=" * 60)
    print("\n" + "=" * 60)
    print("  CURRENT TRADING SIGNAL (Bias Corrected)")
    print("=" * 60)
    current_price = df_5m['Close'].iloc[-1]
    
    # 5M Prediction logic...
    latest_5m = df_5m_scaled.iloc[-LOOKBACK_SMALL:][FEATURE_COLS].values[None, ...]
    pred_5m_raw = close_scaler.inverse_transform(little_model.predict(latest_5m, verbose=0))[0][0]
    target_5m_corrected = pred_5m_raw + final_bias
    
    # 1H Prediction logic...
    latest_1h = df_1h_scaled.iloc[-LOOKBACK_BIG:][FEATURE_COLS].values[None, ...]
    pred_1h_raw = close_scaler.inverse_transform(big_model.predict(latest_1h, verbose=0))[0][0]
    current_price_1h = df_1h['Close'].iloc[-1]
    ema_20_1h = df_1h['EMA_20'].iloc[-1]
    ema_50_1h = df_1h['EMA_50'].iloc[-1]
    
    ema_bullish = (current_price_1h > ema_50_1h) and (ema_20_1h > ema_50_1h)
    ema_bearish = (current_price_1h < ema_50_1h) and (ema_20_1h < ema_50_1h)
    pred_bullish_1h = (pred_1h_raw + final_bias) > current_price_1h
    
    if pred_bullish_1h and ema_bullish:
        trend = "BULLISH (Strong Pattern)"
        trend_icon = "🟢"
    elif not pred_bullish_1h and ema_bearish:
        trend = "BEARISH (Strong Pattern)"
        trend_icon = "🔴"
    else:
        trend = "NEUTRAL (EMA Mismatch)"
        trend_icon = "⚪"
        
    print(f"\n  [Big Brother] 1H Structure: {trend_icon} {trend}")
    print(f"     Price: ${current_price_1h:.2f} | EMA50: ${ema_50_1h:.2f}")
    
    move_pct = (target_5m_corrected - current_price) / current_price
    min_thresh = 0.0015
    if move_pct > min_thresh: entry = "BUY"
    elif move_pct < -min_thresh: entry = "SELL"
    else: entry = "WAIT (Weak Move)"
        
    print(f"\n  [Little Brother] 5M Signal: {entry}")
    print(f"     Current: ${current_price:.2f}")
    print(f"     Target:  ${target_5m_corrected:.2f}")
    
    print(f"\n  {'─' * 40}")
    if "BULLISH" in trend and entry == "BUY": final_sig = "✅ BUY NOW"
    elif "BEARISH" in trend and entry == "SELL": final_sig = "✅ SELL NOW"
    else: final_sig = "⏸️ WAIT / FILTERED"
    print(f"  >> FINAL SIGNAL: {final_sig}")
    print("=" * 60)
    
    print("\n  📊 Charts displayed above")
    
    # Update Fig 1
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                       subplot_titles=(f'Hybrid Performance ({metrics["win_rate"]:.1f}%)', 
                                     f'Account Balance ($200 Start) -> ${metrics["final_balance"]:.2f}', 
                                     'Signals'))
    
    dates = df_5m['Date'].iloc[LOOKBACK_SMALL-1:LOOKBACK_SMALL-1+len(little_predictions)]
    real = df_5m['Close'].iloc[LOOKBACK_SMALL-1:LOOKBACK_SMALL-1+len(little_predictions)]
    
    fig.add_trace(go.Scatter(x=dates, y=real, mode='lines', name='Price', line=dict(color='#00FF00')), row=1, col=1)
    
    # Plot equity curve against trade count
    
    fig.add_trace(go.Scatter(x=list(range(len(equity_curve))), y=equity_curve, 
                             mode='lines', name='Balance ($)', 
                             line=dict(color='#00BFFF', width=2), fill='tozeroy'), row=2, col=1)
        
    fig.update_layout(template='plotly_dark', height=900, title='Phase 2 Results + Paper Trade Simulation')
    fig.show()

