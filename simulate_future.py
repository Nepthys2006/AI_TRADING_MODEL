# ================
# (INFO) SIMULATE 2026 WEALTH
# ================
import numpy as np
import pandas as pd
import pickle
import warnings
import os
import sys
import random
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.models import load_model

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ================
# (INFO) CONFIGURATION
# ================
BIG_BROTHER_MODEL = 'big_brother_v2.keras'
LITTLE_BROTHER_MODEL = 'little_brother_v2.keras'
SCALER_SMALL = 'scaler_small.pkl'
CLOSE_SCALER = 'close_scaler.pkl'

# Filenames to check
TEST_5M_FILENAME = 'TEST_5m.csv'
TEST_1H_FILENAME = 'TEST_1h.csv'

# ================
# (INFO) BACKTEST SETTINGS
# ================
LOOKBACK_BIG = 168
LOOKBACK_SMALL = 48
INITIAL_CAPITAL = 200.0  # User Request

FEATURE_COLS = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'Returns', 'EMA_20', 'EMA_50', 'EMA_Diff',
    'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
    'BB_Width', 'BB_Position', 'ATR', 'Volatility'
]

# ================
# (INFO) HELPER FUNCTIONS
# ================
# ... (Technical indicators reduced for brevity, assuming data has them or we calculate same as before)
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

def find_file(filename):
    path1 = os.path.join('DATA_SET', filename)
    if os.path.exists(path1): return path1
    if os.path.exists(filename): return filename
    return None

def load_and_preprocess(filename, filter_year=None):
    filepath = find_file(filename)
    if not filepath:
        print(f"Skipping {filename} (not found)")
        return None
    print(f"Loading {filepath}...")
    df = pd.read_csv(filepath, sep=';')
    df['Date'] = pd.to_datetime(df['Date'], format='%Y.%m.%d %H:%M')
    
    if filter_year:
        df_filtered = df[df['Date'].dt.year == filter_year]
        if not df_filtered.empty:
            df = df_filtered
            
    df = df.sort_values('Date').reset_index(drop=True)
    df = add_technical_indicators(df)
    df = df.dropna().reset_index(drop=True)
    float_cols = df.select_dtypes(include=['float64']).columns
    df[float_cols] = df[float_cols].astype('float32')
    return df

def hybrid_backtest(df_5m, little_predictions, df_1h, big_predictions):
    # Returns list of trade pct returns
    trades = []
    
    # Simple filters (matching predict_2025.py)
    RSI_BUY_MAX = 70
    RSI_SELL_MIN = 30
    MIN_THRESHOLD = 0.0015
    BIAS_WINDOW_SIZE = 50
    bias_window = []
    
    # Align
    df_5m_subset = df_5m.iloc[LOOKBACK_SMALL-1 : LOOKBACK_SMALL-1+len(little_predictions)].copy()
    df_5m_subset['Predicted_Raw'] = little_predictions
    
    df_1h_subset = df_1h.iloc[LOOKBACK_BIG-1 : LOOKBACK_BIG-1+len(big_predictions)].copy()
    df_1h_subset['Predicted_Raw'] = big_predictions
    df_1h_subset['Hour'] = df_1h_subset['Date'].dt.floor('H')
    hourly_trend = df_1h_subset.set_index('Hour')[['Close', 'Predicted_Raw', 'EMA_20', 'EMA_50']].to_dict('index')
    
    for i in range(len(df_5m_subset) - 1):
        row = df_5m_subset.iloc[i]
        current_price = row['Close']
        actual_next = df_5m_subset.iloc[i + 1]['Close']
        
        # Bias
        raw_pred_5m = row['Predicted_Raw']
        current_bias = current_price - raw_pred_5m
        bias_window.append(current_bias)
        if len(bias_window) > BIAS_WINDOW_SIZE: bias_window.pop(0)
        avg_bias = sum(bias_window) / len(bias_window)
        pred_5m_corrected = raw_pred_5m + avg_bias
        
        # Trend
        hour_key = row['Date'].floor('H')
        trend = "NEUTRAL"
        if hour_key in hourly_trend:
            h_data = hourly_trend[hour_key]
            pred_bullish = (h_data['Predicted_Raw'] - h_data['Close']) > -avg_bias
            ema_bullish = (h_data['Close'] > h_data['EMA_50']) and (h_data['EMA_20'] > h_data['EMA_50'])
            ema_bearish = (h_data['Close'] < h_data['EMA_50']) and (h_data['EMA_20'] < h_data['EMA_50'])
            if pred_bullish and ema_bullish: trend = "BULLISH"
            elif not pred_bullish and ema_bearish: trend = "BEARISH"
            
        # Entry
        pred_change = (pred_5m_corrected - current_price) / current_price
        entry_signal = "WAIT"
        if pred_change > MIN_THRESHOLD: entry_signal = "BUY"
        elif pred_change < -MIN_THRESHOLD: entry_signal = "SELL"
        
        # RSI
        if entry_signal == "BUY" and row['RSI'] > RSI_BUY_MAX: entry_signal = "WAIT"
        if entry_signal == "SELL" and row['RSI'] < RSI_SELL_MIN: entry_signal = "WAIT"
        
        # Execute
        if trend == "BULLISH" and entry_signal == "BUY":
            trades.append((actual_next - current_price) / current_price)
        elif trend == "BEARISH" and entry_signal == "SELL":
            trades.append((current_price - actual_next) / current_price)
            
    return trades

def monte_carlo_simulation(trades_history, num_simulations=50, years=2, initial_balance=2000.0):
    # Estimate trades per year from sample
    # (Assuming sample is roughly 1 year or we just take len(trades) as 1 "unit" of time for simplicity
    # but user wants 2026-2027 (2 years). 
    # Let's assume the provided TEST data is ~1 year. If it's less, this might under-estimate frequency.
    # We will assume trades_history represents the density of trades for the duration of the test file.
    
    trades_per_year = len(trades_history) 
    total_trades_future = trades_per_year * years
    
    simulation_results = []
    
    for _ in range(num_simulations):
        balance = initial_balance
        equity_curve = [balance]
        
        # Bootstrap resampling
        # Randomly choose returns from historical performance
        for _ in range(total_trades_future):
            # Pick a random trade return from history
            r_trade = random.choice(trades_history)
            
            # Apply return (compounded)
            balance = balance * (1 + r_trade)
            equity_curve.append(balance)
            
        simulation_results.append(equity_curve)
        
    return simulation_results, total_trades_future

# ================
# (INFO) MAIN
# ================
if __name__ == "__main__":
    print("=" * 60)
    print("  WEALTH SIMULATION (2026)")
    print(f"  Starting Capital: ${INITIAL_CAPITAL}")
    print("=" * 60)

    # 1. Load Models & Data (2025 Baseline)
    try:
        big_model = load_model(BIG_BROTHER_MODEL)
        little_model = load_model(LITTLE_BROTHER_MODEL)
        with open(SCALER_SMALL, 'rb') as f: scaler = pickle.load(f)
        with open(CLOSE_SCALER, 'rb') as f: close_scaler = pickle.load(f)
    except:
        print("Error loading models. Make sure .keras and .pkl files are present.")
        sys.exit(1)
        
    df_5m = load_and_preprocess(TEST_5M_FILENAME)
    df_1h = load_and_preprocess(TEST_1H_FILENAME)
    
    if df_5m is None or df_1h is None:
        print("Data missing. Cannot benchmark.")
        sys.exit(1)

    # 2. Run Baseline Backtest
    print("\nBenchmarking on Test Data to map probability...")
    
    # Predict 5M
    df_5m_s = df_5m.copy()
    df_5m_s[FEATURE_COLS] = scaler.transform(df_5m[FEATURE_COLS])
    p_5m = little_model.predict(tf.keras.utils.timeseries_dataset_from_array(
        df_5m_s[FEATURE_COLS].values, None, LOOKBACK_SMALL, batch_size=256, shuffle=False
    ), verbose=0)
    pred_5m = close_scaler.inverse_transform(p_5m).flatten()
    
    # Predict 1H
    df_1h_s = df_1h.copy()
    df_1h_s[FEATURE_COLS] = scaler.transform(df_1h[FEATURE_COLS])
    p_1h = big_model.predict(tf.keras.utils.timeseries_dataset_from_array(
        df_1h_s[FEATURE_COLS].values, None, LOOKBACK_BIG, batch_size=64, shuffle=False
    ), verbose=0)
    pred_1h = close_scaler.inverse_transform(p_1h).flatten()
    
    # Extract Trades
    historical_trades = hybrid_backtest(df_5m, pred_5m, df_1h, pred_1h)
    
    if not historical_trades:
        print("No trades found in baseline data. Cannot simulate.")
        sys.exit()

    win_rate = len([x for x in historical_trades if x > 0]) / len(historical_trades)
    avg_return = np.mean(historical_trades) * 100
    print(f"  Baseline Metrics:")
    print(f"  - Win Rate: {win_rate*100:.1f}%")
    print(f"  - Avg Return/Trade: {avg_return:.2f}%")
    print(f"  - Total Samples: {len(historical_trades)}")

    # 3. Simulate Future (Monte Carlo)
    print("\nSimulating 2026 (Monte Carlo Projection)...")
    sim_curves, num_future_trades = monte_carlo_simulation(historical_trades, num_simulations=50, years=1, initial_balance=INITIAL_CAPITAL)
    
    # Calculate Percentiles
    # Transpose to get distribution at each step
    steps = len(sim_curves[0])
    median_curve = []
    p90_curve = []
    p10_curve = []
    
    for i in range(steps):
        step_vals = [curve[i] for curve in sim_curves]
        median_curve.append(np.median(step_vals))
        p90_curve.append(np.percentile(step_vals, 90))
        p10_curve.append(np.percentile(step_vals, 10))
        
    final_median = median_curve[-1]
    final_p90 = p90_curve[-1]
    final_p10 = p10_curve[-1]

    print("\n" + "=" * 60)
    print("  SIMULATION RESULTS (End of 2026)")
    print("=" * 60)
    print(f"  Starting Balance:   ${INITIAL_CAPITAL:,.2f}")
    print(f"  Projected Median:   ${final_median:,.2f} (Likely)")
    print(f"  Optimistic (90%):   ${final_p90:,.2f} (Lucky Streak)")
    print(f"  Pessimistic (10%):  ${final_p10:,.2f} (Bad Streak)")
    print("=" * 60)

    # 4. Visualization
    fig = make_subplots(rows=1, cols=1, subplot_titles=(f"Wealth Projection 2026 (Start: ${INITIAL_CAPITAL})"))
    
    # Plot all simulation traces (faint)
    x_axis = list(range(steps))
    for curve in sim_curves[:20]: # Show first 20 lines
        fig.add_trace(go.Scatter(x=x_axis, y=curve, mode='lines', line=dict(color='rgba(255, 255, 255, 0.1)'), showlegend=False))
        
    # Plot Median (Green - "Real" expectation)
    fig.add_trace(go.Scatter(x=x_axis, y=median_curve, mode='lines', name='Expected Growth (Median)', line=dict(color='#00FF00', width=3)))
    
    # Plot Bands (Red - "Risk/Variance")
    fig.add_trace(go.Scatter(x=x_axis, y=p90_curve, mode='lines', name='Optimistic Top', line=dict(color='#FF0000', dash='dot')))
    fig.add_trace(go.Scatter(x=x_axis, y=p10_curve, mode='lines', name='Pessimistic Bottom', line=dict(color='#FF0000', dash='dot')))
    
    fig.update_layout(
        template='plotly_dark',
        title='Projected Account Balance (2026)',
        xaxis_title='Trade Count (Approx 1 Year)',
        yaxis_title='Balance ($)',
        height=600
    )
    
    print("Displaying Graph...")
    fig.show()
