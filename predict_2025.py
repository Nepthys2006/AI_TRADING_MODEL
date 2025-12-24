# ================
# (INFO) PREDICT 2025 STOCK MOVEMENT
# ================
import numpy as np
import pandas as pd
import pickle
import warnings
import os
import sys
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.models import load_model

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ================
# (INFO) CONFIGURATION
# ================
# Adjust paths if models are in a specific directory
BIG_BROTHER_MODEL = 'big_brother_v2.keras'
LITTLE_BROTHER_MODEL = 'little_brother_v2.keras'
SCALER_SMALL = 'scaler_small.pkl'
CLOSE_SCALER = 'close_scaler.pkl'

# User provided paths
# We will check both 'DATA_SET/...' and just 'filename.csv' for Colab compatibility
TEST_5M_FILENAME = 'TEST_5m.csv'
TEST_1H_FILENAME = 'TEST_1h.csv'

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

def find_file(filename):
    """Helper to find file in DATA_SET or current directory"""
    # 1. Check DATA_SET/filename
    path1 = os.path.join('DATA_SET', filename)
    if os.path.exists(path1):
        return path1
    
    # 2. Check ./filename (Colab root)
    path2 = filename
    if os.path.exists(path2):
        return path2
        
    return None

def load_and_preprocess(filename, filter_year=None):
    filepath = find_file(filename)
    if not filepath:
        print(f"ERROR: File not found: {filename}")
        print(f"  Checked: DATA_SET/{filename} and ./{filename}")
        print(f"  Current Directory: {os.getcwd()}")
        return None
        
    print(f"Loading {filepath}...")
    df = pd.read_csv(filepath, sep=';')
    df['Date'] = pd.to_datetime(df['Date'], format='%Y.%m.%d %H:%M')
    
    if filter_year:
        print(f"Filtering for Year {filter_year}...")
        df_filtered = df[df['Date'].dt.year == filter_year]
        if df_filtered.empty:
            print(f"WARNING: No data found for year {filter_year} in {filepath}")
            print(f"  Available Date Range: {df['Date'].min()} to {df['Date'].max()}")
            print("  -> Proceeding with ALL data in file (assuming this is the intended test set).")
        else:
            print(f"Found {len(df_filtered)} rows for {filter_year}")
            df = df_filtered
            
    df = df.sort_values('Date').reset_index(drop=True)
    df = add_technical_indicators(df)
    df = df.dropna().reset_index(drop=True)
    float_cols = df.select_dtypes(include=['float64']).columns
    df[float_cols] = df[float_cols].astype('float32')
    return df

# ================
# (INFO) HYBRID BACKTEST LOGIC
# ================
def hybrid_backtest(df_5m, little_predictions, df_1h, big_predictions, lookback_5m, lookback_1h):
    results = {
        'total_signals': 0, 'confirmed_trades': 0, 'wins': 0, 'losses': 0,
        'buy_signals': 0, 'sell_signals': 0, 'filtered_signals': 0
    }
    
    trades = []
    
    # --- Bias Correction Setup ---
    bias_window = [] 
    BIAS_WINDOW_SIZE = 50 
    
    # --- Filters ---
    RSI_BUY_MAX = 70
    RSI_SELL_MIN = 30
    MIN_THRESHOLD = 0.0015
    
    # Align Data
    # Ensure indices match the predictions length
    df_5m_subset = df_5m.iloc[lookback_5m-1 : lookback_5m-1+len(little_predictions)].copy()
    df_5m_subset['Predicted_Raw'] = little_predictions
    
    df_1h_subset = df_1h.iloc[lookback_1h-1 : lookback_1h-1+len(big_predictions)].copy()
    df_1h_subset['Predicted_Raw'] = big_predictions
    
    # 1H Mapping
    df_1h_subset['Hour'] = df_1h_subset['Date'].dt.floor('H')
    hourly_trend = df_1h_subset.set_index('Hour')[
        ['Close', 'Predicted_Raw', 'EMA_20', 'EMA_50']
    ].to_dict('index')
    
    print("\nSimulating Trading Logic on Data...")
    
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
        
        # 2. Big Brother Trend
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
        
        if trend == "BULLISH" and entry_signal == "BUY":
            results['buy_signals'] += 1
            results['confirmed_trades'] += 1
            if actual_next > current_price:
                results['wins'] += 1
            else:
                results['losses'] += 1
            trades.append((actual_next - current_price) / current_price)
                
        elif trend == "BEARISH" and entry_signal == "SELL":
            results['sell_signals'] += 1
            results['confirmed_trades'] += 1
            if actual_next < current_price:
                results['wins'] += 1
            else:
                results['losses'] += 1
            trades.append((current_price - actual_next) / current_price)

    # Metrics
    if results['confirmed_trades'] > 0:
        results['win_rate'] = (results['wins'] / results['confirmed_trades']) * 100
    else:
        results['win_rate'] = 0
        
    return results, trades

# ================
# (INFO) MAIN
# ================
if __name__ == "__main__":
    print("=" * 60)
    print("  2025 STOCK PREDICTION & ANALYSIS")
    print("=" * 60)
    
    # Check if models exist (Check current dir for Colab support)
    def check_model_exists(name):
        if os.path.exists(name): return True
        # Check subfolder if needed, but usually models are in root or specified path
        return False
        
    if not check_model_exists(BIG_BROTHER_MODEL) or not check_model_exists(LITTLE_BROTHER_MODEL):
        print("CRITICAL ERROR: Models not found in current directory.")
        print(f"Looking for: {BIG_BROTHER_MODEL} and {LITTLE_BROTHER_MODEL}")
        print("Please upload the .keras model files.")
        sys.exit(1)

    # Load artifacts
    print("Loading models and scalers...")
    try:
        big_model = load_model(BIG_BROTHER_MODEL)
        little_model = load_model(LITTLE_BROTHER_MODEL)
        
        with open(SCALER_SMALL, 'rb') as f:
            scaler = pickle.load(f)
        with open(CLOSE_SCALER, 'rb') as f:
            close_scaler = pickle.load(f)
    except Exception as e:
        print(f"Error loading models/scalers: {e}")
        sys.exit(1)

    # Load 2025 Data
    print("\nLoading Data for 2025...")
    df_5m = load_and_preprocess(TEST_5M_FILENAME, filter_year=2025)
    df_1h = load_and_preprocess(TEST_1H_FILENAME, filter_year=2025)
    
    if df_5m is None or df_5m.empty or df_1h is None or df_1h.empty:
        print("\nCRITICAL ERROR: Could not proceed due to missing or empty data.")
        print("Please check that TEST_5m.csv and TEST_1h.csv are uploaded.")
        # We explicitly exit here to prevent AttributeError downstream
        sys.exit(1)

    # Predictions
    print("\nGenerating Predictions...")
    
    # 5M
    df_5m_scaled = df_5m.copy()
    df_5m_scaled[FEATURE_COLS] = scaler.transform(df_5m[FEATURE_COLS])
    predict_5m = tf.keras.utils.timeseries_dataset_from_array(
        data=df_5m_scaled[FEATURE_COLS].values, targets=None,
        sequence_length=LOOKBACK_SMALL, batch_size=256, shuffle=False
    )
    pred_5m_scaled = little_model.predict(predict_5m, verbose=0)
    little_predictions = close_scaler.inverse_transform(pred_5m_scaled).flatten()
    
    # 1H
    df_1h_scaled = df_1h.copy()
    df_1h_scaled[FEATURE_COLS] = scaler.transform(df_1h[FEATURE_COLS])
    predict_1h = tf.keras.utils.timeseries_dataset_from_array(
        data=df_1h_scaled[FEATURE_COLS].values, targets=None,
        sequence_length=LOOKBACK_BIG, batch_size=64, shuffle=False
    )
    pred_1h_scaled = big_model.predict(predict_1h, verbose=0)
    big_predictions = close_scaler.inverse_transform(pred_1h_scaled).flatten()

    # Run Compare
    results, trades = hybrid_backtest(
        df_5m, little_predictions, 
        df_1h, big_predictions, 
        LOOKBACK_SMALL, LOOKBACK_BIG
    )

    print("\n" + "=" * 60)
    print("  2025 PREDICTION RESULTS")
    print("=" * 60)
    print(f"  Total Opportunities Scanned: {results['total_signals']}")
    print(f"  Trades Taken (Agreement):    {results['confirmed_trades']}")
    print(f"  Win Rate:                    {results['win_rate']:.2f}%")
    print(f"  Wins:                        {results['wins']}")
    print(f"  Losses:                      {results['losses']}")
    print("=" * 60)
    
    if len(trades) > 0:
        total_return_pct = (np.prod([1+t for t in trades]) - 1) * 100
        print(f"  Estimated Return:            {total_return_pct:.2f}% (Compounded)")

    # ================
    # (INFO) VISUALIZATION
    # ================
    print("\nGenerating Graph...")
    
    # Align dates and prices for the graph using the same logic as the main loop
    # The 'little_predictions' array corresponds to windows of size LOOKBACK_SMALL
    # If using timeseries_dataset_from_array with seq_len=48:
    # Prediction[0] is based on rows 0..47.
    # If the model predicts the NEXT candle (t+1), Prediction[0] is for row 48.
    
    # Matching the logic from demo_dual_model.py:
    start_idx = LOOKBACK_SMALL - 1
    end_idx = start_idx + len(little_predictions)
    
    plot_dates = df_5m['Date'].iloc[start_idx:end_idx]
    real_price = df_5m['Close'].iloc[start_idx:end_idx]
    
    # Create Figure
    fig = make_subplots(rows=1, cols=1, subplot_titles=("2025: Real Market vs AI Prediction"))
    
    # 1. Real Data (Green)
    fig.add_trace(go.Scatter(
        x=plot_dates, 
        y=real_price, 
        mode='lines', 
        name='Real Price (Market)', 
        line=dict(color='#00FF00', width=1)
    ))

    # 2. AI Prediction (Red)
    fig.add_trace(go.Scatter(
        x=plot_dates, 
        y=little_predictions, 
        mode='lines', 
        name='AI Prediction', 
        line=dict(color='#FF0000', width=1, dash='dot')
    ))

    fig.update_layout(
        template='plotly_dark',
        title_text='AI Model Performance: Real (Green) vs Predicted (Red)',
        xaxis_title='Date',
        yaxis_title='Price',
        height=600,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Show figure
    print("Graph generated. Displaying...")
    fig.show()
