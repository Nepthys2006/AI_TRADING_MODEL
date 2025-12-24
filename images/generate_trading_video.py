# ===========================================
# (INFO) TRADING TIMELAPSE GENERATOR
# ===========================================

import numpy as np
import pandas as pd
import pickle
import warnings
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import tensorflow as tf
from tensorflow.keras.models import load_model

warnings.filterwarnings('ignore')

# ================
# (INFO) CONFIGURATION
# ================
BIG_BROTHER_MODEL = 'big_brother_v2.keras'
LITTLE_BROTHER_MODEL = 'little_brother_v2.keras'
SCALER_SMALL = 'scaler_small.pkl'
CLOSE_SCALER = 'close_scaler.pkl'
TEST_5M_FILENAME = 'TEST_5m.csv'
TEST_1H_FILENAME = 'TEST_1h.csv'

# Settings
LOOKBACK_BIG = 168
LOOKBACK_SMALL = 48
INITIAL_CAPITAL = 200.0

WEEK_LENGTH = 2016
FAST_SKIP = 24
SLOW_SKIP = 1

FEATURE_COLS = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'Returns', 'EMA_20', 'EMA_50', 'EMA_Diff',
    'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
    'BB_Width', 'BB_Position', 'ATR', 'Volatility'
]

# ================
# (INFO) HELPER FUNCTIONS
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

def check_file(filename):
    if os.path.exists(filename): return filename
    if os.path.exists(os.path.join('DATA_SET', filename)): return os.path.join('DATA_SET', filename)
    return None

def load_and_preprocess(filename):
    filepath = check_file(filename)
    if not filepath: return None
    print(f"Loading {filepath}...")
    df = pd.read_csv(filepath, sep=';')
    df['Date'] = pd.to_datetime(df['Date'], format='%Y.%m.%d %H:%M')
    df = df.sort_values('Date').reset_index(drop=True)
    df = add_technical_indicators(df)
    df = df.dropna().reset_index(drop=True)
    float_cols = df.select_dtypes(include=['float64']).columns
    df[float_cols] = df[float_cols].astype('float32')
    return df

# ================
# (INFO) SIMULATION LOGIC
# ================
def run_simulation(df_5m, little_predictions, df_1h, big_predictions):
    # Align Data
    df_5m_sim = df_5m.iloc[LOOKBACK_SMALL-1 : LOOKBACK_SMALL-1+len(little_predictions)].copy()
    df_5m_sim['Predicted_Raw'] = little_predictions
    df_5m_sim = df_5m_sim.reset_index(drop=True)
    
    df_1h_subset = df_1h.iloc[LOOKBACK_BIG-1 : LOOKBACK_BIG-1+len(big_predictions)].copy()
    df_1h_subset['Predicted_Raw'] = big_predictions
    df_1h_subset['Hour'] = df_1h_subset['Date'].dt.floor('H')
    hourly_trend = df_1h_subset.set_index('Hour')[['Close', 'Predicted_Raw', 'EMA_20', 'EMA_50']].to_dict('index')
    
    bias_window = []
    BIAS_WINDOW_SIZE = 50
    RSI_BUY_MAX = 70
    RSI_SELL_MIN = 30
    MIN_THRESHOLD = 0.0015
    
    # Compute signals and find best week
    signals = []
    
    signals = []
    
    for i in range(len(df_5m_sim) - 1):
        row = df_5m_sim.iloc[i]
        current_price = row['Close']
        actual_next = df_5m_sim.iloc[i + 1]['Close']
        
        # Bias
        current_bias = current_price - row['Predicted_Raw']
        bias_window.append(current_bias)
        if len(bias_window) > BIAS_WINDOW_SIZE: bias_window.pop(0)
        avg_bias = sum(bias_window) / len(bias_window)
        pred_corrected = row['Predicted_Raw'] + avg_bias
        
        # Big Brother
        hour_key = row['Date'].floor('H')
        trend = "NEUTRAL"
        if hour_key in hourly_trend:
            h = hourly_trend[hour_key]
            pred_bullish = (h['Predicted_Raw'] - h['Close']) > -avg_bias
            ema_bullish = (h['Close'] > h['EMA_50']) and (h['EMA_20'] > h['EMA_50'])
            ema_bearish = (h['Close'] < h['EMA_50']) and (h['EMA_20'] < h['EMA_50'])
            if pred_bullish and ema_bullish: trend = "BULLISH"
            elif not pred_bullish and ema_bearish: trend = "BEARISH"
            
        # Entry
        pred_change = (pred_corrected - current_price) / current_price
        signal = "WAIT"
        trade_marker = None 
        
        if pred_change > MIN_THRESHOLD: signal = "BUY"
        elif pred_change < -MIN_THRESHOLD: signal = "SELL"
        
        if signal == "BUY" and row['RSI'] > RSI_BUY_MAX: signal = "WAIT"
        if signal == "SELL" and row['RSI'] < RSI_SELL_MIN: signal = "WAIT"
        
        if trend == "BULLISH" and signal == "BUY": trade_marker = 'BUY'
        elif trend == "BEARISH" and signal == "SELL": trade_marker = 'SELL'
            
        signals.append(trade_marker)
        
    df_5m_sim = df_5m_sim.iloc[:-1] # aligns with loop
    df_5m_sim['TradeMarker'] = signals
    return df_5m_sim

def find_busy_week(sim_data, week_len):
    # Scan for the week with most trades
    max_trades = 0
    best_start = 0
    
    # Step by 1 day (288 candles)
    for i in range(0, len(sim_data) - week_len, 288):
        subset = sim_data.iloc[i : i+week_len]
        trades = subset['TradeMarker'].count()
        if trades > max_trades:
            max_trades = trades
            best_start = i
            
    return best_start, max_trades

# ================
# (INFO) MAIN
# ================
if __name__ == "__main__":
    print("="*60)
    print("  TRADING TIMELAPSE (1 WEEK ACTIVITY)")
    print("="*60)
    
    # 1. Load
    df_5m = load_and_preprocess(TEST_5M_FILENAME)
    df_1h = load_and_preprocess(TEST_1H_FILENAME)
    if df_5m is None or df_1h is None: sys.exit(1)
        
    # 2. Model
    try:
        big_model = load_model(BIG_BROTHER_MODEL)
        little_model = load_model(LITTLE_BROTHER_MODEL)
        with open(SCALER_SMALL, 'rb') as f: scaler = pickle.load(f)
        with open(CLOSE_SCALER, 'rb') as f: close_scaler = pickle.load(f)
    except: sys.exit(1)
        
    # 3. Predict
    print("Generating predictions...")
    df_5m_scaled = df_5m.copy()
    df_5m_scaled[FEATURE_COLS] = scaler.transform(df_5m[FEATURE_COLS])
    p5 = tf.keras.utils.timeseries_dataset_from_array(df_5m_scaled[FEATURE_COLS].values, None, sequence_length=LOOKBACK_SMALL, batch_size=256, shuffle=False)
    little_preds = close_scaler.inverse_transform(little_model.predict(p5, verbose=0)).flatten()
    
    df_1h_scaled = df_1h.copy()
    df_1h_scaled[FEATURE_COLS] = scaler.transform(df_1h[FEATURE_COLS])
    p1 = tf.keras.utils.timeseries_dataset_from_array(df_1h_scaled[FEATURE_COLS].values, None, sequence_length=LOOKBACK_BIG, batch_size=64, shuffle=False)
    big_preds = close_scaler.inverse_transform(big_model.predict(p1, verbose=0)).flatten()
    
    # 5. Find Best Week
    print("Finding 'Busy Week' for Timelapse...")
    start_idx, trade_count = find_busy_week(sim_data, WEEK_LENGTH)
    
    week_data = sim_data.iloc[start_idx : start_idx + WEEK_LENGTH].copy().reset_index(drop=True)
    
    start_date = week_data['Date'].iloc[0]
    end_date = week_data['Date'].iloc[-1]
    print(f"Selected Week: {start_date} -> {end_date}")
    print(f"Total Trades in this week: {trade_count}")
    
    # 6. Calc Weekly Stats State
    # Re-calculate Balance frame-by-frame
    balance = INITIAL_CAPITAL
    wins = 0
    losses = 0
    
    # Add state columns
    week_data['Balance'] = INITIAL_CAPITAL
    week_data['Wins'] = 0
    week_data['Losses'] = 0
    week_data['ResultColor'] = None
    
    for i in range(len(week_data)-1):
        row = week_data.iloc[i]
        
        # Defaults
        week_data.at[i, 'Balance'] = balance
        week_data.at[i, 'Wins'] = wins
        week_data.at[i, 'Losses'] = losses
        
        if row['TradeMarker'] == 'BUY':
            actual_next = week_data.iloc[i+1]['Close']
            pnl = (actual_next - row['Close']) / row['Close']
            balance = balance * (1 + pnl)
            if pnl > 0: 
                wins += 1
                week_data.at[i, 'ResultColor'] = 'green'
            else: 
                losses += 1
                week_data.at[i, 'ResultColor'] = 'red'
                
        elif row['TradeMarker'] == 'SELL':
            actual_next = week_data.iloc[i+1]['Close']
            pnl = (row['Close'] - actual_next) / row['Close']
            balance = balance * (1 + pnl)
            if pnl > 0: 
                wins += 1
                week_data.at[i, 'ResultColor'] = 'green'
            else: 
                losses += 1
                week_data.at[i, 'ResultColor'] = 'red'
                
    # 7. Animation Setup
    print("Starting Live Timelapse...")
    
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(15, 8))
    gs = GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[4, 1])
    
    ax_chart = fig.add_subplot(gs[0, :]) 
    ax_metrics = fig.add_subplot(gs[1, 0]) 
    ax_balance = fig.add_subplot(gs[1, 1]) 
    
    # Fixed X Limits for Timelapse Effect (Chart fills up)
    ax_chart.set_xlim(0, WEEK_LENGTH)
    min_y = week_data['Close'].min() * 0.999
    max_y = week_data['Close'].max() * 1.001
    ax_chart.set_ylim(min_y, max_y)
    
    line_price, = ax_chart.plot([], [], color='cyan', linewidth=1.2, label='Price')
    line_ema20, = ax_chart.plot([], [], color='yellow', linewidth=0.8, alpha=0.7)
    line_ema50, = ax_chart.plot([], [], color='magenta', linewidth=0.8, alpha=0.7)
    scat_buy = ax_chart.scatter([], [], color='lime', marker='^', s=80, zorder=5)
    scat_sell = ax_chart.scatter([], [], color='red', marker='v', s=80, zorder=5)
    
    txt_title = ax_chart.set_title(f"TIMELAPSE: {start_date.date()} to {end_date.date()}", fontsize=14, color='white')
    
    txt_winrate = ax_metrics.text(0.1, 0.7, "", fontsize=12, color='white')
    txt_counts = ax_metrics.text(0.1, 0.4, "", fontsize=12, color='white')
    ax_metrics.axis('off')
    txt_balance = ax_balance.text(0.5, 0.5, "", fontsize=16, color='lime', ha='center', va='center')
    ax_balance.axis('off')
    
    # Rendering Indices
    indices = []
    i = 0
    while i < len(week_data):
        indices.append(i)
        
        # Check trade nearby
        is_trading = False
        if i < len(week_data) - 5:
            if week_data.iloc[i:i+5]['TradeMarker'].notna().any(): is_trading = True
            
        if is_trading: i += SLOW_SKIP
        else: i += FAST_SKIP
        
    def update(frame_idx):
        # Slice 0 to frame_idx (growing line)
        data_slice = week_data.iloc[:frame_idx]
        if data_slice.empty: return
        
        x_data = data_slice.index
        line_price.set_data(x_data, data_slice['Close'])
        line_ema20.set_data(x_data, data_slice['EMA_20'])
        line_ema50.set_data(x_data, data_slice['EMA_50'])
        
        # Scatter
        buys = data_slice[data_slice['TradeMarker'] == 'BUY']
        sells = data_slice[data_slice['TradeMarker'] == 'SELL']
        
        if not buys.empty: scat_buy.set_offsets(np.c_[buys.index, buys['Close']])
        else: scat_buy.set_offsets(np.empty((0, 2)))
            
        if not sells.empty: scat_sell.set_offsets(np.c_[sells.index, sells['Close']])
        else: scat_sell.set_offsets(np.empty((0, 2)))
        
        # Metrics
        curr = week_data.iloc[frame_idx] if frame_idx < len(week_data) else week_data.iloc[-1]
        
        w = int(curr['Wins'])
        l = int(curr['Losses'])
        total = w + l
        wr = (w / total * 100) if total > 0 else 0.0
        bal = curr['Balance']
        ret = (bal - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
        
        txt_winrate.set_text(f"Weekly Win Rate: {wr:.2f}%")
        txt_winrate.set_color('lime' if wr > 50 else 'orange')
        txt_counts.set_text(f"Wins: {w} | Losses: {l} | Total: {total}")
        txt_balance.set_text(f"${bal:.2f}\n({ret:+.2f}%)")
        txt_balance.set_color('lime' if ret >= 0 else 'red')
        
    # Animation Loop
    from IPython.display import display, clear_output
    import time
    
    try:
        handle = display(fig, display_id=True)
    except:
        handle = None
        plt.show()

    if handle:
        for frame_idx in indices:
            update(frame_idx)
            handle.update(fig)
            
            # Speed
            if frame_idx < len(week_data) - 5 and week_data.iloc[frame_idx:frame_idx+5]['TradeMarker'].notna().any():
                time.sleep(0.3)
            else:
                time.sleep(0.005) # Very fast for timelapse
                
    print("\nTimelapse Complete.")
