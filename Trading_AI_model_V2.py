# ===========================================
# GOLD PRICE PREDICTION V2 - IMPROVED MODEL
# ===========================================
# Created by: Daryl James Padogdog
# Improvements: Technical Indicators, Bidirectional LSTM, 
#               Attention Mechanism, Risk Management
# ===========================================

import numpy as np
import pandas as pd
import gc
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Bidirectional, 
    Input, Layer, Attention, Concatenate, GlobalAveragePooling1D
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ===========================================
# 1. CONFIGURATION
# ===========================================
# Data paths - files in current directory (for Colab)
# Training files (2024 and earlier)
TRAIN_BIG = 'TRAIN_1h.csv'
TRAIN_SMALL = 'TRAIN_5m.csv'

# Test files (2025 data only)
TEST_BIG = 'TEST_1h.csv'
TEST_SMALL = 'TEST_5m.csv'

LOOKBACK_BIG = 168    # 1 week of hourly data (7*24)
LOOKBACK_SMALL = 48   # 4 hours of 5-min data

MODEL_BIG_NAME = 'big_brother_v2.keras'
MODEL_SMALL_NAME = 'little_brother_v2.keras'

# Trading Parameters
CONFIDENCE_THRESHOLD = 0.001  # 0.1% minimum price change to trade
STOP_LOSS_PCT = 0.01          # 1% stop loss
TAKE_PROFIT_PCT = 0.02        # 2% take profit

# ===========================================
# 2. TECHNICAL INDICATORS
# ===========================================
def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD and Signal Line"""
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculate_bollinger_bands(prices, period=20, std_dev=2):
    """Calculate Bollinger Bands"""
    middle = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    return upper, middle, lower

def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range"""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

def calculate_ema(prices, period):
    """Calculate Exponential Moving Average"""
    return prices.ewm(span=period, adjust=False).mean()

def add_technical_indicators(df):
    """Add all technical indicators to dataframe"""
    # Price-based features
    df['Returns'] = df['Close'].pct_change()
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Moving Averages
    df['EMA_20'] = calculate_ema(df['Close'], 20)
    df['EMA_50'] = calculate_ema(df['Close'], 50)
    df['EMA_Diff'] = df['EMA_20'] - df['EMA_50']
    
    # RSI
    df['RSI'] = calculate_rsi(df['Close'], 14)
    
    # MACD
    df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = calculate_macd(df['Close'])
    
    # Bollinger Bands
    df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = calculate_bollinger_bands(df['Close'])
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    
    # ATR
    df['ATR'] = calculate_atr(df['High'], df['Low'], df['Close'])
    
    # Price relative to OHLC
    df['HL_Range'] = df['High'] - df['Low']
    df['OC_Range'] = df['Close'] - df['Open']
    
    # Volatility
    df['Volatility'] = df['Returns'].rolling(window=20).std()
    
    return df

# ===========================================
# 3. DATA PREPROCESSING
# ===========================================
def load_and_preprocess(filepath, sep=';'):
    """Load data and add technical indicators"""
    print(f"Loading {filepath}...")
    df = pd.read_csv(filepath, sep=sep)
    
    # Parse date
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], format='%Y.%m.%d %H:%M')
    
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Add technical indicators
    df = add_technical_indicators(df)
    
    # Drop NaN rows (from indicator calculations)
    df = df.dropna().reset_index(drop=True)
    
    # Convert to float32 to save memory
    float_cols = df.select_dtypes(include=['float64']).columns
    df[float_cols] = df[float_cols].astype('float32')
    
    print(f"Loaded {len(df)} rows with {len(df.columns)} features")
    return df

def create_sequences(data, lookback, feature_cols, target_col='Close'):
    """Create sequences for LSTM with multiple features using efficient memory usage"""
    feature_data = data[feature_cols].values.astype('float32')
    target_data = data[target_col].values.astype('float32')
    
    # Use strided slicing to create sequences without copying data (if possible)
    # or just simple list comprehension but ensuring float32
    
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(feature_data[i-lookback:i])
        y.append(target_data[i])
        
    return np.array(X, dtype='float32'), np.array(y, dtype='float32')

def normalize_data(df, feature_cols):
    """Normalize all feature columns"""
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    df_scaled = df.copy()
    # Fit transform and convert to float32
    df_scaled[feature_cols] = scaler.fit_transform(df[feature_cols]).astype('float32')
    
    # Keep a separate scaler for Close price (for inverse transform)
    close_scaler = MinMaxScaler(feature_range=(0, 1))
    close_scaler.fit(df[['Close']])
    
    return df_scaled, scaler, close_scaler

# ===========================================
# 4. MODEL ARCHITECTURE - IMPROVED
# ===========================================
def build_improved_model(input_shape, name="model"):
    """Build Bidirectional LSTM with improved architecture"""
    
    inputs = Input(shape=input_shape, name=f'{name}_input')
    
    # First Bidirectional LSTM layer
    x = Bidirectional(LSTM(128, return_sequences=True, name=f'{name}_lstm1'))(inputs)
    x = Dropout(0.3)(x)
    
    # Second Bidirectional LSTM layer
    x = Bidirectional(LSTM(64, return_sequences=True, name=f'{name}_lstm2'))(x)
    x = Dropout(0.3)(x)
    
    # Global Average Pooling (simple attention mechanism)
    x = GlobalAveragePooling1D()(x)
    
    # Dense layers
    x = Dense(64, activation='relu', name=f'{name}_dense1')(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation='relu', name=f'{name}_dense2')(x)
    
    # Output
    outputs = Dense(1, name=f'{name}_output')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name=name)
    
    # Compile with learning rate scheduler
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    return model

def train_model_v2(X_train, y_train, model_name, epochs=100):
    """Train model with improved callbacks"""
    
    model = build_improved_model(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        name=model_name.replace('.keras', '')
    )
    
    print(f"\nTraining {model_name}...")
    model.summary()
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.0001,
            verbose=1
        )
    ]
    
    # Train
    history = model.fit(
        X_train, y_train,
        batch_size=64,
        epochs=epochs,
        validation_split=0.15,
        callbacks=callbacks,
        verbose=1
    )
    
    model.save(model_name)
    print(f"Saved {model_name}")
    
    return model, history

# ===========================================
# 5. EVALUATION METRICS
# ===========================================
def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """Calculate Sharpe Ratio (annualized)"""
    excess_returns = returns - risk_free_rate/252
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

def calculate_max_drawdown(cumulative_returns):
    """Calculate Maximum Drawdown"""
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns - peak) / peak
    return drawdown.min()

def calculate_profit_factor(trades):
    """Calculate Profit Factor (sum of profits / sum of losses)"""
    profits = trades[trades > 0].sum()
    losses = abs(trades[trades < 0].sum())
    return profits / losses if losses > 0 else float('inf')

def backtest_with_metrics(real_prices, predicted_prices, test_dates):
    """Run backtest and calculate comprehensive metrics"""
    
    results = {
        'total_trades': 0,
        'wins': 0,
        'losses': 0,
        'win_rate': 0,
        'total_return': 0,
        'sharpe_ratio': 0,
        'max_drawdown': 0,
        'profit_factor': 0
    }
    
    trades = []
    returns = []
    signals = []
    
    for i in range(len(real_prices) - 1):
        current_price = real_prices[i]
        predicted_next = predicted_prices[i]
        actual_next = real_prices[i + 1]
        
        # Calculate predicted change percentage
        predicted_change = (predicted_next - current_price) / current_price
        
        # Only trade if confidence threshold met
        if abs(predicted_change) < CONFIDENCE_THRESHOLD:
            signals.append('WAIT')
            continue
        
        # Generate signal
        if predicted_change > 0:
            signal = 'BUY'
        else:
            signal = 'SELL'
        
        signals.append(signal)
        results['total_trades'] += 1
        
        # Calculate actual return
        actual_return = (actual_next - current_price) / current_price
        
        # Apply stop loss / take profit
        if signal == 'BUY':
            if actual_return < -STOP_LOSS_PCT:
                trade_return = -STOP_LOSS_PCT
            elif actual_return > TAKE_PROFIT_PCT:
                trade_return = TAKE_PROFIT_PCT
            else:
                trade_return = actual_return
        else:  # SELL
            if actual_return > STOP_LOSS_PCT:
                trade_return = -STOP_LOSS_PCT
            elif actual_return < -TAKE_PROFIT_PCT:
                trade_return = TAKE_PROFIT_PCT
            else:
                trade_return = -actual_return
        
        trades.append(trade_return)
        returns.append(trade_return)
        
        # Check win/loss
        if signal == 'BUY' and actual_next > current_price:
            results['wins'] += 1
        elif signal == 'SELL' and actual_next < current_price:
            results['wins'] += 1
        else:
            results['losses'] += 1
    
    if results['total_trades'] > 0:
        results['win_rate'] = (results['wins'] / results['total_trades']) * 100
        
        returns_series = pd.Series(returns)
        cumulative_returns = (1 + returns_series).cumprod()
        
        results['total_return'] = (cumulative_returns.iloc[-1] - 1) * 100
        results['sharpe_ratio'] = calculate_sharpe_ratio(returns_series)
        results['max_drawdown'] = calculate_max_drawdown(cumulative_returns) * 100
        results['profit_factor'] = calculate_profit_factor(returns_series)
    
    return results, trades, signals

# ===========================================
# 6. MAIN EXECUTION
# ===========================================
if __name__ == "__main__":
    print("=" * 50)
    print("GOLD PRICE PREDICTION V2 - IMPROVED MODEL")
    print("=" * 50)
    
    # Feature columns to use
    FEATURE_COLS = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'Returns', 'EMA_20', 'EMA_50', 'EMA_Diff',
        'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
        'BB_Width', 'BB_Position', 'ATR', 'Volatility'
    ]
    
    # ==========================================
    # LOAD AND PREPROCESS TRAINING DATA
    # ==========================================
    print("\nLoading TRAINING data (2024 and earlier)...")
    df_train_big = load_and_preprocess(TRAIN_BIG)
    df_train_small = load_and_preprocess(TRAIN_SMALL)
    
    print(f"\nFeatures being used: {len(FEATURE_COLS)}")
    print(FEATURE_COLS)
    
    # ==========================================
    # NORMALIZE TRAINING DATA
    # ==========================================
    df_train_big_scaled, _, scaler_big = normalize_data(df_train_big, FEATURE_COLS)
    df_train_small_scaled, _, scaler_small = normalize_data(df_train_small, FEATURE_COLS)
    
    # ==========================================
    # PREPARE TRAINING SEQUENCES
    # ==========================================
    # Big Brother (1H data) - use ALL training data
    X_train_big, y_train_big = create_sequences(df_train_big_scaled, LOOKBACK_BIG, FEATURE_COLS)
    
    # Free up memory
    del df_train_big, df_train_big_scaled
    gc.collect()
    
    print(f"\nBig Brother - Training samples: {len(X_train_big)}")
    print(f"Input shape: {X_train_big.shape}")
    
    # Little Brother (5M data) - use ALL training data
    X_train_small, y_train_small = create_sequences(df_train_small_scaled, LOOKBACK_SMALL, FEATURE_COLS)
    
    # Free up memory
    del df_train_small, df_train_small_scaled
    gc.collect()
    
    print(f"\nLittle Brother - Training samples: {len(X_train_small)}")
    print(f"Input shape: {X_train_small.shape}")
    
    # ==========================================
    # TRAIN MODELS
    # ==========================================
    big_model, big_history = train_model_v2(X_train_big, y_train_big, MODEL_BIG_NAME)
    small_model, small_history = train_model_v2(X_train_small, y_train_small, MODEL_SMALL_NAME)
    
    # ==========================================
    # BACKTEST ON 2025 TEST DATA
    # ==========================================
    print("\n" + "=" * 50)
    print("BACKTESTING ON 2025 TEST DATA")
    print("=" * 50)
    
    # Load 2025 test data
    print("\nLoading TEST data (2025)...")
    df_test_small = load_and_preprocess(TEST_SMALL)
    
    # Normalize test data using training scaler
    df_test_small_scaled = df_test_small.copy()
    df_test_small_scaled[FEATURE_COLS] = scaler_small.transform(
        df_test_small[FEATURE_COLS].values.reshape(-1, len(FEATURE_COLS))
    ).reshape(-1, len(FEATURE_COLS))
    
    # Create test sequences
    X_test, _ = create_sequences(df_test_small_scaled, LOOKBACK_SMALL, FEATURE_COLS)
    test_dates = df_test_small['Date'].iloc[LOOKBACK_SMALL:].values
    real_prices = df_test_small['Close'].iloc[LOOKBACK_SMALL:].values
    
    print(f"Testing on {len(X_test)} samples (2025 data)...")
    
    # Predict
    predicted_scaled = small_model.predict(X_test, verbose=1)
    predicted_prices = scaler_small.inverse_transform(predicted_scaled).flatten()
    
    # Run backtest
    metrics, trades, signals = backtest_with_metrics(real_prices, predicted_prices, test_dates)
    
    # ==========================================
    # RESULTS
    # ==========================================
    print("\n" + "=" * 50)
    print("BACKTEST RESULTS")
    print("=" * 50)
    print(f"Total Trades:    {metrics['total_trades']}")
    print(f"Wins:            {metrics['wins']}")
    print(f"Losses:          {metrics['losses']}")
    print(f"Win Rate:        {metrics['win_rate']:.2f}%")
    print(f"Total Return:    {metrics['total_return']:.2f}%")
    print(f"Sharpe Ratio:    {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown:    {metrics['max_drawdown']:.2f}%")
    print(f"Profit Factor:   {metrics['profit_factor']:.2f}")
    print("=" * 50)
    
    # ==========================================
    # LIVE SIGNAL
    # ==========================================
    print("\n" + "=" * 50)
    print("CURRENT TRADING SIGNAL")
    print("=" * 50)
    
    # Get latest sequence from test data (most recent 2025 data)
    latest_data = df_test_small_scaled.iloc[-LOOKBACK_SMALL:]
    latest_seq = latest_data[FEATURE_COLS].values
    latest_seq = np.expand_dims(latest_seq, axis=0)
    
    # Predict
    pred_scaled = small_model.predict(latest_seq, verbose=0)
    target_price = scaler_small.inverse_transform(pred_scaled)[0][0]
    current_price = df_test_small['Close'].iloc[-1]
    
    predicted_change = (target_price - current_price) / current_price
    
    print(f"Current Price:   ${current_price:.2f}")
    print(f"Predicted Price: ${target_price:.2f}")
    print(f"Expected Change: {predicted_change * 100:.3f}%")
    
    if abs(predicted_change) < CONFIDENCE_THRESHOLD:
        signal = "WAIT - Low confidence"
    elif predicted_change > 0:
        signal = f"BUY (SL: ${current_price * (1 - STOP_LOSS_PCT):.2f}, TP: ${current_price * (1 + TAKE_PROFIT_PCT):.2f})"
    else:
        signal = f"SELL (SL: ${current_price * (1 + STOP_LOSS_PCT):.2f}, TP: ${current_price * (1 - TAKE_PROFIT_PCT):.2f})"
    
    print(f"\n>> SIGNAL: {signal}")
    print("=" * 50)
    
    # ==========================================
    # VISUALIZATION
    # ==========================================
    print("\nGenerating performance chart...")
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(
            f'Price Prediction (Win Rate: {metrics["win_rate"]:.2f}%)',
            'Cumulative Returns'
        ),
        row_heights=[0.7, 0.3]
    )
    
    # Price chart
    fig.add_trace(
        go.Scatter(
            x=test_dates[:len(real_prices)],
            y=real_prices,
            mode='lines',
            name='Real Price',
            line=dict(color='#00FF00', width=1)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=test_dates[:len(predicted_prices)],
            y=predicted_prices,
            mode='lines',
            name='Predicted Price',
            line=dict(color='#FFFF00', width=1, dash='dot')
        ),
        row=1, col=1
    )
    
    # Cumulative returns
    if trades:
        cum_returns = (pd.Series(trades).fillna(0) + 1).cumprod() - 1
        fig.add_trace(
            go.Scatter(
                x=list(range(len(cum_returns))),
                y=cum_returns * 100,
                mode='lines',
                name='Cumulative Return %',
                line=dict(color='#00BFFF', width=2)
            ),
            row=2, col=1
        )
    
    fig.update_layout(
        template='plotly_dark',
        title=f'Gold Price Prediction V2 - Performance Dashboard',
        height=800,
        showlegend=True
    )
    
    fig.write_html('trading_performance_v2.html')
    print("Chart saved to 'trading_performance_v2.html'")
    
    print("\n✓ V2 Model training and backtesting complete!")
