# ===========================================
# TRAIN LITTLE BROTHER MODEL (5M Data)
# ===========================================
# Run this SECOND in a fresh Colab session
# (After Big Brother is trained and downloaded)
# After training completes, download:
#   - 'little_brother_v2.keras'
#   - 'scaler_small.pkl' (needed for predictions)
# ===========================================

import numpy as np
import pandas as pd
import gc
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Bidirectional, 
    Input, GlobalAveragePooling1D
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# ===========================================
# CONFIGURATION
# ===========================================
TRAIN_FILE = 'TRAIN_5m.csv'
LOOKBACK = 48  # 4 hours of 5-min data
MODEL_NAME = 'little_brother_v2.keras'
SCALER_NAME = 'scaler_small.pkl'

FEATURE_COLS = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'Returns', 'EMA_20', 'EMA_50', 'EMA_Diff',
    'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
    'BB_Width', 'BB_Position', 'ATR', 'Volatility'
]

# ===========================================
# TECHNICAL INDICATORS
# ===========================================
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

# ===========================================
# DATA LOADING
# ===========================================
def load_and_preprocess(filepath):
    print(f"Loading {filepath}...")
    df = pd.read_csv(filepath, sep=';')
    df['Date'] = pd.to_datetime(df['Date'], format='%Y.%m.%d %H:%M')
    df = df.sort_values('Date').reset_index(drop=True)
    df = add_technical_indicators(df)
    df = df.dropna().reset_index(drop=True)
    
    # Convert to float32 to save memory
    float_cols = df.select_dtypes(include=['float64']).columns
    df[float_cols] = df[float_cols].astype('float32')
    
    print(f"Loaded {len(df)} rows")
    return df



# ===========================================
# MODEL
# ===========================================
def build_model(input_shape):
    inputs = Input(shape=input_shape)
    x = Bidirectional(LSTM(128, return_sequences=True))(inputs)
    x = Dropout(0.3)(x)
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = Dropout(0.3)(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation='relu')(x)
    outputs = Dense(1)(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

# ===========================================
# MAIN
# ===========================================
if __name__ == "__main__":
    print("=" * 60)
    print("  LITTLE BROTHER TRAINING (5M Entry Model)")
    print("  Created by: Daryl James Padogdog")
    print("  Purpose:    Analyzes the 5-Minute timeframe to find precise")
    print("              entry points aligned with the Big Brother trend.")
    print("=" * 60)
    print("\n[INFO] Starting process...")
    
    # Load data
    df = load_and_preprocess(TRAIN_FILE)
    
    # Normalize and SAVE the scaler (needed for predictions later)
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = df.copy()
    df_scaled[FEATURE_COLS] = scaler.fit_transform(df[FEATURE_COLS]).astype('float32')
    
    # Save scaler for later use
    with open(SCALER_NAME, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✓ Scaler saved as '{SCALER_NAME}'")
    
    # Also save the close scaler
    close_scaler = MinMaxScaler(feature_range=(0, 1))
    close_scaler.fit(df[['Close']])
    with open('close_scaler.pkl', 'wb') as f:
        pickle.dump(close_scaler, f)
    print("✓ Close scaler saved as 'close_scaler.pkl'")
    
    # Create datasets using tf.data (Streaming)
    # -----------------------------------------
    # Split data indices first (prevent data leakage)
    n = len(df_scaled)
    train_n = int(n * 0.85)
    
    data_array = df_scaled[FEATURE_COLS].values
    target_array = df_scaled['Close'].values
    
    # Train Dataset
    train_dataset = tf.keras.utils.timeseries_dataset_from_array(
        data=data_array[:-LOOKBACK], # slice end to match target
        targets=target_array[LOOKBACK:], # offset targets by lookback
        sequence_length=LOOKBACK,
        sequence_stride=1,
        shuffle=True,
        batch_size=64,
        start_index=0,
        end_index=train_n
    )
    
    # Validation Dataset
    val_dataset = tf.keras.utils.timeseries_dataset_from_array(
        data=data_array[:-LOOKBACK],
        targets=target_array[LOOKBACK:],
        sequence_length=LOOKBACK,
        sequence_stride=1,
        shuffle=False,
        batch_size=64,
        start_index=train_n
    )
    
    print("✓ Datasets created (streaming)")
    
    # Free dataframe memory (the arrays are now referenced by the dataset)
    del df, df_scaled
    gc.collect()
    
    # Build and train
    model = build_model((LOOKBACK, len(FEATURE_COLS)))
    model.summary()
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)
    ]
    
    model.fit(train_dataset, validation_data=val_dataset, epochs=100, callbacks=callbacks)
    
    # Save
    model.save(MODEL_NAME)
    print(f"\n✓ Model saved as '{MODEL_NAME}'")
    print("\nDownload these files:")
    print(f"  - {MODEL_NAME}")
    print(f"  - {SCALER_NAME}")
    print("  - close_scaler.pkl")
