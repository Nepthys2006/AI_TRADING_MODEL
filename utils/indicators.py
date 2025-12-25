# ===========================================
# SHARED TECHNICAL INDICATORS MODULE
# ===========================================
# Centralized indicator calculations with:
# - ADX for trend strength filtering
# - Cyclical time encoding (hour/day sin/cos)
# - Session encoding (London/NY/Asian)
# - ATR-scaled classification thresholds
# ===========================================

import numpy as np
import pandas as pd
from typing import Tuple


# ===========================================
# CORE INDICATORS
# ===========================================

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_macd(
    prices: pd.Series, 
    fast: int = 12, 
    slow: int = 26, 
    signal: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate MACD, Signal Line, and Histogram."""
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram


def calculate_bollinger_bands(
    prices: pd.Series, 
    period: int = 20, 
    std_dev: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Bollinger Bands (upper, middle, lower)."""
    middle = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    return upper, middle, lower


def calculate_atr(
    high: pd.Series, 
    low: pd.Series, 
    close: pd.Series, 
    period: int = 14
) -> pd.Series:
    """Calculate Average True Range."""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def calculate_adx(
    high: pd.Series, 
    low: pd.Series, 
    close: pd.Series, 
    period: int = 14
) -> pd.Series:
    """
    Calculate Average Directional Index (ADX).
    ADX > 25 indicates a trending market, < 20 indicates ranging.
    """
    # True Range
    tr = calculate_atr(high, low, close, period=1) * period  # Unnormalized
    
    # Directional Movement
    up_move = high.diff()
    down_move = -low.diff()
    
    # +DM and -DM
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    plus_dm = pd.Series(plus_dm, index=high.index)
    minus_dm = pd.Series(minus_dm, index=high.index)
    
    # Smoothed values
    atr = calculate_atr(high, low, close, period)
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
    
    # DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.rolling(window=period).mean()
    
    return adx


# ===========================================
# TIME ENCODING
# ===========================================

def add_cyclical_time_features(df: pd.DataFrame, date_column: str = 'Date') -> pd.DataFrame:
    """
    Add sin/cos encoded time features for:
    - Hour of day (captures intraday patterns)
    - Day of week (captures weekly seasonality)
    """
    df = df.copy()
    dates = pd.to_datetime(df[date_column])
    
    # Hour encoding (0-23 -> sin/cos)
    hour = dates.dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    
    # Day of week encoding (0-6 -> sin/cos)
    dow = dates.dt.dayofweek
    df['dow_sin'] = np.sin(2 * np.pi * dow / 7)
    df['dow_cos'] = np.cos(2 * np.pi * dow / 7)
    
    return df


def add_session_encoding(df: pd.DataFrame, date_column: str = 'Date') -> pd.DataFrame:
    """
    Add trading session one-hot encoding:
    - Asian: 22:00-06:00 GMT (suppress in attention)
    - London: 07:00-15:00 GMT
    - New York: 12:00-21:00 GMT (overlap with London 12:00-15:00)
    
    Note: Assumes datetime is in GMT/UTC
    """
    df = df.copy()
    dates = pd.to_datetime(df[date_column])
    hour = dates.dt.hour
    
    # Session flags (with overlap handling)
    df['session_asian'] = ((hour >= 22) | (hour < 6)).astype(float)
    df['session_london'] = ((hour >= 7) & (hour < 16)).astype(float)
    df['session_ny'] = ((hour >= 12) & (hour < 21)).astype(float)
    
    # Session overlap indicator (London/NY overlap is high volatility)
    df['session_overlap'] = ((hour >= 12) & (hour < 16)).astype(float)
    
    return df


# ===========================================
# CLASSIFICATION TARGETS
# ===========================================

def create_classification_targets(
    df: pd.DataFrame,
    close_column: str = 'Close',
    atr_column: str = 'ATR',
    threshold_atr_mult: float = 0.5,
    fixed_threshold: float = None,
    lookahead: int = 1
) -> pd.DataFrame:
    """
    Create 3-class classification targets based on log-returns.
    
    Classes:
        0 = DOWN (return < -threshold)
        1 = FLAT (-threshold <= return <= threshold)
        2 = UP   (return > threshold)
    
    Args:
        df: DataFrame with Close and ATR columns
        close_column: Name of close price column
        atr_column: Name of ATR column
        threshold_atr_mult: Threshold as multiple of ATR/Close (default 0.5)
        fixed_threshold: Optional fixed threshold (overrides ATR-based)
        lookahead: Number of bars to look ahead for return calculation
        
    Returns:
        DataFrame with added 'target_class' and 'log_return' columns
    """
    df = df.copy()
    
    # Calculate log return (future return)
    close = df[close_column]
    future_close = close.shift(-lookahead)
    df['log_return'] = np.log(future_close / close)
    
    # Calculate threshold
    if fixed_threshold is not None:
        threshold = fixed_threshold
        df['threshold'] = threshold
    else:
        # ATR-based threshold: k * (ATR / Close)
        atr_pct = df[atr_column] / close
        threshold = threshold_atr_mult * atr_pct
        df['threshold'] = threshold
    
    # Create class labels
    df['target_class'] = 1  # Default: FLAT
    df.loc[df['log_return'] > df['threshold'], 'target_class'] = 2   # UP
    df.loc[df['log_return'] < -df['threshold'], 'target_class'] = 0  # DOWN
    
    # Remove last rows without valid future return
    df = df.iloc[:-lookahead].copy()
    
    return df


def get_class_distribution(df: pd.DataFrame, target_column: str = 'target_class') -> dict:
    """Print and return class distribution."""
    dist = df[target_column].value_counts(normalize=True).sort_index()
    class_names = {0: 'DOWN', 1: 'FLAT', 2: 'UP'}
    
    print("\nClass Distribution:")
    for cls, pct in dist.items():
        print(f"  {class_names.get(cls, cls)}: {pct*100:.1f}%")
    
    return dist.to_dict()


# ===========================================
# PATTERN CONFIRMATION (Qwen3 VL suggestion)
# ===========================================

def add_pattern_confirmation(
    df: pd.DataFrame,
    high_column: str = 'High',
    low_column: str = 'Low',
    lookback: int = 3
) -> pd.DataFrame:
    """
    Add pattern confirmation features:
    - Higher highs count (last N bars)
    - Lower lows count (last N bars)
    
    Useful for filtering whipsaws during Fed announcements.
    """
    df = df.copy()
    
    highs = df[high_column]
    lows = df[low_column]
    
    # Count higher highs in last N bars
    higher_highs = sum(
        (highs.shift(i) > highs.shift(i+1)).astype(int) 
        for i in range(lookback)
    )
    df['higher_highs_count'] = higher_highs
    
    # Count lower lows in last N bars
    lower_lows = sum(
        (lows.shift(i) < lows.shift(i+1)).astype(int) 
        for i in range(lookback)
    )
    df['lower_lows_count'] = lower_lows
    
    # Pattern strength (-3 to +3)
    df['pattern_strength'] = df['higher_highs_count'] - df['lower_lows_count']
    
    return df


# ===========================================
# FULL FEATURE PIPELINE
# ===========================================

def add_all_indicators(df: pd.DataFrame, include_advanced: bool = True) -> pd.DataFrame:
    """
    Add all technical indicators to DataFrame.
    
    Args:
        df: DataFrame with OHLCV columns
        include_advanced: Include ADX, session encoding, pattern features
        
    Returns:
        DataFrame with all indicators added
    """
    df = df.copy()
    
    # Basic indicators
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
    
    if include_advanced:
        # ADX for trend strength
        df['ADX'] = calculate_adx(df['High'], df['Low'], df['Close'])
        
        # Time encoding
        df = add_cyclical_time_features(df)
        df = add_session_encoding(df)
        
        # Pattern confirmation
        df = add_pattern_confirmation(df)
    
    return df


# Feature column definitions
FEATURE_COLS_BASIC = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'Returns', 'EMA_20', 'EMA_50', 'EMA_Diff',
    'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
    'BB_Width', 'BB_Position', 'ATR', 'Volatility'
]

FEATURE_COLS_ADVANCED = FEATURE_COLS_BASIC + [
    'ADX',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
    'session_asian', 'session_london', 'session_ny', 'session_overlap',
    'higher_highs_count', 'lower_lows_count', 'pattern_strength'
]
