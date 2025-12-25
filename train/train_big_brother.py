# ===========================================
# TRAIN BIG BROTHER MODEL v2.0 (1H Data)
# ===========================================
# AI Council Upgrade: Classification-Based Model
# - 3-class softmax output (UP/FLAT/DOWN)
# - Session-aware MultiHead Attention
# - ATR-scaled classification thresholds
# - Walk-forward validation (150/30 day windows)
# - StandardScaler for fat-tailed gold distribution
# ===========================================

import numpy as np
import pandas as pd
import gc
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Bidirectional, 
    Input, Add, LayerNormalization
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# Import custom modules
import sys
sys.path.append('..')
from utils.attention_layer import SessionAwareAttention
from utils.indicators import (
    add_all_indicators, 
    create_classification_targets,
    get_class_distribution,
    FEATURE_COLS_ADVANCED
)
from utils.walk_forward import create_walk_forward_splits, exclude_weekend_gaps

# ===========================================
# CONFIGURATION
# ===========================================
TRAIN_FILE = 'DATA_SET/TRAIN_1h.csv'
LOOKBACK = 168  # 1 week of hourly data (7*24)
MODEL_NAME = 'big_brother_v2_classification.keras'
SCALER_NAME = 'scaler_big_brother.pkl'

# Date filter for Colab free tier (reduce memory usage)
DATE_START = '2018-01-01'  # Filter start date
DATE_END = '2024-12-31'    # Filter end date

# Classification thresholds
ATR_THRESHOLD_MULT = 0.5  # Threshold = 0.5 * ATR/Close
NUM_CLASSES = 3  # DOWN=0, FLAT=1, UP=2

# Feature columns for v2.0
FEATURE_COLS = FEATURE_COLS_ADVANCED

# ===========================================
# DATA LOADING
# ===========================================
def load_and_preprocess(filepath, date_start=None, date_end=None):
    """Load data and add all technical indicators."""
    print(f"Loading {filepath}...")
    df = pd.read_csv(filepath, sep=';')
    df['Date'] = pd.to_datetime(df['Date'], format='%Y.%m.%d %H:%M')
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Apply date filter for Colab memory optimization
    if date_start:
        df = df[df['Date'] >= date_start]
    if date_end:
        df = df[df['Date'] <= date_end]
    df = df.reset_index(drop=True)
    print(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    # Add all indicators (including ADX, time encoding, session encoding)
    df = add_all_indicators(df, include_advanced=True)
    
    # Drop rows with NaN values
    df = df.dropna().reset_index(drop=True)
    
    # Convert to float32 to save memory
    float_cols = df.select_dtypes(include=['float64']).columns
    df[float_cols] = df[float_cols].astype('float32')
    
    print(f"Loaded {len(df)} rows with {len(FEATURE_COLS)} features")
    return df


# ===========================================
# MODEL v2.0 (Classification with Attention)
# ===========================================
def build_model(input_shape, num_classes=3):
    """
    Build classification model with:
    - Bidirectional LSTM layers
    - Session-aware attention (replaces GlobalAveragePooling)
    - ResNet-style skip connection
    - 3-class softmax output
    """
    inputs = Input(shape=input_shape)
    
    # First LSTM block
    x = Bidirectional(LSTM(128, return_sequences=True))(inputs)
    x = Dropout(0.3)(x)
    
    # Save for skip connection
    skip = x
    
    # Second LSTM block
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = Dropout(0.3)(x)
    
    # Project skip connection to match dimensions
    skip_proj = Dense(128)(skip)  # 64*2 = 128 from Bidirectional
    
    # Add skip connection (ResNet-style)
    # Need to match dimensions - skip has 256 (128*2), x has 128 (64*2)
    x_proj = Dense(256)(x)  # Project x to 256
    skip_aligned = skip  # skip already has 256
    x = Add()([x_proj, skip_aligned])
    x = LayerNormalization()(x)
    
    # Session-aware attention (replaces GlobalAveragePooling)
    x = SessionAwareAttention(
        num_heads=4,
        key_dim=32,
        dropout_rate=0.1,
        recency_weight=2.0
    )(x)
    
    # Classification head
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation='relu')(x)
    
    # 3-class softmax output: [DOWN, FLAT, UP]
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def create_sequences_classification(df, feature_cols, lookback, scaler=None):
    """
    Create sequences for classification training.
    
    Returns:
        X: Feature sequences (samples, lookback, features)
        y: One-hot encoded class labels (samples, 3)
        scaler: Fitted StandardScaler
    """
    # Fit scaler on training data only
    if scaler is None:
        scaler = StandardScaler()
        df_scaled = df.copy()
        df_scaled[feature_cols] = scaler.fit_transform(df[feature_cols])
    else:
        df_scaled = df.copy()
        df_scaled[feature_cols] = scaler.transform(df[feature_cols])
    
    # Create sequences
    data = df_scaled[feature_cols].values
    targets = df['target_class'].values
    
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i])
        y.append(targets[i-1])  # Target for the last bar in sequence
    
    X = np.array(X, dtype='float32')
    y = to_categorical(np.array(y), num_classes=NUM_CLASSES)
    
    return X, y, scaler


# ===========================================
# MAIN
# ===========================================
if __name__ == "__main__":
    print("=" * 60)
    print("  BIG BROTHER v2.0 TRAINING (Classification Model)")
    print("  AI Council Upgrade: UP/FLAT/DOWN Classification")
    print("  Created by: Daryl James Padogdog")
    print("=" * 60)
    print("\n[INFO] Phase 1: Classification Foundation")
    print("[INFO] Features: Session-aware Attention, Walk-Forward Validation")
    print("[INFO] Target: 3-class softmax (ATR-scaled thresholds)")
    
    # Load data with date filter for Colab
    df = load_and_preprocess(TRAIN_FILE, date_start=DATE_START, date_end=DATE_END)
    
    # Exclude weekend gaps (council recommendation)
    df = exclude_weekend_gaps(df)
    
    # Create classification targets with ATR-scaled thresholds
    print("\n[INFO] Creating classification targets...")
    df = create_classification_targets(
        df,
        close_column='Close',
        atr_column='ATR',
        threshold_atr_mult=ATR_THRESHOLD_MULT,
        lookahead=1
    )
    
    # Display class distribution
    get_class_distribution(df)
    
    # Walk-forward validation setup
    print("\n[INFO] Setting up walk-forward validation (150/30 days)...")
    
    all_fold_metrics = []
    best_accuracy = 0
    best_model = None
    
    for train_df, test_df, fold in create_walk_forward_splits(
        df, 
        train_days=150, 
        test_days=30,
        date_column='Date',
        min_train_samples=1000
    ):
        print(f"\n{'='*50}")
        print(f"FOLD {fold}")
        print(f"Train: {train_df['Date'].min()} to {train_df['Date'].max()} ({len(train_df)} samples)")
        print(f"Test:  {test_df['Date'].min()} to {test_df['Date'].max()} ({len(test_df)} samples)")
        
        # Create sequences
        print("Creating sequences...")
        X_train, y_train, scaler = create_sequences_classification(
            train_df, FEATURE_COLS, LOOKBACK, scaler=None
        )
        X_test, y_test, _ = create_sequences_classification(
            test_df, FEATURE_COLS, LOOKBACK, scaler=scaler
        )
        
        print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
        
        if len(X_train) < 100 or len(X_test) < 10:
            print(f"Skipping fold {fold}: insufficient samples")
            continue
        
        # Build fresh model
        model = build_model((LOOKBACK, len(FEATURE_COLS)), NUM_CLASSES)
        
        if fold == 0:
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
            validation_data=(X_test, y_test),
            epochs=100,
            batch_size=64,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        print(f"\nFold {fold} Results:")
        print(f"  Test Accuracy: {test_acc*100:.2f}%")
        print(f"  Test Loss: {test_loss:.4f}")
        
        all_fold_metrics.append({
            'fold': fold,
            'accuracy': test_acc,
            'loss': test_loss
        })
        
        # Track best model
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            best_model = model
            best_scaler = scaler
        
        # Free memory
        del X_train, y_train, X_test, y_test
        gc.collect()
    
    # Summary
    print("\n" + "=" * 60)
    print("  WALK-FORWARD VALIDATION SUMMARY")
    print("=" * 60)
    
    if all_fold_metrics:
        accuracies = [m['accuracy'] for m in all_fold_metrics]
        print(f"\n  Folds Completed: {len(all_fold_metrics)}")
        print(f"  Mean Accuracy:   {np.mean(accuracies)*100:.2f}%")
        print(f"  Std Accuracy:    {np.std(accuracies)*100:.2f}%")
        print(f"  Best Accuracy:   {np.max(accuracies)*100:.2f}%")
        print(f"  Worst Accuracy:  {np.min(accuracies)*100:.2f}%")
        
        # Save best model
        if best_model is not None:
            best_model.save(MODEL_NAME)
            print(f"\n✓ Best model saved as '{MODEL_NAME}'")
            
            # Save scaler
            with open(SCALER_NAME, 'wb') as f:
                pickle.dump(best_scaler, f)
            print(f"✓ Scaler saved as '{SCALER_NAME}'")
        
        # Check if we meet the 65% threshold
        mean_acc = np.mean(accuracies) * 100
        if mean_acc >= 65:
            print(f"\n✅ SUCCESS: Mean accuracy {mean_acc:.1f}% exceeds 65% threshold!")
            print("   Ready to proceed to Phase 2 (Risk Shell)")
        else:
            print(f"\n⚠️  Mean accuracy {mean_acc:.1f}% below 65% target.")
            print("   Consider: adjusting thresholds, more features, or hyperparameter tuning")
    else:
        print("No folds completed. Check data availability.")
    
    print("\n" + "=" * 60)
