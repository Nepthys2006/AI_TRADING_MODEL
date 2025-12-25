# ===========================================
# TRAIN LITTLE BROTHER MODEL v2.0 (5M Data)
# ===========================================
# AI Council Upgrade: Classification-Based Model
# - 3-class softmax output (UP/FLAT/DOWN)
# - Session-aware MultiHead Attention
# - ATR-scaled classification thresholds (calibrated for 5M)
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
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
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
TRAIN_FILE = 'DATA_SET/TRAIN_5m.csv'
MODEL_NAME = 'little_brother_v2_classification.keras'
SCALER_NAME = 'scaler_little_brother.pkl'

# ======== AI Council: Lookback Grid Search ========
# Test different lookback windows to find optimal per council recommendation
# 96 bars = 8 hours = one full London–NY overlap
LOOKBACK_GRID = [48, 64, 80, 96, 120]  # Council recommended range
LOOKBACK = 48  # Default if not using grid search
USE_GRID_SEARCH = True  # Set to True to enable grid search

# Date filter for Colab free tier (reduce memory usage)
DATE_START = '2018-01-01'  # Filter start date
DATE_END = '2024-12-31'    # Filter end date

# Classification thresholds (calibrated for 5M timeframe)
# Lower threshold than 1H since 5M moves are smaller
ATR_THRESHOLD_MULT = 0.3  # Threshold = 0.3 * ATR/Close for 5M
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
    
    Note: Slightly smaller than Big Brother for faster 5M inference
    """
    inputs = Input(shape=input_shape)
    
    # First LSTM block (slightly smaller for 5M)
    x = Bidirectional(LSTM(96, return_sequences=True))(inputs)
    x = Dropout(0.3)(x)
    
    # Save for skip connection
    skip = x
    
    # Second LSTM block
    x = Bidirectional(LSTM(48, return_sequences=True))(x)
    x = Dropout(0.3)(x)
    
    # ResNet-style skip connection
    # skip has 192 (96*2), x has 96 (48*2)
    x_proj = Dense(192)(x)  # Project x to 192
    x = Add()([x_proj, skip])
    x = LayerNormalization()(x)
    
    # Session-aware attention
    x = SessionAwareAttention(
        num_heads=4,
        key_dim=24,  # Smaller key_dim for 5M
        dropout_rate=0.1,
        recency_weight=2.5  # Higher recency weight for short-term
    )(x)
    
    # Classification head
    x = Dense(48, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(24, activation='relu')(x)
    
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
        y.append(targets[i-1])
    
    X = np.array(X, dtype='float32')
    y = to_categorical(np.array(y), num_classes=NUM_CLASSES)
    
    return X, y, scaler


# ===========================================
# MAIN
# ===========================================
if __name__ == "__main__":
    print("=" * 60)
    print("  LITTLE BROTHER v2.0 TRAINING (Classification Model)")
    print("  AI Council Upgrade: UP/FLAT/DOWN Classification")
    print("  Created by: Daryl James Padogdog")
    print("=" * 60)
    print("\n[INFO] Phase 1: Classification Foundation")
    print("[INFO] Features: Session-aware Attention, Walk-Forward Validation")
    print("[INFO] Target: 3-class softmax (ATR-scaled thresholds for 5M)")
    
    if USE_GRID_SEARCH:
        print(f"\n[INFO] AI Council: Lookback Grid Search ENABLED")
        print(f"[INFO] Testing lookbacks: {LOOKBACK_GRID}")
    
    # Load data with date filter for Colab
    df = load_and_preprocess(TRAIN_FILE, date_start=DATE_START, date_end=DATE_END)
    
    # Exclude weekend gaps
    df = exclude_weekend_gaps(df)
    
    # Create classification targets with ATR-scaled thresholds
    print("\n[INFO] Creating classification targets (5M calibration)...")
    df = create_classification_targets(
        df,
        close_column='Close',
        atr_column='ATR',
        threshold_atr_mult=ATR_THRESHOLD_MULT,  # Lower for 5M
        lookahead=1
    )
    
    # Display class distribution
    get_class_distribution(df)
    
    # ======== AI Council: Lookback Grid Search ========
    lookbacks_to_test = LOOKBACK_GRID if USE_GRID_SEARCH else [LOOKBACK]
    grid_results = []
    
    for current_lookback in lookbacks_to_test:
        print(f"\n{'#'*60}")
        print(f"  TESTING LOOKBACK = {current_lookback} bars ({current_lookback * 5} minutes)")
        print(f"{'#'*60}")
        
        all_fold_metrics = []
        best_accuracy = 0
        best_model = None
        best_scaler = None
        
        for train_df, test_df, fold in create_walk_forward_splits(
            df, 
            train_days=150, 
            test_days=30,
            date_column='Date',
            min_train_samples=5000  # Higher min for 5M data
        ):
            print(f"\n{'='*50}")
            print(f"FOLD {fold} (Lookback={current_lookback})")
            print(f"Train: {train_df['Date'].min()} to {train_df['Date'].max()} ({len(train_df)} samples)")
            print(f"Test:  {test_df['Date'].min()} to {test_df['Date'].max()} ({len(test_df)} samples)")
            
            # Create sequences
            X_train, y_train, scaler = create_sequences_classification(
                train_df, FEATURE_COLS, current_lookback, scaler=None
            )
            X_test, y_test, _ = create_sequences_classification(
                test_df, FEATURE_COLS, current_lookback, scaler=scaler
            )
            
            print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
            print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
            
            if len(X_train) < 500 or len(X_test) < 50:
                print(f"Skipping fold {fold}: insufficient samples")
                continue
            
            # Build fresh model
            model = build_model((current_lookback, len(FEATURE_COLS)), NUM_CLASSES)
            
            if fold == 0 and current_lookback == lookbacks_to_test[0]:
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
                batch_size=128,  # Larger batch for 5M data volume
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
                'loss': test_loss,
                'lookback': current_lookback
            })
            
            # Track best model
            if test_acc > best_accuracy:
                best_accuracy = test_acc
                best_model = model
                best_scaler = scaler
            
            # Free memory
            del X_train, y_train, X_test, y_test
            gc.collect()
        
        # Store grid search results for this lookback
        if all_fold_metrics:
            mean_acc = np.mean([m['accuracy'] for m in all_fold_metrics])
            grid_results.append({
                'lookback': current_lookback,
                'mean_accuracy': mean_acc,
                'std_accuracy': np.std([m['accuracy'] for m in all_fold_metrics]),
                'best_accuracy': np.max([m['accuracy'] for m in all_fold_metrics]),
                'folds': len(all_fold_metrics),
                'best_model': best_model,
                'best_scaler': best_scaler
            })
    
    # ======== AI Council: Grid Search Summary ========
    print("\n" + "=" * 60)
    print("  AI COUNCIL: LOOKBACK GRID SEARCH RESULTS")
    print("=" * 60)
    
    if grid_results:
        print(f"\n  {'Lookback':<10} {'Mean Acc':<12} {'Std':<10} {'Best':<10} {'Folds':<8}")
        print("  " + "-" * 50)
        for r in grid_results:
            print(f"  {r['lookback']:<10} {r['mean_accuracy']*100:>8.2f}%   {r['std_accuracy']*100:>6.2f}%   {r['best_accuracy']*100:>6.2f}%   {r['folds']:<8}")
        
        # Find optimal lookback
        best_result = max(grid_results, key=lambda x: x['mean_accuracy'])
        optimal_lookback = best_result['lookback']
        
        print(f"\n  🎯 OPTIMAL LOOKBACK: {optimal_lookback} bars ({optimal_lookback * 5} minutes)")
        print(f"     Mean Accuracy: {best_result['mean_accuracy']*100:.2f}%")
        
        # Save the best model
        if best_result['best_model'] is not None:
            model_name = f'little_brother_v2_lb{optimal_lookback}.keras'
            best_result['best_model'].save(model_name)
            print(f"\n✓ Best model saved as '{model_name}'")
            
            # Also save as default name
            best_result['best_model'].save(MODEL_NAME)
            print(f"✓ Also saved as '{MODEL_NAME}'")
            
            with open(SCALER_NAME, 'wb') as f:
                pickle.dump(best_result['best_scaler'], f)
            print(f"✓ Scaler saved as '{SCALER_NAME}'")
            
            # Save optimal lookback to config file
            with open('optimal_lookback_5m.txt', 'w') as f:
                f.write(f"OPTIMAL_LOOKBACK={optimal_lookback}\n")
                f.write(f"MEAN_ACCURACY={best_result['mean_accuracy']*100:.2f}\n")
            print(f"✓ Config saved to 'optimal_lookback_5m.txt'")
        
        # Check 65% threshold
        mean_acc = best_result['mean_accuracy'] * 100
        if mean_acc >= 65:
            print(f"\n✅ SUCCESS: Mean accuracy {mean_acc:.1f}% exceeds 65% threshold!")
            print("   Ready to proceed to Phase 2 (Risk Shell)")
        else:
            print(f"\n⚠️  Mean accuracy {mean_acc:.1f}% below 65% target.")
            print("   Consider: adjusting thresholds, more features, or hyperparameter tuning")
    else:
        print("No grid search results. Check data availability.")
    
    print("\n" + "=" * 60)
