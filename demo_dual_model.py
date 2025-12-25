# ===========================================
# HYBRID DUAL-TIMEFRAME DEMO v2.0
# ===========================================
# AI Council Upgrade Implementation:
# - 3-class classification (UP/FLAT/DOWN) interpretation
# - Volatility-adaptive triggers (k×ATR thresholds)
# - Session-aware confidence scoring
# - Full risk shell integration (ATR SL/TP, Kelly sizing)
# - Monte-Carlo validation
# ===========================================

import numpy as np
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.models import load_model

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import custom modules
import sys
sys.path.insert(0, '.')
from utils.indicators import (
    add_all_indicators,
    FEATURE_COLS_ADVANCED
)
from utils.risk_shell import RiskShell, RiskConfig, monte_carlo_simulation
from utils.metrics import comprehensive_metrics, print_metrics_report, calculate_sortino_ratio

# ===========================================
# CONFIGURATION
# ===========================================
BIG_BROTHER_MODEL = 'big_brother_v2_classification.keras'
LITTLE_BROTHER_MODEL = 'little_brother_v2_classification.keras'
SCALER_BIG = 'scaler_big_brother.pkl'
SCALER_LITTLE = 'scaler_little_brother.pkl'

TEST_5M = 'DATA_SET/TEST_5m.csv'
TEST_1H = 'DATA_SET/TEST_1h.csv'

LOOKBACK_BIG = 168   # 1 week of hourly data
LOOKBACK_SMALL = 48  # 4 hours of 5-min data

FEATURE_COLS = FEATURE_COLS_ADVANCED
NUM_CLASSES = 3  # DOWN=0, FLAT=1, UP=2

# Risk configuration
RISK_CONFIG = RiskConfig(
    equity_risk=0.01,
    sl_atr_mult=1.5,
    tp_atr_mult=2.5,
    max_hold_bars_5m=576,
    kelly_divisor=20,
    max_consecutive_losses=3,
    min_confidence=0.55,
    min_adx=25.0,
    atr_trigger_mult=0.55
)

# ===========================================
# DATA LOADING
# ===========================================
def load_and_preprocess(filepath):
    """Load data and add all technical indicators."""
    print(f"Loading {filepath}...")
    df = pd.read_csv(filepath, sep=';')
    df['Date'] = pd.to_datetime(df['Date'], format='%Y.%m.%d %H:%M')
    df = df.sort_values('Date').reset_index(drop=True)
    df = add_all_indicators(df, include_advanced=True)
    df = df.dropna().reset_index(drop=True)
    float_cols = df.select_dtypes(include=['float64']).columns
    df[float_cols] = df[float_cols].astype('float32')
    print(f"Loaded {len(df)} rows")
    return df


def interpret_softmax(probs, class_names=['DOWN', 'FLAT', 'UP']):
    """
    Interpret softmax probabilities.
    
    Returns:
        pred_class: 0, 1, or 2
        direction: 'DOWN', 'FLAT', or 'UP'
        confidence: max probability
    """
    pred_class = np.argmax(probs)
    confidence = probs[pred_class]
    direction = class_names[pred_class]
    return pred_class, direction, confidence


# ===========================================
# CLASSIFICATION-BASED BACKTEST
# ===========================================
def classification_backtest(
    df_5m,
    little_probs,  # (N, 3) softmax probabilities
    df_1h,
    big_probs,     # (N, 3) softmax probabilities
    lookback_5m,
    lookback_1h,
    risk_config,
    initial_capital=200.0
):
    """
    Backtest with v2.0 classification models and risk shell.
    """
    # Initialize risk shell
    risk_shell = RiskShell(config=risk_config, initial_equity=initial_capital)
    
    results = {
        'total_signals': 0,
        'entries_attempted': 0,
        'trades_executed': 0,
        'filtered_signals': 0,
        'initial_balance': initial_capital,
    }
    
    signals_log = []
    equity_curve = [initial_capital]
    trade_returns_pct = []
    
    # Align data
    df_5m_subset = df_5m.iloc[lookback_5m-1:lookback_5m-1+len(little_probs)].copy()
    df_1h_subset = df_1h.iloc[lookback_1h-1:lookback_1h-1+len(big_probs)].copy()
    
    # 1H Mapping (hourly trend data)
    df_1h_subset['Hour'] = df_1h_subset['Date'].dt.floor('H')
    hourly_data = df_1h_subset.set_index('Hour').to_dict('index')
    
    # Create hourly probability mapping
    hourly_probs = {}
    for i, row in df_1h_subset.iterrows():
        hour_key = row['Hour']
        if i - (lookback_1h - 1) < len(big_probs):
            prob_idx = i - (lookback_1h - 1)
            if prob_idx >= 0:
                hourly_probs[hour_key] = big_probs[prob_idx]
    
    # Main loop
    for i in range(len(df_5m_subset) - 1):
        row = df_5m_subset.iloc[i]
        next_row = df_5m_subset.iloc[i + 1]
        current_price = row['Close']
        current_high = row['High']
        current_low = row['Low']
        atr = row['ATR']
        adx = row['ADX']
        rsi = row['RSI']
        pattern_strength = row.get('pattern_strength', 0)
        current_time = row['Date']
        
        # Get 5M prediction
        if i < len(little_probs):
            probs_5m = little_probs[i]
        else:
            continue
        
        pred_5m_class, pred_5m_dir, conf_5m = interpret_softmax(probs_5m)
        
        # Get 1H trend prediction
        hour_key = row['Date'].floor('H')
        if hour_key in hourly_probs:
            probs_1h = hourly_probs[hour_key]
            pred_1h_class, pred_1h_dir, conf_1h = interpret_softmax(probs_1h)
        else:
            pred_1h_class, pred_1h_dir, conf_1h = 1, 'FLAT', 0.33
        
        results['total_signals'] += 1
        
        # === CHECK EXIT CONDITIONS ===
        if risk_shell.state.current_position is not None:
            should_exit, exit_reason, pnl = risk_shell.check_exit_conditions(
                current_price, current_high, current_low
            )
            
            if should_exit:
                trade_result = risk_shell.close_position(current_price, exit_reason)
                if trade_result.get('pnl_pct'):
                    trade_returns_pct.append(trade_result['pnl_pct'])
                equity_curve.append(risk_shell.state.equity)
                
                signals_log.append({
                    'date': current_time,
                    'action': 'EXIT',
                    'reason': exit_reason,
                    'price': current_price,
                    'pnl': pnl,
                    'equity': risk_shell.state.equity
                })
                continue
        
        # === CHECK ENTRY CONDITIONS ===
        # Dual-timeframe agreement check
        dual_agree = False
        entry_direction = None
        
        if pred_1h_dir == 'UP' and pred_5m_dir == 'UP':
            dual_agree = True
            entry_direction = 'LONG'
        elif pred_1h_dir == 'DOWN' and pred_5m_dir == 'DOWN':
            dual_agree = True
            entry_direction = 'SHORT'
        
        if not dual_agree:
            results['filtered_signals'] += 1
            signals_log.append({
                'date': current_time,
                'action': 'WAIT',
                'reason': f'No agreement: 1H={pred_1h_dir}, 5M={pred_5m_dir}',
                'price': current_price,
                'conf_1h': conf_1h,
                'conf_5m': conf_5m
            })
            continue
        
        results['entries_attempted'] += 1
        
        # Risk shell entry check
        signal, confidence, reason = risk_shell.check_entry_conditions(
            softmax_probs=probs_5m,
            adx=adx,
            rsi=rsi,
            atr=atr,
            current_price=current_price,
            pattern_strength=int(pattern_strength) if pd.notna(pattern_strength) else 0
        )
        
        if signal is None:
            results['filtered_signals'] += 1
            signals_log.append({
                'date': current_time,
                'action': 'FILTERED',
                'reason': reason,
                'price': current_price,
                'conf': confidence
            })
            continue
        
        # Execute entry
        entry_result = risk_shell.open_position(
            signal=signal,
            entry_price=current_price,
            atr=atr,
            entry_time=current_time
        )
        
        results['trades_executed'] += 1
        
        signals_log.append({
            'date': current_time,
            'action': 'ENTRY',
            'direction': signal,
            'price': current_price,
            'sl': entry_result['stop_loss'],
            'tp': entry_result['take_profit'],
            'size': entry_result['position_size'],
            'conf_1h': conf_1h,
            'conf_5m': conf_5m
        })
        
        equity_curve.append(risk_shell.state.equity)
    
    # Close any remaining position
    if risk_shell.state.current_position is not None:
        final_price = df_5m_subset.iloc[-1]['Close']
        risk_shell.close_position(final_price, "End of backtest")
    
    # Get summary from risk shell
    summary = risk_shell.get_summary()
    
    results.update({
        'wins': summary['wins'],
        'losses': summary['losses'],
        'win_rate': summary['win_rate'],
        'total_pnl': summary['total_pnl'],
        'final_balance': risk_shell.state.equity,
        'net_profit': risk_shell.state.equity - initial_capital,
        'avg_pnl': summary.get('avg_pnl', 0),
        'avg_bars_held': summary.get('avg_bars_held', 0),
        'consecutive_losses': summary['consecutive_losses'],
        'was_halted': summary['is_halted']
    })
    
    # Calculate Sharpe ratio
    if trade_returns_pct:
        returns = np.array(trade_returns_pct)
        if returns.std() > 0:
            # Annualized Sharpe (assuming ~288 5-min bars per day)
            sharpe = np.sqrt(252 * 288) * returns.mean() / returns.std()
        else:
            sharpe = 0
        results['sharpe_ratio'] = sharpe
    else:
        results['sharpe_ratio'] = 0
    
    return results, signals_log, equity_curve, trade_returns_pct, risk_shell


# ===========================================
# MAIN
# ===========================================
if __name__ == "__main__":
    print("=" * 60)
    print("  HYBRID DUAL-TIMEFRAME DEMO v2.0")
    print("  AI Council Classification + Risk Shell")
    print("=" * 60)
    
    # ===== LOAD MODELS =====
    print("\n[1/6] Loading classification models...")
    
    try:
        big_model = load_model(BIG_BROTHER_MODEL, compile=False)
        big_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print(f"  ✓ Big Brother (1H) loaded - Classification model")
    except Exception as e:
        print(f"  ✗ Big Brother not found: {e}")
        print("    Run train_big_brother.py first!")
        big_model = None
    
    try:
        little_model = load_model(LITTLE_BROTHER_MODEL, compile=False)
        little_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print(f"  ✓ Little Brother (5M) loaded - Classification model")
    except Exception as e:
        print(f"  ✗ Little Brother not found: {e}")
        print("    Run train_little_brother.py first!")
        little_model = None
    
    if not big_model or not little_model:
        print("\n⚠️  Models not found. Please train first:")
        print("   1. python train/train_big_brother.py")
        print("   2. python train/train_little_brother.py")
        exit()
    
    # Load scalers
    try:
        with open(SCALER_BIG, 'rb') as f:
            scaler_big = pickle.load(f)
        with open(SCALER_LITTLE, 'rb') as f:
            scaler_little = pickle.load(f)
        print("  ✓ Scalers loaded")
    except Exception as e:
        print(f"  ✗ Scaler error: {e}")
        exit()
    
    # ===== LOAD DATA =====
    print("\n[2/6] Loading test data...")
    df_5m = load_and_preprocess(TEST_5M)
    df_1h = load_and_preprocess(TEST_1H)
    
    # ===== RUN PREDICTIONS =====
    print("\n[3/6] Running classification predictions...")
    
    # Scale data
    df_5m_scaled = df_5m.copy()
    df_5m_scaled[FEATURE_COLS] = scaler_little.transform(df_5m[FEATURE_COLS])
    
    df_1h_scaled = df_1h.copy()
    df_1h_scaled[FEATURE_COLS] = scaler_big.transform(df_1h[FEATURE_COLS])
    
    # Little Brother (5M) predictions
    print("  Running Little Brother (5M)...")
    predict_5m = tf.keras.utils.timeseries_dataset_from_array(
        data=df_5m_scaled[FEATURE_COLS].values,
        targets=None,
        sequence_length=LOOKBACK_SMALL,
        batch_size=256,
        shuffle=False
    )
    little_probs = little_model.predict(predict_5m, verbose=0)
    print(f"  ✓ {len(little_probs)} classifications (5M)")
    
    # Big Brother (1H) predictions
    print("  Running Big Brother (1H)...")
    predict_1h = tf.keras.utils.timeseries_dataset_from_array(
        data=df_1h_scaled[FEATURE_COLS].values,
        targets=None,
        sequence_length=LOOKBACK_BIG,
        batch_size=64,
        shuffle=False
    )
    big_probs = big_model.predict(predict_1h, verbose=0)
    print(f"  ✓ {len(big_probs)} classifications (1H)")
    
    # ===== CLASSIFICATION BACKTEST =====
    print("\n[4/6] Running v2.0 Backtest with Risk Shell...")
    
    results, signals, equity_curve, trade_returns, risk_shell = classification_backtest(
        df_5m, little_probs,
        df_1h, big_probs,
        LOOKBACK_SMALL, LOOKBACK_BIG,
        RISK_CONFIG,
        initial_capital=200.0
    )
    
    # ===== MONTE-CARLO SIMULATION =====
    print("\n[5/6] Running Monte-Carlo Simulation (5k paths)...")
    
    if trade_returns:
        mc_results = monte_carlo_simulation(
            trade_returns,
            n_simulations=5000,
            n_trades=100,
            initial_equity=200.0
        )
    else:
        mc_results = {'error': 'No trades for simulation'}
    
    # ===== DISPLAY RESULTS =====
    print("\n" + "=" * 60)
    print("  v2.0 CLASSIFICATION BACKTEST RESULTS")
    print("  (AI Council: Transaction Costs Applied)")
    print("=" * 60)
    
    print(f"\n  📊 Signal Analysis:")
    print(f"     Total Candles:       {results['total_signals']}")
    print(f"     Entry Attempts:      {results['entries_attempted']}")
    print(f"     Trades Executed:     {results['trades_executed']}")
    print(f"     Filtered Signals:    {results['filtered_signals']}")
    
    print(f"\n  💰 Trading Performance:")
    print(f"     Wins:                {results['wins']}")
    print(f"     Losses:              {results['losses']}")
    print(f"     Win Rate:            {results['win_rate']:.2f}%")
    print(f"     Sharpe Ratio:        {results['sharpe_ratio']:.2f}")
    
    # Calculate Sortino ratio
    if trade_returns:
        returns_arr = np.array(trade_returns) / 100
        sortino = calculate_sortino_ratio(returns_arr)
        print(f"     Sortino Ratio:       {sortino:.2f}")
    
    print(f"     Avg Bars Held:       {results['avg_bars_held']:.1f}")
    
    print(f"\n  💵 Paper Trade Simulation ($200 Start):")
    print(f"     Initial Balance:     ${results['initial_balance']:.2f}")
    print(f"     Final Balance:       ${results['final_balance']:.2f}")
    profit_sign = "+" if results['net_profit'] >= 0 else "-"
    print(f"     Net Profit:          {profit_sign}${abs(results['net_profit']):.2f}")
    
    # Transaction costs summary
    total_costs = risk_shell.state.total_costs_incurred
    if total_costs > 0:
        print(f"\n  💸 AI Council Transaction Costs (0.4 + 0.6 pip):")
        print(f"     Total Costs Paid:    ${total_costs:.2f}")
        cost_pct = (total_costs / results['initial_balance']) * 100
        print(f"     Costs as % Capital:  {cost_pct:.1f}%")
    
    print(f"\n  ⚠️  Risk Shell Status:")
    print(f"     Consecutive Losses:  {results['consecutive_losses']}")
    print(f"     Was Halted:          {'Yes' if results['was_halted'] else 'No'}")
    
    if 'error' not in mc_results:
        print(f"\n  🎲 Monte-Carlo Simulation ({mc_results['n_simulations']} paths, {mc_results['n_trades']} trades):")
        print(f"     Median Equity:       ${mc_results['median_equity']:.2f}")
        print(f"     5th Percentile:      ${mc_results['percentile_5']:.2f}")
        print(f"     95th Percentile:     ${mc_results['percentile_95']:.2f}")
        print(f"     Prob. of Profit:     {mc_results['prob_profit']:.1f}%")
        print(f"     Prob. of 2x:         {mc_results['prob_double']:.1f}%")
        print(f"     Median Max DD:       {mc_results['median_max_drawdown']:.1f}%")
    
    print("=" * 60)
    
    # ===== AI COUNCIL TARGET CHECKS =====
    print("\n  🎯 AI Council Target Checks:")
    sharpe_pass = results['sharpe_ratio'] >= 1.0
    edge_pass = (results['net_profit'] / results['initial_balance']) * 100 >= 5.0
    print(f"     Sharpe > 1.0:        {'✅ PASS' if sharpe_pass else '❌ FAIL'} ({results['sharpe_ratio']:.2f})")
    print(f"     Net Edge > 5%:       {'✅ PASS' if edge_pass else '❌ FAIL'} ({(results['net_profit']/results['initial_balance'])*100:.1f}%)")
    
    # ===== CHECK SUCCESS CRITERIA =====
    if results['win_rate'] >= 60 and results['sharpe_ratio'] >= 1.5:
        print("\n✅ SUCCESS: Win Rate ≥60% and Sharpe ≥1.5!")
        print("   Ready to promote to micro-live trading.")
    elif results['win_rate'] >= 60 and results['sharpe_ratio'] >= 1.0:
        print(f"\n⚠️  Win Rate OK ({results['win_rate']:.1f}%), Sharpe OK ({results['sharpe_ratio']:.2f})")
        print("   Meets AI Council baseline. Consider more optimization.")
    elif results['win_rate'] >= 60:
        print(f"\n⚠️  Win Rate OK ({results['win_rate']:.1f}%), but Sharpe {results['sharpe_ratio']:.2f} < 1.0")
        print("   Consider: tighter risk controls, better entry timing")
    else:
        print(f"\n⚠️  Win Rate {results['win_rate']:.1f}% below 60% target.")
        print("   Consider: adjusting confidence thresholds, more training data")
    
    # ===== CURRENT SIGNAL =====
    print("\n" + "=" * 60)
    print("  CURRENT TRADING SIGNAL")
    print("=" * 60)
    
    current_price = df_5m['Close'].iloc[-1]
    current_atr = df_5m['ATR'].iloc[-1]
    current_adx = df_5m['ADX'].iloc[-1]
    current_rsi = df_5m['RSI'].iloc[-1]
    
    # Get latest predictions
    latest_5m_probs = little_probs[-1]
    latest_1h_probs = big_probs[-1]
    
    _, dir_5m, conf_5m = interpret_softmax(latest_5m_probs)
    _, dir_1h, conf_1h = interpret_softmax(latest_1h_probs)
    
    print(f"\n  [Big Brother] 1H Classification:")
    print(f"     Direction:   {dir_1h}")
    print(f"     Confidence:  {conf_1h*100:.1f}%")
    print(f"     Probs:       DOWN={latest_1h_probs[0]*100:.1f}% | FLAT={latest_1h_probs[1]*100:.1f}% | UP={latest_1h_probs[2]*100:.1f}%")
    
    print(f"\n  [Little Brother] 5M Classification:")
    print(f"     Direction:   {dir_5m}")
    print(f"     Confidence:  {conf_5m*100:.1f}%")
    print(f"     Probs:       DOWN={latest_5m_probs[0]*100:.1f}% | FLAT={latest_5m_probs[1]*100:.1f}% | UP={latest_5m_probs[2]*100:.1f}%")
    
    print(f"\n  [Market State]:")
    print(f"     Price:       ${current_price:.2f}")
    print(f"     ATR:         ${current_atr:.2f}")
    print(f"     ADX:         {current_adx:.1f} {'(Trending)' if current_adx > 25 else '(Ranging)'}")
    print(f"     RSI:         {current_rsi:.1f}")
    
    # Final signal
    print(f"\n  {'─' * 40}")
    if dir_1h == dir_5m and dir_1h != 'FLAT':
        if conf_5m >= 0.55 and current_adx > 25:
            signal = '✅ ' + ('LONG' if dir_1h == 'UP' else 'SHORT') + ' NOW'
            sl, tp = risk_shell.calculate_sl_tp(current_price, current_atr, 'LONG' if dir_1h == 'UP' else 'SHORT')
            print(f"  >> FINAL SIGNAL: {signal}")
            print(f"     Stop-Loss:   ${sl:.2f}")
            print(f"     Take-Profit: ${tp:.2f}")
        else:
            print(f"  >> FINAL SIGNAL: ⚠️ WAIT (Low confidence or ranging)")
    else:
        print(f"  >> FINAL SIGNAL: ⏸️ WAIT (No dual-timeframe agreement)")
    
    print("=" * 60)
    
    # ===== VISUALIZATION =====
    print("\n[6/6] Generating charts...")
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            f'Classification Performance (Win Rate: {results["win_rate"]:.1f}%)',
            f'Equity Curve ($200 → ${results["final_balance"]:.2f})',
            'Classification Probabilities (5M)'
        )
    )
    
    # Row 1: Price with trade signals
    dates = df_5m['Date'].iloc[LOOKBACK_SMALL-1:LOOKBACK_SMALL-1+len(little_probs)]
    prices = df_5m['Close'].iloc[LOOKBACK_SMALL-1:LOOKBACK_SMALL-1+len(little_probs)]
    
    fig.add_trace(
        go.Scatter(x=dates, y=prices, mode='lines', name='Price', line=dict(color='#00FF00')),
        row=1, col=1
    )
    
    # Add entry/exit markers from signals
    entries = [s for s in signals if s['action'] == 'ENTRY']
    exits = [s for s in signals if s['action'] == 'EXIT']
    
    if entries:
        entry_dates = [e['date'] for e in entries]
        entry_prices = [e['price'] for e in entries]
        entry_colors = ['green' if e['direction'] == 'LONG' else 'red' for e in entries]
        fig.add_trace(
            go.Scatter(
                x=entry_dates, y=entry_prices,
                mode='markers', name='Entry',
                marker=dict(size=10, color=entry_colors, symbol='triangle-up')
            ),
            row=1, col=1
        )
    
    # Row 2: Equity curve
    fig.add_trace(
        go.Scatter(
            x=list(range(len(equity_curve))),
            y=equity_curve,
            mode='lines',
            name='Equity ($)',
            line=dict(color='#00BFFF', width=2),
            fill='tozeroy'
        ),
        row=2, col=1
    )
    
    # Row 3: Classification probabilities
    fig.add_trace(
        go.Scatter(x=dates, y=little_probs[:, 2], mode='lines', name='P(UP)', line=dict(color='green')),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(x=dates, y=little_probs[:, 0], mode='lines', name='P(DOWN)', line=dict(color='red')),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(x=dates, y=little_probs[:, 1], mode='lines', name='P(FLAT)', line=dict(color='gray', dash='dash')),
        row=3, col=1
    )
    
    fig.update_layout(
        template='plotly_dark',
        height=900,
        title='v2.0 Classification Results + Risk Shell'
    )
    fig.show()
    
    print("\n  📊 Charts displayed")
    print("=" * 60)
