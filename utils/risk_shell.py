# ===========================================
# RISK SHELL MODULE v2.0
# ===========================================
# AI Council Risk Management Implementation:
# - 1% equity risk per trade
# - ATR-based Stop Loss (1.5×ATR) and Take Profit (2.5×ATR)
# - Kelly/20 position sizing
# - Consecutive loss protection (halt after 3 losses)
# - Max hold time enforcement (48 hours)
# - Trading session restrictions
# ===========================================

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Tuple
from datetime import datetime, timedelta


@dataclass
class RiskConfig:
    """Configuration for risk management."""
    equity_risk: float = 0.01          # 1% equity risk per trade
    sl_atr_mult: float = 2.0           # Stop-loss = 2.0 × ATR (Council: 2× 5-min ATR)
    tp_atr_mult: float = 2.5           # Take-profit = 2.5 × ATR
    max_hold_bars_5m: int = 576        # 48 hours in 5-min bars
    kelly_divisor: int = 20            # Conservative Kelly sizing
    max_consecutive_losses: int = 3    # Halt after 3 losses
    min_confidence: float = 0.55       # Min softmax confidence for entry
    min_adx: float = 25.0              # ADX threshold for trending
    
    # Volatility-adaptive thresholds
    atr_trigger_mult: float = 0.55     # Signal if |pred_return| > 0.55 × ATR/Close
    
    # Dynamic RSI bands
    rsi_base_upper: float = 70.0
    rsi_base_lower: float = 30.0
    rsi_atr_adjustment: float = 0.2    # Adjust by 0.2 × ATR14
    
    # ======== AI Council Additions ========
    # Transaction costs (Council: 0.3-0.5 pip spread, 0.5-0.75 pip slippage)
    spread_pips: float = 0.4           # Realistic spread
    slippage_pips: float = 0.6         # Realistic slippage
    pip_value: float = 0.0001          # Standard pip (adjust for JPY)
    
    # Trade frequency limits (Council: 3-4 per session, max 5/day)
    max_trades_per_session: int = 4    # Cap per trading session
    max_signals_per_day: int = 5       # Hard daily limit


@dataclass
class TradeState:
    """Current trading state."""
    equity: float
    consecutive_losses: int = 0
    is_halted: bool = False
    current_position: Optional[str] = None  # 'LONG', 'SHORT', or None
    entry_price: float = 0.0
    entry_time: Optional[datetime] = None
    stop_loss: float = 0.0
    take_profit: float = 0.0
    position_size: float = 0.0
    bars_held: int = 0
    
    # ======== AI Council Additions ========
    trades_today: int = 0              # Daily trade counter
    last_trade_date: Optional[datetime] = None  # For daily reset
    session_trades: int = 0            # Trades in current session
    total_costs_incurred: float = 0.0  # Cumulative transaction costs


class RiskShell:
    """
    Risk management shell for trading system.
    Implements council-recommended risk controls.
    """
    
    def __init__(self, config: RiskConfig = None, initial_equity: float = 200.0):
        self.config = config or RiskConfig()
        self.state = TradeState(equity=initial_equity)
        self.trade_history = []
        
    def calculate_position_size(
        self,
        entry_price: float,
        atr: float,
        win_rate: float = 0.55
    ) -> float:
        """
        Calculate position size using Kelly/20 formula.
        
        Kelly fraction = (W × R - L) / R
        Where:
            W = win probability
            L = loss probability  
            R = average win / average loss ratio (using TP/SL ratio)
        """
        # Risk-reward ratio from ATR multipliers
        rr_ratio = self.config.tp_atr_mult / self.config.sl_atr_mult
        
        # Kelly formula
        loss_rate = 1 - win_rate
        kelly_fraction = (win_rate * rr_ratio - loss_rate) / rr_ratio
        
        # Conservative Kelly (divide by 20)
        conservative_kelly = max(0, kelly_fraction / self.config.kelly_divisor)
        
        # Apply consecutive loss penalty
        if self.state.consecutive_losses >= 2:
            conservative_kelly *= 0.5  # Halve size after 2 losses
        
        # Maximum risk per trade
        max_risk_amount = self.state.equity * self.config.equity_risk
        
        # Stop-loss distance
        sl_distance = atr * self.config.sl_atr_mult
        
        # Position size based on risk
        if sl_distance > 0:
            risk_based_size = max_risk_amount / sl_distance
        else:
            risk_based_size = 0
        
        # Apply Kelly adjustment
        final_size = risk_based_size * min(conservative_kelly * self.config.kelly_divisor, 1.0)
        
        return final_size
    
    def calculate_sl_tp(
        self,
        entry_price: float,
        atr: float,
        direction: str  # 'LONG' or 'SHORT'
    ) -> Tuple[float, float]:
        """Calculate stop-loss and take-profit levels."""
        sl_distance = atr * self.config.sl_atr_mult
        tp_distance = atr * self.config.tp_atr_mult
        
        if direction == 'LONG':
            stop_loss = entry_price - sl_distance
            take_profit = entry_price + tp_distance
        else:  # SHORT
            stop_loss = entry_price + sl_distance
            take_profit = entry_price - tp_distance
        
        return stop_loss, take_profit
    
    def get_dynamic_rsi_bands(self, atr: float) -> Tuple[float, float]:
        """
        Calculate dynamic RSI bands based on ATR.
        Higher volatility = wider bands.
        """
        adjustment = self.config.rsi_atr_adjustment * atr
        
        upper = self.config.rsi_base_upper - adjustment
        lower = self.config.rsi_base_lower + adjustment
        
        # Clamp to reasonable range
        upper = max(60, min(80, upper))
        lower = max(20, min(40, lower))
        
        return lower, upper
    
    def check_entry_conditions(
        self,
        softmax_probs: np.ndarray,  # [prob_down, prob_flat, prob_up]
        adx: float,
        rsi: float,
        atr: float,
        current_price: float,
        pattern_strength: int = 0,  # From pattern confirmation
        current_time: datetime = None  # For trade frequency tracking
    ) -> Tuple[Optional[str], float, str]:
        """
        Check if entry conditions are met.
        
        Returns:
            (signal, confidence, reason)
            signal: 'LONG', 'SHORT', or None
            confidence: softmax probability
            reason: explanation string
        """
        # Check if halted due to consecutive losses
        if self.state.is_halted:
            return None, 0.0, "HALTED: 3 consecutive losses"
        
        # Check if already in position
        if self.state.current_position is not None:
            return None, 0.0, "Already in position"
        
        # ======== AI Council: Trade Frequency Limits ========
        if current_time is not None:
            # Reset daily counter if new day
            if self.state.last_trade_date is None or current_time.date() != self.state.last_trade_date.date():
                self.state.trades_today = 0
                self.state.session_trades = 0
            
            # Check daily limit
            if self.state.trades_today >= self.config.max_signals_per_day:
                return None, 0.0, f"Daily limit reached: {self.state.trades_today}/{self.config.max_signals_per_day}"
            
            # Check session limit (session changes at 00:00, 08:00, 16:00 UTC)
            if self.state.session_trades >= self.config.max_trades_per_session:
                current_session = (current_time.hour // 8)
                if self.state.last_trade_date:
                    last_session = (self.state.last_trade_date.hour // 8)
                    if current_session != last_session:
                        self.state.session_trades = 0  # Reset for new session
                    else:
                        return None, 0.0, f"Session limit reached: {self.state.session_trades}/{self.config.max_trades_per_session}"
        
        # Get predicted direction and confidence
        pred_class = np.argmax(softmax_probs)
        confidence = softmax_probs[pred_class]
        
        # Minimum confidence check
        if confidence < self.config.min_confidence:
            return None, confidence, f"Low confidence: {confidence:.2f} < {self.config.min_confidence}"
        
        # ADX filter (require trending market)
        if adx < self.config.min_adx:
            return None, confidence, f"Choppy market: ADX {adx:.1f} < {self.config.min_adx}"
        
        # Dynamic RSI bands
        rsi_lower, rsi_upper = self.get_dynamic_rsi_bands(atr)
        
        # Map class to signal
        if pred_class == 2:  # UP
            signal = 'LONG'
            # RSI filter for overbought
            if rsi > rsi_upper:
                return None, confidence, f"Overbought: RSI {rsi:.1f} > {rsi_upper:.1f}"
            # Pattern confirmation (optional, require some bullish pattern)
            if pattern_strength < -2:
                return None, confidence, f"Bearish pattern: strength {pattern_strength}"
                
        elif pred_class == 0:  # DOWN
            signal = 'SHORT'
            # RSI filter for oversold
            if rsi < rsi_lower:
                return None, confidence, f"Oversold: RSI {rsi:.1f} < {rsi_lower:.1f}"
            # Pattern confirmation
            if pattern_strength > 2:
                return None, confidence, f"Bullish pattern: strength {pattern_strength}"
        else:
            # FLAT prediction
            return None, confidence, "FLAT prediction - no trade"
        
        return signal, confidence, "Entry conditions met"
    
    def open_position(
        self,
        signal: str,
        entry_price: float,
        atr: float,
        entry_time: datetime,
        win_rate: float = 0.55
    ) -> dict:
        """Open a new position."""
        # Calculate position size
        position_size = self.calculate_position_size(entry_price, atr, win_rate)
        
        # Calculate SL/TP
        stop_loss, take_profit = self.calculate_sl_tp(entry_price, atr, signal)
        
        # Update state
        self.state.current_position = signal
        self.state.entry_price = entry_price
        self.state.entry_time = entry_time
        self.state.stop_loss = stop_loss
        self.state.take_profit = take_profit
        self.state.position_size = position_size
        self.state.bars_held = 0
        
        # ======== AI Council: Update trade frequency counters ========
        self.state.trades_today += 1
        self.state.session_trades += 1
        self.state.last_trade_date = entry_time
        
        return {
            'action': 'OPEN',
            'direction': signal,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'position_size': position_size,
            'risk_amount': self.state.equity * self.config.equity_risk
        }
    
    def check_exit_conditions(
        self,
        current_price: float,
        current_high: float,
        current_low: float
    ) -> Tuple[bool, str, float]:
        """
        Check if exit conditions are met.
        
        Returns:
            (should_exit, reason, pnl)
        """
        if self.state.current_position is None:
            return False, "No position", 0.0
        
        self.state.bars_held += 1
        
        direction = self.state.current_position
        entry = self.state.entry_price
        sl = self.state.stop_loss
        tp = self.state.take_profit
        
        # Check max hold time
        if self.state.bars_held >= self.config.max_hold_bars_5m:
            pnl = self._calculate_pnl(current_price)
            return True, "Max hold time reached", pnl
        
        if direction == 'LONG':
            # Check stop-loss hit
            if current_low <= sl:
                pnl = self._calculate_pnl(sl)
                return True, "Stop-loss hit", pnl
            # Check take-profit hit
            if current_high >= tp:
                pnl = self._calculate_pnl(tp)
                return True, "Take-profit hit", pnl
        else:  # SHORT
            # Check stop-loss hit
            if current_high >= sl:
                pnl = self._calculate_pnl(sl)
                return True, "Stop-loss hit", pnl
            # Check take-profit hit
            if current_low <= tp:
                pnl = self._calculate_pnl(tp)
                return True, "Take-profit hit", pnl
        
        return False, "Position open", 0.0
    
    def close_position(self, exit_price: float, reason: str) -> dict:
        """Close current position and update state."""
        if self.state.current_position is None:
            return {'action': 'NONE', 'reason': 'No position to close'}
        
        # Calculate gross PnL and transaction costs
        gross_pnl = self._calculate_gross_pnl(exit_price)
        transaction_cost = self._calculate_transaction_cost()
        net_pnl = gross_pnl - transaction_cost
        
        # Track cumulative costs
        self.state.total_costs_incurred += transaction_cost
        
        # Update equity with NET PnL
        self.state.equity += net_pnl
        
        # Update consecutive losses
        if net_pnl < 0:
            self.state.consecutive_losses += 1
            if self.state.consecutive_losses >= self.config.max_consecutive_losses:
                self.state.is_halted = True
        else:
            self.state.consecutive_losses = 0
        
        # Record trade with cost breakdown
        trade_record = {
            'action': 'CLOSE',
            'direction': self.state.current_position,
            'entry_price': self.state.entry_price,
            'exit_price': exit_price,
            'gross_pnl': gross_pnl,
            'transaction_cost': transaction_cost,
            'pnl': net_pnl,  # Net PnL after costs
            'pnl_pct': net_pnl / (self.state.entry_price * self.state.position_size) * 100 if self.state.position_size > 0 else 0,
            'bars_held': self.state.bars_held,
            'reason': reason,
            'equity_after': self.state.equity
        }
        self.trade_history.append(trade_record)
        
        # Reset position state
        self.state.current_position = None
        self.state.entry_price = 0.0
        self.state.entry_time = None
        self.state.stop_loss = 0.0
        self.state.take_profit = 0.0
        self.state.position_size = 0.0
        self.state.bars_held = 0
        
        return trade_record
    
    def _calculate_gross_pnl(self, exit_price: float) -> float:
        """Calculate gross PnL for current position (before costs)."""
        if self.state.current_position == 'LONG':
            return (exit_price - self.state.entry_price) * self.state.position_size
        else:  # SHORT
            return (self.state.entry_price - exit_price) * self.state.position_size
    
    def _calculate_transaction_cost(self) -> float:
        """
        Calculate transaction cost for current position.
        AI Council: 0.4 pip spread + 0.6 pip slippage = 1.0 pip per side
        Round-trip = 2.0 pips
        """
        total_pips = (self.config.spread_pips + self.config.slippage_pips) * 2
        pip_value_usd = self.config.pip_value * self.state.entry_price
        return total_pips * pip_value_usd * self.state.position_size
    
    def _calculate_pnl(self, exit_price: float) -> float:
        """Calculate NET PnL for current position (after costs)."""
        gross = self._calculate_gross_pnl(exit_price)
        cost = self._calculate_transaction_cost()
        return gross - cost
    
    def reset_halt(self):
        """Reset halt status (e.g., after cooling period or manual override)."""
        self.state.is_halted = False
        self.state.consecutive_losses = 0
    
    def get_summary(self) -> dict:
        """Get trading summary statistics."""
        if not self.trade_history:
            return {
                'total_trades': 0,
                'wins': 0,
                'losses': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'current_equity': self.state.equity
            }
        
        trades = pd.DataFrame(self.trade_history)
        wins = (trades['pnl'] > 0).sum()
        losses = (trades['pnl'] <= 0).sum()
        
        return {
            'total_trades': len(trades),
            'wins': int(wins),
            'losses': int(losses),
            'win_rate': wins / len(trades) * 100 if len(trades) > 0 else 0,
            'total_pnl': trades['pnl'].sum(),
            'avg_pnl': trades['pnl'].mean(),
            'max_win': trades['pnl'].max(),
            'max_loss': trades['pnl'].min(),
            'avg_bars_held': trades['bars_held'].mean(),
            'current_equity': self.state.equity,
            'consecutive_losses': self.state.consecutive_losses,
            'is_halted': self.state.is_halted
        }


def monte_carlo_simulation(
    trade_returns: list,
    n_simulations: int = 5000,
    n_trades: int = 100,
    initial_equity: float = 200.0
) -> dict:
    """
    Run Monte-Carlo simulation on historical trade returns.
    
    Args:
        trade_returns: List of trade return percentages
        n_simulations: Number of simulation paths
        n_trades: Number of trades per simulation
        initial_equity: Starting equity
        
    Returns:
        Simulation results with confidence intervals
    """
    if len(trade_returns) < 5:
        return {'error': 'Insufficient trade history for simulation'}
    
    returns = np.array(trade_returns)
    
    # Run simulations
    final_equities = []
    max_drawdowns = []
    
    for _ in range(n_simulations):
        # Random sample with replacement
        sampled_returns = np.random.choice(returns, size=n_trades, replace=True)
        
        # Calculate equity curve
        equity_curve = initial_equity * np.cumprod(1 + sampled_returns / 100)
        final_equity = equity_curve[-1]
        
        # Calculate max drawdown
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / peak
        max_dd = drawdown.max()
        
        final_equities.append(final_equity)
        max_drawdowns.append(max_dd)
    
    final_equities = np.array(final_equities)
    max_drawdowns = np.array(max_drawdowns)
    
    return {
        'median_equity': np.median(final_equities),
        'mean_equity': np.mean(final_equities),
        'percentile_5': np.percentile(final_equities, 5),
        'percentile_25': np.percentile(final_equities, 25),
        'percentile_75': np.percentile(final_equities, 75),
        'percentile_95': np.percentile(final_equities, 95),
        'prob_profit': (final_equities > initial_equity).mean() * 100,
        'prob_double': (final_equities > initial_equity * 2).mean() * 100,
        'median_max_drawdown': np.median(max_drawdowns) * 100,
        'worst_max_drawdown': np.max(max_drawdowns) * 100,
        'n_simulations': n_simulations,
        'n_trades': n_trades
    }
