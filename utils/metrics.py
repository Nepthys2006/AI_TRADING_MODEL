# ===========================================
# TRADING METRICS MODULE v1.0
# ===========================================
# AI Council Implementation:
# - Sharpe/Sortino ratio calculation
# - Transaction cost modeling (spread + slippage)
# - Max drawdown analysis
# - Trade frequency metrics
# ===========================================

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class TransactionCosts:
    """
    Transaction cost model per AI Council recommendation.
    Realistic spread and slippage for forex trading.
    """
    spread_pips: float = 0.4      # 0.3-0.5 pip spread
    slippage_pips: float = 0.6    # 0.5-0.75 pip slippage
    pip_value: float = 0.0001     # Standard pip for major pairs (adjust for JPY pairs)
    
    @property
    def total_cost_pips(self) -> float:
        """Total round-trip cost in pips (entry + exit)."""
        return (self.spread_pips + self.slippage_pips) * 2
    
    def cost_per_trade(self, entry_price: float, position_size: float = 1.0) -> float:
        """
        Calculate transaction cost for a single trade.
        
        Args:
            entry_price: Entry price of the trade
            position_size: Position size in lots/units
            
        Returns:
            Total cost in price units
        """
        cost_per_pip = self.pip_value * entry_price
        return self.total_cost_pips * cost_per_pip * position_size
    
    def cost_percentage(self, entry_price: float) -> float:
        """
        Calculate cost as percentage of entry price.
        
        Returns:
            Cost as decimal (e.g., 0.015 = 1.5%)
        """
        cost_per_pip = self.pip_value * entry_price
        total_cost = self.total_cost_pips * cost_per_pip
        return total_cost / entry_price


def calculate_sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252 * 288,  # 5-min bars per year
    annualize: bool = True
) -> float:
    """
    Calculate Sharpe Ratio.
    
    AI Council Target: Sharpe > 1.0 (cost-adjusted)
    
    Args:
        returns: Array of period returns (as decimals, not percentages)
        risk_free_rate: Annual risk-free rate (default 0)
        periods_per_year: Number of trading periods per year
        annualize: Whether to annualize the ratio
        
    Returns:
        Sharpe ratio
    """
    if len(returns) < 2:
        return 0.0
    
    excess_returns = returns - (risk_free_rate / periods_per_year)
    mean_return = np.mean(excess_returns)
    std_return = np.std(excess_returns, ddof=1)
    
    if std_return == 0 or np.isnan(std_return):
        return 0.0
    
    sharpe = mean_return / std_return
    
    if annualize:
        sharpe *= np.sqrt(periods_per_year)
    
    return sharpe


def calculate_sortino_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252 * 288,
    annualize: bool = True
) -> float:
    """
    Calculate Sortino Ratio (uses downside deviation only).
    
    Better than Sharpe for asymmetric return distributions.
    
    Args:
        returns: Array of period returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Trading periods per year
        annualize: Whether to annualize
        
    Returns:
        Sortino ratio
    """
    if len(returns) < 2:
        return 0.0
    
    target_return = risk_free_rate / periods_per_year
    excess_returns = returns - target_return
    mean_excess = np.mean(excess_returns)
    
    # Downside deviation (only negative returns)
    downside_returns = excess_returns[excess_returns < 0]
    if len(downside_returns) == 0:
        return np.inf if mean_excess > 0 else 0.0
    
    downside_std = np.sqrt(np.mean(downside_returns ** 2))
    
    if downside_std == 0:
        return 0.0
    
    sortino = mean_excess / downside_std
    
    if annualize:
        sortino *= np.sqrt(periods_per_year)
    
    return sortino


def calculate_max_drawdown(equity_curve: np.ndarray) -> Tuple[float, int, int]:
    """
    Calculate maximum drawdown from equity curve.
    
    Args:
        equity_curve: Array of equity values over time
        
    Returns:
        (max_drawdown_pct, peak_idx, trough_idx)
    """
    if len(equity_curve) < 2:
        return 0.0, 0, 0
    
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (peak - equity_curve) / peak
    
    max_dd = np.max(drawdown)
    trough_idx = np.argmax(drawdown)
    peak_idx = np.argmax(equity_curve[:trough_idx + 1]) if trough_idx > 0 else 0
    
    return max_dd * 100, peak_idx, trough_idx


def calculate_calmar_ratio(
    total_return: float,
    max_drawdown_pct: float,
    years: float = 1.0
) -> float:
    """
    Calculate Calmar Ratio (annualized return / max drawdown).
    
    Args:
        total_return: Total return as decimal
        max_drawdown_pct: Maximum drawdown as percentage
        years: Time period in years
        
    Returns:
        Calmar ratio
    """
    if max_drawdown_pct == 0:
        return 0.0
    
    annualized_return = ((1 + total_return) ** (1 / years) - 1) * 100
    return annualized_return / max_drawdown_pct


def apply_transaction_costs(
    trade_returns: List[float],
    entry_prices: List[float],
    costs: TransactionCosts = None
) -> Tuple[List[float], float]:
    """
    Apply transaction costs to trade returns.
    
    Args:
        trade_returns: List of gross trade returns (as decimals)
        entry_prices: Entry price for each trade
        costs: TransactionCosts configuration
        
    Returns:
        (net_returns, total_costs)
    """
    if costs is None:
        costs = TransactionCosts()
    
    net_returns = []
    total_costs = 0.0
    
    for ret, price in zip(trade_returns, entry_prices):
        cost_pct = costs.cost_percentage(price)
        net_ret = ret - cost_pct
        net_returns.append(net_ret)
        total_costs += cost_pct
    
    return net_returns, total_costs


def calculate_trade_frequency(
    trade_times: List[pd.Timestamp],
    period: str = 'D'
) -> dict:
    """
    Calculate trade frequency statistics.
    
    AI Council Limit: Max 5 signals/day, 3-4 per session
    
    Args:
        trade_times: List of trade entry timestamps
        period: Grouping period ('D' for daily, 'H' for hourly)
        
    Returns:
        Frequency statistics
    """
    if not trade_times:
        return {
            'avg_trades_per_day': 0,
            'max_trades_per_day': 0,
            'days_exceeding_limit': 0,
            'total_trading_days': 0
        }
    
    df = pd.DataFrame({'time': trade_times})
    df['date'] = df['time'].dt.date
    
    daily_counts = df.groupby('date').size()
    
    return {
        'avg_trades_per_day': daily_counts.mean(),
        'max_trades_per_day': daily_counts.max(),
        'days_exceeding_limit': (daily_counts > 5).sum(),
        'total_trading_days': len(daily_counts),
        'trades_by_day': daily_counts.to_dict()
    }


def calculate_baseline_returns(
    prices: pd.Series,
    strategy: str = 'buy_hold'
) -> np.ndarray:
    """
    Calculate returns for baseline strategies.
    
    Args:
        prices: Price series
        strategy: 'buy_hold', 'ema_crossover', or 'random'
        
    Returns:
        Array of returns
    """
    if strategy == 'buy_hold':
        # Simple buy and hold
        returns = prices.pct_change().dropna().values
        
    elif strategy == 'ema_crossover':
        # EMA 12/26 crossover
        ema12 = prices.ewm(span=12).mean()
        ema26 = prices.ewm(span=26).mean()
        
        signal = (ema12 > ema26).astype(int).diff()
        position = (ema12 > ema26).astype(int)
        
        returns = (prices.pct_change() * position.shift(1)).dropna().values
        
    elif strategy == 'random':
        # Random entries
        np.random.seed(42)
        position = np.random.choice([0, 1], size=len(prices))
        returns = (prices.pct_change().values * np.roll(position, 1))[1:]
        
    else:
        returns = np.zeros(len(prices) - 1)
    
    return returns


def comprehensive_metrics(
    trade_returns_pct: List[float],
    equity_curve: List[float],
    trade_times: List[pd.Timestamp] = None,
    entry_prices: List[float] = None,
    apply_costs: bool = True,
    initial_equity: float = 200.0
) -> dict:
    """
    Calculate comprehensive performance metrics.
    
    Args:
        trade_returns_pct: List of trade return percentages
        equity_curve: Equity values over time
        trade_times: Entry timestamps for frequency analysis
        entry_prices: Entry prices for cost calculation
        apply_costs: Whether to apply transaction costs
        initial_equity: Starting capital
        
    Returns:
        Dictionary of all metrics
    """
    metrics = {}
    
    # Convert to decimals
    returns = np.array(trade_returns_pct) / 100
    equity = np.array(equity_curve)
    
    # Apply transaction costs if requested
    if apply_costs and entry_prices:
        costs = TransactionCosts()
        net_returns, total_cost = apply_transaction_costs(
            returns.tolist(), entry_prices, costs
        )
        returns = np.array(net_returns)
        metrics['total_transaction_costs_pct'] = total_cost * 100
    
    # Risk metrics
    metrics['sharpe_ratio'] = calculate_sharpe_ratio(returns)
    metrics['sortino_ratio'] = calculate_sortino_ratio(returns)
    
    max_dd, peak_idx, trough_idx = calculate_max_drawdown(equity)
    metrics['max_drawdown_pct'] = max_dd
    metrics['max_drawdown_peak_idx'] = peak_idx
    metrics['max_drawdown_trough_idx'] = trough_idx
    
    # Return metrics
    total_return = (equity[-1] - initial_equity) / initial_equity if len(equity) > 0 else 0
    metrics['total_return_pct'] = total_return * 100
    metrics['calmar_ratio'] = calculate_calmar_ratio(total_return, max_dd) if max_dd > 0 else 0
    
    # Trade statistics
    metrics['total_trades'] = len(returns)
    metrics['winning_trades'] = (returns > 0).sum()
    metrics['losing_trades'] = (returns <= 0).sum()
    metrics['win_rate'] = (returns > 0).mean() * 100 if len(returns) > 0 else 0
    
    if len(returns) > 0:
        metrics['avg_win'] = returns[returns > 0].mean() * 100 if (returns > 0).any() else 0
        metrics['avg_loss'] = returns[returns <= 0].mean() * 100 if (returns <= 0).any() else 0
        metrics['profit_factor'] = abs(returns[returns > 0].sum() / returns[returns <= 0].sum()) if (returns <= 0).any() and returns[returns <= 0].sum() != 0 else 0
    
    # Trade frequency
    if trade_times:
        freq = calculate_trade_frequency(trade_times)
        metrics.update({
            'avg_trades_per_day': freq['avg_trades_per_day'],
            'max_trades_per_day': freq['max_trades_per_day'],
            'days_exceeding_limit': freq['days_exceeding_limit']
        })
    
    # Council target checks
    metrics['passes_sharpe_target'] = metrics['sharpe_ratio'] >= 1.0
    metrics['passes_edge_target'] = metrics['total_return_pct'] >= 5.0
    metrics['passes_frequency_limit'] = metrics.get('max_trades_per_day', 0) <= 5
    
    return metrics


def print_metrics_report(metrics: dict, title: str = "Performance Metrics"):
    """Pretty print metrics report."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    
    print(f"\n  📊 Risk Metrics:")
    print(f"     Sharpe Ratio:        {metrics.get('sharpe_ratio', 0):.2f} {'✅' if metrics.get('passes_sharpe_target') else '❌'}")
    print(f"     Sortino Ratio:       {metrics.get('sortino_ratio', 0):.2f}")
    print(f"     Max Drawdown:        {metrics.get('max_drawdown_pct', 0):.1f}%")
    print(f"     Calmar Ratio:        {metrics.get('calmar_ratio', 0):.2f}")
    
    print(f"\n  💰 Returns:")
    print(f"     Total Return:        {metrics.get('total_return_pct', 0):.1f}% {'✅' if metrics.get('passes_edge_target') else '❌'}")
    print(f"     Win Rate:            {metrics.get('win_rate', 0):.1f}%")
    print(f"     Profit Factor:       {metrics.get('profit_factor', 0):.2f}")
    
    if 'total_transaction_costs_pct' in metrics:
        print(f"\n  💸 Transaction Costs:")
        print(f"     Total Costs:         {metrics['total_transaction_costs_pct']:.2f}%")
    
    if 'avg_trades_per_day' in metrics:
        print(f"\n  📈 Trade Frequency:")
        print(f"     Avg/Day:             {metrics['avg_trades_per_day']:.1f}")
        print(f"     Max/Day:             {metrics['max_trades_per_day']} {'✅' if metrics.get('passes_frequency_limit') else '❌'}")
    
    print(f"\n{'='*60}")
