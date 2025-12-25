# ===========================================
# WALK-FORWARD VALIDATION UTILITIES
# ===========================================
# Implements rolling window train/test splits:
# - 150 trading days train / 30 trading days test
# - No data leakage between folds
# - Retrain schedule every 500 5-min bars (~6 weeks)
# ===========================================

import numpy as np
import pandas as pd
from typing import Generator, Tuple, List
from datetime import timedelta


def create_walk_forward_splits(
    df: pd.DataFrame,
    train_days: int = 150,
    test_days: int = 30,
    date_column: str = 'Date',
    min_train_samples: int = 1000
) -> Generator[Tuple[pd.DataFrame, pd.DataFrame, int], None, None]:
    """
    Generate walk-forward train/test splits.
    
    Args:
        df: DataFrame with datetime index or column
        train_days: Number of trading days for training window
        test_days: Number of trading days for testing window
        date_column: Name of datetime column
        min_train_samples: Minimum samples required in training set
        
    Yields:
        Tuple of (train_df, test_df, fold_number)
    """
    # Ensure datetime
    if date_column in df.columns:
        dates = pd.to_datetime(df[date_column])
    else:
        dates = pd.to_datetime(df.index)
    
    # Get unique trading days
    trading_days = dates.dt.date.unique()
    trading_days = np.sort(trading_days)
    
    total_days = len(trading_days)
    window_size = train_days + test_days
    
    if total_days < window_size:
        raise ValueError(
            f"Insufficient data: {total_days} days available, "
            f"need at least {window_size} days"
        )
    
    fold = 0
    start_idx = 0
    
    while start_idx + window_size <= total_days:
        # Get day ranges
        train_start_day = trading_days[start_idx]
        train_end_day = trading_days[start_idx + train_days - 1]
        test_start_day = trading_days[start_idx + train_days]
        test_end_day = trading_days[min(start_idx + window_size - 1, total_days - 1)]
        
        # Filter dataframe
        train_mask = (dates.dt.date >= train_start_day) & (dates.dt.date <= train_end_day)
        test_mask = (dates.dt.date >= test_start_day) & (dates.dt.date <= test_end_day)
        
        train_df = df[train_mask].copy()
        test_df = df[test_mask].copy()
        
        # Validate minimum samples
        if len(train_df) >= min_train_samples and len(test_df) > 0:
            yield train_df, test_df, fold
            fold += 1
        
        # Slide window by test_days (non-overlapping test sets)
        start_idx += test_days
    
    print(f"Walk-forward: Generated {fold} folds")


def calculate_retrain_schedule(
    df: pd.DataFrame,
    bars_between_retrain: int = 500,
    date_column: str = 'Date'
) -> List[int]:
    """
    Calculate indices where model should be retrained.
    
    Args:
        df: DataFrame with data
        bars_between_retrain: Number of bars between retraining (default 500 5-min bars ≈ 6 weeks)
        date_column: Name of datetime column
        
    Returns:
        List of row indices where retraining should occur
    """
    n = len(df)
    retrain_indices = list(range(bars_between_retrain, n, bars_between_retrain))
    return retrain_indices


class WalkForwardValidator:
    """
    Class to manage walk-forward validation with model checkpointing.
    """
    
    def __init__(
        self,
        train_days: int = 150,
        test_days: int = 30,
        min_train_samples: int = 1000
    ):
        self.train_days = train_days
        self.test_days = test_days
        self.min_train_samples = min_train_samples
        self.fold_results = []
        
    def validate(
        self,
        df: pd.DataFrame,
        model_builder_fn,
        train_fn,
        evaluate_fn,
        date_column: str = 'Date'
    ) -> dict:
        """
        Run walk-forward validation.
        
        Args:
            df: Full dataset
            model_builder_fn: Function that returns a fresh model
            train_fn: Function(model, train_df) -> trained_model
            evaluate_fn: Function(model, test_df) -> metrics_dict
            
        Returns:
            Aggregated metrics across all folds
        """
        all_metrics = []
        
        for train_df, test_df, fold in create_walk_forward_splits(
            df, 
            self.train_days, 
            self.test_days,
            date_column,
            self.min_train_samples
        ):
            print(f"\n{'='*50}")
            print(f"Fold {fold}: Train {len(train_df)} samples, Test {len(test_df)} samples")
            print(f"Train: {train_df[date_column].min()} to {train_df[date_column].max()}")
            print(f"Test:  {test_df[date_column].min()} to {test_df[date_column].max()}")
            
            # Build fresh model
            model = model_builder_fn()
            
            # Train
            model = train_fn(model, train_df)
            
            # Evaluate
            metrics = evaluate_fn(model, test_df)
            metrics['fold'] = fold
            metrics['train_size'] = len(train_df)
            metrics['test_size'] = len(test_df)
            
            all_metrics.append(metrics)
            self.fold_results.append({
                'fold': fold,
                'metrics': metrics,
                'train_range': (train_df[date_column].min(), train_df[date_column].max()),
                'test_range': (test_df[date_column].min(), test_df[date_column].max())
            })
            
            print(f"Fold {fold} Results: {metrics}")
        
        # Aggregate results
        aggregated = self._aggregate_metrics(all_metrics)
        return aggregated
    
    def _aggregate_metrics(self, all_metrics: List[dict]) -> dict:
        """Aggregate metrics across folds."""
        if not all_metrics:
            return {}
        
        # Get numeric keys
        numeric_keys = [k for k in all_metrics[0].keys() 
                       if isinstance(all_metrics[0][k], (int, float))]
        
        aggregated = {}
        for key in numeric_keys:
            values = [m[key] for m in all_metrics if key in m]
            aggregated[f'{key}_mean'] = np.mean(values)
            aggregated[f'{key}_std'] = np.std(values)
            aggregated[f'{key}_min'] = np.min(values)
            aggregated[f'{key}_max'] = np.max(values)
        
        aggregated['n_folds'] = len(all_metrics)
        
        return aggregated


def exclude_weekend_gaps(df: pd.DataFrame, date_column: str = 'Date') -> pd.DataFrame:
    """
    Remove rows immediately after weekend gaps to prevent
    training on discontinuous price action.
    
    Args:
        df: DataFrame with datetime column
        date_column: Name of datetime column
        
    Returns:
        DataFrame with weekend gap rows removed
    """
    df = df.copy()
    dates = pd.to_datetime(df[date_column])
    
    # Calculate time difference between consecutive rows
    time_diff = dates.diff()
    
    # Flag rows that follow a gap > 2 days (weekend)
    weekend_gap_mask = time_diff > timedelta(days=2)
    
    # Also flag the next few rows after gap (market adjustment period)
    adjustment_rows = 5  # Skip 5 rows after weekend
    for i in range(1, adjustment_rows + 1):
        weekend_gap_mask = weekend_gap_mask | weekend_gap_mask.shift(-i).fillna(False)
    
    print(f"Excluding {weekend_gap_mask.sum()} rows near weekend gaps")
    
    return df[~weekend_gap_mask].reset_index(drop=True)
