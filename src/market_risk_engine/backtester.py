import pandas as pd
from typing import Dict

class Backtester:
    """
    Backtesting Engine: Validates the effectiveness of the VaR model.
    Calculates the number of breaches by comparing actual returns against VaR forecasts.
    """
    def __init__(self, actual_returns: pd.Series, var_forecasts: pd.Series):
        """
        Initializes the backtester.
        
        :param actual_returns: Series of actual realized portfolio returns.
        :param var_forecasts: Series of VaR forecasts for corresponding time points (expected to be positive values).
        """
        if len(actual_returns) != len(var_forecasts):
            raise ValueError("The length of actual returns and VaR forecasts must be identical.")
            
        self.actual_returns = actual_returns
        self.var_forecasts = var_forecasts

    def run_test(self) -> Dict[str, float]:
        """
        Executes the backtest and returns breach statistics.
        A breach is defined as: absolute value of actual loss > forecasted VaR.
        """
        # Negate actual returns to represent actual losses
        actual_losses = -self.actual_returns
        
        # Determine if a breach occurred on each respective day
        breaches = actual_losses > self.var_forecasts
        
        num_breaches = int(breaches.sum())
        total_days = len(self.actual_returns)
        breach_ratio = num_breaches / total_days if total_days > 0 else 0.0
        
        return {
            "total_days": total_days,
            "num_breaches": num_breaches,
            "breach_ratio": breach_ratio
        }