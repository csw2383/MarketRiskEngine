import pytest
import pandas as pd
from market_risk_engine.backtester import Backtester

def test_backtester_logic():
    """Tests the core breach statistics logic of the backtesting engine."""
    # Actual returns: Day 1 gained 2%, Day 2 lost 5%, Day 3 lost 1%
    actual_returns = pd.Series([0.02, -0.05, -0.01])
    
    # Forecasted VaR: A constant 3% (0.03) risk tolerance for all days
    var_forecasts = pd.Series([0.03, 0.03, 0.03])
    
    backtester = Backtester(actual_returns, var_forecasts)
    results = backtester.run_test()
    
    # Expected result: Only Day 2's loss (0.05) exceeds the VaR (0.03), 
    # resulting in exactly 1 breach.
    assert results["total_days"] == 3
    assert results["num_breaches"] == 1
    assert results["breach_ratio"] == 1/3

def test_backtester_length_mismatch():
    """Tests if a ValueError is correctly raised when sequence lengths do not match."""
    actual_returns = pd.Series([0.02, -0.05])
    var_forecasts = pd.Series([0.03, 0.03, 0.03])
    
    # The match string must perfectly align with the error message in backtester.py
    with pytest.raises(ValueError, match="The length of actual returns and VaR forecasts must be identical."):
        Backtester(actual_returns, var_forecasts)