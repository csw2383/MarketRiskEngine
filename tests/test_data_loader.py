import pytest
import pandas as pd
import numpy as np
from market_risk_engine.data_loader import DataLoader

def test_returns_calculation():
    # Prepare minimal mock data (avoiding actual network requests)
    mock_dates = pd.date_range("2023-01-01", periods=3)
    # Assume stock price goes 100 -> 110 -> 121 (a 10% daily increase)
    mock_prices = pd.DataFrame({"AAPL": [100.0, 110.0, 121.0]}, index=mock_dates)
    
    # Initialize the DataLoader (dates are arbitrary since we bypass fetching)
    loader = DataLoader(["AAPL"], "2023-01-01", "2023-01-03")
    
    # Inject the mock data directly into the object to bypass the API call
    loader.clean_prices = mock_prices
    
    # Test the simple return calculation
    simple_returns = loader.get_returns(return_type='simple')
    
    # Day 0 becomes NaN and is dropped. Day 1 and Day 2 returns should strictly equal 0.1
    np.testing.assert_almost_equal(simple_returns.iloc[0]["AAPL"], 0.1)
    np.testing.assert_almost_equal(simple_returns.iloc[1]["AAPL"], 0.1)

def test_invalid_return_type():
    loader = DataLoader(["AAPL"], "2023-01-01", "2023-01-03")
    loader.clean_prices = pd.DataFrame({"AAPL": [100.0, 110.0]})
    
    # Test if a ValueError is correctly raised when an invalid parameter is passed
    with pytest.raises(ValueError):
        loader.get_returns(return_type='magic_math')