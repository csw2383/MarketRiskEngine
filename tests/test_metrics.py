import pytest
import pandas as pd
import numpy as np
from market_risk_engine.metrics import RiskEngine

@pytest.fixture
def sample_returns():
    """Fixture: Provides a stable set of simulated returns for testing."""
    dates = pd.date_range("2023-01-01", periods=100)
    np.random.seed(42)
    # Simulate two stocks with mean 0 and standard deviation 0.01
    data = np.random.normal(0, 0.01, (100, 2))
    return pd.DataFrame(data, index=dates, columns=['StockA', 'StockB'])

def test_risk_engine_init(sample_returns):
    """Tests the initialization and weight allocation logic."""
    engine = RiskEngine(sample_returns)
    assert len(engine.weights) == 2
    assert np.isclose(np.sum(engine.weights), 1.0)
    
    # Test custom weight assignment
    engine_custom = RiskEngine(sample_returns, weights=[0.6, 0.4])
    assert engine_custom.weights[0] == 0.6

def test_invalid_weights(sample_returns):
    """Tests if a ValueError is raised for invalid weights (sum not equal to 1.0)."""
    # The match string must align with the English error message in metrics.py
    with pytest.raises(ValueError, match="Portfolio weights must sum to exactly 1.0"):
        RiskEngine(sample_returns, weights=[0.5, 0.1])

def test_calculate_var_historical(sample_returns):
    """Tests the calculation of VaR using the Historical Simulation method."""
    engine = RiskEngine(sample_returns, weights=[1.0, 0.0])
    var_95 = engine.calculate_var_historical(confidence=0.95)
    
    # Verify that a reasonable positive risk value is returned
    assert isinstance(var_95, float)
    assert var_95 > 0

def test_calculate_var_parametric(sample_returns):
    """Tests the calculation of VaR using the Parametric (Variance-Covariance) method."""
    engine = RiskEngine(sample_returns, weights=[1.0, 0.0])
    var_95 = engine.calculate_var_parametric(confidence=0.95)
    assert isinstance(var_95, float)
    assert var_95 > 0

def test_calculate_var_monte_carlo(sample_returns):
    """Tests the calculation of VaR using the Monte Carlo Simulation method."""
    engine = RiskEngine(sample_returns, weights=[1.0, 0.0])
    var_95 = engine.calculate_var_monte_carlo(confidence=0.95, n_sims=1000)
    assert isinstance(var_95, float)
    assert var_95 > 0