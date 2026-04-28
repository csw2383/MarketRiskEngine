import numpy as np
import pandas as pd
from scipy.stats import norm
import multiprocessing as mp
from typing import Union, List


def _simulate_mc_worker(args: tuple) -> np.ndarray:
    """
    Independent worker process for Monte Carlo simulation.
    
    :param args: A tuple containing (mu, sigma, n_sims, seed).
    """
    mu, sigma, n_sims, seed = args
    
    # CRITICAL: Each process must have an independent random seed. 
    # Otherwise, all CPU cores will generate the exact same sequence of random numbers!
    np.random.seed(seed)
    return np.random.normal(mu, sigma, n_sims)


class RiskEngine:
    """
    Quantitative Risk Engine: Calculates the Value at Risk (VaR) of a portfolio.
    Supports Historical, Parametric, and Monte Carlo (Single & Multi-core) methodologies.
    """
    def __init__(self, returns: pd.DataFrame, weights: Union[List[float], np.ndarray] = None):
        """
        Initializes the risk engine.
        
        :param returns: DataFrame containing asset return time series.
        :param weights: Initial asset weights. Defaults to an equally weighted portfolio if None.
        """
        self.returns = returns
        num_assets = returns.shape[1]
        
        if weights is None:
            self.weights = np.array([1.0 / num_assets] * num_assets)
        else:
            self.weights = np.array(weights)
            
        if not np.isclose(np.sum(self.weights), 1.0):
            raise ValueError("Portfolio weights must sum to exactly 1.0")

    def get_portfolio_returns(self) -> pd.Series:
        """Calculates the historical return series at the portfolio level."""
        return self.returns.dot(self.weights)

    def calculate_var_historical(self, confidence: float = 0.95) -> float:
        """
        Calculates VaR using the Historical Simulation method.
        """
        port_rets = self.get_portfolio_returns()
        var = -np.percentile(port_rets, (1 - confidence) * 100)
        return var

    def calculate_var_parametric(self, confidence: float = 0.95) -> float:
        """
        Calculates VaR using the Variance-Covariance (Parametric) method.
        Assumes returns follow a normal distribution.
        """
        port_rets = self.get_portfolio_returns()
        mu = np.mean(port_rets)
        sigma = np.std(port_rets)
        
        var = -(mu + sigma * norm.ppf(1 - confidence))
        return var

    def calculate_var_monte_carlo(self, confidence: float = 0.95, n_sims: int = 10000, seed: int = 42) -> float:
        """
        Single-core version of Monte Carlo VaR.
        Simulates future return distributions based on Geometric Brownian Motion assumptions.
        """
        port_rets = self.get_portfolio_returns()
        mu = np.mean(port_rets)
        sigma = np.std(port_rets)
        
        # guarantees perfect reproducibility
        np.random.seed(seed)
        sim_returns = np.random.normal(mu, sigma, n_sims)
        
        var = -np.percentile(sim_returns, (1 - confidence) * 100)
        return float(var)

    def calculate_var_mc_parallel(self, confidence: float = 0.95, n_sims: int = 100000) -> float:
        """
        [W8D2 Core Feature] Multi-process parallel version of Monte Carlo VaR.
        Leverages all available CPU cores to accelerate simulations of hundreds of thousands or millions of paths.
        """
        port_rets = self.get_portfolio_returns()
        mu = np.mean(port_rets)
        sigma = np.std(port_rets)
        
        # Get the number of physical/logical CPU cores available on the machine
        cores = mp.cpu_count()
        
        # Evenly distribute the total number of simulations across all cores
        sims_per_core = n_sims // cores
        
        args_list = [(mu, sigma, sims_per_core, 42 + i) for i in range(cores)]
        
        # Initialize the Process Pool
        with mp.Pool(processes=cores) as pool:
            # The map function automatically distributes tasks to idle cores 
            # and blocks execution until all results are ready.
            results = pool.map(_simulate_mc_worker, args_list)
            
        # Concatenate the results from all parallel cores into a single, massive 1D array
        all_simulations = np.concatenate(results)
        
        var = -np.percentile(all_simulations, (1 - confidence) * 100)
        return float(var)