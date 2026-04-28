import yfinance as yf
import pandas as pd
import numpy as np
import logging
from typing import List, Union

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataLoader:
    """
    Responsible for fetching historical financial data from Yahoo Finance, 
    handling data cleaning, and calculating asset returns.
    """
    
    def __init__(self, tickers: List[str], start_date: str, end_date: str):
        self.tickers = [t.upper() for t in tickers]
        self.start_date = start_date
        self.end_date = end_date
        self.raw_data = pd.DataFrame()
        self.clean_prices = pd.DataFrame()

    def fetch_data(self) -> pd.DataFrame:
        """
        Downloads historical adjusted close prices for the specified tickers.
        """
        logging.info(f"Downloading data for: {self.tickers} from {self.start_date} to {self.end_date}")
        
        df = yf.download(self.tickers, start=self.start_date, end=self.end_date, progress=False, auto_adjust=False)
        
        if df.empty:
            logging.error("Failed to download any data. Please check the Tickers or dates.")
            raise ValueError("No data fetched.")
            
        price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
        logging.info(f"Successfully identified price column: '{price_col}'")
            
        if len(self.tickers) == 1:
            self.raw_data = df[[price_col]].rename(columns={price_col: self.tickers[0]})
        else:
            self.raw_data = df[price_col]
            
        logging.info("Data download complete.")
        return self.raw_data

    def handle_missing_data(self, method: str = 'ffill') -> pd.DataFrame:
        """
        Handles missing values (NaN). Due to holidays or trading halts, 
        trading days across different markets/stocks may not align perfectly.
        """
        if self.raw_data.empty:
            raise ValueError("Data not fetched yet. Call fetch_data() first.")
            
        logging.info(f"Cleaning missing data using method: {method}")
        
        if method == 'ffill':
            # Forward Fill (ffill): If there is no trading on a given day, 
            # assume the price remains the same as the previous day's close.
            self.clean_prices = self.raw_data.ffill()
            # If the first few days are NaN, use backward fill (bfill) to populate them.
            self.clean_prices = self.clean_prices.bfill()
        elif method == 'drop':
            # Aggressive approach: If any stock has a NaN on a given day, 
            # drop the entire row for that day.
            self.clean_prices = self.raw_data.dropna()
        else:
            raise ValueError("Unsupported fill method. Please use 'ffill' or 'drop'.")
            
        return self.clean_prices

    def get_returns(self, return_type: str = 'log') -> pd.DataFrame:
        """
        Calculates daily returns.
        
        :param return_type: 'simple' (simple percentage) or 'log' (logarithmic returns).
        """
        if self.clean_prices.empty:
            self.handle_missing_data()
            
        if return_type == 'simple':
            # R_t = (P_t / P_{t-1}) - 1
            returns = self.clean_prices.pct_change().dropna()
        elif return_type == 'log':
            # R_t = ln(P_t / P_{t-1})
            returns = np.log(self.clean_prices / self.clean_prices.shift(1)).dropna()
        else:
            raise ValueError("return_type must be either 'simple' or 'log'.")
            
        logging.info(f"Calculated {return_type} returns.")
        return returns