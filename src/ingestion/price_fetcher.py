"""
Stock Price and Technical Indicators Fetcher Module
Downloads historical stock prices and calculates technical indicators using Yahoo Finance
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yfinance as yf
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


class PriceFetcher:
    """
    Fetches historical stock prices and calculates technical indicators
    """
    
    def __init__(self,
                 cache_dir: str = "./data/raw/prices",
                 cache_expiry_hours: int = 24):
        """
        Initialize price fetcher
        
        Args:
            cache_dir: Directory to cache price data
            cache_expiry_hours: Hours before cache expires
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_expiry_hours = cache_expiry_hours
        
        logger.info(f"PriceFetcher initialized with cache at {self.cache_dir}")
    
    def _get_cache_path(self, ticker: str, start: str, end: str) -> Path:
        """Generate cache file path for price data"""
        cache_key = f"{ticker}_{start}_{end}".replace('-', '')
        return self.cache_dir / f"{cache_key}_prices.parquet"
    
    def _is_cache_valid(self, cache_path: Path) -> bool:
        """Check if cache file is still valid"""
        if not cache_path.exists():
            return False
        
        mod_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
        expiry_time = datetime.now() - timedelta(hours=self.cache_expiry_hours)
        
        return mod_time > expiry_time
    
    def fetch_prices(self,
                    ticker: str,
                    start_date: str,
                    end_date: str,
                    interval: str = '1d',
                    use_cache: bool = True) -> pd.DataFrame:
        """
        Fetch historical stock prices
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Data interval (1d, 1wk, 1mo)
            use_cache: Whether to use cached data
            
        Returns:
            DataFrame with OHLCV data
        """
        ticker = ticker.upper()
        
        # Check cache
        cache_path = self._get_cache_path(ticker, start_date, end_date)
        if use_cache and self._is_cache_valid(cache_path):
            logger.info(f"Loading cached prices for {ticker}")
            return pd.read_parquet(cache_path)
        
        try:
            # Download from Yahoo Finance
            logger.info(f"Downloading prices for {ticker} from {start_date} to {end_date}")
            
            stock = yf.Ticker(ticker)
            df = stock.history(
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=True,  # Adjust for splits and dividends
                prepost=False
            )
            
            if df.empty:
                logger.warning(f"No price data found for {ticker}")
                return pd.DataFrame()
            
            # Clean column names
            df.columns = [col.replace(' ', '_').lower() for col in df.columns]
            
            # Add ticker column
            df['ticker'] = ticker
            
            # Reset index to have date as column
            df = df.reset_index()
            if 'Date' in df.columns:
                df.rename(columns={'Date': 'date'}, inplace=True)
            
            # Calculate additional price metrics
            df = self._add_price_metrics(df)
            
            # Cache the data
            if use_cache:
                df.to_parquet(cache_path, index=False)
                logger.info(f"Cached prices for {ticker}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching prices for {ticker}: {e}")
            return pd.DataFrame()
    
    def _add_price_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add additional price-based metrics"""
        if df.empty:
            return df
        
        # Returns
        df['daily_return'] = df['close'].pct_change(fill_method=None)
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        
        # Price changes
        df['intraday_range'] = df['high'] - df['low']
        df['intraday_range_pct'] = df['intraday_range'] / df['close']
        
        # Gap analysis
        df['gap'] = df['open'] - df['close'].shift(1)
        df['gap_pct'] = df['gap'] / df['close'].shift(1)
        
        # Volume metrics
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
        df['dollar_volume'] = df['volume'] * df['close']
        
        return df
    
    def calculate_indicators(self,
                           df: pd.DataFrame,
                           indicators: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Calculate technical indicators
        
        Args:
            df: DataFrame with OHLCV data
            indicators: List of indicators to calculate (None = all)
            
        Returns:
            DataFrame with additional indicator columns
        """
        if df.empty:
            return df
        
        df = df.copy()
        
        # Default indicators
        if indicators is None:
            indicators = ['sma', 'ema', 'rsi', 'macd', 'bbands', 'atr', 'obv', 'adx', 'stoch']
        
        # Simple Moving Averages
        if 'sma' in indicators:
            for period in [5, 10, 20, 50, 200]:
                if len(df) >= period:
                    df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
        
        # Exponential Moving Averages
        if 'ema' in indicators:
            for period in [12, 26, 50]:
                if len(df) >= period:
                    df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        
        # Relative Strength Index
        if 'rsi' in indicators:
            df['rsi_14'] = self._calculate_rsi(df['close'], 14)
        
        # MACD
        if 'macd' in indicators:
            macd_result = self._calculate_macd(df['close'])
            df['macd'] = macd_result['macd']
            df['macd_signal'] = macd_result['signal']
            df['macd_histogram'] = macd_result['histogram']
        
        # Bollinger Bands
        if 'bbands' in indicators:
            bb_result = self._calculate_bollinger_bands(df['close'])
            df['bb_upper'] = bb_result['upper']
            df['bb_middle'] = bb_result['middle']
            df['bb_lower'] = bb_result['lower']
            df['bb_width'] = bb_result['width']
            df['bb_pct'] = bb_result['pct']
        
        # Average True Range
        if 'atr' in indicators:
            df['atr_14'] = self._calculate_atr(df, 14)
        
        # On Balance Volume
        if 'obv' in indicators:
            df['obv'] = self._calculate_obv(df)
        
        # Average Directional Index
        if 'adx' in indicators:
            adx_result = self._calculate_adx(df, 14)
            df['adx'] = adx_result['adx']
            df['plus_di'] = adx_result['plus_di']
            df['minus_di'] = adx_result['minus_di']
        
        # Stochastic Oscillator
        if 'stoch' in indicators:
            stoch_result = self._calculate_stochastic(df, 14)
            df['stoch_k'] = stoch_result['k']
            df['stoch_d'] = stoch_result['d']
        
        # Support and Resistance Levels
        if 'support_resistance' in indicators:
            sr_levels = self._calculate_support_resistance(df)
            df['nearest_support'] = sr_levels['support']
            df['nearest_resistance'] = sr_levels['resistance']
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, 
                       fast: int = 12, slow: int = 26, signal: int = 9) -> Dict:
        """Calculate MACD indicator"""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        macd_histogram = macd - macd_signal
        
        return {
            'macd': macd,
            'signal': macd_signal,
            'histogram': macd_histogram
        }
    
    def _calculate_bollinger_bands(self, prices: pd.Series, 
                                  period: int = 20, std_dev: int = 2) -> Dict:
        """Calculate Bollinger Bands"""
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        width = upper - lower
        pct = (prices - lower) / (upper - lower)
        
        return {
            'upper': upper,
            'middle': middle,
            'lower': lower,
            'width': width,
            'pct': pct
        }
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    def _calculate_obv(self, df: pd.DataFrame) -> pd.Series:
        """Calculate On Balance Volume"""
        obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        return obv
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> Dict:
        """Calculate Average Directional Index"""
        # Calculate directional movements
        plus_dm = df['high'].diff()
        minus_dm = -df['low'].diff()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        # When both are positive, select the larger one
        mask = (plus_dm > 0) & (minus_dm > 0)
        plus_dm[mask] = np.where(plus_dm[mask] > minus_dm[mask], plus_dm[mask], 0)
        minus_dm[mask] = np.where(minus_dm[mask] > plus_dm[mask], minus_dm[mask], 0)
        
        # Calculate ATR
        atr = self._calculate_atr(df, period)
        
        # Calculate directional indicators
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        # Calculate ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return {
            'adx': adx,
            'plus_di': plus_di,
            'minus_di': minus_di
        }
    
    def _calculate_stochastic(self, df: pd.DataFrame, 
                             period: int = 14, smooth_k: int = 3, smooth_d: int = 3) -> Dict:
        """Calculate Stochastic Oscillator"""
        lowest_low = df['low'].rolling(window=period).min()
        highest_high = df['high'].rolling(window=period).max()
        
        k_percent = 100 * ((df['close'] - lowest_low) / (highest_high - lowest_low))
        k_percent = k_percent.rolling(window=smooth_k).mean()
        d_percent = k_percent.rolling(window=smooth_d).mean()
        
        return {
            'k': k_percent,
            'd': d_percent
        }
    
    def _calculate_support_resistance(self, df: pd.DataFrame, window: int = 20) -> Dict:
        """Calculate support and resistance levels"""
        # Simple approach using rolling min/max
        support = df['low'].rolling(window=window).min()
        resistance = df['high'].rolling(window=window).max()
        
        return {
            'support': support,
            'resistance': resistance
        }
    
    def get_returns(self, df: pd.DataFrame, periods: List[int] = None) -> pd.DataFrame:
        """
        Calculate returns for various periods
        
        Args:
            df: DataFrame with price data
            periods: List of periods for return calculation
            
        Returns:
            DataFrame with return columns
        """
        if df.empty:
            return df
        
        df = df.copy()
        
        if periods is None:
            periods = [1, 5, 10, 20, 60, 120, 252]  # Daily, weekly, bi-weekly, monthly, quarterly, semi-annual, annual
        
        for period in periods:
            if len(df) > period:
                df[f'return_{period}d'] = df['close'].pct_change(periods=period, fill_method=None)
                df[f'log_return_{period}d'] = np.log(df['close'] / df['close'].shift(period))
        
        # Forward returns for prediction targets
        for period in [1, 5, 10, 20]:
            if len(df) > period:
                df[f'forward_return_{period}d'] = df['close'].shift(-period) / df['close'] - 1
        
        return df
    
    def calculate_volatility(self, df: pd.DataFrame, windows: List[int] = None) -> pd.DataFrame:
        """
        Calculate various volatility measures
        
        Args:
            df: DataFrame with price data
            windows: List of rolling windows for volatility
            
        Returns:
            DataFrame with volatility columns
        """
        if df.empty:
            return df
        
        df = df.copy()
        
        if windows is None:
            windows = [10, 20, 30, 60]
        
        # Historical volatility (standard deviation of returns)
        for window in windows:
            if len(df) > window:
                df[f'volatility_{window}d'] = df['daily_return'].rolling(window=window).std() * np.sqrt(252)
        
        # Parkinson volatility (using high-low)
        for window in windows:
            if len(df) > window:
                hl_ratio = np.log(df['high'] / df['low'])
                df[f'parkinson_vol_{window}d'] = np.sqrt(
                    252 / (4 * window * np.log(2)) * (hl_ratio ** 2).rolling(window=window).sum()
                )
        
        # Garman-Klass volatility
        for window in windows:
            if len(df) > window:
                log_hl = np.log(df['high'] / df['low']) ** 2
                log_co = np.log(df['close'] / df['open']) ** 2
                
                df[f'garman_klass_vol_{window}d'] = np.sqrt(
                    252 / window * (
                        0.5 * log_hl.rolling(window=window).sum() -
                        (2 * np.log(2) - 1) * log_co.rolling(window=window).sum()
                    )
                )
        
        return df
    
    def fetch_multiple_tickers(self,
                             tickers: List[str],
                             start_date: str,
                             end_date: str,
                             parallel: bool = True,
                             max_workers: int = 5) -> Dict[str, pd.DataFrame]:
        """
        Fetch prices for multiple tickers
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date
            end_date: End date
            parallel: Whether to fetch in parallel
            max_workers: Maximum parallel workers
            
        Returns:
            Dictionary of DataFrames by ticker
        """
        results = {}
        
        if parallel and len(tickers) > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_ticker = {
                    executor.submit(self.fetch_prices, ticker, start_date, end_date): ticker
                    for ticker in tickers
                }
                
                for future in as_completed(future_to_ticker):
                    ticker = future_to_ticker[future]
                    try:
                        df = future.result()
                        if not df.empty:
                            results[ticker] = df
                            logger.info(f"Successfully fetched {ticker}")
                    except Exception as e:
                        logger.error(f"Error fetching {ticker}: {e}")
        else:
            for ticker in tickers:
                df = self.fetch_prices(ticker, start_date, end_date)
                if not df.empty:
                    results[ticker] = df
        
        logger.info(f"Fetched prices for {len(results)}/{len(tickers)} tickers")
        return results
    
    def get_earnings_dates(self, ticker: str) -> pd.DataFrame:
        """
        Get historical and upcoming earnings dates
        
        Returns:
            DataFrame with earnings dates and actual/estimate EPS
        """
        try:
            stock = yf.Ticker(ticker)
            
            # Get earnings dates
            earnings = stock.earnings_dates
            
            if earnings is not None and not earnings.empty:
                earnings = earnings.reset_index()
                earnings.columns = [col.replace(' ', '_').lower() for col in earnings.columns]
                return earnings
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error fetching earnings dates for {ticker}: {e}")
            return pd.DataFrame()
    
    def get_company_info(self, ticker: str) -> Dict:
        """
        Get basic company information
        
        Returns:
            Dictionary with company info
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Extract key information
            key_fields = [
                'longName', 'sector', 'industry', 'marketCap', 'enterpriseValue',
                'trailingPE', 'forwardPE', 'priceToBook', 'profitMargins',
                'returnOnEquity', 'returnOnAssets', 'revenueGrowth', 'earningsGrowth',
                'currentRatio', 'debtToEquity', 'freeCashflow', 'operatingCashflow',
                'totalRevenue', 'totalDebt', 'totalCash', 'sharesOutstanding',
                'floatShares', 'beta', 'dividendYield', 'payoutRatio'
            ]
            
            company_info = {field: info.get(field) for field in key_fields if field in info}
            company_info['ticker'] = ticker
            company_info['retrieved_at'] = datetime.now().isoformat()
            
            return company_info
            
        except Exception as e:
            logger.error(f"Error fetching company info for {ticker}: {e}")
            return {'ticker': ticker, 'error': str(e)}
    
    def create_features_for_ml(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features for machine learning models
        
        Returns:
            DataFrame with ML-ready features
        """
        if df.empty:
            return df
        
        df = df.copy()
        
        # Calculate all indicators
        df = self.calculate_indicators(df)
        df = self.get_returns(df)
        df = self.calculate_volatility(df)
        
        # Price position features
        df['price_to_sma20'] = df['close'] / df['sma_20'] - 1
        df['price_to_sma50'] = df['close'] / df['sma_50'] - 1
        df['price_to_sma200'] = df['close'] / df['sma_200'] - 1
        
        # Trend features
        df['sma20_trend'] = df['sma_20'].diff(5) / df['sma_20'].shift(5)
        df['sma50_trend'] = df['sma_50'].diff(10) / df['sma_50'].shift(10)
        
        # Volume features
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_trend'] = df['volume_sma'].diff(5) / df['volume_sma'].shift(5)
        
        # Momentum features
        df['rsi_oversold'] = (df['rsi_14'] < 30).astype(int)
        df['rsi_overbought'] = (df['rsi_14'] > 70).astype(int)
        
        # MACD features
        df['macd_crossover'] = ((df['macd'] > df['macd_signal']) & 
                                (df['macd'].shift(1) <= df['macd_signal'].shift(1))).astype(int)
        
        # Time features
        if 'date' in df.columns:
            df['day_of_week'] = pd.to_datetime(df['date']).dt.dayofweek
            df['month'] = pd.to_datetime(df['date']).dt.month
            df['quarter'] = pd.to_datetime(df['date']).dt.quarter
        
        return df