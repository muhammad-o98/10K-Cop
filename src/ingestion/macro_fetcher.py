"""
Macroeconomic Data Fetcher Module
Retrieves macroeconomic indicators from FRED (Federal Reserve Economic Data)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from fredapi import Fred
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


class MacroFetcher:
    """
    Fetches macroeconomic data from FRED API
    """
    
    # Key economic indicators and their FRED series IDs
    MACRO_SERIES = {
        # Interest Rates
        'fed_funds_rate': 'FEDFUNDS',
        'treasury_10y': 'DGS10',
        'treasury_2y': 'DGS2',
        'treasury_3m': 'DGS3MO',
        'term_spread_10y2y': 'T10Y2Y',
        'term_spread_10y3m': 'T10Y3M',
        'real_interest_rate': 'REAINTRATREARAT10Y',
        
        # Inflation
        'cpi_all': 'CPIAUCSL',
        'cpi_core': 'CPILFESL',
        'pce': 'PCEPI',
        'pce_core': 'PCEPILFE',
        'inflation_expectations_5y': 'T5YIE',
        'inflation_expectations_10y': 'T10YIE',
        'ppi': 'PPIACO',
        
        # Employment
        'unemployment_rate': 'UNRATE',
        'nonfarm_payrolls': 'PAYEMS',
        'initial_jobless_claims': 'ICSA',
        'continuing_claims': 'CCSA',
        'labor_force_participation': 'CIVPART',
        'employment_population_ratio': 'EMRATIO',
        'average_hourly_earnings': 'CES0500000003',
        'job_openings': 'JTSJOL',
        
        # Economic Activity
        'gdp': 'GDP',
        'real_gdp': 'GDPC1',
        'gdp_growth': 'A191RL1Q225SBEA',
        'industrial_production': 'INDPRO',
        'capacity_utilization': 'TCU',
        'retail_sales': 'RSXFS',
        'personal_income': 'PI',
        'personal_consumption': 'PCE',
        'personal_savings_rate': 'PSAVERT',
        
        # Housing
        'housing_starts': 'HOUST',
        'building_permits': 'PERMIT',
        'existing_home_sales': 'EXHOSLUSM495S',
        'case_shiller_index': 'CSUSHPISA',
        'mortgage_rate_30y': 'MORTGAGE30US',
        'mortgage_rate_15y': 'MORTGAGE15US',
        
        # Business & Manufacturing
        'ism_manufacturing': 'MANEMP',
        'ism_services': 'NMFBAI',
        'durable_goods_orders': 'DGORDER',
        'business_inventories': 'BUSINV',
        'consumer_sentiment': 'UMCSENT',
        'consumer_confidence': 'CSCICP03USM665S',
        
        # Financial Markets
        'vix': 'VIXCLS',
        'dollar_index': 'DTWEXBGS',
        'sp500': 'SP500',
        'corporate_bond_spread': 'BAMLC0A0CM',
        'high_yield_spread': 'BAMLH0A0HYM2',
        
        # Money Supply & Credit
        'm2_money_supply': 'M2SL',
        'commercial_bank_credit': 'TOTBKCR',
        'consumer_credit': 'TOTALSL',
        'bank_lending_standards': 'DRTSCILM',
        
        # Commodities
        'oil_price_wti': 'DCOILWTICO',
        'gold_price': 'GOLDPMGBD228NLBM',
        'copper_price': 'PCOPPUSDM',
        'natural_gas': 'DHHNGSP',
        
        # Global Indicators
        'china_gdp': 'NYGDPMKTPCDWLDCHN',
        'euro_area_gdp': 'NAEXKP01EZQ657S',
        'global_uncertainty': 'GEPUCURRENT',
    }
    
    # Categories for organizing indicators
    CATEGORIES = {
        'rates': ['fed_funds_rate', 'treasury_10y', 'treasury_2y', 'treasury_3m', 
                  'term_spread_10y2y', 'term_spread_10y3m', 'real_interest_rate'],
        'inflation': ['cpi_all', 'cpi_core', 'pce', 'pce_core', 'ppi',
                      'inflation_expectations_5y', 'inflation_expectations_10y'],
        'employment': ['unemployment_rate', 'nonfarm_payrolls', 'initial_jobless_claims',
                      'labor_force_participation', 'average_hourly_earnings'],
        'activity': ['gdp', 'real_gdp', 'gdp_growth', 'industrial_production',
                    'retail_sales', 'personal_consumption'],
        'housing': ['housing_starts', 'building_permits', 'case_shiller_index',
                   'mortgage_rate_30y'],
        'sentiment': ['consumer_sentiment', 'consumer_confidence', 'vix'],
        'commodities': ['oil_price_wti', 'gold_price', 'copper_price']
    }
    
    def __init__(self,
                 api_key: str,
                 cache_dir: str = "./data/raw/macro",
                 cache_expiry_hours: int = 24):
        """
        Initialize macro data fetcher
        
        Args:
            api_key: FRED API key
            cache_dir: Directory to cache macro data
            cache_expiry_hours: Hours before cache expires
        """
        self.fred = Fred(api_key=api_key)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_expiry_hours = cache_expiry_hours
        
        logger.info(f"MacroFetcher initialized with cache at {self.cache_dir}")
    
    def _get_cache_path(self, series_id: str, start: str, end: str) -> Path:
        """Generate cache file path for series data"""
        cache_key = f"{series_id}_{start}_{end}".replace('-', '').replace(':', '')
        return self.cache_dir / f"{cache_key}.parquet"
    
    def _is_cache_valid(self, cache_path: Path) -> bool:
        """Check if cache file is still valid"""
        if not cache_path.exists():
            return False
        
        mod_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
        expiry_time = datetime.now() - timedelta(hours=self.cache_expiry_hours)
        
        return mod_time > expiry_time
    
    def fetch_series(self,
                    series_id: str,
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None,
                    use_cache: bool = True) -> pd.DataFrame:
        """
        Fetch a single FRED series
        
        Args:
            series_id: FRED series ID
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            use_cache: Whether to use cached data
            
        Returns:
            DataFrame with series data
        """
        # Default date range if not specified
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365*10)).strftime('%Y-%m-%d')
        
        # Check cache
        cache_path = self._get_cache_path(series_id, start_date, end_date)
        if use_cache and self._is_cache_valid(cache_path):
            logger.info(f"Loading cached data for {series_id}")
            return pd.read_parquet(cache_path)
        
        try:
            # Fetch from FRED
            logger.info(f"Fetching {series_id} from FRED API")
            
            data = self.fred.get_series(
                series_id,
                observation_start=start_date,
                observation_end=end_date
            )
            
            if data is None or data.empty:
                logger.warning(f"No data found for series {series_id}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(data, columns=['value'])
            df.index.name = 'date'
            df = df.reset_index()
            
            # Add metadata
            df['series_id'] = series_id
            df['series_name'] = self._get_series_name(series_id)
            
            # Calculate changes and growth rates
            df = self._calculate_changes(df)
            
            # Cache the data
            if use_cache:
                df.to_parquet(cache_path, index=False)
                logger.info(f"Cached data for {series_id}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching series {series_id}: {e}")
            return pd.DataFrame()
    
    def _get_series_name(self, series_id: str) -> str:
        """Get human-readable name for series ID"""
        for name, sid in self.MACRO_SERIES.items():
            if sid == series_id:
                return name
        return series_id
    
    def _calculate_changes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate period-over-period changes and growth rates"""
        if df.empty or 'value' not in df.columns:
            return df
        
        df = df.copy()
        
        # Absolute changes
        df['change_1d'] = df['value'].diff()
        df['change_1w'] = df['value'].diff(5)  # Assuming daily data
        df['change_1m'] = df['value'].diff(21)
        df['change_3m'] = df['value'].diff(63)
        df['change_1y'] = df['value'].diff(252)
        
        # Percentage changes (for non-rate series)
        if not any(term in df['series_name'].iloc[0] for term in ['rate', 'spread', 'ratio']):
            df['pct_change_1m'] = df['value'].pct_change(21, fill_method=None)
            df['pct_change_3m'] = df['value'].pct_change(63, fill_method=None)
            df['pct_change_1y'] = df['value'].pct_change(252, fill_method=None)
        
        # Moving averages
        df['ma_30d'] = df['value'].rolling(window=30).mean()
        df['ma_90d'] = df['value'].rolling(window=90).mean()
        df['ma_252d'] = df['value'].rolling(window=252).mean()
        
        # Z-score (standardized value)
        df['z_score_1y'] = (df['value'] - df['value'].rolling(252).mean()) / df['value'].rolling(252).std()
        
        return df
    
    def get_macro_features(self,
                          categories: Optional[List[str]] = None,
                          start_date: Optional[str] = None,
                          end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Get macro features for specified categories
        
        Args:
            categories: List of categories to fetch (None = all)
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with all macro features
        """
        if categories is None:
            categories = list(self.CATEGORIES.keys())
        
        # Collect series IDs for specified categories
        series_ids = []
        for category in categories:
            if category in self.CATEGORIES:
                for indicator in self.CATEGORIES[category]:
                    if indicator in self.MACRO_SERIES:
                        series_ids.append(self.MACRO_SERIES[indicator])
        
        # Fetch all series
        all_data = []
        for series_id in series_ids:
            df = self.fetch_series(series_id, start_date, end_date)
            if not df.empty:
                all_data.append(df)
        
        if not all_data:
            return pd.DataFrame()
        
        # Combine all series
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Pivot to wide format
        pivot_df = combined_df.pivot_table(
            index='date',
            columns='series_name',
            values='value',
            aggfunc='first'
        ).reset_index()
        
        # Forward fill missing values (many series have different frequencies)
        pivot_df = pivot_df.ffill()
        
        return pivot_df
    
    def get_recession_indicators(self,
                                start_date: Optional[str] = None,
                                end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Get recession probability indicators
        
        Returns:
            DataFrame with recession indicators
        """
        recession_series = {
            'recession_periods': 'USREC',  # NBER recession indicator
            'sahm_rule': 'SAHMCURRENT',  # Sahm Rule recession indicator
            'smoothed_recession_prob': 'RECPROUSM156N',  # Smoothed U.S. Recession Probabilities
            'yield_curve_prob': 'THREEFYTP10',  # 3-Month/10-Year Treasury Spread
        }
        
        recession_data = []
        
        for name, series_id in recession_series.items():
            try:
                data = self.fred.get_series(
                    series_id,
                    observation_start=start_date,
                    observation_end=end_date
                )
                
                if data is not None and not data.empty:
                    df = pd.DataFrame(data, columns=[name])
                    df.index.name = 'date'
                    recession_data.append(df)
                    
            except Exception as e:
                logger.warning(f"Could not fetch {name}: {e}")
        
        if recession_data:
            result = pd.concat(recession_data, axis=1)
            result = result.reset_index()
            
            # Add composite recession score
            result['recession_score'] = result[['sahm_rule', 'smoothed_recession_prob']].mean(axis=1)
            
            return result
        
        return pd.DataFrame()
    
    def get_financial_conditions(self,
                                start_date: Optional[str] = None,
                                end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Get financial conditions indices
        
        Returns:
            DataFrame with financial conditions data
        """
        conditions_series = {
            'chicago_fed_conditions': 'NFCI',  # Chicago Fed National Financial Conditions Index
            'st_louis_stress': 'STLFSI2',  # St. Louis Fed Financial Stress Index
            'kansas_city_stress': 'KCFSI',  # Kansas City Financial Stress Index
            'credit_spread_aaa': 'AAA10Y',  # Moody's Seasoned Aaa Corporate Bond Yield Relative to 10-Year
            'credit_spread_baa': 'BAA10Y',  # Moody's Seasoned Baa Corporate Bond Yield Relative to 10-Year
            'ted_spread': 'TEDRATE',  # TED Spread
        }
        
        conditions_data = []
        
        for name, series_id in conditions_series.items():
            df = self.fetch_series(series_id, start_date, end_date)
            if not df.empty:
                df = df[['date', 'value']].rename(columns={'value': name})
                conditions_data.append(df)
        
        if conditions_data:
            # Merge all conditions data
            result = conditions_data[0]
            for df in conditions_data[1:]:
                result = pd.merge(result, df, on='date', how='outer')
            
            # Sort by date and forward fill
            result = result.sort_values('date')
            result = result.ffill()
            
            # Calculate composite stress index
            stress_cols = [col for col in result.columns if 'stress' in col or 'spread' in col]
            if stress_cols:
                # Standardize each stress measure
                for col in stress_cols:
                    result[f'{col}_zscore'] = (result[col] - result[col].mean()) / result[col].std()
                
                # Average z-scores for composite
                zscore_cols = [col for col in result.columns if '_zscore' in col]
                result['composite_stress'] = result[zscore_cols].mean(axis=1)
            
            return result
        
        return pd.DataFrame()
    
    def get_nowcast_data(self) -> Dict[str, Any]:
        """
        Get real-time nowcasting data (latest values for key indicators)
        
        Returns:
            Dictionary with latest macro indicators
        """
        nowcast_series = {
            'gdp_now': 'GDPNOW',  # Atlanta Fed GDPNow
            'aruoba_index': 'ARUOBA',  # Aruoba-Diebold-Scotti Business Conditions Index
            'weekly_economic_index': 'WEI',  # NY Fed Weekly Economic Index
            'chicago_fed_national_activity': 'CFNAI',  # Chicago Fed National Activity Index
        }
        
        nowcast_data = {}
        
        for name, series_id in nowcast_series.items():
            try:
                # Get only the latest value
                data = self.fred.get_series(series_id)
                if data is not None and not data.empty:
                    latest_value = data.iloc[-1]
                    latest_date = data.index[-1]
                    nowcast_data[name] = {
                        'value': latest_value,
                        'date': latest_date.strftime('%Y-%m-%d'),
                        'series_id': series_id
                    }
            except Exception as e:
                logger.warning(f"Could not fetch {name}: {e}")
        
        return nowcast_data
    
    def create_macro_features_for_ml(self,
                                    target_date: str,
                                    lookback_days: int = 252) -> pd.DataFrame:
        """
        Create macro features for ML models at a specific date
        
        Args:
            target_date: Date for which to create features
            lookback_days: Number of days to look back for features
            
        Returns:
            DataFrame with ML-ready macro features
        """
        end_date = pd.to_datetime(target_date)
        start_date = end_date - timedelta(days=lookback_days * 2)  # Extra buffer for calculations
        
        # Get all macro data
        macro_df = self.get_macro_features(
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )
        
        if macro_df.empty:
            return pd.DataFrame()
        
        # Convert date column to datetime
        macro_df['date'] = pd.to_datetime(macro_df['date'])
        
        # Filter to target date
        target_data = macro_df[macro_df['date'] <= end_date].copy()
        
        if target_data.empty:
            return pd.DataFrame()
        
        # Get latest values
        latest_idx = target_data['date'].idxmax()
        features = {}
        
        # For each macro indicator
        for col in target_data.columns:
            if col == 'date':
                continue
            
            # Latest value
            features[f'{col}_current'] = target_data.loc[latest_idx, col]
            
            # Changes over various periods
            for days in [5, 21, 63, 252]:
                if len(target_data) > days:
                    features[f'{col}_change_{days}d'] = (
                        target_data[col].iloc[-1] - target_data[col].iloc[-days-1]
                    )
            
            # Moving averages
            for window in [21, 63, 252]:
                if len(target_data) > window:
                    features[f'{col}_ma_{window}d'] = target_data[col].rolling(window).mean().iloc[-1]
            
            # Volatility
            if len(target_data) > 21:
                features[f'{col}_vol_21d'] = target_data[col].rolling(21).std().iloc[-1]
            
            # Z-score
            if len(target_data) > 252:
                mean_252 = target_data[col].rolling(252).mean().iloc[-1]
                std_252 = target_data[col].rolling(252).std().iloc[-1]
                if std_252 > 0:
                    features[f'{col}_zscore'] = (target_data[col].iloc[-1] - mean_252) / std_252
        
        # Add date
        features['date'] = target_date
        
        # Convert to DataFrame
        features_df = pd.DataFrame([features])
        
        # Add derived features
        features_df = self._add_derived_macro_features(features_df)
        
        return features_df
    
    def _add_derived_macro_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived macro features for better predictive power"""
        df = df.copy()
        
        # Yield curve features
        if 'treasury_10y_current' in df.columns and 'treasury_2y_current' in df.columns:
            df['yield_curve_slope'] = df['treasury_10y_current'] - df['treasury_2y_current']
            df['yield_curve_inverted'] = (df['yield_curve_slope'] < 0).astype(int)
        
        # Real rates
        if 'fed_funds_rate_current' in df.columns and 'cpi_all_change_252d' in df.columns:
            df['real_fed_funds'] = df['fed_funds_rate_current'] - df['cpi_all_change_252d']
        
        # Employment strength
        if 'unemployment_rate_current' in df.columns and 'labor_force_participation_current' in df.columns:
            df['employment_strength'] = (
                (100 - df['unemployment_rate_current']) * 
                df['labor_force_participation_current'] / 100
            )
        
        # Growth momentum
        growth_cols = [col for col in df.columns if 'gdp' in col and 'change' in col]
        if growth_cols:
            df['growth_momentum'] = df[growth_cols].mean(axis=1)
        
        # Inflation pressure
        inflation_cols = [col for col in df.columns if ('cpi' in col or 'pce' in col) and 'change' in col]
        if inflation_cols:
            df['inflation_pressure'] = df[inflation_cols].mean(axis=1)
        
        # Financial conditions composite
        stress_cols = [col for col in df.columns if 'spread' in col or 'vix' in col]
        if stress_cols:
            df['financial_stress'] = df[stress_cols].mean(axis=1)
        
        return df
    
    def get_series_info(self, series_id: str) -> Dict:
        """
        Get metadata about a FRED series
        
        Returns:
            Dictionary with series information
        """
        try:
            info = self.fred.get_series_info(series_id)
            return {
                'id': info['id'],
                'title': info['title'],
                'units': info['units'],
                'frequency': info['frequency'],
                'seasonal_adjustment': info['seasonal_adjustment'],
                'last_updated': info['last_updated'],
                'observation_start': info['observation_start'],
                'observation_end': info['observation_end'],
                'popularity': info['popularity'],
                'notes': info['notes'][:500] if 'notes' in info else None  # Limit notes length
            }
        except Exception as e:
            logger.error(f"Error fetching series info for {series_id}: {e}")
            return {}
    
    def search_series(self, search_text: str, limit: int = 10) -> pd.DataFrame:
        """
        Search for FRED series by text
        
        Returns:
            DataFrame with search results
        """
        try:
            results = self.fred.search(search_text, limit=limit)
            if results is not None and not results.empty:
                # Get series info for each result
                series_info = []
                for series_id in results.index[:limit]:
                    info = self.get_series_info(series_id)
                    if info:
                        series_info.append(info)
                
                return pd.DataFrame(series_info)
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error searching for '{search_text}': {e}")
            return pd.DataFrame()