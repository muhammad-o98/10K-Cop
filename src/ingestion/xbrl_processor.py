"""
XBRL Financial Data Processor Module
Extracts structured financial metrics from SEC XBRL companyfacts API
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import requests
from collections import defaultdict

logger = logging.getLogger(__name__)


class XBRLProcessor:
    """
    Processes XBRL financial data from SEC companyfacts API
    """
    
    COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{}.json"
    
    # Key financial statement items mapping
    GAAP_MAPPING = {
        # Balance Sheet Items
        'Assets': 'us-gaap:Assets',
        'CurrentAssets': 'us-gaap:AssetsCurrent',
        'Cash': 'us-gaap:CashAndCashEquivalentsAtCarryingValue',
        'AccountsReceivable': 'us-gaap:AccountsReceivableNetCurrent',
        'Inventory': 'us-gaap:InventoryNet',
        'PPE': 'us-gaap:PropertyPlantAndEquipmentNet',
        'Liabilities': 'us-gaap:Liabilities',
        'CurrentLiabilities': 'us-gaap:LiabilitiesCurrent',
        'LongTermDebt': 'us-gaap:LongTermDebtNoncurrent',
        'Equity': 'us-gaap:StockholdersEquity',
        
        # Income Statement Items
        'Revenue': 'us-gaap:Revenues',
        'RevenueAlt': 'us-gaap:SalesRevenueNet',
        'CostOfRevenue': 'us-gaap:CostOfRevenue',
        'GrossProfit': 'us-gaap:GrossProfit',
        'OperatingExpenses': 'us-gaap:OperatingExpenses',
        'OperatingIncome': 'us-gaap:OperatingIncomeLoss',
        'InterestExpense': 'us-gaap:InterestExpense',
        'IncomeTaxExpense': 'us-gaap:IncomeTaxExpenseBenefit',
        'NetIncome': 'us-gaap:NetIncomeLoss',
        'EPS': 'us-gaap:EarningsPerShareDiluted',
        
        # Cash Flow Items
        'OperatingCashFlow': 'us-gaap:NetCashProvidedByUsedInOperatingActivities',
        'InvestingCashFlow': 'us-gaap:NetCashProvidedByUsedInInvestingActivities',
        'FinancingCashFlow': 'us-gaap:NetCashProvidedByUsedInFinancingActivities',
        'FreeCashFlow': 'us-gaap:FreeCashFlow',
        'CapEx': 'us-gaap:PaymentsToAcquirePropertyPlantAndEquipment',
        
        # Additional Metrics
        'SharesOutstanding': 'us-gaap:WeightedAverageNumberOfDilutedSharesOutstanding',
        'CommonStock': 'us-gaap:CommonStockSharesOutstanding',
        'RetainedEarnings': 'us-gaap:RetainedEarningsAccumulatedDeficit',
        'WorkingCapital': 'us-gaap:WorkingCapital',
    }
    
    def __init__(self,
                 user_agent: str,
                 cache_dir: str = "./data/raw/xbrl",
                 rate_limit_delay: float = 0.1):
        """
        Initialize XBRL processor
        
        Args:
            user_agent: Required user agent for SEC compliance
            cache_dir: Directory to cache companyfacts data
            rate_limit_delay: Delay between API requests
        """
        self.user_agent = user_agent
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limit_delay = rate_limit_delay
        self.last_request_time = 0
        
        # Session setup
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': user_agent,
            'Accept-Encoding': 'gzip, deflate',
            'Host': 'data.sec.gov'
        })
        
        logger.info(f"XBRLProcessor initialized with cache at {self.cache_dir}")
    
    def _rate_limit(self):
        """Enforce rate limiting"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = time.time()
    
    def fetch_companyfacts(self, cik: str, force_refresh: bool = False) -> Optional[Dict]:
        """
        Fetch company facts from SEC API
        
        Args:
            cik: Company CIK (will be zero-padded)
            force_refresh: Force download even if cached
            
        Returns:
            Company facts JSON data
        """
        cik = str(cik).zfill(10)
        cache_file = self.cache_dir / f"{cik}_facts.json"
        
        # Check cache (valid for 1 day)
        if not force_refresh and cache_file.exists():
            mod_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
            if datetime.now() - mod_time < timedelta(days=1):
                logger.info(f"Loading cached companyfacts for CIK {cik}")
                with open(cache_file, 'r') as f:
                    return json.load(f)
        
        # Download fresh data
        try:
            url = self.COMPANYFACTS_URL.format(cik)
            self._rate_limit()
            
            response = self.session.get(url)
            response.raise_for_status()
            
            data = response.json()
            
            # Cache the data
            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Successfully fetched companyfacts for CIK {cik}")
            return data
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logger.warning(f"No companyfacts found for CIK {cik}")
            else:
                logger.error(f"HTTP error fetching companyfacts for CIK {cik}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error fetching companyfacts for CIK {cik}: {e}")
            return None
    
    def extract_metrics(self, 
                       facts_json: Dict,
                       fiscal_years: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Extract key financial metrics from companyfacts JSON
        
        Args:
            facts_json: Raw companyfacts JSON
            fiscal_years: Filter for specific fiscal years
            
        Returns:
            DataFrame with financial metrics by fiscal period
        """
        if not facts_json or 'facts' not in facts_json:
            return pd.DataFrame()
        
        metrics_data = []
        facts = facts_json.get('facts', {})
        
        # Process us-gaap namespace (standard GAAP metrics)
        us_gaap = facts.get('us-gaap', {})
        
        for metric_name, gaap_field in self.GAAP_MAPPING.items():
            field_key = gaap_field.replace('us-gaap:', '')
            
            if field_key in us_gaap:
                field_data = us_gaap[field_key]
                units_data = field_data.get('units', {})
                
                # Process different unit types
                for unit_type, values in units_data.items():
                    for value_item in values:
                        # Extract filing metadata
                        fy = value_item.get('fy')
                        fp = value_item.get('fp')
                        form = value_item.get('form')
                        
                        # Filter by fiscal years if specified
                        if fiscal_years and fy not in fiscal_years:
                            continue
                        
                        # Only use 10-K/10-Q data
                        if form not in ['10-K', '10-Q', '8-K']:
                            continue
                        
                        metrics_data.append({
                            'metric': metric_name,
                            'gaap_field': gaap_field,
                            'value': value_item.get('val'),
                            'unit': unit_type,
                            'fiscal_year': fy,
                            'fiscal_period': fp,
                            'form': form,
                            'filed': value_item.get('filed'),
                            'accession': value_item.get('accn'),
                            'start_date': value_item.get('start'),
                            'end_date': value_item.get('end'),
                            'instant': value_item.get('instant'),
                        })
        
        df = pd.DataFrame(metrics_data)
        
        if df.empty:
            return df
        
        # Convert dates
        date_cols = ['filed', 'start_date', 'end_date', 'instant']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Sort and deduplicate
        df = df.sort_values(['fiscal_year', 'fiscal_period', 'filed'])
        
        # Keep the most recent filing for each metric/period combination
        df = df.drop_duplicates(
            subset=['metric', 'fiscal_year', 'fiscal_period'],
            keep='last'
        )
        
        return df
    
    def parse_footnotes(self, facts_json: Dict) -> Dict[str, Any]:
        """
        Extract footnote information from companyfacts
        
        Returns:
            Dictionary of footnotes by accession number
        """
        footnotes = {}
        
        if not facts_json or 'facts' not in facts_json:
            return footnotes
        
        # Note: Full footnote parsing would require additional XBRL processing
        # This is a placeholder for more complex footnote extraction
        
        entity_info = facts_json.get('entityInformation', {})
        footnotes['entity'] = {
            'name': entity_info.get('entityName'),
            'cik': entity_info.get('cik'),
            'ein': entity_info.get('ein'),
            'incorporation': entity_info.get('stateCountryIncorporation'),
            'fiscal_year_end': entity_info.get('fiscalYearEnd'),
        }
        
        return footnotes
    
    def standardize_accounts(self, 
                           metrics_df: pd.DataFrame,
                           target_currency: str = 'USD') -> pd.DataFrame:
        """
        Standardize account values to common units and currency
        
        Args:
            metrics_df: DataFrame from extract_metrics
            target_currency: Target currency for standardization
            
        Returns:
            Standardized DataFrame
        """
        if metrics_df.empty:
            return metrics_df
        
        df = metrics_df.copy()
        
        # Standardize units (convert to base units)
        unit_multipliers = {
            'USD': 1,
            'USD/shares': 1,  # Per share metrics
            'shares': 1,
            'pure': 1,  # Ratios
            'USD_per_shares': 1,
        }
        
        # Handle millions/thousands notation in units
        def parse_unit_multiplier(unit):
            if pd.isna(unit):
                return 1
            unit = str(unit)
            if 'Millions' in unit or 'InMillions' in unit:
                return 1_000_000
            elif 'Thousands' in unit or 'InThousands' in unit:
                return 1_000
            elif 'Hundreds' in unit or 'InHundreds' in unit:
                return 100
            return 1
        
        df['multiplier'] = df['unit'].apply(parse_unit_multiplier)
        df['standardized_value'] = df['value'] * df['multiplier']
        
        # If standardized_value is same as value for all rows, keep it for consistency
        return df
    
    def calculate_derived_metrics(self, metrics_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate additional derived metrics from base XBRL data
        
        Returns:
            DataFrame with additional calculated metrics
        """
        if metrics_df.empty:
            return metrics_df
        
        df = metrics_df.copy()
        
        # Use standardized_value if it exists, otherwise use value
        value_column = 'standardized_value' if 'standardized_value' in df.columns else 'value'
        
        # Pivot to get metrics as columns for easier calculation
        pivot_df = df.pivot_table(
            index=['fiscal_year', 'fiscal_period'],
            columns='metric',
            values=value_column,
            aggfunc='first'
        ).reset_index()
        
        # Calculate derived metrics
        if 'Revenue' in pivot_df.columns and 'CostOfRevenue' in pivot_df.columns:
            pivot_df['GrossProfitCalc'] = pivot_df['Revenue'] - pivot_df['CostOfRevenue']
            pivot_df['GrossMargin'] = pivot_df['GrossProfitCalc'] / pivot_df['Revenue']
        
        if 'OperatingIncome' in pivot_df.columns and 'Revenue' in pivot_df.columns:
            pivot_df['OperatingMargin'] = pivot_df['OperatingIncome'] / pivot_df['Revenue']
        
        if 'NetIncome' in pivot_df.columns and 'Revenue' in pivot_df.columns:
            pivot_df['NetMargin'] = pivot_df['NetIncome'] / pivot_df['Revenue']
        
        if 'CurrentAssets' in pivot_df.columns and 'CurrentLiabilities' in pivot_df.columns:
            pivot_df['WorkingCapitalCalc'] = pivot_df['CurrentAssets'] - pivot_df['CurrentLiabilities']
            pivot_df['CurrentRatio'] = pivot_df['CurrentAssets'] / pivot_df['CurrentLiabilities']
        
        if 'NetIncome' in pivot_df.columns and 'Assets' in pivot_df.columns:
            pivot_df['ROA'] = pivot_df['NetIncome'] / pivot_df['Assets']
        
        if 'NetIncome' in pivot_df.columns and 'Equity' in pivot_df.columns:
            pivot_df['ROE'] = pivot_df['NetIncome'] / pivot_df['Equity']
        
        if 'LongTermDebt' in pivot_df.columns and 'Equity' in pivot_df.columns:
            pivot_df['DebtToEquity'] = pivot_df['LongTermDebt'] / pivot_df['Equity']
        
        # Melt back to long format
        derived_metrics = pivot_df.melt(
            id_vars=['fiscal_year', 'fiscal_period'],
            var_name='metric',
            value_name='value'
        )
        
        derived_metrics['source'] = 'calculated'
        
        return derived_metrics
    
    def get_financial_statements(self,
                                cik: str,
                                fiscal_years: List[int],
                                statements: List[str] = ['BS', 'IS', 'CF']) -> Dict[str, pd.DataFrame]:
        """
        Get structured financial statements
        
        Args:
            cik: Company CIK
            fiscal_years: Years to retrieve
            statements: List of statements ('BS'=Balance Sheet, 'IS'=Income Statement, 'CF'=Cash Flow)
            
        Returns:
            Dictionary of DataFrames by statement type
        """
        # Fetch company facts
        facts = self.fetch_companyfacts(cik)
        if not facts:
            return {}
        
        # Extract metrics
        metrics_df = self.extract_metrics(facts, fiscal_years)
        if metrics_df.empty:
            return {}
        
        # Standardize values
        metrics_df = self.standardize_accounts(metrics_df)
        
        # Calculate derived metrics
        derived_df = self.calculate_derived_metrics(metrics_df)
        
        # Organize by statement type
        statement_mapping = {
            'BS': ['Assets', 'CurrentAssets', 'Cash', 'AccountsReceivable', 'Inventory',
                   'PPE', 'Liabilities', 'CurrentLiabilities', 'LongTermDebt', 'Equity',
                   'CurrentRatio', 'WorkingCapitalCalc', 'DebtToEquity'],
            'IS': ['Revenue', 'RevenueAlt', 'CostOfRevenue', 'GrossProfit', 'GrossProfitCalc',
                   'OperatingExpenses', 'OperatingIncome', 'InterestExpense', 
                   'IncomeTaxExpense', 'NetIncome', 'EPS', 'GrossMargin', 
                   'OperatingMargin', 'NetMargin'],
            'CF': ['OperatingCashFlow', 'InvestingCashFlow', 'FinancingCashFlow',
                   'FreeCashFlow', 'CapEx'],
        }
        
        results = {}
        
        # Filter annual data only for 10-K
        annual_df = metrics_df[metrics_df['form'] == '10-K'].copy()
        
        # Use standardized_value if it exists, otherwise use value
        value_column = 'standardized_value' if 'standardized_value' in annual_df.columns else 'value'
        
        for statement_type in statements:
            if statement_type in statement_mapping:
                statement_metrics = statement_mapping[statement_type]
                
                # Filter for relevant metrics
                statement_df = annual_df[annual_df['metric'].isin(statement_metrics)]
                
                if not statement_df.empty:
                    # Pivot to create statement format
                    pivot = statement_df.pivot_table(
                        index='metric',
                        columns='fiscal_year',
                        values=value_column,
                        aggfunc='first'
                    )
                    
                    # Add derived metrics
                    derived_metrics = derived_df[derived_df['metric'].isin(statement_metrics)]
                    if not derived_metrics.empty:
                        derived_pivot = derived_metrics.pivot_table(
                            index='metric',
                            columns='fiscal_year', 
                            values='value',
                            aggfunc='first'
                        )
                        pivot = pd.concat([pivot, derived_pivot])
                    
                    results[statement_type] = pivot
        
        return results
    
    def get_peer_metrics(self,
                        ciks: List[str],
                        metrics: List[str],
                        fiscal_year: int) -> pd.DataFrame:
        """
        Get metrics for peer comparison
        
        Args:
            ciks: List of company CIKs
            metrics: List of metric names to retrieve
            fiscal_year: Fiscal year for comparison
            
        Returns:
            DataFrame with peer metrics
        """
        peer_data = []
        
        for cik in ciks:
            facts = self.fetch_companyfacts(cik)
            if not facts:
                continue
            
            metrics_df = self.extract_metrics(facts, [fiscal_year])
            if metrics_df.empty:
                continue
            
            metrics_df = self.standardize_accounts(metrics_df)
            
            # Get annual data only
            annual = metrics_df[metrics_df['form'] == '10-K']
            
            for metric in metrics:
                metric_data = annual[annual['metric'] == metric]
                if not metric_data.empty:
                    # Use standardized_value if available, otherwise use value
                    value_col = 'standardized_value' if 'standardized_value' in metric_data.columns else 'value'
                    value = metric_data.iloc[0][value_col]
                    peer_data.append({
                        'cik': cik,
                        'metric': metric,
                        'value': value,
                        'fiscal_year': fiscal_year
                    })
        
        return pd.DataFrame(peer_data)
    
    def validate_data_quality(self, metrics_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate extracted XBRL data quality
        
        Returns:
            Dictionary with validation results
        """
        validation = {
            'total_records': len(metrics_df),
            'missing_values': {},
            'outliers': {},
            'coverage': {},
            'consistency_checks': []
        }
        
        if metrics_df.empty:
            return validation
        
        # Check missing values by metric
        for metric in metrics_df['metric'].unique():
            metric_data = metrics_df[metrics_df['metric'] == metric]
            missing_pct = metric_data['value'].isna().sum() / len(metric_data) * 100
            validation['missing_values'][metric] = f"{missing_pct:.1f}%"
        
        # Check for outliers using IQR method (use 'value' column if 'standardized_value' doesn't exist)
        value_col = 'standardized_value' if 'standardized_value' in metrics_df.columns else 'value'
        for metric in metrics_df['metric'].unique():
            metric_data = metrics_df[metrics_df['metric'] == metric][value_col].dropna()
            if len(metric_data) > 3:
                Q1 = metric_data.quantile(0.25)
                Q3 = metric_data.quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((metric_data < (Q1 - 1.5 * IQR)) | (metric_data > (Q3 + 1.5 * IQR))).sum()
                if outliers > 0:
                    validation['outliers'][metric] = outliers
        
        # Check data coverage by fiscal year
        if 'fiscal_year' in metrics_df.columns:
            years_coverage = metrics_df.groupby('fiscal_year')['metric'].nunique()
            validation['coverage'] = years_coverage.to_dict()
        
        # Consistency checks
        # Example: Assets = Liabilities + Equity
        annual_data = metrics_df[metrics_df['form'] == '10-K']
        for year in annual_data['fiscal_year'].unique():
            year_data = annual_data[annual_data['fiscal_year'] == year]
            
            # Get values (use standardized_value if available, otherwise use value)
            assets = year_data[year_data['metric'] == 'Assets'][value_col].values
            liabilities = year_data[year_data['metric'] == 'Liabilities'][value_col].values
            equity = year_data[year_data['metric'] == 'Equity'][value_col].values
            
            if len(assets) > 0 and len(liabilities) > 0 and len(equity) > 0:
                diff = abs(assets[0] - (liabilities[0] + equity[0]))
                if diff > assets[0] * 0.01:  # More than 1% difference
                    validation['consistency_checks'].append({
                        'year': year,
                        'check': 'Assets = Liabilities + Equity',
                        'difference': diff,
                        'percentage': (diff / assets[0]) * 100
                    })
        
        return validation