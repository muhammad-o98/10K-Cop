"""
Financial Ratio Calculator Module
Calculates 20+ financial ratios and metrics from XBRL data
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import pandas as pd
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RatioResult:
    """Container for ratio calculation results"""
    ratio_name: str
    value: float
    category: str
    fiscal_year: int
    fiscal_period: str
    interpretation: str = ""
    benchmark: Optional[float] = None
    percentile: Optional[float] = None
    trend: Optional[str] = None
    health_score: Optional[float] = None


class RatioCalculator:
    """
    Calculates comprehensive financial ratios from financial statements
    """
    
    # Ratio categories
    LIQUIDITY_RATIOS = [
        'current_ratio', 'quick_ratio', 'cash_ratio', 'operating_cash_ratio',
        'defensive_interval_ratio', 'cash_conversion_cycle'
    ]
    
    LEVERAGE_RATIOS = [
        'debt_to_equity', 'debt_to_assets', 'equity_ratio', 'equity_multiplier',
        'interest_coverage', 'debt_service_coverage', 'fixed_charge_coverage'
    ]
    
    PROFITABILITY_RATIOS = [
        'gross_margin', 'operating_margin', 'net_margin', 'ebitda_margin',
        'roa', 'roe', 'roic', 'roce', 'ros', 'eps_growth'
    ]
    
    EFFICIENCY_RATIOS = [
        'asset_turnover', 'fixed_asset_turnover', 'working_capital_turnover',
        'inventory_turnover', 'receivables_turnover', 'payables_turnover',
        'days_inventory', 'days_receivables', 'days_payables'
    ]
    
    VALUATION_RATIOS = [
        'pe_ratio', 'peg_ratio', 'price_to_book', 'price_to_sales',
        'ev_to_ebitda', 'ev_to_revenue', 'dividend_yield', 'dividend_payout'
    ]
    
    GROWTH_RATIOS = [
        'revenue_growth', 'earnings_growth', 'asset_growth', 'equity_growth',
        'operating_cash_flow_growth', 'free_cash_flow_growth'
    ]
    
    # Industry benchmarks (simplified - in production, load from database)
    INDUSTRY_BENCHMARKS = {
        'Technology': {
            'current_ratio': 2.0,
            'debt_to_equity': 0.5,
            'gross_margin': 0.60,
            'operating_margin': 0.25,
            'roe': 0.20,
            'asset_turnover': 0.8
        },
        'Healthcare': {
            'current_ratio': 1.8,
            'debt_to_equity': 0.6,
            'gross_margin': 0.65,
            'operating_margin': 0.20,
            'roe': 0.15,
            'asset_turnover': 0.9
        },
        'Consumer': {
            'current_ratio': 1.5,
            'debt_to_equity': 0.8,
            'gross_margin': 0.40,
            'operating_margin': 0.15,
            'roe': 0.18,
            'asset_turnover': 1.2
        }
    }
    
    def __init__(self, industry: str = "Technology"):
        """
        Initialize ratio calculator
        
        Args:
            industry: Industry for benchmarking
        """
        self.industry = industry
        self.benchmarks = self.INDUSTRY_BENCHMARKS.get(industry, self.INDUSTRY_BENCHMARKS['Technology'])
        logger.info(f"RatioCalculator initialized for {industry} industry")
    
    def calculate_all_ratios(self, 
                            financial_data: Dict[str, float],
                            fiscal_year: int,
                            fiscal_period: str = "FY") -> Dict[str, RatioResult]:
        """
        Calculate all available ratios from financial data
        
        Args:
            financial_data: Dictionary of financial metrics
            fiscal_year: Fiscal year
            fiscal_period: Fiscal period (FY, Q1, Q2, etc.)
            
        Returns:
            Dictionary of ratio results
        """
        results = {}
        
        # Calculate each category of ratios
        results.update(self.calculate_liquidity_ratios(financial_data, fiscal_year, fiscal_period))
        results.update(self.calculate_leverage_ratios(financial_data, fiscal_year, fiscal_period))
        results.update(self.calculate_profitability_ratios(financial_data, fiscal_year, fiscal_period))
        results.update(self.calculate_efficiency_ratios(financial_data, fiscal_year, fiscal_period))
        
        # Add interpretations and health scores
        for ratio_name, result in results.items():
            result.interpretation = self._interpret_ratio(ratio_name, result.value)
            result.health_score = self._calculate_health_score(ratio_name, result.value)
            
            # Add benchmark if available
            if ratio_name in self.benchmarks:
                result.benchmark = self.benchmarks[ratio_name]
        
        logger.info(f"Calculated {len(results)} ratios for fiscal year {fiscal_year}")
        return results
    
    def calculate_liquidity_ratios(self,
                                  data: Dict[str, float],
                                  fiscal_year: int,
                                  fiscal_period: str) -> Dict[str, RatioResult]:
        """Calculate liquidity ratios"""
        ratios = {}
        
        # Current Ratio
        if 'CurrentAssets' in data and 'CurrentLiabilities' in data and data['CurrentLiabilities'] > 0:
            current_ratio = data['CurrentAssets'] / data['CurrentLiabilities']
            ratios['current_ratio'] = RatioResult(
                ratio_name='current_ratio',
                value=current_ratio,
                category='Liquidity',
                fiscal_year=fiscal_year,
                fiscal_period=fiscal_period
            )
        
        # Quick Ratio (Acid Test)
        if all(k in data for k in ['CurrentAssets', 'Inventory', 'CurrentLiabilities']):
            if data['CurrentLiabilities'] > 0:
                quick_assets = data['CurrentAssets'] - data.get('Inventory', 0)
                quick_ratio = quick_assets / data['CurrentLiabilities']
                ratios['quick_ratio'] = RatioResult(
                    ratio_name='quick_ratio',
                    value=quick_ratio,
                    category='Liquidity',
                    fiscal_year=fiscal_year,
                    fiscal_period=fiscal_period
                )
        
        # Cash Ratio
        if 'Cash' in data and 'CurrentLiabilities' in data and data['CurrentLiabilities'] > 0:
            cash_ratio = data['Cash'] / data['CurrentLiabilities']
            ratios['cash_ratio'] = RatioResult(
                ratio_name='cash_ratio',
                value=cash_ratio,
                category='Liquidity',
                fiscal_year=fiscal_year,
                fiscal_period=fiscal_period
            )
        
        # Operating Cash Flow Ratio
        if 'OperatingCashFlow' in data and 'CurrentLiabilities' in data and data['CurrentLiabilities'] > 0:
            ocf_ratio = data['OperatingCashFlow'] / data['CurrentLiabilities']
            ratios['operating_cash_ratio'] = RatioResult(
                ratio_name='operating_cash_ratio',
                value=ocf_ratio,
                category='Liquidity',
                fiscal_year=fiscal_year,
                fiscal_period=fiscal_period
            )
        
        # Cash Conversion Cycle
        if all(k in data for k in ['days_inventory', 'days_receivables', 'days_payables']):
            ccc = data['days_inventory'] + data['days_receivables'] - data.get('days_payables', 0)
            ratios['cash_conversion_cycle'] = RatioResult(
                ratio_name='cash_conversion_cycle',
                value=ccc,
                category='Liquidity',
                fiscal_year=fiscal_year,
                fiscal_period=fiscal_period
            )
        
        return ratios
    
    def calculate_leverage_ratios(self,
                                 data: Dict[str, float],
                                 fiscal_year: int,
                                 fiscal_period: str) -> Dict[str, RatioResult]:
        """Calculate leverage/solvency ratios"""
        ratios = {}
        
        # Debt to Equity
        if 'LongTermDebt' in data and 'Equity' in data and data['Equity'] > 0:
            debt_to_equity = data.get('LongTermDebt', 0) / data['Equity']
            ratios['debt_to_equity'] = RatioResult(
                ratio_name='debt_to_equity',
                value=debt_to_equity,
                category='Leverage',
                fiscal_year=fiscal_year,
                fiscal_period=fiscal_period
            )
        
        # Debt to Assets
        if 'Liabilities' in data and 'Assets' in data and data['Assets'] > 0:
            debt_to_assets = data['Liabilities'] / data['Assets']
            ratios['debt_to_assets'] = RatioResult(
                ratio_name='debt_to_assets',
                value=debt_to_assets,
                category='Leverage',
                fiscal_year=fiscal_year,
                fiscal_period=fiscal_period
            )
        
        # Equity Ratio
        if 'Equity' in data and 'Assets' in data and data['Assets'] > 0:
            equity_ratio = data['Equity'] / data['Assets']
            ratios['equity_ratio'] = RatioResult(
                ratio_name='equity_ratio',
                value=equity_ratio,
                category='Leverage',
                fiscal_year=fiscal_year,
                fiscal_period=fiscal_period
            )
        
        # Interest Coverage Ratio
        if 'OperatingIncome' in data and 'InterestExpense' in data and data['InterestExpense'] > 0:
            interest_coverage = data['OperatingIncome'] / data['InterestExpense']
            ratios['interest_coverage'] = RatioResult(
                ratio_name='interest_coverage',
                value=interest_coverage,
                category='Leverage',
                fiscal_year=fiscal_year,
                fiscal_period=fiscal_period
            )
        
        # Debt Service Coverage Ratio
        if 'OperatingCashFlow' in data and 'InterestExpense' in data:
            debt_payments = data.get('InterestExpense', 0) + data.get('DebtPayments', 0)
            if debt_payments > 0:
                dscr = data['OperatingCashFlow'] / debt_payments
                ratios['debt_service_coverage'] = RatioResult(
                    ratio_name='debt_service_coverage',
                    value=dscr,
                    category='Leverage',
                    fiscal_year=fiscal_year,
                    fiscal_period=fiscal_period
                )
        
        return ratios
    
    def calculate_profitability_ratios(self,
                                      data: Dict[str, float],
                                      fiscal_year: int,
                                      fiscal_period: str) -> Dict[str, RatioResult]:
        """Calculate profitability ratios"""
        ratios = {}
        
        # Gross Margin
        if 'GrossProfit' in data and 'Revenue' in data and data['Revenue'] > 0:
            gross_margin = data['GrossProfit'] / data['Revenue']
            ratios['gross_margin'] = RatioResult(
                ratio_name='gross_margin',
                value=gross_margin,
                category='Profitability',
                fiscal_year=fiscal_year,
                fiscal_period=fiscal_period
            )
        elif 'Revenue' in data and 'CostOfRevenue' in data and data['Revenue'] > 0:
            gross_margin = (data['Revenue'] - data['CostOfRevenue']) / data['Revenue']
            ratios['gross_margin'] = RatioResult(
                ratio_name='gross_margin',
                value=gross_margin,
                category='Profitability',
                fiscal_year=fiscal_year,
                fiscal_period=fiscal_period
            )
        
        # Operating Margin
        if 'OperatingIncome' in data and 'Revenue' in data and data['Revenue'] > 0:
            operating_margin = data['OperatingIncome'] / data['Revenue']
            ratios['operating_margin'] = RatioResult(
                ratio_name='operating_margin',
                value=operating_margin,
                category='Profitability',
                fiscal_year=fiscal_year,
                fiscal_period=fiscal_period
            )
        
        # Net Margin
        if 'NetIncome' in data and 'Revenue' in data and data['Revenue'] > 0:
            net_margin = data['NetIncome'] / data['Revenue']
            ratios['net_margin'] = RatioResult(
                ratio_name='net_margin',
                value=net_margin,
                category='Profitability',
                fiscal_year=fiscal_year,
                fiscal_period=fiscal_period
            )
        
        # Return on Assets (ROA)
        if 'NetIncome' in data and 'Assets' in data and data['Assets'] > 0:
            roa = data['NetIncome'] / data['Assets']
            ratios['roa'] = RatioResult(
                ratio_name='roa',
                value=roa,
                category='Profitability',
                fiscal_year=fiscal_year,
                fiscal_period=fiscal_period
            )
        
        # Return on Equity (ROE)
        if 'NetIncome' in data and 'Equity' in data and data['Equity'] > 0:
            roe = data['NetIncome'] / data['Equity']
            ratios['roe'] = RatioResult(
                ratio_name='roe',
                value=roe,
                category='Profitability',
                fiscal_year=fiscal_year,
                fiscal_period=fiscal_period
            )
        
        # Return on Invested Capital (ROIC)
        if 'OperatingIncome' in data and 'IncomeTaxExpense' in data:
            nopat = data['OperatingIncome'] * (1 - data.get('TaxRate', 0.21))
            invested_capital = data.get('Equity', 0) + data.get('LongTermDebt', 0)
            if invested_capital > 0:
                roic = nopat / invested_capital
                ratios['roic'] = RatioResult(
                    ratio_name='roic',
                    value=roic,
                    category='Profitability',
                    fiscal_year=fiscal_year,
                    fiscal_period=fiscal_period
                )
        
        # EBITDA Margin
        if all(k in data for k in ['OperatingIncome', 'Depreciation', 'Amortization', 'Revenue']):
            if data['Revenue'] > 0:
                ebitda = data['OperatingIncome'] + data.get('Depreciation', 0) + data.get('Amortization', 0)
                ebitda_margin = ebitda / data['Revenue']
                ratios['ebitda_margin'] = RatioResult(
                    ratio_name='ebitda_margin',
                    value=ebitda_margin,
                    category='Profitability',
                    fiscal_year=fiscal_year,
                    fiscal_period=fiscal_period
                )
        
        return ratios
    
    def calculate_efficiency_ratios(self,
                                   data: Dict[str, float],
                                   fiscal_year: int,
                                   fiscal_period: str) -> Dict[str, RatioResult]:
        """Calculate efficiency/activity ratios"""
        ratios = {}
        
        # Asset Turnover
        if 'Revenue' in data and 'Assets' in data and data['Assets'] > 0:
            asset_turnover = data['Revenue'] / data['Assets']
            ratios['asset_turnover'] = RatioResult(
                ratio_name='asset_turnover',
                value=asset_turnover,
                category='Efficiency',
                fiscal_year=fiscal_year,
                fiscal_period=fiscal_period
            )
        
        # Fixed Asset Turnover
        if 'Revenue' in data and 'PPE' in data and data['PPE'] > 0:
            fixed_asset_turnover = data['Revenue'] / data['PPE']
            ratios['fixed_asset_turnover'] = RatioResult(
                ratio_name='fixed_asset_turnover',
                value=fixed_asset_turnover,
                category='Efficiency',
                fiscal_year=fiscal_year,
                fiscal_period=fiscal_period
            )
        
        # Working Capital Turnover
        working_capital = data.get('CurrentAssets', 0) - data.get('CurrentLiabilities', 0)
        if 'Revenue' in data and working_capital > 0:
            wc_turnover = data['Revenue'] / working_capital
            ratios['working_capital_turnover'] = RatioResult(
                ratio_name='working_capital_turnover',
                value=wc_turnover,
                category='Efficiency',
                fiscal_year=fiscal_year,
                fiscal_period=fiscal_period
            )
        
        # Inventory Turnover
        if 'CostOfRevenue' in data and 'Inventory' in data and data['Inventory'] > 0:
            inventory_turnover = data['CostOfRevenue'] / data['Inventory']
            ratios['inventory_turnover'] = RatioResult(
                ratio_name='inventory_turnover',
                value=inventory_turnover,
                category='Efficiency',
                fiscal_year=fiscal_year,
                fiscal_period=fiscal_period
            )
            
            # Days Inventory Outstanding
            ratios['days_inventory'] = RatioResult(
                ratio_name='days_inventory',
                value=365 / inventory_turnover,
                category='Efficiency',
                fiscal_year=fiscal_year,
                fiscal_period=fiscal_period
            )
        
        # Receivables Turnover
        if 'Revenue' in data and 'AccountsReceivable' in data and data['AccountsReceivable'] > 0:
            receivables_turnover = data['Revenue'] / data['AccountsReceivable']
            ratios['receivables_turnover'] = RatioResult(
                ratio_name='receivables_turnover',
                value=receivables_turnover,
                category='Efficiency',
                fiscal_year=fiscal_year,
                fiscal_period=fiscal_period
            )
            
            # Days Sales Outstanding
            ratios['days_receivables'] = RatioResult(
                ratio_name='days_receivables',
                value=365 / receivables_turnover,
                category='Efficiency',
                fiscal_year=fiscal_year,
                fiscal_period=fiscal_period
            )
        
        # Payables Turnover (if accounts payable data available)
        if 'CostOfRevenue' in data and 'AccountsPayable' in data and data['AccountsPayable'] > 0:
            payables_turnover = data['CostOfRevenue'] / data['AccountsPayable']
            ratios['payables_turnover'] = RatioResult(
                ratio_name='payables_turnover',
                value=payables_turnover,
                category='Efficiency',
                fiscal_year=fiscal_year,
                fiscal_period=fiscal_period
            )
            
            # Days Payables Outstanding
            ratios['days_payables'] = RatioResult(
                ratio_name='days_payables',
                value=365 / payables_turnover,
                category='Efficiency',
                fiscal_year=fiscal_year,
                fiscal_period=fiscal_period
            )
        
        return ratios
    
    def calculate_growth_ratios(self,
                               current_data: Dict[str, float],
                               prior_data: Dict[str, float],
                               fiscal_year: int) -> Dict[str, RatioResult]:
        """Calculate growth ratios (year-over-year)"""
        ratios = {}
        
        # Revenue Growth
        if 'Revenue' in current_data and 'Revenue' in prior_data and prior_data['Revenue'] > 0:
            revenue_growth = (current_data['Revenue'] - prior_data['Revenue']) / prior_data['Revenue']
            ratios['revenue_growth'] = RatioResult(
                ratio_name='revenue_growth',
                value=revenue_growth,
                category='Growth',
                fiscal_year=fiscal_year,
                fiscal_period='YoY'
            )
        
        # Earnings Growth
        if 'NetIncome' in current_data and 'NetIncome' in prior_data and prior_data['NetIncome'] > 0:
            earnings_growth = (current_data['NetIncome'] - prior_data['NetIncome']) / prior_data['NetIncome']
            ratios['earnings_growth'] = RatioResult(
                ratio_name='earnings_growth',
                value=earnings_growth,
                category='Growth',
                fiscal_year=fiscal_year,
                fiscal_period='YoY'
            )
        
        # Asset Growth
        if 'Assets' in current_data and 'Assets' in prior_data and prior_data['Assets'] > 0:
            asset_growth = (current_data['Assets'] - prior_data['Assets']) / prior_data['Assets']
            ratios['asset_growth'] = RatioResult(
                ratio_name='asset_growth',
                value=asset_growth,
                category='Growth',
                fiscal_year=fiscal_year,
                fiscal_period='YoY'
            )
        
        # Operating Cash Flow Growth
        if 'OperatingCashFlow' in current_data and 'OperatingCashFlow' in prior_data:
            if prior_data['OperatingCashFlow'] > 0:
                ocf_growth = (current_data['OperatingCashFlow'] - prior_data['OperatingCashFlow']) / prior_data['OperatingCashFlow']
                ratios['operating_cash_flow_growth'] = RatioResult(
                    ratio_name='operating_cash_flow_growth',
                    value=ocf_growth,
                    category='Growth',
                    fiscal_year=fiscal_year,
                    fiscal_period='YoY'
                )
        
        return ratios
    
    def _interpret_ratio(self, ratio_name: str, value: float) -> str:
        """
        Provide interpretation for a ratio value
        
        Args:
            ratio_name: Name of the ratio
            value: Ratio value
            
        Returns:
            Text interpretation
        """
        interpretations = {
            'current_ratio': {
                'excellent': (2.0, float('inf'), "Strong liquidity position"),
                'good': (1.5, 2.0, "Adequate liquidity"),
                'fair': (1.0, 1.5, "Sufficient liquidity"),
                'poor': (0, 1.0, "Potential liquidity concerns")
            },
            'debt_to_equity': {
                'excellent': (0, 0.3, "Very low leverage"),
                'good': (0.3, 0.5, "Conservative leverage"),
                'fair': (0.5, 1.0, "Moderate leverage"),
                'poor': (1.0, float('inf'), "High leverage risk")
            },
            'gross_margin': {
                'excellent': (0.5, 1.0, "Strong pricing power"),
                'good': (0.35, 0.5, "Healthy margins"),
                'fair': (0.2, 0.35, "Average margins"),
                'poor': (0, 0.2, "Low margins, cost pressure")
            },
            'roe': {
                'excellent': (0.20, float('inf'), "Excellent returns"),
                'good': (0.15, 0.20, "Good returns"),
                'fair': (0.10, 0.15, "Average returns"),
                'poor': (-float('inf'), 0.10, "Poor returns")
            }
        }
        
        if ratio_name in interpretations:
            for category, (min_val, max_val, interpretation) in interpretations[ratio_name].items():
                if min_val <= value < max_val:
                    return interpretation
        
        return "Within normal range"
    
    def _calculate_health_score(self, ratio_name: str, value: float) -> float:
        """
        Calculate health score for a ratio (0-100)
        
        Args:
            ratio_name: Name of the ratio
            value: Ratio value
            
        Returns:
            Health score (0-100)
        """
        # Simplified scoring - in production, use more sophisticated models
        ideal_values = {
            'current_ratio': 2.0,
            'quick_ratio': 1.5,
            'debt_to_equity': 0.4,
            'gross_margin': 0.5,
            'operating_margin': 0.25,
            'net_margin': 0.15,
            'roe': 0.20,
            'roa': 0.10,
            'asset_turnover': 1.0
        }
        
        if ratio_name not in ideal_values:
            return 50.0  # Default neutral score
        
        ideal = ideal_values[ratio_name]
        
        # Calculate score based on distance from ideal
        if ratio_name in ['debt_to_equity']:  # Lower is better
            if value <= ideal:
                score = 100
            else:
                score = max(0, 100 - (value - ideal) / ideal * 50)
        else:  # Higher is generally better
            if value >= ideal:
                score = min(100, 50 + (value / ideal) * 50)
            else:
                score = max(0, (value / ideal) * 50)
        
        return round(score, 1)
    
    def compare_to_industry(self, 
                           ratios: Dict[str, RatioResult],
                           industry_data: pd.DataFrame) -> Dict[str, float]:
        """
        Compare ratios to industry percentiles
        
        Args:
            ratios: Company's calculated ratios
            industry_data: DataFrame with industry ratio distributions
            
        Returns:
            Dictionary of percentile rankings
        """
        percentiles = {}
        
        for ratio_name, result in ratios.items():
            if ratio_name in industry_data.columns:
                # Calculate percentile rank
                industry_values = industry_data[ratio_name].dropna()
                if len(industry_values) > 0:
                    percentile = (industry_values < result.value).mean() * 100
                    percentiles[ratio_name] = percentile
                    result.percentile = percentile
        
        return percentiles
    
    def generate_ratio_report(self, ratios: Dict[str, RatioResult]) -> pd.DataFrame:
        """
        Generate a comprehensive ratio report
        
        Args:
            ratios: Dictionary of calculated ratios
            
        Returns:
            DataFrame with ratio report
        """
        report_data = []
        
        for ratio_name, result in ratios.items():
            report_data.append({
                'Ratio': ratio_name.replace('_', ' ').title(),
                'Value': round(result.value, 3),
                'Category': result.category,
                'Interpretation': result.interpretation,
                'Health Score': result.health_score,
                'Benchmark': result.benchmark,
                'Percentile': result.percentile
            })
        
        df = pd.DataFrame(report_data)
        
        # Sort by category and ratio name
        df = df.sort_values(['Category', 'Ratio'])
        
        return df