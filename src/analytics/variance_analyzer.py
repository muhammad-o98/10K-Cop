"""
Variance Analyzer Module
Performs year-over-year variance analysis and waterfall charts
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import pandas as pd
import numpy as np
from dataclasses import dataclass
import plotly.graph_objects as go
import plotly.express as px

logger = logging.getLogger(__name__)


@dataclass
class VarianceItem:
    """Container for variance analysis results"""
    metric_name: str
    current_value: float
    prior_value: float
    absolute_change: float
    percent_change: float
    category: str
    impact_on_parent: Optional[float] = None
    explanation: Optional[str] = None
    is_favorable: Optional[bool] = None


class VarianceAnalyzer:
    """
    Performs comprehensive variance analysis on financial statements
    """
    
    # Define metric hierarchies for waterfall analysis
    INCOME_STATEMENT_HIERARCHY = {
        'Revenue': {
            'level': 1,
            'components': [],
            'favorable_direction': 'increase'
        },
        'CostOfRevenue': {
            'level': 2,
            'components': [],
            'favorable_direction': 'decrease'
        },
        'GrossProfit': {
            'level': 1,
            'components': ['Revenue', 'CostOfRevenue'],
            'favorable_direction': 'increase'
        },
        'OperatingExpenses': {
            'level': 2,
            'components': ['ResearchDevelopment', 'SellingGeneral', 'Other'],
            'favorable_direction': 'decrease'
        },
        'OperatingIncome': {
            'level': 1,
            'components': ['GrossProfit', 'OperatingExpenses'],
            'favorable_direction': 'increase'
        },
        'InterestExpense': {
            'level': 2,
            'components': [],
            'favorable_direction': 'decrease'
        },
        'IncomeTaxExpense': {
            'level': 2,
            'components': [],
            'favorable_direction': 'decrease'
        },
        'NetIncome': {
            'level': 1,
            'components': ['OperatingIncome', 'InterestExpense', 'IncomeTaxExpense', 'OtherIncome'],
            'favorable_direction': 'increase'
        }
    }
    
    BALANCE_SHEET_HIERARCHY = {
        'CurrentAssets': {
            'level': 2,
            'components': ['Cash', 'AccountsReceivable', 'Inventory', 'OtherCurrentAssets'],
            'favorable_direction': 'increase'
        },
        'NonCurrentAssets': {
            'level': 2,
            'components': ['PPE', 'Intangibles', 'Goodwill', 'OtherNonCurrentAssets'],
            'favorable_direction': 'context'
        },
        'Assets': {
            'level': 1,
            'components': ['CurrentAssets', 'NonCurrentAssets'],
            'favorable_direction': 'increase'
        },
        'CurrentLiabilities': {
            'level': 2,
            'components': ['AccountsPayable', 'ShortTermDebt', 'OtherCurrentLiabilities'],
            'favorable_direction': 'context'
        },
        'NonCurrentLiabilities': {
            'level': 2,
            'components': ['LongTermDebt', 'DeferredTaxes', 'OtherNonCurrentLiabilities'],
            'favorable_direction': 'decrease'
        },
        'Liabilities': {
            'level': 1,
            'components': ['CurrentLiabilities', 'NonCurrentLiabilities'],
            'favorable_direction': 'decrease'
        },
        'Equity': {
            'level': 1,
            'components': ['CommonStock', 'RetainedEarnings', 'AOCI'],
            'favorable_direction': 'increase'
        }
    }
    
    CASH_FLOW_HIERARCHY = {
        'OperatingCashFlow': {
            'level': 1,
            'components': ['NetIncome', 'Depreciation', 'WorkingCapitalChanges'],
            'favorable_direction': 'increase'
        },
        'InvestingCashFlow': {
            'level': 1,
            'components': ['CapEx', 'Acquisitions', 'InvestmentPurchases'],
            'favorable_direction': 'context'
        },
        'FinancingCashFlow': {
            'level': 1,
            'components': ['DebtIssuance', 'DebtRepayment', 'Dividends', 'ShareBuybacks'],
            'favorable_direction': 'context'
        },
        'FreeCashFlow': {
            'level': 1,
            'components': ['OperatingCashFlow', 'CapEx'],
            'favorable_direction': 'increase'
        }
    }
    
    def __init__(self):
        """Initialize variance analyzer"""
        logger.info("VarianceAnalyzer initialized")
    
    def analyze_period_variance(self,
                               current_period: Dict[str, float],
                               prior_period: Dict[str, float],
                               statement_type: str = "income_statement") -> List[VarianceItem]:
        """
        Analyze variance between two periods
        
        Args:
            current_period: Current period financial data
            prior_period: Prior period financial data
            statement_type: Type of financial statement
            
        Returns:
            List of variance items
        """
        variances = []
        
        # Get hierarchy based on statement type
        if statement_type == "income_statement":
            hierarchy = self.INCOME_STATEMENT_HIERARCHY
        elif statement_type == "balance_sheet":
            hierarchy = self.BALANCE_SHEET_HIERARCHY
        elif statement_type == "cash_flow":
            hierarchy = self.CASH_FLOW_HIERARCHY
        else:
            hierarchy = {}
        
        # Calculate variances for all metrics
        all_metrics = set(current_period.keys()) | set(prior_period.keys())
        
        for metric in all_metrics:
            current_val = current_period.get(metric, 0)
            prior_val = prior_period.get(metric, 0)
            
            # Calculate changes
            absolute_change = current_val - prior_val
            
            if prior_val != 0:
                percent_change = (absolute_change / abs(prior_val)) * 100
            else:
                percent_change = 100 if current_val > 0 else -100 if current_val < 0 else 0
            
            # Determine if change is favorable
            if metric in hierarchy:
                direction = hierarchy[metric]['favorable_direction']
                if direction == 'increase':
                    is_favorable = absolute_change > 0
                elif direction == 'decrease':
                    is_favorable = absolute_change < 0
                else:  # context-dependent
                    is_favorable = None
            else:
                is_favorable = None
            
            # Create variance item
            variance_item = VarianceItem(
                metric_name=metric,
                current_value=current_val,
                prior_value=prior_val,
                absolute_change=absolute_change,
                percent_change=percent_change,
                category=statement_type,
                is_favorable=is_favorable,
                explanation=self._generate_explanation(metric, percent_change)
            )
            
            variances.append(variance_item)
        
        # Sort by absolute change magnitude
        variances.sort(key=lambda x: abs(x.absolute_change), reverse=True)
        
        logger.info(f"Analyzed {len(variances)} variance items for {statement_type}")
        return variances
    
    def create_waterfall_data(self,
                            variances: List[VarianceItem],
                            start_metric: str = "Revenue",
                            end_metric: str = "NetIncome") -> pd.DataFrame:
        """
        Create data for waterfall chart
        
        Args:
            variances: List of variance items
            start_metric: Starting metric for waterfall
            end_metric: Ending metric for waterfall
            
        Returns:
            DataFrame formatted for waterfall chart
        """
        waterfall_data = []
        
        # Find start and end values
        start_item = next((v for v in variances if v.metric_name == start_metric), None)
        end_item = next((v for v in variances if v.metric_name == end_metric), None)
        
        if not start_item or not end_item:
            logger.warning(f"Could not find start ({start_metric}) or end ({end_metric}) metrics")
            return pd.DataFrame()
        
        # Add starting point
        waterfall_data.append({
            'Metric': f"{start_metric} (Prior)",
            'Value': start_item.prior_value,
            'Type': 'Start',
            'Cumulative': start_item.prior_value
        })
        
        # Add variances as steps
        cumulative = start_item.prior_value
        
        for variance in variances:
            if variance.metric_name in [start_metric, end_metric]:
                continue
            
            if abs(variance.absolute_change) > 0.01:  # Filter small changes
                cumulative += variance.absolute_change
                
                waterfall_data.append({
                    'Metric': variance.metric_name,
                    'Value': variance.absolute_change,
                    'Type': 'Increase' if variance.absolute_change > 0 else 'Decrease',
                    'Cumulative': cumulative
                })
        
        # Add ending point
        waterfall_data.append({
            'Metric': f"{end_metric} (Current)",
            'Value': end_item.current_value,
            'Type': 'End',
            'Cumulative': end_item.current_value
        })
        
        return pd.DataFrame(waterfall_data)
    
    def create_waterfall_chart(self,
                              waterfall_df: pd.DataFrame,
                              title: str = "Year-over-Year Variance Analysis") -> go.Figure:
        """
        Create interactive waterfall chart
        
        Args:
            waterfall_df: DataFrame with waterfall data
            title: Chart title
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Define colors
        colors = []
        for typ in waterfall_df['Type']:
            if typ == 'Start':
                colors.append('lightblue')
            elif typ == 'End':
                colors.append('darkblue')
            elif typ == 'Increase':
                colors.append('green')
            else:  # Decrease
                colors.append('red')
        
        # Create waterfall chart
        fig.add_trace(go.Waterfall(
            name="Variance",
            orientation="v",
            x=waterfall_df['Metric'],
            y=waterfall_df['Value'],
            text=[f"{v:,.0f}" for v in waterfall_df['Value']],
            textposition="outside",
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            increasing={"marker": {"color": "green"}},
            decreasing={"marker": {"color": "red"}},
            totals={"marker": {"color": "blue"}}
        ))
        
        fig.update_layout(
            title=title,
            showlegend=False,
            xaxis_title="Components",
            yaxis_title="Value ($)",
            height=600,
            template="plotly_white"
        )
        
        return fig
    
    def analyze_margin_changes(self,
                              current_period: Dict[str, float],
                              prior_period: Dict[str, float]) -> Dict[str, VarianceItem]:
        """
        Analyze changes in profit margins
        
        Args:
            current_period: Current period data
            prior_period: Prior period data
            
        Returns:
            Dictionary of margin variances
        """
        margin_variances = {}
        
        # Gross Margin
        if all(k in current_period for k in ['Revenue', 'CostOfRevenue']):
            current_gross_margin = (current_period['Revenue'] - current_period.get('CostOfRevenue', 0)) / current_period['Revenue']
            prior_gross_margin = (prior_period['Revenue'] - prior_period.get('CostOfRevenue', 0)) / prior_period['Revenue']
            
            margin_variances['gross_margin'] = VarianceItem(
                metric_name='Gross Margin',
                current_value=current_gross_margin * 100,
                prior_value=prior_gross_margin * 100,
                absolute_change=(current_gross_margin - prior_gross_margin) * 100,
                percent_change=((current_gross_margin - prior_gross_margin) / prior_gross_margin) * 100 if prior_gross_margin != 0 else 0,
                category='Margins',
                is_favorable=current_gross_margin > prior_gross_margin
            )
        
        # Operating Margin
        if all(k in current_period for k in ['Revenue', 'OperatingIncome']):
            current_op_margin = current_period['OperatingIncome'] / current_period['Revenue']
            prior_op_margin = prior_period['OperatingIncome'] / prior_period['Revenue']
            
            margin_variances['operating_margin'] = VarianceItem(
                metric_name='Operating Margin',
                current_value=current_op_margin * 100,
                prior_value=prior_op_margin * 100,
                absolute_change=(current_op_margin - prior_op_margin) * 100,
                percent_change=((current_op_margin - prior_op_margin) / prior_op_margin) * 100 if prior_op_margin != 0 else 0,
                category='Margins',
                is_favorable=current_op_margin > prior_op_margin
            )
        
        # Net Margin
        if all(k in current_period for k in ['Revenue', 'NetIncome']):
            current_net_margin = current_period['NetIncome'] / current_period['Revenue']
            prior_net_margin = prior_period['NetIncome'] / prior_period['Revenue']
            
            margin_variances['net_margin'] = VarianceItem(
                metric_name='Net Margin',
                current_value=current_net_margin * 100,
                prior_value=prior_net_margin * 100,
                absolute_change=(current_net_margin - prior_net_margin) * 100,
                percent_change=((current_net_margin - prior_net_margin) / prior_net_margin) * 100 if prior_net_margin != 0 else 0,
                category='Margins',
                is_favorable=current_net_margin > prior_net_margin
            )
        
        return margin_variances
    
    def decompose_roe_change(self,
                            current_period: Dict[str, float],
                            prior_period: Dict[str, float]) -> Dict[str, float]:
        """
        Decompose ROE change using DuPont analysis
        
        Args:
            current_period: Current period data
            prior_period: Prior period data
            
        Returns:
            Dictionary with DuPont components
        """
        decomposition = {}
        
        # Calculate current period components
        if all(k in current_period for k in ['NetIncome', 'Revenue', 'Assets', 'Equity']):
            current_npm = current_period['NetIncome'] / current_period['Revenue']
            current_asset_turnover = current_period['Revenue'] / current_period['Assets']
            current_equity_multiplier = current_period['Assets'] / current_period['Equity']
            current_roe = current_npm * current_asset_turnover * current_equity_multiplier
            
            decomposition['current_roe'] = current_roe
            decomposition['current_npm'] = current_npm
            decomposition['current_asset_turnover'] = current_asset_turnover
            decomposition['current_equity_multiplier'] = current_equity_multiplier
        
        # Calculate prior period components
        if all(k in prior_period for k in ['NetIncome', 'Revenue', 'Assets', 'Equity']):
            prior_npm = prior_period['NetIncome'] / prior_period['Revenue']
            prior_asset_turnover = prior_period['Revenue'] / prior_period['Assets']
            prior_equity_multiplier = prior_period['Assets'] / prior_period['Equity']
            prior_roe = prior_npm * prior_asset_turnover * prior_equity_multiplier
            
            decomposition['prior_roe'] = prior_roe
            decomposition['prior_npm'] = prior_npm
            decomposition['prior_asset_turnover'] = prior_asset_turnover
            decomposition['prior_equity_multiplier'] = prior_equity_multiplier
            
            # Calculate contribution of each component to ROE change
            decomposition['roe_change'] = current_roe - prior_roe
            
            # Impact of NPM change
            decomposition['npm_impact'] = (current_npm - prior_npm) * prior_asset_turnover * prior_equity_multiplier
            
            # Impact of asset turnover change
            decomposition['turnover_impact'] = current_npm * (current_asset_turnover - prior_asset_turnover) * prior_equity_multiplier
            
            # Impact of equity multiplier change
            decomposition['leverage_impact'] = current_npm * current_asset_turnover * (current_equity_multiplier - prior_equity_multiplier)
        
        return decomposition
    
    def analyze_working_capital_changes(self,
                                       current_bs: Dict[str, float],
                                       prior_bs: Dict[str, float]) -> Dict[str, VarianceItem]:
        """
        Analyze changes in working capital components
        
        Args:
            current_bs: Current balance sheet
            prior_bs: Prior balance sheet
            
        Returns:
            Dictionary of working capital variances
        """
        wc_variances = {}
        
        # Calculate working capital
        current_wc = current_bs.get('CurrentAssets', 0) - current_bs.get('CurrentLiabilities', 0)
        prior_wc = prior_bs.get('CurrentAssets', 0) - prior_bs.get('CurrentLiabilities', 0)
        
        wc_variances['working_capital'] = VarianceItem(
            metric_name='Working Capital',
            current_value=current_wc,
            prior_value=prior_wc,
            absolute_change=current_wc - prior_wc,
            percent_change=((current_wc - prior_wc) / abs(prior_wc)) * 100 if prior_wc != 0 else 0,
            category='Working Capital',
            is_favorable=current_wc > prior_wc
        )
        
        # Analyze components
        wc_components = ['Cash', 'AccountsReceivable', 'Inventory', 'AccountsPayable']
        
        for component in wc_components:
            if component in current_bs or component in prior_bs:
                current_val = current_bs.get(component, 0)
                prior_val = prior_bs.get(component, 0)
                
                wc_variances[component.lower()] = VarianceItem(
                    metric_name=component,
                    current_value=current_val,
                    prior_value=prior_val,
                    absolute_change=current_val - prior_val,
                    percent_change=((current_val - prior_val) / abs(prior_val)) * 100 if prior_val != 0 else 0,
                    category='Working Capital',
                    is_favorable=None  # Context-dependent
                )
        
        return wc_variances
    
    def _generate_explanation(self, metric: str, percent_change: float) -> str:
        """
        Generate explanation for variance
        
        Args:
            metric: Metric name
            percent_change: Percentage change
            
        Returns:
            Explanation text
        """
        direction = "increased" if percent_change > 0 else "decreased"
        magnitude = abs(percent_change)
        
        if magnitude < 1:
            size = "slightly"
        elif magnitude < 5:
            size = "modestly"
        elif magnitude < 10:
            size = "notably"
        elif magnitude < 20:
            size = "significantly"
        else:
            size = "substantially"
        
        explanations = {
            'Revenue': f"Revenue {size} {direction} by {magnitude:.1f}%, indicating {'growth' if percent_change > 0 else 'contraction'} in business activity",
            'NetIncome': f"Net income {size} {direction} by {magnitude:.1f}%, reflecting {'improved' if percent_change > 0 else 'reduced'} profitability",
            'OperatingExpenses': f"Operating expenses {size} {direction} by {magnitude:.1f}%, suggesting {'higher' if percent_change > 0 else 'lower'} cost structure",
            'Cash': f"Cash position {size} {direction} by {magnitude:.1f}%, affecting liquidity",
            'Debt': f"Debt levels {size} {direction} by {magnitude:.1f}%, impacting leverage ratios"
        }
        
        return explanations.get(metric, f"{metric} {direction} by {magnitude:.1f}%")
    
    def create_variance_report(self, variances: List[VarianceItem]) -> pd.DataFrame:
        """
        Create variance report DataFrame
        
        Args:
            variances: List of variance items
            
        Returns:
            DataFrame with variance report
        """
        report_data = []
        
        for variance in variances:
            report_data.append({
                'Metric': variance.metric_name,
                'Prior Period': f"${variance.prior_value:,.0f}",
                'Current Period': f"${variance.current_value:,.0f}",
                'Absolute Change': f"${variance.absolute_change:,.0f}",
                'Percent Change': f"{variance.percent_change:.1f}%",
                'Favorable': '✓' if variance.is_favorable else '✗' if variance.is_favorable is False else '-',
                'Explanation': variance.explanation
            })
        
        return pd.DataFrame(report_data)
    
    def identify_key_drivers(self, 
                            variances: List[VarianceItem],
                            threshold: float = 0.1) -> List[VarianceItem]:
        """
        Identify key drivers of change
        
        Args:
            variances: List of variance items
            threshold: Minimum absolute percent change to be considered key
            
        Returns:
            List of key variance drivers
        """
        key_drivers = [
            v for v in variances 
            if abs(v.percent_change) >= threshold * 100
        ]
        
        # Sort by impact magnitude
        key_drivers.sort(key=lambda x: abs(x.absolute_change), reverse=True)
        
        logger.info(f"Identified {len(key_drivers)} key variance drivers")
        return key_drivers